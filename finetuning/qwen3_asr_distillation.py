#!/usr/bin/env python3
"""MAC-SLU teacher/student distillation (no weight quantization)."""
from __future__ import annotations
import argparse, json, os, random, shutil
import numpy as np
import torch
from datasets import load_dataset
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

from distillation_losses import (RepresentationProjector, combine_distillation_losses,
    masked_token_kl, representation_contrastive_loss)
from qwen3_asr_common import freeze_teacher, load_asr, model_metadata, resolve_checkpoint, save_json
from qwen3_asr_sft import (DataCollatorForQwen3ASRFinetuning, MakeEveryCheckpointInferableCallback,
    extract_default_prompt, make_preprocess_fn_prefix_only, patch_outer_forward, save_best_checkpoint,
    save_prompt_txt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_conf", required=True); p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--resume_from", default=""); p.add_argument("--validate_only", action="store_true")
    return p.parse_args()


class ProjectionCheckpointCallback(TrainerCallback):
    def __init__(self, projections): self.projections = projections
    def on_save(self, args, state, control, **kwargs):
        if args.process_index == 0:
            torch.save(self.projections.state_dict(), os.path.join(args.output_dir,
                       f"checkpoint-{state.global_step}", "distillation_projections.pt"))
        return control


class QuantizerCheckpointCallback(TrainerCallback):
    def __init__(self, quantizer, schedule): self.quantizer, self.schedule = quantizer, schedule
    def _quantization_phase(self, state):
        per = self.schedule["distillation_epochs_per_cycle"] + self.schedule["quantization_epochs_per_cycle"]
        return int(state.epoch or 0) % per >= self.schedule["distillation_epochs_per_cycle"]
    def on_step_end(self, args, state, control, **kwargs):
        if self._quantization_phase(state):
            step = self.quantizer.step_counter + 1
            self.quantizer.update_and_project(self.schedule["update_centroids"],
                self.schedule["update_assignments"] and step % self.schedule["recluster_interval_steps"] == 0)
        return control
    def on_save(self, args, state, control, **kwargs):
        if args.process_index == 0:
            torch.save({"current_cycle": int(state.epoch or 0) // (
                self.schedule["distillation_epochs_per_cycle"] + self.schedule["quantization_epochs_per_cycle"]),
                "current_phase": "quantization" if self._quantization_phase(state) else "distillation",
                "current_phase_epoch": int(state.epoch or 0), "quantizer": self.quantizer.state_dict()},
                os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "quantization_state.pt"))
        return control


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher, projections, kd, quantizer=None, schedule=None, **kwargs):
        super().__init__(*args, **kwargs); self.teacher = teacher.to(self.args.device); self.projections = projections.to(self.args.device)
        self.kd, self.quantizer, self.schedule = kd, quantizer, schedule

    def create_optimizer(self):
        super().create_optimizer()
        known = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        extra = [p for p in self.projections.parameters() if p.requires_grad and id(p) not in known]
        if extra: self.optimizer.add_param_group({"params": extra})
        return self.optimizer

    def _quantization_phase(self):
        if not self.schedule: return False
        epoch = int(self.state.epoch or 0)
        per = self.schedule["distillation_epochs_per_cycle"] + self.schedule["quantization_epochs_per_cycle"]
        return epoch % per >= self.schedule["distillation_epochs_per_cycle"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        quant_phase = self._quantization_phase()
        outputs = model(**inputs, output_hidden_states=not quant_phase)
        if quant_phase:
            loss = outputs.loss
        else:
            self.teacher.eval()
            teacher_inputs = {k: v for k, v in inputs.items()}
            with torch.no_grad():
                teacher_outputs = self.teacher(**teacher_inputs, output_hidden_states=True)
            kl = masked_token_kl(outputs.logits, teacher_outputs.logits, inputs["labels"], self.kd["temperature"])
            contrastive = representation_contrastive_loss(
                outputs.hidden_states, teacher_outputs.hidden_states, inputs["labels"],
                self.kd["student_layers"], self.kd["teacher_layers"], self.projections,
                self.kd["contrastive_temperature"], self.kd.get("allow_batch_size_one_contrastive", False))
            loss = combine_distillation_losses(outputs.loss, kl, contrastive,
                ce_weight=self.kd["ce_weight"], kl_weight=self.kd["kl_weight"],
                contrastive_weight=self.kd["contrastive_weight"])
        return (loss, outputs) if return_outputs else loss

def run(quantized=False):
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with open(args.train_conf, encoding="utf-8") as f: conf = json.load(f)
    training_conf, model_conf = dict(conf[0]), conf[1]; kd = model_conf["distillation"]
    teacher_conf = model_conf.get("teacher", {})
    vocabulary_pruning = model_conf.get("vocabulary_pruning", {})
    if vocabulary_pruning.get("enabled", False):
        if not teacher_conf.get("exp_dir"):
            raise ValueError("Vocabulary-pruned distillation requires teacher.exp_dir")
        teacher_path = resolve_checkpoint(teacher_conf["exp_dir"],
                                          teacher_conf.get("checkpoint_mode", "latest"))
    else:
        teacher_path = teacher_conf.get("teacher_source_checkpoint", "")
        if not teacher_path:
            raise ValueError("Full-vocabulary distillation requires teacher.teacher_source_checkpoint")
    if not kd.get("enabled"): raise ValueError("distillation.enabled must be true")
    if not quantized and "quantization" in model_conf:
        raise ValueError("Distillation-only configuration must not contain quantization")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    student_wrapper, student, processor = load_asr(model_conf["model_path"], dtype)
    teacher_wrapper, teacher, teacher_processor = load_asr(teacher_path, dtype)
    patch_outer_forward(student)
    vocabulary_mapping = None
    if vocabulary_pruning.get("enabled", False):
        from vocabulary_pruning import apply_structural_vocabulary_pruning, load_vocabulary_mapping
        vocabulary_mapping = load_vocabulary_mapping(vocabulary_pruning["manifest"])
        apply_structural_vocabulary_pruning(student, vocabulary_mapping)
        apply_structural_vocabulary_pruning(teacher, vocabulary_mapping)
    # A PEFT teacher already exposes CausalLM.forward; dense Qwen outer models need the repository patch.
    if hasattr(teacher, "thinker"):
        patch_outer_forward(teacher)
    freeze_teacher(teacher)
    metadata = {"teacher": model_metadata(teacher, teacher_processor, teacher_path),
                "student": model_metadata(student, processor, model_conf["model_path"])}
    if metadata["teacher"]["vocabulary_size"] != metadata["student"]["vocabulary_size"]:
        raise ValueError(f"Teacher/student tokenizer vocabulary sizes differ: {metadata}")
    teacher_vocab = teacher_processor.tokenizer.get_vocab()
    student_vocab = processor.tokenizer.get_vocab()
    if teacher_vocab != student_vocab:
        mismatch = next((token for token in set(teacher_vocab) | set(student_vocab)
                         if teacher_vocab.get(token) != student_vocab.get(token)), "unknown")
        raise ValueError(f"Teacher/student token IDs are incompatible at token {mismatch!r}: "
                         f"teacher={teacher_vocab.get(mismatch)}, student={student_vocab.get(mismatch)}")
    metadata["tokenizer_vocabulary_compatible"] = True
    save_json(os.path.join(args.output_dir, "teacher_metadata.json"), metadata)
    student.generation_config = GenerationConfig.from_model_config(student.config)
    raw = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})
    ds = raw.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds:
        ds[split] = ds[split].remove_columns([x for x in ds[split].column_names if x not in keep])
    collator = DataCollatorForQwen3ASRFinetuning(processor, int(model_conf.get("sr", 16000)))
    if vocabulary_mapping is not None:
        from vocabulary_pruning import VocabularyRemappingCollator
        collator = VocabularyRemappingCollator(collator, vocabulary_mapping)
    sample = collator([ds["train"][0]])
    device = next(student.parameters()).device
    sample = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None) for k, v in sample.items()}
    with torch.no_grad():
        so = student(**sample, output_hidden_states=True); to = teacher(**sample, output_hidden_states=True)
    # This explicit preflight validates logits, labels and token-aligned hidden states.
    masked_token_kl(so.logits, to.logits, sample["labels"], kd["temperature"])
    sd = [so.hidden_states[i].size(-1) for i in kd["student_layers"]]
    td = [to.hidden_states[i].size(-1) for i in kd["teacher_layers"]]
    projections = RepresentationProjector(sd, td, kd.get("projection_dimension")).to(device)
    metadata.update({"student_logit_shape": list(so.logits.shape), "teacher_logit_shape": list(to.logits.shape),
                     "labels_shape": list(sample["labels"].shape), "student_hidden_dimensions": sd,
                     "teacher_hidden_dimensions": td, "hidden_states_token_aligned": True})
    save_json(os.path.join(args.output_dir, "teacher_metadata.json"), metadata)
    del so, to, sample
    if args.validate_only:
        print(json.dumps(metadata, indent=2)); return
    schedule = model_conf.get("quantization") if quantized else None; quantizer = None
    if quantized:
        from kmeans_quantizer import ScalarKMeansQuantizer
        expected = schedule["num_cycles"] * (schedule["distillation_epochs_per_cycle"] + schedule["quantization_epochs_per_cycle"])
        if float(training_conf["num_train_epochs"]) != expected:
            raise ValueError(f"num_train_epochs={training_conf['num_train_epochs']} but quantization schedule requires {expected}")
        quantizer = ScalarKMeansQuantizer(student, schedule["bit_width"], schedule["include_patterns"],
                                          schedule["exclude_patterns"], args.seed)
        if args.resume_from and os.path.isfile(os.path.join(args.resume_from, "quantization_state.pt")):
            quantizer.load_state_dict(torch.load(os.path.join(args.resume_from, "quantization_state.pt"),
                                                  map_location="cpu", weights_only=False)["quantizer"])
            quantizer.update_and_project(False, False)
    training_conf["run_name"] = os.path.basename(args.output_dir)
    targs = TrainingArguments(output_dir=args.output_dir, do_eval=True, bf16=use_bf16,
                              fp16=torch.cuda.is_available() and not use_bf16, **training_conf)
    prompt = extract_default_prompt(ds["train"]); callbacks = [MakeEveryCheckpointInferableCallback(processor, student, prompt),
                                                                ProjectionCheckpointCallback(projections)]
    if args.resume_from and os.path.isfile(os.path.join(args.resume_from, "distillation_projections.pt")):
        projections.load_state_dict(torch.load(os.path.join(args.resume_from, "distillation_projections.pt"),
                                                map_location="cpu", weights_only=True))
    if quantizer is not None: callbacks.append(QuantizerCheckpointCallback(quantizer, schedule))
    trainer = DistillationTrainer(model=student, teacher=teacher, projections=projections, kd=kd,
        quantizer=quantizer, schedule=schedule, args=targs, train_dataset=ds["train"],
        eval_dataset=ds["validation"], data_collator=collator,
        processing_class=processor.tokenizer, callbacks=callbacks)
    os.makedirs(args.output_dir, exist_ok=True); save_json(os.path.join(args.output_dir, "train_conf.json"), conf)
    save_json(os.path.join(args.output_dir, "distillation_runtime_conf.json"), {"seed": args.seed,
              "teacher_path": teacher_path, "distillation": kd, "quantization": schedule})
    processor.save_pretrained(args.output_dir)
    trainer.train(resume_from_checkpoint=args.resume_from or None)
    best = trainer.state.best_model_checkpoint
    save_best_checkpoint(best, args.output_dir, processor, student, prompt)
    if best:
        projection_file = os.path.join(best, "distillation_projections.pt")
        if os.path.isfile(projection_file): shutil.copy2(projection_file, os.path.join(args.output_dir, "checkpoint-best"))
    if quantizer is not None: torch.save(quantizer.state_dict(), os.path.join(args.output_dir, "quantizer_state.pt"))


if __name__ == "__main__": run(False)
