# coding=utf-8
"""Qwen3-ASR SFT with auxiliary domain/intent prototype prediction."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, GenerationConfig, Trainer, TrainerCallback, TrainingArguments

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.prototype_prompt_utils import (  # noqa: E402
    build_training_candidate_labels,
    format_domain_intent_candidates,
    get_prompt_template,
)
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def save_prompt_txt(save_dir: str, prompt: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt or "")


def save_best_checkpoint(best_src: str, output_dir: str, processor=None, model=None, default_prompt: str = "", best_ckpt_name: str = "checkpoint-best"):
    if not best_src or not os.path.isdir(best_src):
        print("[best] checkpoint-best not created: no best_model_checkpoint was selected.")
        return
    best_ckpt_dir = os.path.join(output_dir, best_ckpt_name)
    if os.path.exists(best_ckpt_dir):
        shutil.rmtree(best_ckpt_dir)
    shutil.copytree(best_src, best_ckpt_dir)
    if processor is not None:
        processor.save_pretrained(best_ckpt_dir)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.save_pretrained(best_ckpt_dir)
    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(best_ckpt_dir)
    save_prompt_txt(best_ckpt_dir, default_prompt)
    print(f"[best] Saved best checkpoint from {best_src} to {best_ckpt_dir}")


from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig  # noqa: E402
from qwen_asr.core.transformers_backend.modeling_qwen3_asr_prototype import (  # noqa: E402
    Qwen3ASRPrototypeForConditionalGeneration,
)
from slu_decoding.prototypes import MACSLULabelSchema, PrototypeIndex  # noqa: E402


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR prototype finetuning")
    p.add_argument("--train_conf", type=str, required=True)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="dev.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-prototype-finetuning-out")
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    return p.parse_args()


def load_train_conf(train_conf_path: str) -> List[Dict[str, Any]]:
    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or len(cfg) != 2 or not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("train_conf must be [training_args, model_args]")
    return cfg


def _schema_intent_labels(schema: MACSLULabelSchema) -> List[str]:
    labels: List[str] = []
    seen = set()
    for domain in schema.valid_domains():
        for intent in schema.valid_intents(domain):
            if intent and intent not in seen:
                labels.append(intent)
                seen.add(intent)
    return labels


def resolve_prototype_json_path(prototype_conf: Dict[str, Any]) -> str:
    """Return the prototype JSON path shared with qwen3_asr_test_prototype.py.

    ``prototype_json`` is the preferred key because it is the same artifact passed
    to ``finetuning/qwen3_asr_test_prototype.py --prototype_json``.  ``init_path``
    is kept as a backward-compatible alias for the previous draft config.
    """
    return str(prototype_conf.get("prototype_json") or prototype_conf.get("init_path") or "")


def _merge_prototype_schema(label_schema: MACSLULabelSchema, prototype_index: Optional[PrototypeIndex]) -> None:
    if prototype_index is None:
        return
    proto_schema = prototype_index.data.get("label_schema", {}) or {}
    for domain in proto_schema.get("domains", []) or []:
        label_schema.domains.add(str(domain))
    for domain, intents in (proto_schema.get("domain2intents", {}) or {}).items():
        for intent in intents or []:
            label_schema.add_domain_intent(str(domain), str(intent))
    for domain_intent, slots in (proto_schema.get("domain_intent2slot_keys", {}) or {}).items():
        parts = str(domain_intent).split("|||")
        if len(parts) >= 2:
            for slot in slots or []:
                label_schema.add_slot_key(parts[0], parts[1], str(slot))


def _prototype_labels(index: Optional[PrototypeIndex], kind: str) -> List[str]:
    if index is None:
        return []
    section = {"domain": index.domain, "intent": index.intent}.get(kind, {})
    labels: List[str] = []
    seen = set()
    for key, item in section.items():
        if not isinstance(item, dict):
            continue
        label = str((item.get("meta", {}) or {}).get("label", key)).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def maybe_add_empty_label(labels: List[str], prototype_conf: Dict[str, Any]) -> List[str]:
    if not bool(prototype_conf.get("add_empty_prototype", True)):
        return labels
    empty_label = str(prototype_conf.get("empty_label", "__empty__"))
    if empty_label and empty_label not in labels:
        return [empty_label] + labels
    return labels


def build_prototype_label_maps(
    prototype_conf: Dict[str, Any],
    prototype_index: Optional[PrototypeIndex] = None,
) -> tuple[MACSLULabelSchema, List[str], List[str], Dict[str, int], Dict[str, int]]:
    labels_path = prototype_conf.get("labels_path", "")
    schema_path = prototype_conf.get("schema_path", "")
    schema = MACSLULabelSchema(labels_path=labels_path, schema_path=schema_path)
    _merge_prototype_schema(schema, prototype_index)
    domains = maybe_add_empty_label(
        list(prototype_conf.get("domain_labels", []) or _prototype_labels(prototype_index, "domain") or schema.valid_domains()),
        prototype_conf,
    )
    intents = maybe_add_empty_label(
        list(prototype_conf.get("intent_labels", []) or _prototype_labels(prototype_index, "intent") or _schema_intent_labels(schema)),
        prototype_conf,
    )
    if not domains:
        raise ValueError("No domain labels found. Set prototype.labels_path, prototype.domain_labels, or prototype.prototype_json.")
    if not intents:
        raise ValueError("No intent labels found. Set prototype.labels_path, prototype.intent_labels, or prototype.prototype_json.")
    return schema, domains, intents, {x: i for i, x in enumerate(domains)}, {x: i for i, x in enumerate(intents)}


def _prototype_vectors_by_label(index: PrototypeIndex, kind: str) -> Dict[str, List[float]]:
    section = {"domain": index.domain, "intent": index.intent}.get(kind, {})
    out: Dict[str, List[float]] = {}
    for key, item in section.items():
        if not isinstance(item, dict):
            continue
        label = str((item.get("meta", {}) or {}).get("label", key))
        vec = item.get("vector") or []
        if label and vec and label not in out:
            out[label] = [float(x) for x in vec]
    return out


def initialize_prototype_embeddings(
    model,
    prototype_conf: Dict[str, Any],
    domain_labels: List[str],
    intent_labels: List[str],
    prototype_index: Optional[PrototypeIndex] = None,
) -> None:
    init_path = resolve_prototype_json_path(prototype_conf)
    head = getattr(model.thinker, "prototype_head", None)
    if head is None:
        raise RuntimeError("prototype_head is not enabled")
    if not init_path:
        print("[prototype] no prototype_json set; using random prototype initialization")
        return
    index = prototype_index or PrototypeIndex.load(init_path)
    domain_vecs = _prototype_vectors_by_label(index, "domain")
    intent_vecs = _prototype_vectors_by_label(index, "intent")

    def copy_vectors(weight: torch.Tensor, labels: List[str], vecs: Dict[str, List[float]], kind: str):
        dim = weight.size(1)
        copied = 0
        with torch.no_grad():
            for idx, label in enumerate(labels):
                vec = vecs.get(label)
                if not vec:
                    continue
                if len(vec) != dim:
                    raise ValueError(f"{kind} prototype dimension mismatch for {label}: {len(vec)} != {dim}")
                weight[idx].copy_(torch.tensor(vec, dtype=weight.dtype, device=weight.device))
                copied += 1
        print(f"[prototype] initialized {copied}/{len(labels)} {kind} prototypes from {init_path}")

    copy_vectors(head.domain_prototypes.weight, domain_labels, domain_vecs, "domain")
    copy_vectors(head.intent_prototypes.weight, intent_labels, intent_vecs, "intent")
    if prototype_conf.get("freeze_initial_prototypes", False):
        head.domain_prototypes.weight.requires_grad = False
        head.intent_prototypes.weight.requires_grad = False


def label_ids(labels: List[str], label2id: Dict[str, int]) -> List[int]:
    return [int(label2id[x]) for x in labels if x in label2id]


def multi_hot(label_ids: List[int], size: int) -> List[float]:
    vec = [0.0] * int(size)
    for idx in label_ids:
        if 0 <= int(idx) < size:
            vec[int(idx)] = 1.0
    return vec


def make_preprocess_fn_prototype(processor, schema, domain2id, intent2id, prototype_conf, seed: int):
    k = int(prototype_conf.get("k", 5))
    template = get_prompt_template(prototype_conf)
    domain_aware = bool(prototype_conf.get("domain_aware_intents", True))

    def _preprocess(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        rng = random.Random(seed + int(idx))
        prompt = ex.get("prompt", "")
        domains, intents, gold_domains, gold_intents = build_training_candidate_labels(
            ex, schema=schema, k=k, rng=rng, domain_aware_intents=domain_aware
        )
        domain_label_ids = label_ids(gold_domains, domain2id)
        intent_label_ids = label_ids(gold_intents, intent2id)
        empty_label = str(prototype_conf.get("empty_label", "__empty__"))
        if bool(prototype_conf.get("add_empty_prototype", True)):
            if not domain_label_ids and empty_label in domain2id:
                domain_label_ids = [domain2id[empty_label]]
                if empty_label not in domains:
                    domains = [empty_label] + domains
            if not intent_label_ids and empty_label in intent2id:
                intent_label_ids = [intent2id[empty_label]]
                if empty_label not in intents:
                    intents = [empty_label] + intents
        augmented_prompt = format_domain_intent_candidates(prompt, domains, intents, **template)
        prefix_msgs = build_prefix_messages(augmented_prompt, None)
        prefix_text = processor.apply_chat_template([prefix_msgs], add_generation_prompt=True, tokenize=False)[0]
        return {
            "prompt": augmented_prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
            "domain_label_ids": domain_label_ids,
            "intent_label_ids": intent_label_ids,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRPrototypeFinetuning:
    processor: Any
    sampling_rate: int = 16000
    num_domains: int = 0
    num_intents: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]
        full_inputs = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix_inputs = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_inputs["labels"] = labels
        full_inputs["domain_labels"] = torch.tensor(
            [multi_hot([int(x) for x in f.get("domain_label_ids", [])], self.num_domains) for f in features],
            dtype=torch.float32,
        )
        full_inputs["intent_labels"] = torch.tensor(
            [multi_hot([int(x) for x in f.get("intent_label_ids", [])], self.num_intents) for f in features],
            dtype=torch.float32,
        )
        full_inputs["prototype_prefix_lengths"] = torch.tensor(prefix_lens, dtype=torch.long)
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, processor, model=None, default_prompt: str = ""):
        self.processor = processor
        self.model = model
        self.default_prompt = default_prompt

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.save_pretrained(ckpt_dir)
        if self.model is not None and getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.save_pretrained(ckpt_dir)
        save_prompt_txt(ckpt_dir, self.default_prompt)
        return control


def main():
    args_cli = parse_args()
    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

    train_conf = load_train_conf(args_cli.train_conf)
    training_args_conf, model_args_conf = train_conf
    training_args_conf = dict(training_args_conf)
    model_args_conf = dict(model_args_conf)
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    prototype_conf["enabled"] = True

    prototype_json = resolve_prototype_json_path(prototype_conf)
    prototype_index = PrototypeIndex.load(prototype_json) if prototype_json else None
    if prototype_json:
        prototype_conf["prototype_json"] = prototype_json
        prototype_conf.pop("init_path", None)
    schema, domain_labels, intent_labels, domain2id, intent2id = build_prototype_label_maps(prototype_conf, prototype_index)
    if prototype_index is not None:
        prototype_conf.setdefault("pooling", prototype_index.data.get("prototype_pooling", prototype_conf.get("pooling", "mean_pooling")))
        prototype_conf.setdefault("prototype_source", prototype_index.prototype_source)
        prototype_conf.setdefault("embedding_backend", prototype_index.embedding_backend)
    prototype_conf["domain_labels"] = domain_labels
    prototype_conf["intent_labels"] = intent_labels
    prototype_conf["num_domains"] = len(domain_labels)
    prototype_conf["num_intents"] = len(intent_labels)
    model_args_conf["prototype"] = prototype_conf
    train_conf = [training_args_conf, model_args_conf]

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")
    sr = int(model_args_conf.get("sr", 16000))
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    config = Qwen3ASRConfig.from_pretrained(model_path)
    config.thinker_config.prototype_config = prototype_conf
    lora_type = model_args_conf.get("lora_type", "default")
    if lora_type == "qlora":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(model_path, config=config, dtype=dtype, quantization_config=bnb_config, device_map=None)
    else:
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(model_path, config=config, dtype=dtype, device_map=None)
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    initialize_prototype_embeddings(model, prototype_conf, domain_labels, intent_labels, prototype_index)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        if lora_type not in ["default", "qlora"]:
            raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")
        lora_config = dict(lora_config)
        modules_to_save = list(lora_config.get("modules_to_save", []) or [])
        if "prototype_head" not in modules_to_save:
            modules_to_save.append("prototype_head")
        lora_config["modules_to_save"] = modules_to_save
        model_args_conf["lora_config"] = lora_config
        train_conf[1]["lora_config"] = lora_config
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_config))
        model.print_trainable_parameters()
    else:
        print("Full Finetuning")

    if training_args_conf.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    from datasets import load_dataset

    raw_ds = load_dataset("json", data_files={"train": args_cli.train_file, "validation": args_cli.eval_file})
    ds = raw_ds.map(make_preprocess_fn_prototype(processor, schema, domain2id, intent2id, prototype_conf, seed), with_indices=True, num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text", "domain_label_ids", "intent_label_ids"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = ds["train"][0]["prompt"] if len(ds["train"]) else ""
    collator = DataCollatorForQwen3ASRPrototypeFinetuning(
        processor=processor,
        sampling_rate=sr,
        num_domains=len(domain_labels),
        num_intents=len(intent_labels),
    )
    training_args_conf["run_name"] = os.path.basename(args_cli.output_dir)
    if model_args_conf.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = model_args_conf["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = str(model_args_conf.get("wandb_log_model", "false")).lower()
    training_args = TrainingArguments(output_dir=args_cli.output_dir, do_eval=True, bf16=use_bf16, fp16=not use_bf16, **training_args_conf)
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(processor=processor, model=model, default_prompt=default_prompt)],
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    if trainer.args.process_index == 0:
        with open(os.path.join(training_args.output_dir, "train_conf.json"), "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)
    processor.save_pretrained(training_args.output_dir)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.save_pretrained(training_args.output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(training_args.output_dir)

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)
    if trainer.args.process_index == 0:
        save_best_checkpoint(getattr(trainer.state, "best_model_checkpoint", None), training_args.output_dir, processor=processor, model=model, default_prompt=default_prompt)
        save_prompt_txt(training_args.output_dir, default_prompt)


if __name__ == "__main__":
    main()
