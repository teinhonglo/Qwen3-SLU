# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import random

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModelForCausalLM

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "chosen": str(ex["chosen"]),
            "rejected": str(ex["rejected"]),
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRSimPO:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = []
        prefix_texts = []
        responses = []
        pair_ids = []

        for pair_id, feature in enumerate(features):
            for candidate in (feature["chosen"], feature["rejected"]):
                audio_paths.append(feature["audio"])
                prefix_texts.append(feature["prefix_text"])
                responses.append(candidate)
                pair_ids.append(pair_id)

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + resp + eos for pfx, resp in zip(prefix_texts, responses)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        full_inputs["simpo_pair_ids"] = torch.tensor(pair_ids, dtype=torch.long)
        return full_inputs


def extract_default_prompt(dataset) -> str:
    prompts = []
    for ex in dataset:
        p = str(ex.get("prompt", "") or "").strip()
        if p:
            prompts.append(p)

    if not prompts:
        return ""

    first = prompts[0]
    if any(p != first for p in prompts[1:]):
        print("[warn] Multiple prompt values found in train set; using the first non-empty prompt for prompt.txt")
    return first


def save_prompt_txt(save_dir: str, prompt: str):
    os.makedirs(save_dir, exist_ok=True)
    prompt_path = os.path.join(save_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt or "")

class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

class SimPOTrainer(CastFloatInputsTrainer):
    def __init__(
        self,
        *args,
        simpo_beta: float = 2.0,
        simpo_gamma: float = 1.0,
        simpo_length_normalization: bool = True,
        simpo_sft_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not simpo_length_normalization:
            raise ValueError("SimPO length normalization is required and must remain enabled.")
        self.simpo_beta = float(simpo_beta)
        self.simpo_gamma = float(simpo_gamma)
        self.simpo_sft_loss_weight = float(simpo_sft_loss_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pair_ids = inputs.pop("simpo_pair_ids")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        token_log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        gathered = token_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = (gathered * valid_mask.float()).sum(dim=-1)
        response_lens = valid_mask.sum(dim=-1).clamp_min(1).float()
        avg_log_probs = seq_log_probs / response_lens

        losses = []
        chosen_scores = []
        rejected_scores = []
        chosen_lens = []
        rejected_lens = []
        for pair_id in pair_ids.unique(sorted=True).tolist():
            idx = (pair_ids == pair_id).nonzero(as_tuple=False).flatten()
            if idx.numel() != 2:
                raise ValueError(f"Each SimPO pair must contain exactly chosen/rejected rows, got {idx.numel()}")
            chosen_idx, rejected_idx = idx[0], idx[1]
            chosen_avg_logp = avg_log_probs[chosen_idx]
            rejected_avg_logp = avg_log_probs[rejected_idx]
            logits_margin = self.simpo_beta * chosen_avg_logp - self.simpo_beta * rejected_avg_logp - self.simpo_gamma
            losses.append(-torch.nn.functional.logsigmoid(logits_margin))
            chosen_scores.append(chosen_avg_logp.detach())
            rejected_scores.append(rejected_avg_logp.detach())
            chosen_lens.append(response_lens[chosen_idx].detach())
            rejected_lens.append(response_lens[rejected_idx].detach())

        loss = torch.stack(losses).mean()
        if self.simpo_sft_loss_weight > 0:
            chosen_mask = torch.zeros_like(seq_log_probs, dtype=torch.bool)
            for pair_id in pair_ids.unique(sorted=True).tolist():
                idx = (pair_ids == pair_id).nonzero(as_tuple=False).flatten()
                chosen_mask[idx[0]] = True
            sft_loss = -(seq_log_probs[chosen_mask] / response_lens[chosen_mask]).mean()
            loss = loss + self.simpo_sft_loss_weight * sft_loss

        if chosen_scores:
            chosen_t = torch.stack(chosen_scores)
            rejected_t = torch.stack(rejected_scores)
            margin_t = chosen_t - rejected_t
            self.log({
                "simpo/chosen_avg_logp": chosen_t.mean().item(),
                "simpo/rejected_avg_logp": rejected_t.mean().item(),
                "simpo/margin": margin_t.mean().item(),
                "simpo/reward_accuracy": (margin_t > 0).float().mean().item(),
                "simpo/chosen_response_length": torch.stack(chosen_lens).float().mean().item(),
                "simpo/rejected_response_length": torch.stack(rejected_lens).float().mean().item(),
            })
        return (loss, outputs) if return_outputs else loss


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, processor, model=None, default_prompt: str = ""):
        self.processor = processor
        self.model = model
        self.default_prompt = default_prompt

    def _save_infer_files(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        self.processor.save_pretrained(save_dir)

        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.save_pretrained(save_dir)

        if self.model is not None and getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.save_pretrained(save_dir)

        save_prompt_txt(save_dir, self.default_prompt)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        self._save_infer_files(ckpt_dir)
        return control

def save_best_checkpoint(
    best_src: str,
    output_dir: str,
    processor=None,
    model=None,
    default_prompt: str = "",
    best_ckpt_name: str = "checkpoint-best",
):
    if not best_src or not os.path.isdir(best_src):
        print(
            "[best] checkpoint-best not created: no best_model_checkpoint was selected. "
            "Please make sure evaluation runs and load_best_model_at_end=true."
        )
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


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SimPO preference finetuning")

    # Paths
    p.add_argument("--train_conf", type=str, default="",
                   help=("JSON config path with format: [training_args, model_args]. "
                         "If omitted, load train_conf.json from --init_model_dir/its parent."))
    p.add_argument('--seed', type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="dev.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")
    p.add_argument("--init_model_dir", type=str, default="",
                   help="Optional source experiment/checkpoint to initialize SimPO training from")
    p.add_argument("--auto_latest_init_checkpoint", action="store_true",
                   help="Resolve --init_model_dir to its latest checkpoint-* before loading")
    p.add_argument("--auto_best_init_checkpoint", action="store_true",
                   help="Resolve --init_model_dir to checkpoint-best before loading")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # SimPO hyperparameters. Length normalization is intentionally required:
    # reward = response-token logp_sum / response_length.
    p.add_argument("--simpo_beta", type=float, default=None,
                   help="Reward-difference scale. Recommended sweep: 2.0, 2.5, 3.0")
    p.add_argument("--simpo_gamma", type=float, default=None,
                   help="Target reward margin. Recommended sweep: 0.5, 1.0, 1.5")
    p.add_argument("--simpo_gamma_beta_ratio", type=float, default=None,
                   help=("Optional SimPO paper-style normalized margin. "
                         "When --simpo_gamma is omitted, gamma = beta * gamma_beta_ratio."))
    p.add_argument("--simpo_sft_loss_weight", type=float, default=None,
                   help="Optional auxiliary SFT loss on chosen responses; default comes from config or 0.0")
    p.add_argument("--disable_length_normalization", action="store_true",
                   help="Rejected by design. SimPO reward must use response length-normalized log probability.")

    return p.parse_args()

def resolve_init_model_dir(init_model_dir: str, auto_latest: bool = False, auto_best: bool = False) -> str:
    init_model_dir = (init_model_dir or "").strip()
    if not init_model_dir:
        return ""
    if auto_latest and auto_best:
        raise ValueError("Only one of --auto_latest_init_checkpoint and --auto_best_init_checkpoint may be set")
    if auto_best:
        best_dir = os.path.join(init_model_dir, "checkpoint-best")
        if not os.path.isdir(best_dir):
            raise FileNotFoundError(f"checkpoint-best not found under init_model_dir: {best_dir}")
        return best_dir
    if auto_latest:
        latest = find_latest_checkpoint(init_model_dir)
        if latest is None:
            print(f"[init] no checkpoint-* found under {init_model_dir}; using init_model_dir directly")
            return init_model_dir
        return latest
    return init_model_dir


def find_train_conf_path(*dirs: str) -> str:
    for d in dirs:
        d = (d or "").strip()
        if not d:
            continue
        candidates = [
            os.path.join(d, "train_conf.json"),
            os.path.join(os.path.dirname(d), "train_conf.json"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    return ""


def resolve_train_conf_path(train_conf_path: str, raw_init_model_dir: str, resolved_init_model_dir: str) -> str:
    train_conf_path = (train_conf_path or "").strip()
    if train_conf_path:
        return train_conf_path

    inferred = find_train_conf_path(raw_init_model_dir, resolved_init_model_dir)
    if inferred:
        print(f"[init] using train_conf from source model: {inferred}")
        return inferred

    raise ValueError(
        "--train_conf is required unless --init_model_dir points to an experiment "
        "or checkpoint whose experiment root contains train_conf.json"
    )


def load_train_conf(train_conf_path: str) -> Optional[List[Dict[str, Any]]]:
    if not train_conf_path:
        return None

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf must be a list in format: [training_args, model_args]")

    training_args, model_args = cfg
    if not isinstance(training_args, dict) or not isinstance(model_args, dict):
        raise ValueError("train_conf entries must both be dictionaries")
    return [training_args, model_args]

def main():
    args_cli = parse_args()

    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    init_model_dir = resolve_init_model_dir(
        args_cli.init_model_dir,
        auto_latest=args_cli.auto_latest_init_checkpoint,
        auto_best=args_cli.auto_best_init_checkpoint,
    )
    if init_model_dir:
        print(f"[init] initialize SimPO training from: {init_model_dir}")

    train_conf_path = resolve_train_conf_path(
        args_cli.train_conf,
        raw_init_model_dir=args_cli.init_model_dir,
        resolved_init_model_dir=init_model_dir,
    )
    train_conf = load_train_conf(train_conf_path)
    if train_conf is None:
        raise ValueError("Unable to load train_conf")

    training_args_conf, model_args_conf = train_conf
    training_args_conf = dict(training_args_conf)

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, chosen, rejected, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(model_args_conf.get("sr", 16000))
    simpo_conf = dict(model_args_conf.get("simpo_config", {}))
    simpo_beta = float(args_cli.simpo_beta if args_cli.simpo_beta is not None else simpo_conf.get("beta", 2.0))
    if args_cli.simpo_gamma is not None:
        simpo_gamma = float(args_cli.simpo_gamma)
        simpo_gamma_beta_ratio = simpo_gamma / simpo_beta if simpo_beta else None
    else:
        configured_gamma = simpo_conf.get("gamma", None)
        simpo_gamma_beta_ratio = (
            args_cli.simpo_gamma_beta_ratio
            if args_cli.simpo_gamma_beta_ratio is not None
            else simpo_conf.get("gamma_beta_ratio", None)
        )
        if configured_gamma is not None:
            simpo_gamma = float(configured_gamma)
        elif simpo_gamma_beta_ratio is not None:
            simpo_gamma = simpo_beta * float(simpo_gamma_beta_ratio)
        else:
            simpo_gamma = 1.0
            simpo_gamma_beta_ratio = simpo_gamma / simpo_beta if simpo_beta else None
    simpo_sft_loss_weight = float(
        args_cli.simpo_sft_loss_weight
        if args_cli.simpo_sft_loss_weight is not None
        else simpo_conf.get("sft_loss_weight", 0.0)
    )
    simpo_length_normalization = bool(simpo_conf.get("length_normalization", True)) and not args_cli.disable_length_normalization
    if not simpo_length_normalization:
        raise ValueError("Length-normalized response log probability is required for SimPO.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    train_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # LoRA / QLoRA: mirror finetuning/qwen3_asr_test.py loading order.
    # First load the base model from model_args.model_path, then attach the
    # source adapter checkpoint. This keeps SimPO continuation consistent with
    # how the same checkpoint is used for inference.
    lora_config = model_args_conf.get("lora_config", None)
    lora_type = model_args_conf.get("lora_type", "default")
    init_is_lora_adapter = bool(init_model_dir and os.path.isfile(os.path.join(init_model_dir, "adapter_config.json")))
    if init_is_lora_adapter and not lora_config:
        raise ValueError(
            "init_model_dir looks like a LoRA adapter checkpoint, but train_conf has no lora_config. "
            "Use the source model's train_conf.json or pass a LoRA train_conf."
        )

    if lora_config:
        if lora_type not in ["default", "qlora"]:
            raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")

        print(f"LoRA Finetuning {lora_type}")
        from_pretrained_kwargs = {
            "dtype": train_dtype,
            "device_map": None,
            "attn_implementation": "flash_attention_2",
        }
        if lora_type == "qlora":
            from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print(f"[init] loading base model from: {model_path}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            **from_pretrained_kwargs,
        )
        model = asr_wrapper.model
        processor = asr_wrapper.processor

        patch_outer_forward(model)
        model.generation_config = GenerationConfig.from_model_config(model.config)

        if init_model_dir:
            print(f"[init] loading trainable LoRA adapter from: {init_model_dir}")
            model = PeftModelForCausalLM.from_pretrained(
                model,
                init_model_dir,
                torch_dtype=train_dtype,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **lora_config
            )
            model = get_peft_model(model, peft_config)
        print("="*100)
        model.print_trainable_parameters()
        print("="*100)
    else:
        load_model_path = init_model_dir or model_path
        if init_model_dir:
            print(f"[init] loading full model weights from: {load_model_path}")
        print("Full Finetuning")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            load_model_path,
            dtype=train_dtype,
            device_map=None,
        )
        model = asr_wrapper.model
        processor = asr_wrapper.processor

        patch_outer_forward(model)
        model.generation_config = GenerationConfig.from_model_config(model.config)

    if training_args_conf["gradient_checkpointing"]:
        model.config.use_cache = False
        model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            "validation": args_cli.eval_file,
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "chosen", "rejected", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = extract_default_prompt(ds["train"])

    collator = DataCollatorForQwen3ASRSimPO(processor=processor, sampling_rate=sr)

    training_args_conf["run_name"] = os.path.basename(args_cli.output_dir)
    if model_args_conf.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = model_args_conf["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = str(model_args_conf.get("wandb_log_model", "false")).lower()

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        do_eval=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        **training_args_conf
    )

    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        simpo_beta=simpo_beta,
        simpo_gamma=simpo_gamma,
        simpo_length_normalization=simpo_length_normalization,
        simpo_sft_loss_weight=simpo_sft_loss_weight,
        callbacks=[
            MakeEveryCheckpointInferableCallback(
                processor=processor,
                model=model,
                default_prompt=default_prompt,
            ),
        ],
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if train_conf is not None and trainer.args.process_index == 0:
        saved_train_conf = os.path.join(training_args.output_dir, "train_conf.json")
        with open(saved_train_conf, "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)
        init_info = {
            "train_conf_path": train_conf_path,
            "raw_init_model_dir": args_cli.init_model_dir,
            "resolved_init_model_dir": init_model_dir,
            "simpo_beta": simpo_beta,
            "simpo_gamma": simpo_gamma,
            "simpo_gamma_beta_ratio": simpo_gamma_beta_ratio,
            "simpo_length_normalization": simpo_length_normalization,
            "simpo_sft_loss_weight": simpo_sft_loss_weight,
        }
        with open(os.path.join(training_args.output_dir, "simpo_init_info.json"), "w", encoding="utf-8") as f:
            json.dump(init_info, f, ensure_ascii=False, indent=4)

    processor.save_pretrained(training_args.output_dir)

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.save_pretrained(training_args.output_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(training_args.output_dir)

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if trainer.args.process_index == 0:
        save_best_checkpoint(
            best_src=getattr(trainer.state, "best_model_checkpoint", None),
            output_dir=training_args.output_dir,
            processor=processor,
            model=model,
            default_prompt=default_prompt,
        )
        save_prompt_txt(training_args.output_dir, default_prompt)


if __name__ == "__main__":
    main()
