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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import random

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


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
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
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


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--train_conf", type=str, required=True,
                   help="JSON config path with format: [training_args, model_args]")
    p.add_argument('--seed', type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


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


def enable_lora(model, model_args_conf: Dict[str, Any]):
    finetune_type = str(model_args_conf.get("finetune_type", "full")).strip().lower()
    if finetune_type == "full":
        return model
    if finetune_type != "lora":
        raise ValueError(
            "model_args.finetune_type must be one of: ['full', 'lora']"
        )

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA finetuning requires `peft`. Please install it first, e.g. `pip install peft`."
        ) from e

    lora_mode = str(model_args_conf.get("lora_mode", "llm_backbone")).strip().lower()
    if lora_mode not in {"llm_backbone", "audio_encoder_llm_backbone"}:
        raise ValueError(
            "model_args.lora_mode must be one of: ['llm_backbone', 'audio_encoder_llm_backbone']"
        )

    lora_r = int(model_args_conf.get("lora_r", 8))
    lora_alpha = int(model_args_conf.get("lora_alpha", 16))
    lora_dropout = float(model_args_conf.get("lora_dropout", 0.05))
    lora_bias = str(model_args_conf.get("lora_bias", "none"))

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules="all-linear",
        modules_to_save=["lm_head"],
        exclude_modules=["audio_tower"] if lora_mode == "llm_backbone" else None,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def apply_freeze_components(model, model_args_conf: Dict[str, Any]):
    freeze_components = model_args_conf.get("freeze_components", [])
    if isinstance(freeze_components, str):
        freeze_components = re.split(r"[\s,+]+", freeze_components.strip()) if freeze_components.strip() else []
    elif not isinstance(freeze_components, list):
        raise ValueError("model_args.freeze_components must be a string or a list of strings")

    aliases = {
        "audio_encoder": "audio_encoder",
        "audio": "audio_encoder",
        "audio_tower": "audio_encoder",
        "token_embedding": "token_embedding",
        "token_embeddings": "token_embedding",
        "embedding": "token_embedding",
        "embed_tokens": "token_embedding",
    }

    normalized = []
    for item in freeze_components:
        key = aliases.get(str(item).strip().lower())
        if key is None:
            raise ValueError(
                "Unsupported freeze component: "
                f"{item}. Supported values: audio_encoder, token_embedding"
            )
        if key not in normalized:
            normalized.append(key)

    for comp in normalized:
        if comp == "audio_encoder":
            module = model.thinker.audio_tower
        else:
            module = model.thinker.model.embed_tokens
        for p in module.parameters():
            p.requires_grad = False

    if normalized:
        print(f"[freeze] Components frozen: {', '.join(normalized)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    print(f"[params] trainable={trainable:,} / total={total:,} ({pct:.2f}%)")
    return model


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

    train_conf = load_train_conf(args_cli.train_conf)
    if train_conf is None:
        raise ValueError("--train_conf is required")

    training_args_conf, model_args_conf = train_conf

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(training_args_conf.get("sr", 16000))
    batch_size = int(training_args_conf.get("per_device_train_batch_size", 32))
    grad_acc = int(training_args_conf.get("gradient_accumulation_steps", 4))
    learning_rate = float(training_args_conf.get("learning_rate", 2e-5))
    num_train_epochs = float(training_args_conf.get("num_train_epochs", 1))
    logging_steps = int(training_args_conf.get("logging_steps", 10))
    lr_scheduler_type = training_args_conf.get("lr_scheduler_type", "linear")
    warmup_ratio = float(training_args_conf.get("warmup_ratio", 0.02))
    num_workers = int(training_args_conf.get("dataloader_num_workers", 4))
    pin_memory = bool(training_args_conf.get("dataloader_pin_memory", True))
    persistent_workers = bool(training_args_conf.get("dataloader_persistent_workers", True))
    prefetch_factor = int(training_args_conf.get("dataloader_prefetch_factor", 2))
    save_strategy = training_args_conf.get("save_strategy", "steps")
    save_steps = int(training_args_conf.get("save_steps", 200))
    save_total_limit = int(training_args_conf.get("save_total_limit", 5))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    model = enable_lora(model, model_args_conf)
    model = apply_freeze_components(model, model_args_conf)

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = extract_default_prompt(ds["train"])

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=persistent_workers,
        dataloader_prefetch_factor=prefetch_factor if num_workers > 0 else None,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            MakeEveryCheckpointInferableCallback(
                processor=processor,
                model=model,
                default_prompt=default_prompt,
            )
        ],
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if train_conf is not None and trainer.args.process_index == 0:
        saved_train_conf = os.path.join(training_args.output_dir, "train_conf.json")
        with open(saved_train_conf, "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)

    processor.save_pretrained(training_args.output_dir)

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.save_pretrained(training_args.output_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(training_args.output_dir)

    if trainer.args.process_index == 0:
        save_prompt_txt(training_args.output_dir, default_prompt)

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
