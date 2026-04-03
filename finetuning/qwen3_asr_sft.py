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
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel

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
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--train_conf", type=str, required=True,
                   help="JSON config path with format: [training_args, model_args]")
    p.add_argument('--seed', type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="dev.jsonl")
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
    training_args_conf = dict(training_args_conf)

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(model_args_conf.get("sr", 16000))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    if training_args_conf["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # LoRA
    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        lora_type = model_args_conf.get("lora_type", "default")
        print(f"LoRA Finetuning {lora_type}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_config
        )
        
        model = get_peft_model(model, peft_config)
        print("="*100)
        model.print_trainable_parameters()
        print("="*100)
    else:
        print("Full Finetuning")

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            "validation": args_cli.eval_file,
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

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
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
