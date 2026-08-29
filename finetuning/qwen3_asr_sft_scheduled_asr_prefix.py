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
from peft.peft_model import PeftModel


TARGET_MARKER = "<asr_text>"

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
        target_asr = ex.get("text_asr") or ex["text"]
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            # Validation remains clean; only the generated train JSONL is required
            # to carry text_asr.
            "target_asr": target_asr,
            "prefix_text": prefix_text,
        }

    return _preprocess



def mask_leading_valid_tokens(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_lens: List[int],
) -> torch.Tensor:
    """Mask the first N non-padding tokens for each sample.

    Qwen3-ASR uses left padding by default. Therefore a prefix length cannot be
    translated to labels[i, :prefix_len]. Instead, locate the valid-token
    positions from attention_mask and mask the first prefix_len valid tokens.
    This is padding-side agnostic and works for both left and right padding.
    """
    if labels.shape != attention_mask.shape:
        raise ValueError(
            f"labels/attention_mask shape mismatch: {labels.shape} vs {attention_mask.shape}"
        )
    if len(prefix_lens) != labels.size(0):
        raise ValueError(
            f"prefix_lens batch mismatch: {len(prefix_lens)} vs {labels.size(0)}"
        )

    for i, prefix_len in enumerate(prefix_lens):
        prefix_len = int(prefix_len)
        valid_positions = torch.nonzero(
            attention_mask[i].to(dtype=torch.bool),
            as_tuple=False,
        ).squeeze(-1)
        if prefix_len < 0 or prefix_len > valid_positions.numel():
            raise ValueError(
                f"Invalid prefix_len={prefix_len} for sample {i} with "
                f"{valid_positions.numel()} valid tokens"
            )
        labels[i, valid_positions[:prefix_len]] = -100

    # Padding tokens are never supervised, regardless of tokenizer pad id.
    labels[attention_mask == 0] = -100
    return labels


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
        labels = mask_leading_valid_tokens(
            labels,
            full_inputs["attention_mask"],
            prefix_lens,
        )

        full_inputs["labels"] = labels
        return full_inputs


def get_generated_asr_value_char_span(target_asr: str):
    """Return the character span of only the generated ASR JSON value.

    The returned [start, end) span is relative to target_asr.  We include the
    JSON string quotes in the masked value, while keeping the surrounding
    output structure (e.g. "asr_text": and "semantics":) supervised.
    """
    if not isinstance(target_asr, str):
        raise ValueError("text_asr must be a string")
    if TARGET_MARKER not in target_asr:
        raise ValueError(f"text_asr does not contain {TARGET_MARKER!r}")

    header, payload_text = target_asr.split(TARGET_MARKER, 1)
    payload = json.loads(payload_text)
    if not isinstance(payload, dict) or "asr_text" not in payload or "semantics" not in payload:
        raise ValueError("text_asr payload must contain asr_text and semantics")

    key_prefix = '"asr_text": '
    key_pos = payload_text.find(key_prefix)
    if key_pos < 0:
        raise ValueError("text_asr payload does not contain the expected asr_text key")

    value_text = json.dumps(payload["asr_text"], ensure_ascii=False)
    value_start_in_payload = key_pos + len(key_prefix)
    if not payload_text.startswith(value_text, value_start_in_payload):
        raise ValueError("text_asr does not use the expected canonical JSON serialization")

    value_end_in_payload = value_start_in_payload + len(value_text)
    payload_offset = len(header) + len(TARGET_MARKER)
    return (
        payload_offset + value_start_in_payload,
        payload_offset + value_end_in_payload,
    )


@dataclass
class DataCollatorForQwen3ASRScheduledPrefix:
    processor: Any
    sampling_rate: int = 16000

    def _mask_generated_asr_values(
        self,
        labels,
        full_inputs,
        full_texts,
        targets,
        target_spans,
        eos,
    ):
        """Mask only generated ASR value tokens, independent of padding side."""
        tokenizer = self.processor.tokenizer
        audio_token_id = tokenizer.convert_tokens_to_ids(self.processor.audio_token)

        for i, (span_start, span_end) in enumerate(target_spans):
            if not 0 <= span_start <= span_end <= len(targets[i]):
                raise ValueError(
                    f"Invalid generated ASR char span {(span_start, span_end)} "
                    f"for target length {len(targets[i])}"
                )

            valid_positions = torch.nonzero(
                full_inputs["attention_mask"][i].to(dtype=torch.bool),
                as_tuple=False,
            ).squeeze(-1)
            valid_ids = full_inputs["input_ids"][i, valid_positions].tolist()

            # Reconstruct exactly the text tokenized by Qwen3ASRProcessor,
            # but without recomputing audio features.
            audio_token_count = sum(token_id == audio_token_id for token_id in valid_ids)
            expanded_full_text = self.processor.replace_multimodal_special_tokens(
                [full_texts[i]],
                iter([audio_token_count]),
            )[0]

            target_suffix = targets[i] + eos
            if not expanded_full_text.endswith(target_suffix):
                raise RuntimeError(
                    "Unable to align target text with the processor-expanded full sequence"
                )
            target_char_offset = len(expanded_full_text) - len(target_suffix)
            absolute_span_start = target_char_offset + span_start
            absolute_span_end = target_char_offset + span_end

            tokenized = tokenizer(
                expanded_full_text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=False,
                truncation=False,
            )
            token_ids = tokenized["input_ids"]
            offsets = tokenized["offset_mapping"]

            if token_ids != valid_ids:
                raise RuntimeError(
                    "Tokenizer alignment mismatch while locating generated ASR tokens"
                )

            masked_tokens = 0
            for token_index, (char_start, char_end) in enumerate(offsets):
                # Special tokens generally have a zero-length offset.
                if char_end <= char_start:
                    continue
                if char_start < absolute_span_end and char_end > absolute_span_start:
                    labels[i, valid_positions[token_index]] = -100
                    masked_tokens += 1

            if span_end > span_start and masked_tokens == 0:
                raise RuntimeError(
                    "Generated ASR span is non-empty but no tokens were masked"
                )

        return labels

    def _encode(self, prefix_texts, targets, audios, generated_asr_spans=None):
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + target + eos for pfx, target in zip(prefix_texts, targets)]
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

        labels = full_inputs["input_ids"].clone()
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

        # Always exclude system/user/audio/chat-template prefix tokens from loss.
        labels = mask_leading_valid_tokens(
            labels,
            full_inputs["attention_mask"],
            prefix_lens,
        )

        # For the generated branch, additionally exclude only the generated
        # ASR value.  The JSON structure and ground-truth semantics stay supervised.
        if generated_asr_spans is not None:
            labels = self._mask_generated_asr_values(
                labels=labels,
                full_inputs=full_inputs,
                full_texts=full_texts,
                targets=targets,
                target_spans=generated_asr_spans,
                eos=eos,
            )

        full_inputs["labels"] = labels
        return full_inputs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        prefix_texts = [feature["prefix_text"] for feature in features]
        audios = [load_audio(feature["audio"], sr=self.sampling_rate) for feature in features]
        clean = self._encode(
            prefix_texts,
            [feature["target"] for feature in features],
            audios,
        )
        generated_targets = [feature["target_asr"] for feature in features]
        generated = self._encode(
            prefix_texts,
            generated_targets,
            audios,
            generated_asr_spans=[
                get_generated_asr_value_char_span(target) for target in generated_targets
            ],
        )
        return {"clean": clean, "generated": generated}


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
        if model_dtype is None:
            return inputs

        def cast_floats(value):
            if torch.is_tensor(value) and value.is_floating_point():
                return value.to(dtype=model_dtype)
            if isinstance(value, dict):
                return {key: cast_floats(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cast_floats(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cast_floats(item) for item in value)
            return value

        return cast_floats(inputs)


class ScheduledASRPrefixTrainer(CastFloatInputsTrainer):
    def __init__(self, *args, scheduled_asr_prefix_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = scheduled_asr_prefix_config or {}
        self.schedule_start_ratio = float(config.get("start_ratio", 0.3))
        self.schedule_end_ratio = float(config.get("end_ratio", 0.8))
        self.max_generated_loss_weight = float(
            config.get("max_generated_loss_weight", 0.5)
        )
        if not 0.0 <= self.schedule_start_ratio <= self.schedule_end_ratio <= 1.0:
            raise ValueError("Expected 0 <= start_ratio <= end_ratio <= 1")
        if not 0.0 <= self.max_generated_loss_weight <= 1.0:
            raise ValueError("max_generated_loss_weight must be between 0 and 1")

    def generated_loss_weight(self) -> float:
        if not self.model.training:
            return 0.0
        total_steps = max(int(self.state.max_steps), 1)
        progress = min(max(float(self.state.global_step) / total_steps, 0.0), 1.0)
        if progress <= self.schedule_start_ratio:
            return 0.0
        if progress >= self.schedule_end_ratio:
            return self.max_generated_loss_weight
        span = self.schedule_end_ratio - self.schedule_start_ratio
        if span == 0.0:
            return self.max_generated_loss_weight
        return self.max_generated_loss_weight * (
            progress - self.schedule_start_ratio
        ) / span

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        # Evaluation unwraps the nested batch to the clean branch before the
        # base Trainer calls compute_loss again.
        if "clean" not in inputs:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        
        clean_inputs = inputs["clean"]
        generated_inputs = inputs["generated"]
        clean_outputs = model(**clean_inputs)
        clean_loss = clean_outputs.loss
        generated_weight = self.generated_loss_weight()

        if generated_weight > 0.0:
            generated_outputs = model(**generated_inputs)
            loss = (
                (1.0 - generated_weight) * clean_loss
                + generated_weight * generated_outputs.loss
            )
        else:
            loss = clean_loss

        return (loss, clean_outputs) if return_outputs else loss
      
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        # Trainer.prediction_step bypasses compute_loss when labels are not at
        # the top level. Evaluation always uses the clean branch, so unwrap it
        # before Trainer checks for labels and calls the model.
        if "clean" in inputs:
            inputs = inputs["clean"]
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

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

    # Resume / warm start
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="",
        help="Warm-start model/adapter weights from a checkpoint without resuming optimizer/scheduler state",
    )

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
    scheduled_asr_prefix_config = model_args_conf.get("scheduled_asr_prefix", {})

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(model_args_conf.get("sr", 16000))
    eval_max_new_tokens = int(model_args_conf.get("eval_max_new_tokens", 256))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    # LoRA
    lora_config = model_args_conf.get("lora_config", None)
    lora_type = model_args_conf.get("lora_type", "default")

    if lora_type == "qlora":
        # load pretrained model (reload)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            quantization_config=bnb_config,
            device_map=None,
        )
    else:
        # load pretrained model
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            device_map=None,
        )

    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    init_from_checkpoint = (args_cli.init_from_checkpoint or "").strip()
    if init_from_checkpoint and not os.path.isdir(init_from_checkpoint):
        raise FileNotFoundError(f"init_from_checkpoint not found: {init_from_checkpoint}")

    if lora_config:
        if lora_type not in ["default", "qlora"]:
            raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")

        print(f"LoRA Finetuning {lora_type}")
        if init_from_checkpoint:
            print(f"[init] warm-start LoRA adapter from checkpoint = {init_from_checkpoint}")
            model = PeftModel.from_pretrained(
                model,
                init_from_checkpoint,
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
        if init_from_checkpoint:
            raise ValueError("--init_from_checkpoint currently supports LoRA/QLoRA checkpoints only")
        print("Full Finetuning")

    if training_args_conf["gradient_checkpointing"]:
        model.config.use_cache = False
        model.enable_input_require_grads()

        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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

    keep = {"prompt", "audio", "target", "target_asr", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = extract_default_prompt(ds["train"])

    collator = DataCollatorForQwen3ASRScheduledPrefix(
        processor=processor,
        sampling_rate=sr,
    )

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

    trainer = ScheduledASRPrefixTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        scheduled_asr_prefix_config=scheduled_asr_prefix_config,
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
    if init_from_checkpoint and (resume_from or args_cli.resume == 1):
        raise ValueError("--init_from_checkpoint warm-starts weights and cannot be combined with --resume/--resume_from")
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
