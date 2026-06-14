#!/usr/bin/env python3
"""Prototype-only Qwen3-ASR finetuning for MAC-SLU domain/intent labels.

The training target is the auxiliary prototype loss only: the collator does not
provide autoregressive ``labels``. When the train config contains LoRA options,
the prototype objective is optimized with LoRA/QLoRA; otherwise it falls back to
full finetuning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
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
from finetuning.prototype_joint_utils import (  # noqa: E402
    DOMAIN_INTENT_SEP,
    extract_gold_domain_intent_labels,
    make_domain_intent_label,
)
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig  # noqa: E402
from qwen_asr.core.transformers_backend.modeling_qwen3_asr_prototype_joint import (  # noqa: E402
    Qwen3ASRJointPrototypeForConditionalGeneration,
)
from slu_decoding.prototypes import MACSLULabelSchema, PrototypeIndex  # noqa: E402


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


def save_prompt_txt(save_dir: str, prompt: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt or "")


def save_best_checkpoint(best_src: str, output_dir: str, processor=None, model=None, default_prompt: str = "", best_ckpt_name: str = "checkpoint-best") -> None:
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


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_text = processor.apply_chat_template([build_prefix_messages(prompt, None)], add_generation_prompt=True, tokenize=False)
    return prefix_text[0] if isinstance(prefix_text, list) else prefix_text


def load_train_conf(train_conf_path: str) -> List[Dict[str, Any]]:
    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or len(cfg) != 2 or not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("train_conf must be [training_args, model_args]")
    return cfg


def _schema_domain_intent_labels(schema: MACSLULabelSchema) -> List[str]:
    labels: List[str] = []
    seen = set()
    for domain in schema.valid_domains():
        for intent in schema.valid_intents(domain):
            label = make_domain_intent_label(domain, intent)
            if domain and intent and label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def resolve_prototype_json_path(prototype_conf: Dict[str, Any]) -> str:
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


def _prototype_domain_intent_labels(index: Optional[PrototypeIndex]) -> List[str]:
    if index is None:
        return []
    labels: List[str] = []
    seen = set()
    for key, item in index.intent.items():
        if not isinstance(item, dict):
            continue
        meta = item.get("meta", {}) or {}
        domain = str(meta.get("domain", "")).strip()
        intent = str(meta.get("intent", meta.get("label", ""))).strip()
        label = make_domain_intent_label(domain, intent) if domain and intent else str(key).strip()
        if label and DOMAIN_INTENT_SEP in label and label not in seen:
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
) -> tuple[MACSLULabelSchema, List[str], Dict[str, int]]:
    labels_path = prototype_conf.get("labels_path", "")
    schema_path = prototype_conf.get("schema_path", "")
    schema = MACSLULabelSchema(labels_path=labels_path, schema_path=schema_path)
    _merge_prototype_schema(schema, prototype_index)
    domain_intents = maybe_add_empty_label(
        list(
            prototype_conf.get("domain_intent_labels", [])
            or _schema_domain_intent_labels(schema)
            or _prototype_domain_intent_labels(prototype_index)
        ),
        prototype_conf,
    )
    if not domain_intents:
        raise ValueError("No domain-intent labels found. Set prototype.labels_path, prototype.domain_intent_labels, or prototype.prototype_json.")
    return schema, domain_intents, {x: i for i, x in enumerate(domain_intents)}


def _prototype_domain_intent_vectors(index: PrototypeIndex) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for key, item in index.intent.items():
        if not isinstance(item, dict):
            continue
        meta = item.get("meta", {}) or {}
        domain = str(meta.get("domain", "")).strip()
        intent = str(meta.get("intent", meta.get("label", ""))).strip()
        label = make_domain_intent_label(domain, intent) if domain and intent else str(key).strip()
        vec = item.get("vector") or []
        if label and vec and label not in out:
            out[label] = [float(x) for x in vec]
    return out


def initialize_prototype_embeddings(
    model,
    prototype_conf: Dict[str, Any],
    domain_intent_labels: List[str],
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
    domain_intent_vecs = _prototype_domain_intent_vectors(index)

    weight = head.domain_intent_prototypes.weight
    dim = weight.size(1)
    copied = 0
    with torch.no_grad():
        for idx, label in enumerate(domain_intent_labels):
            vec = domain_intent_vecs.get(label)
            if not vec:
                continue
            if len(vec) != dim:
                raise ValueError(f"domain-intent prototype dimension mismatch for {label}: {len(vec)} != {dim}")
            weight[idx].copy_(torch.tensor(vec, dtype=weight.dtype, device=weight.device))
            copied += 1
    print(f"[prototype] initialized {copied}/{len(domain_intent_labels)} domain-intent prototypes from {init_path}")
    if prototype_conf.get("freeze_initial_prototypes", False):
        head.domain_intent_prototypes.weight.requires_grad = False

def label_ids(labels: List[str], label2id: Dict[str, int]) -> List[int]:
    return [int(label2id[x]) for x in labels if x in label2id]


def multi_hot(label_ids: List[int], size: int) -> List[float]:
    vec = [0.0] * int(size)
    for idx in label_ids:
        if 0 <= int(idx) < size:
            vec[int(idx)] = 1.0
    return vec


def prototype_feature_prompt(base_prompt: str, augmented_prompt: str, prototype_source: str) -> str:
    """Match the original prototype feature source defaults from build_macslu_prototypes.py."""
    if prototype_source == "audio_only":
        return ""
    if prototype_source == "audio_prompt":
        return base_prompt or ""
    if prototype_source == "audio_prefix":
        return augmented_prompt or base_prompt or ""
    if prototype_source == "text_prefix":
        # The prototype-only trainer remains audio-capable; this keeps the decoded-prefix
        # style prompt closest to the legacy mode while preserving the audio batch path.
        return augmented_prompt or base_prompt or ""
    raise ValueError(f"Unsupported prototype_source: {prototype_source}")


def make_preprocess_fn_prototype(processor, schema, domain_intent2id, prototype_conf, seed: int):
    k = int(prototype_conf.get("k", 5))
    template = get_prompt_template(prototype_conf)
    domain_aware = bool(prototype_conf.get("domain_aware_intents", True))
    prototype_source = str(prototype_conf.get("prototype_source", "audio_only"))

    def _preprocess(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        rng = random.Random(seed + int(idx))
        prompt = ex.get("prompt", "")
        domains, intents, gold_domains, gold_intents = build_training_candidate_labels(
            ex, schema=schema, k=k, rng=rng, domain_aware_intents=domain_aware
        )
        gold_domain_intents = extract_gold_domain_intent_labels(ex)
        domain_intent_label_ids = label_ids(gold_domain_intents, domain_intent2id)
        empty_label = str(prototype_conf.get("empty_label", "__empty__"))
        if bool(prototype_conf.get("add_empty_prototype", True)) and not domain_intent_label_ids and empty_label in domain_intent2id:
            domain_intent_label_ids = [domain_intent2id[empty_label]]
            if empty_label not in domains:
                domains = [empty_label] + domains
            if empty_label not in intents:
                intents = [empty_label] + intents

        augmented_prompt = format_domain_intent_candidates(prompt, domains, intents, **template)
        feature_prompt = prototype_feature_prompt(prompt, augmented_prompt, prototype_source)
        prefix_text = build_prefix_text(processor, feature_prompt)
        return {
            "prompt": augmented_prompt,
            "audio": ex["audio"],
            "prefix_text": prefix_text,
            "domain_intent_label_ids": domain_intent_label_ids,
        }

    return _preprocess


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    inputs[key] = value.to(dtype=model_dtype)
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Return prototype logits so Trainer can compute eval_f1.

        The base Trainer would collect the language-model logits from the Qwen3-ASR
        output, but the prototype-only objective is evaluated on the auxiliary
        domain/intent heads.  Run the regular forward pass for loss, then collect
        the prototype logits used by those heads as evaluation predictions.
        """
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        inputs = self._prepare_inputs(inputs)
        domain_intent_labels = inputs.get("domain_intent_labels")
        proto_inputs = {
            key: inputs[key]
            for key in [
                "input_ids",
                "input_features",
                "attention_mask",
                "feature_attention_mask",
                "audio_feature_lengths",
                "prototype_prefix_lengths",
            ]
            if key in inputs
        }
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.detach() if getattr(outputs, "loss", None) is not None else None
            proto_model = get_prototype_base_model(model)
            domain_intent_logits = proto_model.thinker.prototype_logits(**proto_inputs)
        logits = domain_intent_logits.detach()
        labels = domain_intent_labels.detach() if torch.is_tensor(domain_intent_labels) else domain_intent_labels
        return loss, logits, labels


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



def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            major = torch.cuda.get_device_capability(device=device)[0]
        except Exception:
            major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

@dataclass
class DataCollatorForPrototypeOnlyFinetuning:
    """Batch audio/prompt inputs and provide only prototype classification labels."""

    processor: Any
    sampling_rate: int = 16000
    num_domain_intents: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]
        inputs = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix_lens = inputs["attention_mask"].sum(dim=1).tolist()
        inputs["domain_intent_labels"] = torch.tensor(
            [multi_hot([int(x) for x in f.get("domain_intent_label_ids", [])], self.num_domain_intents) for f in features],
            dtype=torch.float32,
        )
        inputs["prototype_prefix_lengths"] = torch.tensor(prefix_lens, dtype=torch.long)
        return inputs



def _micro_f1_from_logits(logits: Any, labels: Any, threshold: float = 0.5) -> float:
    pred = torch.as_tensor(logits).float().sigmoid().ge(threshold)
    gold = torch.as_tensor(labels).float().ge(0.5)
    tp = (pred & gold).sum().item()
    fp = (pred & ~gold).sum().item()
    fn = (~pred & gold).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_prototype_metrics(eval_pred) -> Dict[str, float]:
    domain_intent_f1 = _micro_f1_from_logits(eval_pred.predictions, eval_pred.label_ids)
    return {
        "domain_intent_micro_f1": domain_intent_f1,
        "f1": domain_intent_f1,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Qwen3-ASR MAC-SLU prototype-only finetuning")
    p.add_argument("--train_conf", type=str, required=True)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--init_from_checkpoint", type=str, default="", help="Warm-start a LoRA/QLoRA adapter checkpoint")
    return p.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_full_finetune_conf(train_conf_path: str) -> Tuple[List[Dict[str, Any]], PrototypeIndex | None, List[str]]:
    train_conf = load_train_conf(train_conf_path)
    training_args_conf, model_args_conf = dict(train_conf[0]), dict(train_conf[1])
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    prototype_conf["enabled"] = True

    prototype_json = resolve_prototype_json_path(prototype_conf)
    prototype_index = PrototypeIndex.load(prototype_json) if prototype_json else None
    if prototype_json:
        prototype_conf["prototype_json"] = prototype_json
        prototype_conf.pop("init_path", None)
    schema, domain_intent_labels, _ = build_prototype_label_maps(prototype_conf, prototype_index)
    if prototype_index is not None:
        prototype_conf.setdefault("pooling", prototype_index.data.get("prototype_pooling", prototype_conf.get("pooling", "mean_pooling")))
        prototype_conf.setdefault("prototype_source", prototype_index.prototype_source)
        prototype_conf.setdefault("embedding_backend", prototype_index.embedding_backend)
    # Independent domain/intent label maps are intentionally not populated; the
    # prototype-only objective now predicts joint domain-intent labels.
    prototype_conf.pop("domain_labels", None)
    prototype_conf.pop("intent_labels", None)
    prototype_conf.pop("num_domains", None)
    prototype_conf.pop("num_intents", None)
    prototype_conf["domain_intent_labels"] = domain_intent_labels
    prototype_conf["num_domain_intents"] = len(domain_intent_labels)

    model_args_conf["prototype"] = prototype_conf
    return [training_args_conf, model_args_conf], prototype_index, domain_intent_labels


def get_prototype_base_model(model):
    """Return the prototype-aware base model, unwrapping PEFT adapters if needed."""
    if hasattr(model, "thinker") and hasattr(model.thinker, "prototype_logits"):
        return model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if hasattr(base, "thinker") and hasattr(base.thinker, "prototype_logits"):
            return base
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None) if base_model is not None else None
    if inner is not None and hasattr(inner, "thinker") and hasattr(inner.thinker, "prototype_logits"):
        return inner
    raise RuntimeError("Unable to locate prototype-aware base model")


def apply_lora_if_configured(model, model_args_conf: Dict[str, Any], init_from_checkpoint: str = ""):
    lora_config = model_args_conf.get("lora_config", None)
    lora_type = str(model_args_conf.get("lora_type", "default")).lower()
    if not lora_config:
        if init_from_checkpoint:
            raise ValueError("--init_from_checkpoint currently supports LoRA/QLoRA checkpoints only")
        print("Full Finetuning (prototype-only objective)")
        return model
    if lora_type not in {"default", "qlora"}:
        raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")
    print(f"LoRA Finetuning {lora_type} (prototype-only objective)")
    if init_from_checkpoint:
        print(f"[init] warm-start LoRA adapter from checkpoint = {init_from_checkpoint}")
        model = PeftModel.from_pretrained(
            model,
            init_from_checkpoint,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_config)
        model = get_peft_model(model, peft_config)
    print("=" * 100)
    model.print_trainable_parameters()
    print("=" * 100)
    return model


def train_prototype_only(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_conf, prototype_index, domain_intent_labels = prepare_full_finetune_conf(args.train_conf)
    training_args_conf, model_args_conf = train_conf
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")
    sr = int(model_args_conf.get("sr", 16000))
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    use_bf16 = dtype == torch.bfloat16

    config = Qwen3ASRConfig.from_pretrained(model_path)
    config.thinker_config.prototype_config = prototype_conf
    if str(model_args_conf.get("lora_type", "")).lower() == "qlora":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen3ASRJointPrototypeForConditionalGeneration.from_pretrained(model_path, config=config, dtype=dtype, quantization_config=bnb_config, device_map=None)
    else:
        model = Qwen3ASRJointPrototypeForConditionalGeneration.from_pretrained(model_path, config=config, dtype=dtype, device_map=None)
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    initialize_prototype_embeddings(model, prototype_conf, domain_intent_labels, prototype_index)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    init_from_checkpoint = (getattr(args, "init_from_checkpoint", "") or "").strip()
    if init_from_checkpoint and not os.path.isdir(init_from_checkpoint):
        raise FileNotFoundError(f"init_from_checkpoint not found: {init_from_checkpoint}")
    model = apply_lora_if_configured(model, model_args_conf, init_from_checkpoint=init_from_checkpoint)

    if training_args_conf.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    from datasets import load_dataset

    schema, _, domain_intent2id = build_prototype_label_maps(prototype_conf, prototype_index)
    raw_ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})
    ds = raw_ds.map(make_preprocess_fn_prototype(processor, schema, domain_intent2id, prototype_conf, args.seed), with_indices=True, num_proc=1)
    keep = {"prompt", "audio", "prefix_text", "domain_intent_label_ids"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = ds["train"][0]["prompt"] if len(ds["train"]) else ""
    collator = DataCollatorForPrototypeOnlyFinetuning(
        processor=processor,
        sampling_rate=sr,
        num_domain_intents=len(domain_intent_labels),
    )

    training_args_conf = dict(training_args_conf)
    # Transformers only reports eval_loss when it knows which batch fields are labels.
    # The prototype-only objective uses custom multi-hot labels instead of the usual
    # causal-LM `labels`, so make them explicit for evaluation and best-checkpoint
    # selection.
    training_args_conf.setdefault("label_names", ["domain_intent_labels"])
    training_args_conf["run_name"] = os.path.basename(args.output_dir)
    if model_args_conf.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = model_args_conf["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = str(model_args_conf.get("wandb_log_model", "false")).lower()
    training_args = TrainingArguments(output_dir=args.output_dir, do_eval=True, bf16=use_bf16, fp16=not use_bf16, **training_args_conf)
    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_prototype_metrics,
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

    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)
    if trainer.args.process_index == 0:
        trainer.save_model(training_args.output_dir)
        save_best_checkpoint(getattr(trainer.state, "best_model_checkpoint", None), training_args.output_dir, processor=processor, model=model, default_prompt=default_prompt)
        save_prompt_txt(training_args.output_dir, default_prompt)



def main() -> None:
    args = parse_args()
    train_prototype_only(args)


if __name__ == "__main__":
    main()
