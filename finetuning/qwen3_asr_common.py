"""Shared checkpoint resolution and teacher/student loading helpers."""
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict
import torch
from qwen_asr import Qwen3ASRModel


def resolve_checkpoint(exp_dir: str, mode: str) -> str:
    root = Path(exp_dir)
    if mode == "exp_dir":
        candidate = root
    elif mode == "best":
        candidate = root / "checkpoint-best"
    elif mode == "latest":
        choices = [(int(m.group(1)), p) for p in root.glob("checkpoint-*")
                   if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir()]
        if not choices:
            raise FileNotFoundError(f"No checkpoint-N directories under {exp_dir}")
        candidate = max(choices)[1]
    else:
        raise ValueError(f"checkpoint_mode must be best, latest, or exp_dir; got {mode}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {candidate}")
    return str(candidate.resolve())


def load_asr(path: str, dtype: torch.dtype, device_map=None):
    checkpoint = Path(path)
    adapter_config = checkpoint / "adapter_config.json"
    experiment_conf = None
    if adapter_config.is_file():
        from peft import PeftModel
        # Trainer checkpoints live immediately below the experiment directory.
        conf_path = checkpoint.parent / "train_conf.json"
        if not conf_path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint requires experiment train_conf.json: {conf_path}")
        model_conf = json.loads(conf_path.read_text(encoding="utf-8"))[1]
        experiment_conf = model_conf
        wrapper = Qwen3ASRModel.from_pretrained(model_conf["model_path"], dtype=dtype,
                                                device_map=device_map)
        wrapper.model = PeftModel.from_pretrained(wrapper.model, path, is_trainable=False)
    else:
        wrapper = Qwen3ASRModel.from_pretrained(path, dtype=dtype, device_map=device_map)
        for conf_path in (checkpoint / "train_conf.json", checkpoint.parent / "train_conf.json"):
            if conf_path.is_file():
                experiment_conf = json.loads(conf_path.read_text(encoding="utf-8"))[1]
                break
    vocabulary_pruning = (experiment_conf or {}).get("vocabulary_pruning", {})
    if vocabulary_pruning.get("enabled", False):
        from vocabulary_pruning import apply_structural_vocabulary_pruning, load_vocabulary_mapping
        apply_structural_vocabulary_pruning(
            wrapper.model,
            load_vocabulary_mapping(vocabulary_pruning["manifest"]),
        )
    return wrapper, wrapper.model, wrapper.processor


def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def model_metadata(model, processor, checkpoint: str) -> Dict[str, Any]:
    tokenizer = processor.tokenizer
    config = getattr(getattr(model, "thinker", model), "config", model.config)
    return {"checkpoint": checkpoint, "outer_wrapper": type(model).__name__,
            "trainable_model": type(getattr(model, "thinker", model)).__name__,
            "processor_class": type(processor).__name__, "tokenizer_class": type(tokenizer).__name__,
            "vocabulary_size": len(tokenizer),
            "hidden_size": getattr(config, "hidden_size", None),
            "teacher_eval": not model.training,
            "teacher_trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def save_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
