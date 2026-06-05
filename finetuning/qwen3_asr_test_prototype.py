#!/usr/bin/env python3
"""Prototype-domain/intent inference and JSONL generation for MAC-SLU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoProcessor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.prototype_prompt_utils import (  # noqa: E402
    extract_gold_domain_intents,
    format_domain_intent_candidates,
    get_prompt_template,
)
from finetuning.qwen3_asr_sft_prototype import (  # noqa: E402
    build_prefix_text,
    find_latest_checkpoint,
    load_audio,
    load_train_conf,
    resolve_dtype,
)
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig  # noqa: E402
from qwen_asr.core.transformers_backend.modeling_qwen3_asr_prototype import (  # noqa: E402
    Qwen3ASRPrototypeForConditionalGeneration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Qwen3-ASR MAC-SLU prototype domain/intent inference")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--output_jsonl_dir", type=str, default="data-json/macslu_prototype")
    p.add_argument("--prediction_root", type=str, default="")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--prototype_top_k", type=int, default=0)
    p.add_argument("--checkpoint_mode", choices=["best", "latest", "exp_dir"], default="best")
    return p.parse_args()


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype) -> Dict[str, Any]:
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        out[key] = value
    return out


def resolve_checkpoint_path(args) -> str:
    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        return os.path.join(model_path, "checkpoint-best")
    if args.auto_latest_checkpoint:
        latest = find_latest_checkpoint(model_path)
        if latest is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        return latest
    return model_path


def load_prototype_model(args, model_args_conf: Dict[str, Any], dtype: torch.dtype):
    ckpt_path = resolve_checkpoint_path(args)
    print(f"[info] use checkpoint: {ckpt_path}")
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    if not prototype_conf.get("enabled", False):
        raise ValueError("train_conf model_args.prototype.enabled must be true for prototype testing")

    if model_args_conf.get("lora_config", None):
        from peft.peft_model import PeftModelForCausalLM

        base_path = model_args_conf["model_path"]
        config = Qwen3ASRConfig.from_pretrained(base_path)
        config.thinker_config.prototype_config = prototype_conf
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(
            base_path,
            config=config,
            dtype=dtype,
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        model = PeftModelForCausalLM.from_pretrained(model, ckpt_path, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(base_path, fix_mistral_regex=True)
    else:
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(ckpt_path, dtype=dtype, device_map=args.device)
        processor = AutoProcessor.from_pretrained(ckpt_path, fix_mistral_regex=True)
    model.eval()
    return model, processor, ckpt_path

def get_predict_model(model):
    if hasattr(model, "predict_prototypes"):
        return model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if hasattr(base, "predict_prototypes"):
            return base
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None) if base_model is not None else None
    if inner is not None and hasattr(inner, "predict_prototypes"):
        return inner
    raise RuntimeError("Unable to locate prototype-aware base model for predict_prototypes")


def strip_empty(labels: Sequence[str]) -> List[str]:
    return [str(x) for x in labels if str(x) and str(x) != "__empty__"]

def prototype_feature_prompt(base_prompt: str, augmented_prompt: str, prototype_source: str) -> str:
    if prototype_source == "audio_only":
        return ""
    if prototype_source == "audio_prompt":
        return base_prompt or ""
    if prototype_source == "audio_prefix":
        return augmented_prompt or base_prompt or ""
    if prototype_source == "text_prefix":
        return augmented_prompt or base_prompt or ""
    raise ValueError(f"Unsupported prototype_source: {prototype_source}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_empty(labels: Sequence[str]) -> List[str]:
    return [str(x) for x in labels if str(x) and str(x) != "__empty__"]


def pack_hit_labels_and_confs(hits: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    confs: List[float] = []
    for hit in hits:
        label = str(hit.get("label", ""))
        if not label or label == "__empty__":
            continue
        labels.append(label)
        confs.append(float(hit.get("score", 0.0)))
    return labels, confs


def score_one_kind(rows: Sequence[Dict[str, Any]], pred_key: str, gold_key: str) -> Dict[str, float]:
    tp = fp = fn = exact = 0
    per_label: Dict[str, List[int]] = {}
    for row in rows:
        pred = set(strip_empty(row.get(pred_key, [])))
        gold = set(strip_empty(row.get(gold_key, [])))
        exact += int(pred == gold)
        for label in pred | gold:
            stat = per_label.setdefault(label, [0, 0, 0])
            if label in pred and label in gold:
                stat[0] += 1
            elif label in pred:
                stat[1] += 1
            else:
                stat[2] += 1
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1s = []
    for l_tp, l_fp, l_fn in per_label.values():
        l_p = l_tp / (l_tp + l_fp) if l_tp + l_fp else 0.0
        l_r = l_tp / (l_tp + l_fn) if l_tp + l_fn else 0.0
        macro_f1s.append(2 * l_p * l_r / (l_p + l_r) if l_p + l_r else 0.0)
    return {
        "exact_match": exact / len(rows) if rows else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0,
    }

def format_metrics(split: str, rows: Sequence[Dict[str, Any]]) -> str:
    domain = score_one_kind(rows, "pred_domains", "gold_domains")
    intent = score_one_kind(rows, "pred_intents", "gold_intents")
    both_exact = sum(
        int(set(strip_empty(r.get("pred_domains", []))) == set(strip_empty(r.get("gold_domains", []))) and set(strip_empty(r.get("pred_intents", []))) == set(strip_empty(r.get("gold_intents", []))))
        for r in rows
    )
    lines = [f"split: {split}", f"count: {len(rows)}", f"joint_exact_match: {both_exact / len(rows) if rows else 0.0:.6f}"]
    for name, metrics in [("domain", domain), ("intent", intent)]:
        lines.append(f"[{name}]")
        for key in ["exact_match", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
            lines.append(f"{key}: {metrics[key]:.6f}")
    return "\n".join(lines) + "\n"

def infer_split(
    model: Any,
    processor: Any,
    input_jsonl: str,
    output_jsonl: str,
    metrics_path: str,
    split: str,
    sr: int,
    prototype_top_k: int,
    prototype_source: str,
) -> List[Dict[str, Any]]:
    rows = read_jsonl(input_jsonl)
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)
    proto_model = get_predict_model(model)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{idx}")).strip()
        wav = load_audio(row.get("audio", ""), sr=sr)
        base_prompt = row.get("prompt", "")
        # During prototype lookup there is no decoded semantic prefix yet; keep
        # audio_only/audio_prompt defaults aligned with local/build_macslu_prototypes.py.
        feature_prompt = prototype_feature_prompt(base_prompt, base_prompt, prototype_source)
        prefix_text = build_prefix_text(processor, feature_prompt)
        inputs = processor(text=[prefix_text], audio=[wav], return_tensors="pt", padding=True, truncation=False)
        prefix_len = int(inputs["attention_mask"][0].sum().item())
        inputs["prototype_prefix_lengths"] = torch.tensor([prefix_len], dtype=torch.long)
        inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)
        with torch.inference_mode():
            hits = proto_model.predict_prototypes(top_k=prototype_top_k, **inputs)
        gold_domains, gold_intents = extract_gold_domain_intents(row)
        pred_domains, domain_confs = pack_hit_labels_and_confs(hits["domains"][0])
        pred_intents, intent_confs = pack_hit_labels_and_confs(hits["intents"][0])
        out_rows.append(
            {
                "text_id": text_id,
                "pred_domains": pred_domains,
                "pred_intents": pred_intents,
                "domain_confs": domain_confs,
                "intent_conf": intent_confs,
                "intent_confs": intent_confs,
                "gold_domains": gold_domains,
                "gold_intents": gold_intents,
            }
        )
        print(f"[{split} {idx}/{len(rows)}] prototype predicted: {text_id}")

    write_jsonl(output_jsonl, out_rows)
    metrics_text = format_metrics(split, out_rows)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(metrics_text, end="")
    print(f"[info] saved prototype predictions: {output_jsonl}")
    print(f"[info] saved prototype metrics: {metrics_path}")
    return out_rows

  
def build_augmented_data(input_jsonl: str, pred_rows: Sequence[Dict[str, Any]], output_jsonl: str, prompt_template: Dict[str, str]) -> None:
    rows = read_jsonl(input_jsonl)
    by_id = {str(r.get("text_id", "")): r for r in pred_rows}
    augmented = []
    for idx, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{idx}")).strip()
        pred = by_id.get(text_id, {})
        item = dict(row)
        item["prompt"] = format_domain_intent_candidates(
            row.get("prompt", ""),
            pred.get("pred_domains", []),
            pred.get("pred_intents", []),
            **prompt_template,
        )
        item["prototype_pred_domains"] = pred.get("pred_domains", [])
        item["prototype_pred_intents"] = pred.get("pred_intents", [])
        augmented.append(item)
    write_jsonl(output_jsonl, augmented)
    print(f"[info] saved augmented MAC-SLU jsonl: {output_jsonl}")


def run_inference_and_build_data(args: argparse.Namespace) -> None:
    train_conf_path = os.path.join(args.exp_dir, "train_conf.json")
    train_conf = load_train_conf(train_conf_path)
    _, model_args_conf = train_conf
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    sr = int(model_args_conf.get("sr", 16000))
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    checkpoint_args = SimpleNamespace(
        exp_dir=args.exp_dir,
        auto_best_checkpoint=args.checkpoint_mode == "best",
        auto_latest_checkpoint=args.checkpoint_mode == "latest",
        device=args.device,
    )
    model, processor, _ = load_prototype_model(checkpoint_args, model_args_conf, dtype)
    prototype_top_k = int(args.prototype_top_k or prototype_conf.get("k", 5))
    prompt_template = get_prompt_template(prototype_conf)
    prototype_source = str(prototype_conf.get("prototype_source", "audio_only"))
    split_to_file = {"train": args.train_file, "dev": args.eval_file, "test": args.test_file}
    prediction_root = args.prediction_root or args.exp_dir
    for split in args.splits:
        input_jsonl = split_to_file[split]
        split_dir = os.path.join(prediction_root, split)
        pred_path = os.path.join(split_dir, "prototype_predictions.jsonl")
        metrics_path = os.path.join(split_dir, "metrics_proto.txt")
        pred_rows = infer_split(
            model=model,
            processor=processor,
            input_jsonl=input_jsonl,
            output_jsonl=pred_path,
            metrics_path=metrics_path,
            split=split,
            sr=sr,
            prototype_top_k=prototype_top_k,
            prototype_source=prototype_source,
        )
        build_augmented_data(input_jsonl, pred_rows, os.path.join(args.output_jsonl_dir, f"{split}.jsonl"), prompt_template)


def main() -> None:
    args = parse_args()
    run_inference_and_build_data(args)


if __name__ == "__main__":
    main()
