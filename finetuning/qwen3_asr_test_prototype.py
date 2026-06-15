#!/usr/bin/env python3
"""Prototype-domain/intent inference and JSONL generation for MAC-SLU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    p.add_argument("--dev_file", type=str, required=True)
    p.add_argument("--test_file", type=str, required=True)
    p.add_argument("--output_jsonl_dir", type=str, default="data-json/macslu_prototype")
    p.add_argument("--prediction_root", type=str, default="")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--prototype_top_k", type=int, default=0)
    p.add_argument("--prototype_min_similarity", type=float, default=-1.0, help="Filter candidates when building augmented JSONL/prompts; -1 auto-selects on dev F1")
    p.add_argument("--prototype_metric_ks", nargs="+", type=int, default=[1, 3, 5], help="K values for prototype ranking metrics")
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


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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


def semantic_frame_count(row: Dict[str, Any]) -> int:
    semantics = row.get("semantics", [])
    if semantics is None:
        return 0
    if isinstance(semantics, str):
        try:
            semantics = json.loads(semantics)
        except json.JSONDecodeError:
            return 0
    return len(semantics) if isinstance(semantics, list) else 0


def pack_hit_labels_confs_and_similarities(hits: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[float], List[float]]:
    labels: List[str] = []
    confs: List[float] = []
    similarities: List[float] = []
    for hit in hits:
        label = str(hit.get("label", ""))
        if not label or label == "__empty__":
            continue
        labels.append(label)
        confs.append(float(hit.get("score", 0.0)))
        similarities.append(float(hit.get("similarity", hit.get("score", 0.0))))
    return labels, confs, similarities


def filter_by_similarity(labels: Sequence[str], similarities: Sequence[float], min_similarity: Optional[float]) -> Tuple[List[str], List[float]]:
    labels = list(labels)
    similarities = [float(x) for x in similarities]
    if min_similarity is None:
        return labels, similarities
    kept_labels: List[str] = []
    kept_similarities: List[float] = []
    for label, similarity in zip(labels, similarities):
        if similarity >= float(min_similarity):
            kept_labels.append(label)
            kept_similarities.append(similarity)
    return kept_labels, kept_similarities


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


def average_precision_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(strip_empty(gold))
    if not gold_set or k <= 0:
        return 0.0
    hits = 0
    score = 0.0
    seen = set()
    for rank, label in enumerate(strip_empty(pred)[:k], start=1):
        if label in seen:
            continue
        seen.add(label)
        if label in gold_set:
            hits += 1
            score += hits / rank
    return score / min(len(gold_set), k)


def reciprocal_rank_at_k(pred: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(strip_empty(gold))
    if not gold_set or k <= 0:
        return 0.0
    for rank, label in enumerate(strip_empty(pred)[:k], start=1):
        if label in gold_set:
            return 1.0 / rank
    return 0.0


def ranking_metrics(rows: Sequence[Dict[str, Any]], pred_key: str, gold_key: str, ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    clean_ks = sorted({int(k) for k in ks if int(k) > 0})
    for k in clean_ks:
        precision_sum = recall_sum = hit_sum = covered_sum = ap_sum = rr_sum = 0.0
        for row in rows:
            pred = strip_empty(row.get(pred_key, []))
            gold = strip_empty(row.get(gold_key, []))
            pred_at_k = pred[:k]
            gold_set = set(gold)
            hits = len(set(pred_at_k) & gold_set)
            precision_sum += hits / k if k else 0.0
            recall_sum += hits / len(gold_set) if gold_set else 0.0
            hit_sum += 1.0 if hits > 0 else 0.0
            covered_sum += 1.0 if gold_set and gold_set.issubset(set(pred_at_k)) else 0.0
            ap_sum += average_precision_at_k(pred, gold, k)
            rr_sum += reciprocal_rank_at_k(pred, gold, k)
        denom = len(rows) if rows else 1
        metrics[f"precision@{k}"] = precision_sum / denom if rows else 0.0
        metrics[f"recall@{k}"] = recall_sum / denom if rows else 0.0
        metrics[f"hit@{k}"] = hit_sum / denom if rows else 0.0
        metrics[f"all_gold_covered@{k}"] = covered_sum / denom if rows else 0.0
        metrics[f"map@{k}"] = ap_sum / denom if rows else 0.0
        metrics[f"mrr@{k}"] = rr_sum / denom if rows else 0.0
    return metrics


def joint_coverage_metrics(rows: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    clean_ks = sorted({int(k) for k in ks if int(k) > 0})
    for k in clean_ks:
        covered_sum = 0.0
        for row in rows:
            pred_domains = set(strip_empty(row.get("pred_domains", []))[:k])
            gold_domains = set(strip_empty(row.get("gold_domains", [])))
            pred_intents = set(strip_empty(row.get("pred_intents", []))[:k])
            gold_intents = set(strip_empty(row.get("gold_intents", [])))
            domains_covered = bool(gold_domains) and gold_domains.issubset(pred_domains)
            intents_covered = bool(gold_intents) and gold_intents.issubset(pred_intents)
            covered_sum += 1.0 if domains_covered and intents_covered else 0.0
        metrics[f"all_gold_covered@{k}"] = covered_sum / len(rows) if rows else 0.0
    return metrics


def thresholded_rows(rows: Sequence[Dict[str, Any]], pred_key: str, sim_key: str, out_key: str, min_similarity: Optional[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        labels, _ = filter_by_similarity(row.get(pred_key, []), row.get(sim_key, []), min_similarity)
        new_row[out_key] = labels
        out.append(new_row)
    return out


def thresholded_metrics(rows: Sequence[Dict[str, Any]], pred_key: str, sim_key: str, gold_key: str, min_similarity: Optional[float]) -> Dict[str, float]:
    filtered_key = f"{pred_key}_thresholded"
    filtered = thresholded_rows(rows, pred_key, sim_key, filtered_key, min_similarity)
    base = score_one_kind(filtered, filtered_key, gold_key)
    example_p = example_r = example_f = covered = empty = total_candidates = 0.0
    for row in filtered:
        pred = set(strip_empty(row.get(filtered_key, [])))
        gold = set(strip_empty(row.get(gold_key, [])))
        total_candidates += len(pred)
        empty += 1.0 if not pred else 0.0
        p = len(pred & gold) / len(pred) if pred else (1.0 if not gold else 0.0)
        r = len(pred & gold) / len(gold) if gold else (1.0 if not pred else 0.0)
        f = 2 * p * r / (p + r) if p + r else 0.0
        example_p += p
        example_r += r
        example_f += f
        covered += 1.0 if gold and gold.issubset(pred) else 0.0
    denom = len(filtered) if filtered else 1
    base.update(
        {
            "min_similarity": float(min_similarity) if min_similarity is not None else None,
            "avg_candidates": total_candidates / denom if filtered else 0.0,
            "empty_candidate_rate": empty / denom if filtered else 0.0,
            "example_precision": example_p / denom if filtered else 0.0,
            "example_recall": example_r / denom if filtered else 0.0,
            "example_f1": example_f / denom if filtered else 0.0,
            "all_gold_covered_rate": covered / denom if filtered else 0.0,
        }
    )
    return base


def compute_metrics(split: str, rows: Sequence[Dict[str, Any]], metric_ks: Sequence[int], min_similarity: Optional[float]) -> Dict[str, Any]:
    domain_set = score_one_kind(rows, "pred_domains", "gold_domains")
    intent_set = score_one_kind(rows, "pred_intents", "gold_intents")
    both_exact = sum(
        int(set(strip_empty(r.get("pred_domains", []))) == set(strip_empty(r.get("gold_domains", []))) and set(strip_empty(r.get("pred_intents", []))) == set(strip_empty(r.get("gold_intents", []))))
        for r in rows
    )
    out: Dict[str, Any] = {
        "split": split,
        "count": len(rows),
        "joint_exact_match": both_exact / len(rows) if rows else 0.0,
        "domain": {
            "set": domain_set,
            "ranking": ranking_metrics(rows, "pred_domains", "gold_domains", metric_ks),
            "thresholded": thresholded_metrics(rows, "pred_domains", "pred_domains_similarity", "gold_domains", min_similarity),
        },
        "intent": {
            "set": intent_set,
            "ranking": ranking_metrics(rows, "pred_intents", "gold_intents", metric_ks),
            "thresholded": thresholded_metrics(rows, "pred_intents", "pred_intents_similarity", "gold_intents", min_similarity),
        },
        "domain_intent": {
            "ranking": joint_coverage_metrics(rows, metric_ks),
        },
        "by_semantic_frame_count": {},
    }
    for count in sorted({int(r.get("semantic_frame_count", 0)) for r in rows}):
        group = [r for r in rows if int(r.get("semantic_frame_count", 0)) == count]
        out["by_semantic_frame_count"][str(count)] = {
            "count": len(group),
            "domain_ranking": ranking_metrics(group, "pred_domains", "gold_domains", metric_ks),
            "intent_ranking": ranking_metrics(group, "pred_intents", "gold_intents", metric_ks),
            "domain_intent_ranking": joint_coverage_metrics(group, metric_ks),
            "domain_thresholded": thresholded_metrics(group, "pred_domains", "pred_domains_similarity", "gold_domains", min_similarity),
            "intent_thresholded": thresholded_metrics(group, "pred_intents", "pred_intents_similarity", "gold_intents", min_similarity),
        }
    return out


def format_metrics(split: str, rows: Sequence[Dict[str, Any]], metric_ks: Sequence[int], min_similarity: Optional[float]) -> str:
    metrics = compute_metrics(split, rows, metric_ks, min_similarity)
    lines = [f"split: {split}", f"count: {metrics['count']}", f"joint_exact_match: {metrics['joint_exact_match']:.6f}"]
    for name in ["domain", "intent"]:
        lines.append(f"[{name}/set]")
        for key in ["exact_match", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
            lines.append(f"{key}: {metrics[name]['set'][key]:.6f}")
        lines.append(f"[{name}/ranking]")
        for key, value in metrics[name]["ranking"].items():
            lines.append(f"{key}: {value:.6f}")
        lines.append(f"[{name}/thresholded]")
        for key, value in metrics[name]["thresholded"].items():
            if value is None:
                lines.append(f"{key}: none")
            else:
                lines.append(f"{key}: {value:.6f}")
    lines.append("[domain_intent/ranking]")
    for key, value in metrics["domain_intent"]["ranking"].items():
        lines.append(f"{key}: {value:.6f}")
    lines.append("[by_semantic_frame_count]")
    max_k = max([int(k) for k in metric_ks if int(k) > 0], default=0)
    for count, group in metrics["by_semantic_frame_count"].items():
        lines.append(f"{count}_intent count: {group['count']}")
        if max_k > 0:
            lines.append(f"{count}_intent domain_recall@{max_k}: {group['domain_ranking'].get(f'recall@{max_k}', 0.0):.6f}")
            lines.append(f"{count}_intent domain_all_gold_covered@{max_k}: {group['domain_ranking'].get(f'all_gold_covered@{max_k}', 0.0):.6f}")
            lines.append(f"{count}_intent intent_recall@{max_k}: {group['intent_ranking'].get(f'recall@{max_k}', 0.0):.6f}")
            lines.append(f"{count}_intent intent_all_gold_covered@{max_k}: {group['intent_ranking'].get(f'all_gold_covered@{max_k}', 0.0):.6f}")
            lines.append(f"{count}_intent domain_intent_all_gold_covered@{max_k}: {group['domain_intent_ranking'].get(f'all_gold_covered@{max_k}', 0.0):.6f}")
            lines.append(f"{count}_intent avg_thresholded_intent_candidates: {group['intent_thresholded'].get('avg_candidates', 0.0):.6f}")
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
    metric_ks: Sequence[int],
    min_similarity: Optional[float],
    metrics_json_path: str,
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
        pred_domains, domain_confs, domain_similarities = pack_hit_labels_confs_and_similarities(hits["domains"][0])
        pred_intents, intent_confs, intent_similarities = pack_hit_labels_confs_and_similarities(hits["intents"][0])
        out_rows.append(
            {
                "text_id": text_id,
                "pred_domains": pred_domains,
                "pred_intents": pred_intents,
                "domain_confs": domain_confs,
                "intent_conf": intent_confs,
                "intent_confs": intent_confs,
                "pred_domains_similarity": domain_similarities,
                "pred_intents_similarity": intent_similarities,
                "gold_domains": gold_domains,
                "gold_intents": gold_intents,
                "semantic_frame_count": semantic_frame_count(row),
            }
        )
        print(f"[{split} {idx}/{len(rows)}] prototype predicted: {text_id}")

    write_jsonl(output_jsonl, out_rows)
    metrics_obj = compute_metrics(split, out_rows, metric_ks, min_similarity)
    metrics_text = format_metrics(split, out_rows, metric_ks, min_similarity)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(metrics_text, end="")
    print(f"[info] saved prototype predictions: {output_jsonl}")
    print(f"[info] saved prototype metrics: {metrics_path}")
    print(f"[info] saved prototype metrics json: {metrics_json_path}")
    return out_rows

def build_augmented_data(
    input_jsonl: str,
    pred_rows: Sequence[Dict[str, Any]],
    output_jsonl: str,
    prompt_template: Dict[str, str],
    min_similarity: Optional[float],
) -> None:
    rows = read_jsonl(input_jsonl)
    by_id = {str(r.get("text_id", "")): r for r in pred_rows}
    augmented = []
    for idx, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{idx}")).strip()
        pred = by_id.get(text_id, {})
        item = dict(row)
        filtered_domains, filtered_domain_similarities = filter_by_similarity(
            pred.get("pred_domains", []), pred.get("pred_domains_similarity", []), min_similarity
        )
        filtered_intents, filtered_intent_similarities = filter_by_similarity(
            pred.get("pred_intents", []), pred.get("pred_intents_similarity", []), min_similarity
        )
        item["prompt"] = format_domain_intent_candidates(
            row.get("prompt", ""),
            filtered_domains,
            filtered_intents,
            **prompt_template,
        )
        item["prototype_pred_domains"] = filtered_domains
        item["prototype_pred_intents"] = filtered_intents
        item["prototype_pred_domains_similarity"] = filtered_domain_similarities
        item["prototype_pred_intents_similarity"] = filtered_intent_similarities
        item["prototype_min_similarity"] = float(min_similarity) if min_similarity is not None else None
        augmented.append(item)
    write_jsonl(output_jsonl, augmented)
    print(f"[info] saved augmented MAC-SLU jsonl: {output_jsonl}")


def is_auto_min_similarity(value: Optional[float]) -> bool:
    return value is not None and float(value) == -1.0


def candidate_similarity_thresholds(rows: Sequence[Dict[str, Any]], sim_keys: Sequence[str]) -> List[float]:
    values = sorted(
        {
            float(sim)
            for row in rows
            for key in sim_keys
            for sim in (row.get(key, []) or [])
        }
    )
    if not values:
        return []
    return [values[0] - 1e-6] + values


def auto_select_min_similarity(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Select one threshold on dev by maximizing domain/intent thresholded F1."""
    candidates = candidate_similarity_thresholds(rows, ["pred_domains_similarity", "pred_intents_similarity"])
    if not candidates:
        return None
    best_threshold = candidates[0]
    best_score = float("-inf")
    for threshold in candidates:
        metrics = compute_metrics("dev", rows, [], threshold)
        domain_f1 = float(metrics["domain"]["thresholded"].get("micro_f1", 0.0))
        intent_f1 = float(metrics["intent"]["thresholded"].get("micro_f1", 0.0))
        score = (domain_f1 + intent_f1) / 2.0
        if score > best_score or (score == best_score and threshold < best_threshold):
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def write_metrics_files(
    split: str,
    rows: Sequence[Dict[str, Any]],
    metric_ks: Sequence[int],
    min_similarity: Optional[float],
    metrics_path: str,
    metrics_json_path: str,
) -> None:
    metrics_obj = compute_metrics(split, rows, metric_ks, min_similarity)
    metrics_text = format_metrics(split, rows, metric_ks, min_similarity)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(metrics_text, end="")
    print(f"[info] saved prototype metrics: {metrics_path}")
    print(f"[info] saved prototype metrics json: {metrics_json_path}")


def write_selected_min_similarity(path_root: str, value: Optional[float], auto: bool) -> None:
    os.makedirs(path_root, exist_ok=True)
    obj = {"prototype_min_similarity": float(value) if value is not None else None, "auto_selected": bool(auto)}
    json_path = os.path.join(path_root, "prototype_min_similarity.json")
    txt_path = os.path.join(path_root, "prototype_min_similarity.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("none\n" if value is None else f"{float(value):.10f}\n")
    print(f"[info] selected prototype_min_similarity={obj['prototype_min_similarity']} auto={auto}")


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
    metric_ks = sorted({int(k) for k in args.prototype_metric_ks if int(k) > 0 and int(k) <= prototype_top_k} | {prototype_top_k})
    prompt_template = get_prompt_template(prototype_conf)
    prototype_source = str(prototype_conf.get("prototype_source", "audio_only"))
    split_to_file = {"train": args.train_file, "dev": args.dev_file, "test": args.test_file}
    prediction_root = args.prediction_root or args.exp_dir
    auto_min_similarity = is_auto_min_similarity(args.prototype_min_similarity)
    initial_min_similarity = None if auto_min_similarity else args.prototype_min_similarity
    split_results: Dict[str, Dict[str, Any]] = {}
    for split in args.splits:
        input_jsonl = split_to_file[split]
        split_dir = os.path.join(prediction_root, split)
        pred_path = os.path.join(split_dir, "prototype_predictions.jsonl")
        metrics_path = os.path.join(split_dir, "metrics_proto.txt")
        metrics_json_path = os.path.join(split_dir, "metrics_proto.json")
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
            metric_ks=metric_ks,
            min_similarity=initial_min_similarity,
            metrics_json_path=metrics_json_path,
        )
        split_results[split] = {
            "input_jsonl": input_jsonl,
            "pred_rows": pred_rows,
            "metrics_path": metrics_path,
            "metrics_json_path": metrics_json_path,
        }

    selected_min_similarity = initial_min_similarity
    if auto_min_similarity:
        if "dev" not in split_results:
            raise ValueError("--prototype_min_similarity -1 requires the dev split for threshold selection")
        selected_min_similarity = auto_select_min_similarity(split_results["dev"]["pred_rows"])
    write_selected_min_similarity(prediction_root, selected_min_similarity, auto_min_similarity)

    for split in args.splits:
        result = split_results[split]
        write_metrics_files(
            split,
            result["pred_rows"],
            metric_ks,
            selected_min_similarity,
            result["metrics_path"],
            result["metrics_json_path"],
        )
        build_augmented_data(
            result["input_jsonl"],
            result["pred_rows"],
            os.path.join(args.output_jsonl_dir, f"{split}.jsonl"),
            prompt_template,
            selected_min_similarity,
        )


def main() -> None:
    args = parse_args()
    run_inference_and_build_data(args)


if __name__ == "__main__":
    main()
