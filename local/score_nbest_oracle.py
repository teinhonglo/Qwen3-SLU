#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from local.metrics import (  # noqa: E402
    collect_slot_set,
    edit_distance,
    normalize_semantics,
    normalize_text,
    slot_mer_metric,
    tokenize_for_mer,
)


def extract_payload_text(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    m = re.match(r"^language\s+.+?<asr_text>(.*)$", raw_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text


def try_parse_score_dict(text: str) -> Dict[str, Any]:
    payload = extract_payload_text(text)
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def parse_hypothesis(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        raw_text = str(raw.get("text", raw.get("pred_raw", raw.get("hyp", ""))))
    else:
        raw_text = str(raw or "")
    pred_json = try_parse_score_dict(raw_text)
    valid_json = int(bool(pred_json))
    pred_query = pred_json.get("asr_text", "") if isinstance(pred_json, dict) else ""
    pred_semantics = []
    try:
        semantics = pred_json.get("semantics", []) if isinstance(pred_json, dict) else []
        if isinstance(semantics, list):
            pred_semantics = semantics
        elif isinstance(semantics, str):
            loaded = json.loads(semantics)
            pred_semantics = loaded if isinstance(loaded, list) else []
    except Exception:
        valid_json = 0
        pred_semantics = []
    return {
        "raw": raw_text,
        "pred_json": pred_json,
        "pred_query": pred_query,
        "pred_semantics": pred_semantics,
        "valid_json": valid_json,
    }


def calculate_one_prediction_metrics(pred_data: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate one n-best hypothesis' oracle EMA and SLU metrics."""
    pred_semantics = normalize_semantics(pred_data.get("pred_semantics", []))
    gt_semantics = normalize_semantics(gt_data.get("semantics", []))

    pred_intents = sorted([(s.get("domain"), s.get("intent")) for s in pred_semantics])
    gt_intents = sorted([(s.get("domain"), s.get("intent")) for s in gt_semantics])
    intent_match = int(pred_intents == gt_intents)
    overall_match = int(pred_semantics == gt_semantics)

    pred_explicit_slot_set = collect_slot_set(pred_semantics, "slots")
    gt_explicit_slot_set = collect_slot_set(gt_semantics, "slots")
    explicit_slot_tp = len(pred_explicit_slot_set & gt_explicit_slot_set)
    explicit_slot_fp = len(pred_explicit_slot_set - gt_explicit_slot_set)
    explicit_slot_fn = len(gt_explicit_slot_set - pred_explicit_slot_set)

    pred_implicit_slot_set = collect_slot_set(pred_semantics, "implicit_slots")
    gt_implicit_slot_set = collect_slot_set(gt_semantics, "implicit_slots")
    implicit_slot_tp = len(pred_implicit_slot_set & gt_implicit_slot_set)
    implicit_slot_fp = len(pred_implicit_slot_set - gt_implicit_slot_set)
    implicit_slot_fn = len(gt_implicit_slot_set - pred_implicit_slot_set)

    slot_tp = explicit_slot_tp + implicit_slot_tp
    slot_fp = explicit_slot_fp + implicit_slot_fp
    slot_fn = explicit_slot_fn + implicit_slot_fn
    slot_precision, slot_recall, slot_f1 = slot_mer_metric(slot_tp, slot_fp, slot_fn)
    explicit_slot_precision, explicit_slot_recall, explicit_slot_f1 = slot_mer_metric(
        explicit_slot_tp, explicit_slot_fp, explicit_slot_fn
    )
    implicit_slot_precision, implicit_slot_recall, implicit_slot_f1 = slot_mer_metric(
        implicit_slot_tp, implicit_slot_fp, implicit_slot_fn
    )

    query_ref_tokens = tokenize_for_mer(normalize_text(gt_data.get("query", "")))
    query_hyp_tokens = tokenize_for_mer(normalize_text(pred_data.get("pred_query", "")))
    mer_error = edit_distance(query_ref_tokens, query_hyp_tokens)
    mer_ref_len = len(query_ref_tokens)
    mer = mer_error / mer_ref_len if mer_ref_len else 0.0

    pred_slots_value = []
    for s in pred_data.get("pred_semantics", []):
        slots = s.get("slots", {}) if isinstance(s, dict) else {}
        if isinstance(slots, dict):
            pred_slots_value.extend(slots.values())

    slot_match_count = 0
    valid_slots = len(pred_slots_value)
    pred_query_ori = pred_data.get("pred_query", "")
    gt_semantics_ori = gt_data.get("semantics", [])
    for slot in pred_slots_value:
        if slot in pred_query_ori:
            slot_match_count += 1
        elif slot in str(gt_semantics_ori):
            valid_slots -= 1
    slot_match_acc = slot_match_count / valid_slots if valid_slots else 0.0

    return {
        "oracle_ema": overall_match,
        "overall_match": overall_match,
        "intent_match": intent_match,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "explicit_slot_tp": explicit_slot_tp,
        "explicit_slot_fp": explicit_slot_fp,
        "explicit_slot_fn": explicit_slot_fn,
        "explicit_slot_precision": explicit_slot_precision,
        "explicit_slot_recall": explicit_slot_recall,
        "explicit_slot_f1": explicit_slot_f1,
        "implicit_slot_tp": implicit_slot_tp,
        "implicit_slot_fp": implicit_slot_fp,
        "implicit_slot_fn": implicit_slot_fn,
        "implicit_slot_precision": implicit_slot_precision,
        "implicit_slot_recall": implicit_slot_recall,
        "implicit_slot_f1": implicit_slot_f1,
        "query_mer_errors": mer_error,
        "query_mer_ref_lens": mer_ref_len,
        "query_mer": mer,
        "slot_match_acc": slot_match_acc,
    }


def score_key(item: Dict[str, Any]) -> float:
    s = item.get("score", {})
    return (
        1000.0 * float(s.get("oracle_ema", 0))
        + 100.0 * float(s.get("intent_match", 0))
        + 10.0 * float(s.get("slot_f1", 0.0))
        + float(s.get("explicit_slot_f1", 0.0))
        + float(s.get("implicit_slot_f1", 0.0))
        - float(s.get("query_mer", 0.0))
        + 0.1 * float(s.get("slot_match_acc", 0.0))
        + 0.01 * float(s.get("valid_json", 0))
    )


def format_metrics_text(stats: Dict[str, Any]) -> str:
    lines = [
        "N-best oracle scoring metrics",
        "=============================",
        f"samples: {stats['samples']}",
        f"hypotheses: {stats['hypotheses']}",
        f"valid_json: {stats['valid_json']}",
        f"valid_json_rate: {stats['valid_json_rate']:.6f}",
        f"oracle_hit_samples: {stats['oracle_hit_samples']}",
        f"oracle_ema_coverage: {stats['oracle_ema_coverage']:.6f}",
        f"average_oracle_rank: {stats['average_oracle_rank']:.6f}",
        f"oracle_rank_count: {stats['oracle_rank_count']}",
    ]
    return "\n".join(lines) + "\n"


def score_file(input_jsonl: str, output_jsonl: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    stats = {
        "samples": 0,
        "hypotheses": 0,
        "valid_json": 0,
        "oracle_hit_samples": 0,
        "oracle_rank_sum": 0.0,
        "oracle_rank_count": 0,
    }
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            gt = {"query": row.get("query", ""), "semantics": row.get("semantics", [])}
            scored: List[Dict[str, Any]] = []
            for rank, hyp in enumerate(row.get("nbest", []) or []):
                parsed = parse_hypothesis(hyp)
                score = calculate_one_prediction_metrics(parsed, gt)
                score["valid_json"] = parsed["valid_json"]
                item = {
                    "rank": rank,
                    "raw": parsed["raw"],
                    "pred_query": parsed["pred_query"],
                    "pred_semantics": parsed["pred_semantics"],
                    "score": score,
                    "preference_score": score_key({"score": score}),
                }
                scored.append(item)
                stats["hypotheses"] += 1
                stats["valid_json"] += parsed["valid_json"]
            oracle_items = [h for h in scored if h["score"].get("oracle_ema", 0)]
            if oracle_items:
                stats["oracle_hit_samples"] += 1
                # Rank is reported as 1-based original n-best position.
                stats["oracle_rank_sum"] += min(h["rank"] for h in oracle_items) + 1
                stats["oracle_rank_count"] += 1
            scored.sort(key=lambda x: (x["preference_score"], -x["rank"]), reverse=True)
            out = dict(row)
            out["scored_nbest"] = scored
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["samples"] += 1
    stats["oracle_ema_coverage"] = stats["oracle_hit_samples"] / stats["samples"] if stats["samples"] else 0.0
    stats["valid_json_rate"] = stats["valid_json"] / stats["hypotheses"] if stats["hypotheses"] else 0.0
    stats["average_oracle_rank"] = (
        stats["oracle_rank_sum"] / stats["oracle_rank_count"] if stats["oracle_rank_count"] else 0.0
    )
    with open(output_jsonl + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    metrics_path = os.path.join(os.path.dirname(output_jsonl) or ".", "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(format_metrics_text(stats))
    return stats


def main():
    p = argparse.ArgumentParser("Score n-best hypotheses with oracle EMA and SLU metrics")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    args = p.parse_args()
    stats = score_file(args.input_jsonl, args.output_jsonl)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
