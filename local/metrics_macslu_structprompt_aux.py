#!/usr/bin/env python3
"""Evaluate PII/CDI predictions produced by qwen3_asr_test.py."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


LANGUAGE_PREFIX_RE = re.compile(r"^language\s+.+?<asr_text>(.*)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StructSFT PII/CDI tasks")
    parser.add_argument("--task", choices=["pii", "cdi"], required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def extract_payload_text(raw_text: str) -> str:
    raw_text = str(raw_text or "").strip()
    match = LANGUAGE_PREFIX_RE.match(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text


def parse_json_list(raw_text: str) -> list | None:
    payload = extract_payload_text(raw_text)
    try:
        value = json.loads(payload)
        return value if isinstance(value, list) else None
    except Exception:
        pass

    start = payload.find("[")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(payload[start:])
        return value if isinstance(value, list) else None
    except Exception:
        return None


def normalize_pii(value: Any) -> dict[tuple[str, str], frozenset[str]] | None:
    if not isinstance(value, list):
        return None

    relations: dict[tuple[str, str], set[str]] = {}
    for item in value:
        if not isinstance(item, dict):
            return None
        domain = item.get("domain")
        intent = item.get("intent")
        slots = item.get("slots")
        if not isinstance(domain, str) or not domain.strip():
            return None
        if not isinstance(intent, str):
            return None
        if not isinstance(slots, list):
            return None
        if not all(isinstance(slot, str) and slot.strip() for slot in slots):
            return None

        key = (domain.strip(), intent.strip())
        slot_set = relations.setdefault(key, set())
        slot_set.update(slot.strip() for slot in slots)

    return {key: frozenset(slots) for key, slots in relations.items()}


def relation_pair_set(
    mapping: dict[tuple[str, str], frozenset[str]]
) -> set[tuple[str, str, str]]:
    return {
        (domain, intent, slot)
        for (domain, intent), slots in mapping.items()
        for slot in slots
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def candidate_valid(
    pred: dict[tuple[str, str], frozenset[str]] | None,
    gt_row: dict,
) -> bool:
    if pred is None:
        return False

    candidate_di = {
        (str(item.get("domain", "")).strip(), str(item.get("intent", "")).strip())
        for item in gt_row.get("pii_domain_intents", [])
        if isinstance(item, dict)
    }
    candidate_slots = {
        str(slot).strip()
        for slot in gt_row.get("pii_slots", [])
        if str(slot).strip()
    }

    for key, slots in pred.items():
        if key not in candidate_di:
            return False
        if any(slot not in candidate_slots for slot in slots):
            return False
    return True


def evaluate_pii(pred_rows: list[dict], gt_rows: list[dict]) -> tuple[dict, list[dict]]:
    pred_by_id = {str(row.get("text_id", "")): row for row in pred_rows}

    total = len(gt_rows)
    valid_json = 0
    relation_exact = 0
    candidate_valid_count = 0
    missing_predictions = 0

    di_tp = di_fp = di_fn = 0
    pair_tp = pair_fp = pair_fn = 0
    scored_rows = []

    for gt in gt_rows:
        text_id = str(gt.get("text_id", ""))
        pred_row = pred_by_id.get(text_id)
        raw_pred = "" if pred_row is None else str(pred_row.get("pred_raw", ""))
        if pred_row is None:
            missing_predictions += 1

        gold_list = parse_json_list(str(gt.get("text", "")))
        gold = normalize_pii(gold_list)
        if gold is None:
            raise ValueError(f"Invalid PII gold target for {text_id}: {gt.get('text')}")

        pred_list = parse_json_list(raw_pred)
        pred = normalize_pii(pred_list)
        is_valid = pred is not None
        if is_valid:
            valid_json += 1
        is_candidate_valid = candidate_valid(pred, gt)
        if is_candidate_valid:
            candidate_valid_count += 1

        if pred is None:
            pred = {}

        is_exact = pred == gold
        if is_exact:
            relation_exact += 1

        gold_di = set(gold)
        pred_di = set(pred)
        di_tp_i = len(gold_di & pred_di)
        di_fp_i = len(pred_di - gold_di)
        di_fn_i = len(gold_di - pred_di)
        di_tp += di_tp_i
        di_fp += di_fp_i
        di_fn += di_fn_i

        gold_pairs = relation_pair_set(gold)
        pred_pairs = relation_pair_set(pred)
        pair_tp_i = len(gold_pairs & pred_pairs)
        pair_fp_i = len(pred_pairs - gold_pairs)
        pair_fn_i = len(gold_pairs - pred_pairs)
        pair_tp += pair_tp_i
        pair_fp += pair_fp_i
        pair_fn += pair_fn_i

        scored_rows.append(
            {
                "text_id": text_id,
                "task": "pii",
                "pred_raw": raw_pred,
                "valid_json": is_valid,
                "candidate_valid": is_candidate_valid,
                "relation_exact": is_exact,
                "gold_relation": [
                    {"domain": d, "intent": i, "slots": sorted(slots)}
                    for (d, i), slots in sorted(gold.items())
                ],
                "pred_relation": [
                    {"domain": d, "intent": i, "slots": sorted(slots)}
                    for (d, i), slots in sorted(pred.items())
                ],
                "di_tp": di_tp_i,
                "di_fp": di_fp_i,
                "di_fn": di_fn_i,
                "pair_tp": pair_tp_i,
                "pair_fp": pair_fp_i,
                "pair_fn": pair_fn_i,
            }
        )

    metrics = {
        "task": "pii",
        "num_examples": total,
        "missing_predictions": missing_predictions,
        "valid_json_rate": safe_div(valid_json, total),
        "candidate_valid_rate": safe_div(candidate_valid_count, total),
        "relation_exact_match": safe_div(relation_exact, total),
        "domain_intent_micro": {
            **prf(di_tp, di_fp, di_fn),
            "tp": di_tp,
            "fp": di_fp,
            "fn": di_fn,
        },
        "intent_slot_pair_micro": {
            **prf(pair_tp, pair_fp, pair_fn),
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
        },
    }
    return metrics, scored_rows


def parse_bool(raw_text: str) -> bool | None:
    payload = extract_payload_text(raw_text).strip().lower()
    if payload == "true":
        return True
    if payload == "false":
        return False
    return None


def class_stats(
    gold_rows: list[bool], pred_rows: list[bool | None], positive_label: bool
) -> dict[str, float | int]:
    tp = fp = fn = 0
    for gold, pred in zip(gold_rows, pred_rows):
        if pred == positive_label and gold == positive_label:
            tp += 1
        elif pred == positive_label and gold != positive_label:
            fp += 1
        elif gold == positive_label and pred != positive_label:
            fn += 1
    return {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def evaluate_cdi(pred_rows: list[dict], gt_rows: list[dict]) -> tuple[dict, list[dict]]:
    pred_by_id = {str(row.get("text_id", "")): row for row in pred_rows}

    gold_labels: list[bool] = []
    pred_labels: list[bool | None] = []
    scored_rows = []
    missing_predictions = 0
    valid_outputs = 0
    correct = 0
    correct_valid = 0
    pair_stats: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "valid": 0}
    )

    for gt in gt_rows:
        text_id = str(gt.get("text_id", ""))
        if not isinstance(gt.get("cdi_label"), bool):
            raise ValueError(f"Missing boolean cdi_label for {text_id}")
        gold = bool(gt["cdi_label"])
        pred_row = pred_by_id.get(text_id)
        raw_pred = "" if pred_row is None else str(pred_row.get("pred_raw", ""))
        if pred_row is None:
            missing_predictions += 1
        pred = parse_bool(raw_pred)
        valid = pred is not None
        is_correct = valid and pred == gold

        gold_labels.append(gold)
        pred_labels.append(pred)
        valid_outputs += int(valid)
        correct += int(is_correct)
        correct_valid += int(is_correct)

        current_count = int(gt.get("current_intent_count", -1))
        reference_count = int(gt.get("reference_intent_count", -1))
        key = (current_count, reference_count)
        pair_stats[key]["total"] += 1
        pair_stats[key]["correct"] += int(is_correct)
        pair_stats[key]["valid"] += int(valid)

        scored_rows.append(
            {
                "text_id": text_id,
                "task": "cdi",
                "pred_raw": raw_pred,
                "gold_label": gold,
                "pred_label": pred,
                "valid_output": valid,
                "correct": is_correct,
                "current_intent_count": current_count,
                "reference_intent_count": reference_count,
                "reference_text_id": gt.get("reference_text_id", ""),
                "reference_query": gt.get("reference_query", ""),
            }
        )

    total = len(gt_rows)
    true_stats = class_stats(gold_labels, pred_labels, True)
    false_stats = class_stats(gold_labels, pred_labels, False)
    macro_f1 = (float(true_stats["f1"]) + float(false_stats["f1"])) / 2.0
    balanced_accuracy = (
        float(true_stats["recall"]) + float(false_stats["recall"])
    ) / 2.0

    per_count_pair = []
    for (current_count, reference_count), stats in sorted(pair_stats.items()):
        per_count_pair.append(
            {
                "current_intent_count": current_count,
                "reference_intent_count": reference_count,
                "num_examples": stats["total"],
                "valid_output_rate": safe_div(stats["valid"], stats["total"]),
                "accuracy": safe_div(stats["correct"], stats["total"]),
            }
        )

    metrics = {
        "task": "cdi",
        "num_examples": total,
        "missing_predictions": missing_predictions,
        "num_true": sum(gold_labels),
        "num_false": total - sum(gold_labels),
        "valid_output_rate": safe_div(valid_outputs, total),
        "accuracy": safe_div(correct, total),
        "accuracy_on_valid_outputs": safe_div(correct_valid, valid_outputs),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "true_class": true_stats,
        "false_class": false_stats,
        "invalid_outputs": total - valid_outputs,
        "per_intent_count_pair": per_count_pair,
    }
    return metrics, scored_rows


def pct(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


def format_metrics(metrics: dict) -> str:
    if metrics["task"] == "pii":
        di = metrics["domain_intent_micro"]
        pair = metrics["intent_slot_pair_micro"]
        return "\n".join(
            [
                "PII Evaluation",
                f"Examples: {metrics['num_examples']}",
                f"Missing predictions: {metrics['missing_predictions']}",
                f"Valid JSON Rate: {pct(metrics['valid_json_rate'])}",
                f"Candidate Valid Rate: {pct(metrics['candidate_valid_rate'])}",
                f"Relation Exact Match: {pct(metrics['relation_exact_match'])}",
                f"Domain-Intent Micro P/R/F1: {pct(di['precision'])} / {pct(di['recall'])} / {pct(di['f1'])}",
                f"Intent-Slot Pair Micro P/R/F1: {pct(pair['precision'])} / {pct(pair['recall'])} / {pct(pair['f1'])}",
            ]
        )

    true_stats = metrics["true_class"]
    false_stats = metrics["false_class"]
    return "\n".join(
        [
            "CDI Evaluation",
            f"Examples: {metrics['num_examples']} (true={metrics['num_true']}, false={metrics['num_false']})",
            f"Missing predictions: {metrics['missing_predictions']}",
            f"Valid Output Rate: {pct(metrics['valid_output_rate'])}",
            f"Accuracy: {pct(metrics['accuracy'])}",
            f"Accuracy on Valid Outputs: {pct(metrics['accuracy_on_valid_outputs'])}",
            f"Macro-F1: {pct(metrics['macro_f1'])}",
            f"Balanced Accuracy: {pct(metrics['balanced_accuracy'])}",
            f"True-class P/R/F1: {pct(true_stats['precision'])} / {pct(true_stats['recall'])} / {pct(true_stats['f1'])}",
            f"False-class P/R/F1: {pct(false_stats['precision'])} / {pct(false_stats['recall'])} / {pct(false_stats['f1'])}",
            f"Invalid outputs: {metrics['invalid_outputs']}",
        ]
    )


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_file)
    gt_path = Path(args.gt_file)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    pred_rows = load_jsonl(pred_path)
    gt_rows = load_jsonl(gt_path)
    if args.task == "pii":
        metrics, scored_rows = evaluate_pii(pred_rows, gt_rows)
    else:
        metrics, scored_rows = evaluate_cdi(pred_rows, gt_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = output_dir / "metrics.json"
    metrics_txt = output_dir / "metrics.txt"
    scored_jsonl = output_dir / "scored_predictions.jsonl"

    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_txt.write_text(format_metrics(metrics) + "\n", encoding="utf-8")
    with scored_jsonl.open("w", encoding="utf-8") as output:
        for row in scored_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(format_metrics(metrics))
    print(f"[INFO] metrics.json -> {metrics_json}")
    print(f"[INFO] scored predictions -> {scored_jsonl}")


if __name__ == "__main__":
    main()
