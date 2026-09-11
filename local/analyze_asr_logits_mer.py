#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from local.metrics import edit_distance, normalize_text, tokenize_for_mer  # noqa: E402


METRIC_KEYS = ("asr_avg_logit", "asr_avg_logprob", "asr_avg_prob")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _index_by_text_id(rows: List[Dict[str, Any]], path: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for line_no, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", "")).strip()
        if not text_id:
            raise ValueError(f"Missing text_id at {path}:{line_no}")
        if text_id in indexed:
            raise ValueError(f"Duplicate text_id={text_id!r} in {path}")
        indexed[text_id] = row
    return indexed


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = sum(v * v for v in dx)
    denom_y = sum(v * v for v in dy)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(denom_x * denom_y)


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average_rank
        i = j
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _sample_mer(reference: Any, hypothesis: Any) -> float:
    ref_tokens = tokenize_for_mer(normalize_text(reference))
    hyp_tokens = tokenize_for_mer(normalize_text(hypothesis))
    errors = edit_distance(ref_tokens, hyp_tokens)
    return errors / len(ref_tokens) if ref_tokens else 0.0


def build_analysis_rows(
    pred_rows: List[Dict[str, Any]],
    gt_rows: List[Dict[str, Any]],
    pred_path: str = "predictions",
    gt_path: str = "ground_truth",
) -> List[Dict[str, Any]]:
    gt_by_id = _index_by_text_id(gt_rows, gt_path)
    analysis_rows: List[Dict[str, Any]] = []

    for pred in pred_rows:
        text_id = str(pred.get("text_id", "")).strip()
        if not text_id:
            continue
        gt = gt_by_id.get(text_id)
        if gt is None:
            raise ValueError(f"text_id={text_id!r} from {pred_path} is missing in {gt_path}")

        row: Dict[str, Any] = {
            "text_id": text_id,
            "query": gt.get("query", ""),
            "pred_query": pred.get("pred_query", ""),
            "mer": _sample_mer(gt.get("query", ""), pred.get("pred_query", "")),
            "asr_token_count": int(pred.get("asr_token_count", 0) or 0),
        }
        valid_metric = False
        for key in METRIC_KEYS:
            value = _finite_float(pred.get(key))
            row[key] = value
            valid_metric = valid_metric or value is not None
        if valid_metric:
            analysis_rows.append(row)

    return analysis_rows


def _correlation_summary(rows: List[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    pairs = [
        (row[metric_key], row["mer"])
        for row in rows
        if row.get(metric_key) is not None and _finite_float(row.get("mer")) is not None
    ]
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    return {
        "count": len(pairs),
        "pearson_r": _pearson(xs, ys),
        "spearman_rho": _spearman(xs, ys),
        "metric_mean": _mean(xs) if xs else None,
        "mer_mean": _mean(ys) if ys else None,
    }


def _equal_count_bins(
    rows: List[Dict[str, Any]], metric_key: str, num_bins: int
) -> List[Dict[str, Any]]:
    pairs = sorted(
        (
            float(row[metric_key]),
            float(row["mer"]),
        )
        for row in rows
        if row.get(metric_key) is not None and _finite_float(row.get("mer")) is not None
    )
    if not pairs:
        return []

    bin_count = min(max(1, num_bins), len(pairs))
    bins: List[Dict[str, Any]] = []
    for bin_idx in range(bin_count):
        start = bin_idx * len(pairs) // bin_count
        end = (bin_idx + 1) * len(pairs) // bin_count
        chunk = pairs[start:end]
        if not chunk:
            continue
        xs = [x for x, _ in chunk]
        ys = [y for _, y in chunk]
        bins.append(
            {
                "bin": bin_idx + 1,
                "count": len(chunk),
                "metric_min": min(xs),
                "metric_max": max(xs),
                "metric_mean": _mean(xs),
                "mer_mean": _mean(ys),
            }
        )
    return bins


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "text_id",
        "query",
        "pred_query",
        "mer",
        "asr_token_count",
        *METRIC_KEYS,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: List[Dict[str, Any]], bins: List[Dict[str, Any]], path: str) -> None:
    pairs = [
        (float(row["asr_avg_logit"]), float(row["mer"]))
        for row in rows
        if row.get("asr_avg_logit") is not None
    ]
    if not pairs:
        return

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.35, s=14, label="utterances")
    if bins:
        plt.plot(
            [row["metric_mean"] for row in bins],
            [row["mer_mean"] for row in bins],
            marker="o",
            linewidth=2,
            label="equal-count bin mean",
        )
    plt.xlabel("Average selected logit over generated asr_text tokens")
    plt.ylabel("Utterance MER")
    plt.title("ASR token confidence vs. MER")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def analyze(
    pred_file: str,
    gt_file: str,
    output_dir: str,
    num_bins: int = 10,
) -> Dict[str, Any]:
    pred_rows = _load_jsonl(pred_file)
    gt_rows = _load_jsonl(gt_file)
    rows = build_analysis_rows(pred_rows, gt_rows, pred_file, gt_file)
    if not rows:
        raise ValueError(
            "No ASR logit statistics found in predictions. "
            "Rerun inference with --save_asr_logits."
        )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "asr_logits_mer.csv")
    summary_path = os.path.join(output_dir, "asr_logits_mer_summary.json")
    plot_path = os.path.join(output_dir, "asr_avg_logit_vs_mer.png")

    _write_csv(rows, csv_path)
    correlations = {key: _correlation_summary(rows, key) for key in METRIC_KEYS}
    bins = _equal_count_bins(rows, "asr_avg_logit", num_bins)
    summary = {
        "num_predictions": len(pred_rows),
        "num_analyzed": len(rows),
        "coverage": len(rows) / len(pred_rows) if pred_rows else 0.0,
        "metric_definition": {
            "asr_avg_logit": (
                "Mean raw pre-softmax model logit assigned to generated tokens "
                "overlapping the JSON asr_text string value, recomputed by teacher "
                "forcing the top-1 generated sequence."
            ),
            "asr_avg_logprob": (
                "Mean log-probability of the same generated asr_text tokens."
            ),
            "asr_avg_prob": (
                "Mean probability of the same generated asr_text tokens."
            ),
            "mer": (
                "Per-utterance MER computed with local.metrics tokenization and "
                "edit distance."
            ),
        },
        "correlations_with_mer": correlations,
        "asr_avg_logit_equal_count_bins": bins,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _plot(rows, bins, plot_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze generated asr_text token confidence against utterance MER."
    )
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_bins", type=int, default=10)
    args = parser.parse_args()

    summary = analyze(
        pred_file=args.pred_file,
        gt_file=args.gt_file,
        output_dir=args.output_dir,
        num_bins=args.num_bins,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
