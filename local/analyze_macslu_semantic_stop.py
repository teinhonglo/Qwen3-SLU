#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, median


COUNT_ORDER = ["no_intent", "correct_count", "under_by_1", "under_by_2plus", "overprediction"]
COUNT_LABELS = {
    "no_intent": "No intent",
    "correct_count": "Correct count",
    "under_by_1": "Under by 1",
    "under_by_2plus": "Under by 2+",
    "overprediction": "Overprediction",
}
OUTCOME_ORDER = ["exact_missing_frame_recovered", "duplicate_existing_frame", "wrong_extra_frame", "no_complete_extra_frame"]
OUTCOME_LABELS = {
    "exact_missing_frame_recovered": "Exact missing frame recovered",
    "duplicate_existing_frame": "Duplicate existing frame",
    "wrong_extra_frame": "Wrong extra frame",
    "no_complete_extra_frame": "No complete extra frame",
}


def load_records(root, splits, records_subdir=""):
    rows = []
    for split in splits:
        if records_subdir:
            path = os.path.join(root, split, records_subdir, "records.jsonl")
        else:
            path = os.path.join(root, split, "records.jsonl")
        if not os.path.isfile(path):
            print(f"[warning] missing records: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def finite_values(records, key):
    vals = []
    for r in records:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            vals.append(float(v))
    return vals


def rate(records, pred):
    return sum(1 for r in records if pred(r)) / len(records) if records else 0.0


def write_summary(rows, output_csv):
    fields = [
        "split", "count_group", "count", "mean_stop_logprob", "median_stop_logprob",
        "mean_stop_probability", "median_stop_probability", "mean_continue_logprob", "median_continue_logprob",
        "mean_stop_margin", "median_stop_margin", "exact_missing_frame_recovery_rate",
        "duplicate_existing_frame_rate", "wrong_extra_frame_rate", "no_complete_extra_frame_rate", "forced_full_exact_rate",
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("split", ""), r.get("count_status", ""))].append(r)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split in sorted({r.get("split", "") for r in rows}):
            for group in COUNT_ORDER:
                recs = groups.get((split, group), [])
                def mm(key, fn):
                    vals = finite_values(recs, key)
                    return fn(vals) if vals else ""
                w.writerow({
                    "split": split,
                    "count_group": group,
                    "count": len(recs),
                    "mean_stop_logprob": mm("stop_logprob", mean),
                    "median_stop_logprob": mm("stop_logprob", median),
                    "mean_stop_probability": mm("stop_probability", mean),
                    "median_stop_probability": mm("stop_probability", median),
                    "mean_continue_logprob": mm("continue_logprob", mean),
                    "median_continue_logprob": mm("continue_logprob", median),
                    "mean_stop_margin": mm("stop_margin", mean),
                    "median_stop_margin": mm("stop_margin", median),
                    "exact_missing_frame_recovery_rate": rate(recs, lambda r: r.get("forced_outcome") == "exact_missing_frame_recovered"),
                    "duplicate_existing_frame_rate": rate(recs, lambda r: r.get("forced_outcome") == "duplicate_existing_frame"),
                    "wrong_extra_frame_rate": rate(recs, lambda r: r.get("forced_outcome") == "wrong_extra_frame"),
                    "no_complete_extra_frame_rate": rate(recs, lambda r: r.get("forced_outcome") == "no_complete_extra_frame"),
                    "forced_full_exact_rate": rate(recs, lambda r: bool(r.get("forced_full_exact"))),
                })
    print(f"[info] saved: {output_csv}")


def plot_stop_logprob(rows, root, splits):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.6), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    all_vals = finite_values(rows, "stop_logprob")
    ylim = (min(all_vals) - 0.5, max(all_vals) + 0.5) if all_vals else (-20, 1)
    rng = np.random.default_rng(0)
    for ax, split in zip(axes, splits):
        data = [[r["stop_logprob"] for r in rows if r.get("split") == split and r.get("count_status") == g and isinstance(r.get("stop_logprob"), (int, float))] for g in COUNT_ORDER]
        
        ax.boxplot(data)
        ax.set_xticks(range(1, len(COUNT_ORDER) + 1))
        ax.set_xticklabels([COUNT_LABELS[g] for g in COUNT_ORDER])
        
        for i, vals in enumerate(data, start=1):
            if vals:
                ax.scatter(rng.normal(i, 0.04, len(vals)), vals, alpha=0.35, s=12)
                ax.text(i, ylim[1], f"n={len(vals)}", ha="center", va="top", fontsize=9)
        ax.set_title(f"({chr(96 + list(splits).index(split) + 1)}) {split.capitalize()}")
        ax.set_xlabel("Count status")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("Log probability of the first semantics-list closing token")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(root, f"fig_stop_logprob.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"[info] saved: {path}")
    plt.close(fig)


def plot_forced_outcome(rows, root, splits):
    import matplotlib.pyplot as plt
    import numpy as np
    groups = COUNT_ORDER[:4]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.8), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        bottoms = np.zeros(len(groups))
        totals = np.array([sum(1 for r in rows if r.get("split") == split and r.get("count_status") == g) for g in groups], dtype=float)
        for outcome, color in zip(OUTCOME_ORDER, colors):
            counts = np.array([sum(1 for r in rows if r.get("split") == split and r.get("count_status") == g and r.get("forced_outcome") == outcome) for g in groups], dtype=float)
            pct = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) * 100.0
            ax.bar(range(len(groups)), pct, bottom=bottoms, label=OUTCOME_LABELS[outcome], color=color)
            bottoms += pct
        for x, (g, total) in enumerate(zip(groups, totals)):
            ax.text(x, 102, f"n={int(total)}", ha="center", va="bottom", fontsize=9)
            recs = [r for r in rows if r.get("split") == split and r.get("count_status") == g]
            if g.startswith("under"):
                val = rate(recs, lambda r: r.get("forced_outcome") == "exact_missing_frame_recovered") * 100
                label = f"recover={val:.1f}%"
            else:
                val = rate(recs, lambda r: len(r.get("added_frames", [])) > 0) * 100
                label = f"extra={val:.1f}%"
            ax.text(x, 94, label, ha="center", va="top", fontsize=8, rotation=90)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([COUNT_LABELS[g] for g in groups], rotation=25, ha="right")
        ax.set_ylim(0, 112)
        ax.set_title(f"({chr(96 + list(splits).index(split) + 1)}) {split.capitalize()}")
        ax.set_xlabel("Count status")
    axes[0].set_ylabel("Forced outcome percentage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ["pdf", "png"]:
        path = os.path.join(root, f"fig_forced_outcome.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"[info] saved: {path}")
    plt.close(fig)


def print_key_stats(rows):
    for split in ["dev", "test"]:
        under = [r for r in rows if r.get("split") == split and r.get("count_status") in {"under_by_1", "under_by_2plus"}]
        no_intent = [r for r in rows if r.get("split") == split and r.get("count_status") == "no_intent"]
        print(f"{split.capitalize()} underprediction exact recovery rate: {rate(under, lambda r: r.get('forced_outcome') == 'exact_missing_frame_recovered'):.4f} ({len(under)} samples)")
        print(f"{split.capitalize()} no-intent extra-frame rate: {rate(no_intent, lambda r: len(r.get('added_frames', [])) > 0):.4f} ({len(no_intent)} samples)")


def parse_args():
    p = argparse.ArgumentParser("Analyze MAC-SLU semantic STOP diagnostic records")
    p.add_argument("--root", default="exp/macslu_semantic_stop")
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    p.add_argument("--records_subdir", default="", help="Optional per-split subdirectory containing records.jsonl")
    p.add_argument("--output_root", default="", help="Directory for summary.csv and figures; defaults to --root")
    p.add_argument("--skip_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_records(args.root, args.splits, records_subdir=args.records_subdir)
    output_root = args.output_root or args.root
    write_summary(rows, os.path.join(output_root, "summary.csv"))
    if not args.skip_plots:
        plot_stop_logprob(rows, output_root, args.splits)
        plot_forced_outcome(rows, output_root, args.splits)
    print_key_stats(rows)


if __name__ == "__main__":
    main()
