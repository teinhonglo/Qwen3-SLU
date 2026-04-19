#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}") from e
    return rows


def normalize_sem(sem):
    if not isinstance(sem, list):
        return []
    out = []
    for frame in sem:
        if not isinstance(frame, dict):
            continue
        domain = str(frame.get("domain", "")).strip().lower()
        intent = str(frame.get("intent", "")).strip().lower()
        slots = frame.get("slots", {})
        if isinstance(slots, dict):
            nslots = {str(k).strip().lower(): str(v).strip().lower() for k, v in slots.items()}
        else:
            nslots = {}
        out.append({"domain": domain, "intent": intent, "slots": nslots})
    return out


def frame_keys(sem):
    return Counter((f["domain"], f["intent"]) for f in sem)


def slot_items(sem):
    c = Counter()
    for f in sem:
        for k, v in f["slots"].items():
            c[(f["domain"], f["intent"], k, v)] += 1
    return c


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser("Evaluate qwen-format SLURP predictions")
    parser.add_argument("prediction_jsonl")
    parser.add_argument("ground_truth_jsonl")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    preds = load_jsonl(args.prediction_jsonl)
    gts = load_jsonl(args.ground_truth_jsonl)

    by_id_pred = {str(x.get("text_id")): x for x in preds}
    by_id_gt = {str(x.get("text_id")): x for x in gts}

    common_ids = sorted(set(by_id_pred) & set(by_id_gt))
    missing_pred = sorted(set(by_id_gt) - set(by_id_pred))

    intent_correct = 0
    exact_correct = 0

    frame_tp = frame_fp = frame_fn = 0
    slot_tp = slot_fp = slot_fn = 0

    for sid in common_ids:
        pred_sem = normalize_sem(by_id_pred[sid].get("pred_semantics", []))
        gt_sem = normalize_sem(by_id_gt[sid].get("semantics", []))

        pred_frames = frame_keys(pred_sem)
        gt_frames = frame_keys(gt_sem)
        frame_tp += sum((pred_frames & gt_frames).values())
        frame_fp += sum((pred_frames - gt_frames).values())
        frame_fn += sum((gt_frames - pred_frames).values())

        pred_slots = slot_items(pred_sem)
        gt_slots = slot_items(gt_sem)
        slot_tp += sum((pred_slots & gt_slots).values())
        slot_fp += sum((pred_slots - gt_slots).values())
        slot_fn += sum((gt_slots - pred_slots).values())

        if pred_frames == gt_frames:
            intent_correct += 1
        if pred_sem == gt_sem:
            exact_correct += 1

    n = len(common_ids)
    frame_p, frame_r, frame_f1 = f1_from_counts(frame_tp, frame_fp, frame_fn)
    slot_p, slot_r, slot_f1 = f1_from_counts(slot_tp, slot_fp, slot_fn)

    lines = [
        f"samples_eval={n}",
        f"samples_missing_pred={len(missing_pred)}",
        f"intent_acc={intent_correct / n if n else 0.0:.6f}",
        f"exact_match={exact_correct / n if n else 0.0:.6f}",
        f"frame_precision={frame_p:.6f}",
        f"frame_recall={frame_r:.6f}",
        f"frame_f1={frame_f1:.6f}",
        f"slot_precision={slot_p:.6f}",
        f"slot_recall={slot_r:.6f}",
        f"slot_f1={slot_f1:.6f}",
    ]

    report = "\n".join(lines)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
