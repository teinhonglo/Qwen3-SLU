#!/usr/bin/env python3
"""Create task-selectable StructPrompt training data for MAC-SLU ablations.

This reuses the exact SLU/PII/CDI construction and sampling seeds from
prepare_macslu_structprompt_jsonl.py. Selected training tasks are balanced to
the same per-task size used by the full SLU+PII+CDI setup, so an ablation
removes tasks without changing the exposure count of the remaining tasks.
Dev/test remain full-SLU only.
"""

import argparse
import json
import random
from pathlib import Path

from prepare_macslu_structprompt_jsonl import (
    build_cdi_rows,
    load_jsonl,
    pii_row,
    repeat_rows_to_size,
    slu_row,
)


TASK_ORDER = ("slu", "pii", "cdi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create selectable SLU/PII/CDI StructPrompt MAC-SLU JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--expand-splits", nargs="+", default=["train"])
    parser.add_argument(
        "--train-tasks",
        nargs="+",
        default=list(TASK_ORDER),
        help="Training tasks selected from: slu pii cdi. SLU is required.",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--cdi-pairs-per-class", type=int, default=1)
    parser.add_argument("--cdi-max-anchor-intent-count", type=int, default=3)
    return parser.parse_args()


def normalize_train_tasks(tasks: list[str]) -> list[str]:
    requested = [str(task).strip().lower() for task in tasks if str(task).strip()]
    unknown = sorted(set(requested).difference(TASK_ORDER))
    if unknown:
        raise ValueError(
            f"Unsupported --train-tasks: {unknown}. Allowed tasks: {list(TASK_ORDER)}"
        )
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate --train-tasks are not allowed: {requested}")
    if "slu" not in requested:
        raise ValueError("--train-tasks must include the main task: slu")
    return [task for task in TASK_ORDER if task in requested]


def convert_split(
    src_path: Path,
    output_path: Path,
    expand: bool,
    seed: int,
    train_tasks: list[str],
    cdi_pairs_per_class: int,
    cdi_max_anchor_intent_count: int,
) -> dict[str, int]:
    source_rows = load_jsonl(src_path)
    counts = {
        "source_rows": len(source_rows),
        "slu": 0,
        "pii": 0,
        "cdi": 0,
        "cdi_true": 0,
        "cdi_false": 0,
        "cdi_anchors": 0,
        "balance_target_per_task": 0,
        "total": 0,
    }

    if not expand:
        output_rows = [slu_row(row) for row in source_rows]
    else:
        # Always construct CDI with the same seed as the full setup. Its size is
        # the balancing target even when CDI itself is ablated, which preserves
        # the exposure count of SLU/PII relative to full StructSFT.
        cdi_groups = build_cdi_rows(
            source_rows,
            random.Random(seed + 1),
            pairs_per_class=cdi_pairs_per_class,
            max_anchor_intent_count=cdi_max_anchor_intent_count,
        )
        counts["cdi_anchors"] = sum(bool(group) for group in cdi_groups)
        flat_cdi_rows = [row for group in cdi_groups for row in group]
        target_size = len(flat_cdi_rows)
        counts["balance_target_per_task"] = target_size
        if target_size == 0:
            raise ValueError("CDI balancing target is empty; cannot build task ablation")

        task_rows: dict[str, list[dict]] = {}
        if "slu" in train_tasks:
            task_rows["slu"] = repeat_rows_to_size(
                [slu_row(row) for row in source_rows],
                target_size,
                random.Random(seed + 2),
                "SLU",
            )
        if "pii" in train_tasks:
            # Use one shared RNG exactly as in the full-data preparation.
            pii_rng = random.Random(seed)
            pii_rows = [pii_row(row, pii_rng) for row in source_rows]
            task_rows["pii"] = repeat_rows_to_size(
                pii_rows,
                target_size,
                random.Random(seed + 3),
                "PII",
            )
        if "cdi" in train_tasks:
            task_rows["cdi"] = flat_cdi_rows

        # Interleave selected tasks so their local ordering mirrors the original
        # 1:1:1 construction, with omitted tasks simply removed.
        output_rows = []
        for row_index in range(target_size):
            for task in train_tasks:
                output_rows.append(task_rows[task][row_index])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for result in output_rows:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            task = result["task"]
            counts[task] += 1
            if task == "cdi":
                counts["cdi_true" if result["cdi_label"] else "cdi_false"] += 1
            counts["total"] += 1

    if expand:
        expected = counts["balance_target_per_task"] * len(train_tasks)
    else:
        expected = counts["source_rows"]
    if counts["total"] != expected:
        raise RuntimeError(
            f"Sanity check failed for {src_path}: total={counts['total']}, expected={expected}"
        )
    return counts


def main() -> None:
    args = parse_args()
    train_tasks = normalize_train_tasks(args.train_tasks)

    src_root = Path(args.src_json_root)
    output_root = Path(args.json_root)
    expand_splits = set(args.expand_splits)
    unknown_splits = expand_splits.difference(args.splits)
    if unknown_splits:
        raise ValueError(f"--expand-splits not present in --splits: {sorted(unknown_splits)}")

    print(f"[INFO] train_tasks={' '.join(train_tasks)}")
    for split_index, split in enumerate(args.splits):
        src_path = src_root / f"{split}.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"Required source JSONL not found: {src_path}")
        counts = convert_split(
            src_path=src_path,
            output_path=output_root / f"{split}.jsonl",
            expand=split in expand_splits,
            seed=args.seed + split_index * 10000,
            train_tasks=train_tasks,
            cdi_pairs_per_class=args.cdi_pairs_per_class,
            cdi_max_anchor_intent_count=args.cdi_max_anchor_intent_count,
        )
        print(f"[INFO] {split}:")
        print(f"source_rows={counts['source_rows']}")
        print(f"slu={counts['slu']}")
        print(f"pii={counts['pii']}")
        print(f"cdi={counts['cdi']}")
        if split in expand_splits:
            print(f"balance_target_per_task={counts['balance_target_per_task']}")
            print(f"cdi_anchors={counts['cdi_anchors']}")
        if counts["cdi"]:
            print(f"cdi_true={counts['cdi_true']}")
            print(f"cdi_false={counts['cdi_false']}")
        print(f"total={counts['total']}")


if __name__ == "__main__":
    main()
