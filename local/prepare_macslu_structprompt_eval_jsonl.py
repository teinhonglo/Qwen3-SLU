#!/usr/bin/env python3
"""Prepare deterministic PII/CDI evaluation JSONL for MAC-SLU StructSFT."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prepare_macslu_structprompt_jsonl import (
    CDI_PROMPT_TEMPLATE,
    bool_target_text,
    intent_count,
    load_jsonl,
    pii_row,
    semantic_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare held-out PII/CDI evaluation data from MAC-SLU JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--cdi-pairs-per-class", type=int, default=1)
    parser.add_argument("--cdi-max-anchor-intent-count", type=int, default=3)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_cdi_eval_rows(
    rows: list[dict],
    rng: random.Random,
    pairs_per_class: int,
    max_anchor_intent_count: int,
) -> tuple[list[dict], dict[str, int]]:
    """Build balanced CDI pairs without failing on sparse held-out intent counts.

    An anchor is evaluated only when at least K positive and K negative references
    exist in the same held-out split. This keeps each evaluated anchor balanced and
    makes the reported CDI accuracy directly interpretable.
    """
    if pairs_per_class < 1:
        raise ValueError("--cdi-pairs-per-class must be at least 1")
    if max_anchor_intent_count < 0:
        raise ValueError("--cdi-max-anchor-intent-count must be at least 0")

    infos = []
    pools: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        query = str(row.get("query", "") or "").strip()
        if not query:
            raise ValueError(
                f"CDI requires non-empty query text: {row.get('text_id', index)}"
            )
        count = intent_count(semantic_frames(row))
        info = {
            "index": index,
            "row": row,
            "query": query,
            "count": count,
            "text_id": str(row.get("text_id", "")),
        }
        infos.append(info)
        pools[count].append(info)

    stats = {
        "source_rows": len(rows),
        "eligible_anchors": 0,
        "evaluated_anchors": 0,
        "skipped_over_intent_limit": 0,
        "skipped_no_positive": 0,
        "skipped_no_negative": 0,
        "cdi_rows": 0,
        "cdi_true": 0,
        "cdi_false": 0,
    }
    output_rows = []

    for current in infos:
        if current["count"] > max_anchor_intent_count:
            stats["skipped_over_intent_limit"] += 1
            continue

        stats["eligible_anchors"] += 1
        positive = [
            candidate
            for candidate in pools[current["count"]]
            if candidate["index"] != current["index"]
        ]
        negative = [
            candidate
            for count, candidates in pools.items()
            if count != current["count"]
            for candidate in candidates
        ]

        if len(positive) < pairs_per_class:
            stats["skipped_no_positive"] += 1
            continue
        if len(negative) < pairs_per_class:
            stats["skipped_no_negative"] += 1
            continue

        sampled_pairs = [
            (True, pair_index, reference)
            for pair_index, reference in enumerate(
                rng.sample(positive, pairs_per_class), start=1
            )
        ]
        sampled_pairs.extend(
            (False, pair_index, reference)
            for pair_index, reference in enumerate(
                rng.sample(negative, pairs_per_class), start=1
            )
        )
        rng.shuffle(sampled_pairs)

        stats["evaluated_anchors"] += 1
        for label, pair_index, reference in sampled_pairs:
            pair_type = "positive" if label else "negative"
            result = dict(current["row"])
            result["text_id"] = (
                f"{current['text_id']}__cdi_eval_{pair_type}_{pair_index}"
            )
            result["task"] = "cdi"
            result["prompt"] = CDI_PROMPT_TEMPLATE.format(
                reference_query=reference["query"]
            )
            result["text"] = bool_target_text(label)
            result["reference_text_id"] = reference["text_id"]
            result["reference_query"] = reference["query"]
            result["current_intent_count"] = current["count"]
            result["reference_intent_count"] = reference["count"]
            result["cdi_label"] = label
            output_rows.append(result)
            stats["cdi_rows"] += 1
            stats["cdi_true" if label else "cdi_false"] += 1

    return output_rows, stats


def main() -> None:
    args = parse_args()
    src_path = Path(args.src_json_root) / f"{args.split}.jsonl"
    if not src_path.is_file():
        raise FileNotFoundError(f"Required source JSONL not found: {src_path}")

    rows = load_jsonl(src_path)
    pii_rng = random.Random(args.seed)
    cdi_rng = random.Random(args.seed + 1)

    pii_rows = [pii_row(row, pii_rng) for row in rows]
    cdi_rows, cdi_stats = build_cdi_eval_rows(
        rows,
        cdi_rng,
        pairs_per_class=args.cdi_pairs_per_class,
        max_anchor_intent_count=args.cdi_max_anchor_intent_count,
    )

    output_root = Path(args.output_root)
    pii_path = output_root / f"{args.split}_pii.jsonl"
    cdi_path = output_root / f"{args.split}_cdi.jsonl"
    summary_path = output_root / f"{args.split}_summary.json"

    write_jsonl(pii_path, pii_rows)
    write_jsonl(cdi_path, cdi_rows)

    summary = {
        "split": args.split,
        "seed": args.seed,
        "pii_rows": len(pii_rows),
        "cdi_pairs_per_class": args.cdi_pairs_per_class,
        "cdi_max_anchor_intent_count": args.cdi_max_anchor_intent_count,
        **cdi_stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] PII evaluation rows: {len(pii_rows)} -> {pii_path}")
    print(
        "[INFO] CDI evaluation: "
        f"anchors={cdi_stats['evaluated_anchors']}/"
        f"{cdi_stats['eligible_anchors']}, "
        f"rows={cdi_stats['cdi_rows']}, "
        f"true={cdi_stats['cdi_true']}, false={cdi_stats['cdi_false']}"
    )
    print(f"[INFO] Evaluation summary -> {summary_path}")


if __name__ == "__main__":
    main()
