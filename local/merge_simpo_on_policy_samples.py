#!/usr/bin/env python3
"""Merge one stochastic response per seed into SimPO-style candidate sets."""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


IDENTITY_FIELDS = ("query", "audio", "prompt", "text", "semantics")


def load_seed_file(path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text_id = str(row.get("text_id", "")).strip()
            if not text_id:
                raise ValueError(f"Missing text_id in {path}:{line_number}")
            if text_id in rows:
                raise ValueError(f"Duplicate text_id {text_id!r} in {path}")
            nbest = row.get("nbest", [])
            if not isinstance(nbest, list) or len(nbest) != 1:
                raise ValueError(
                    f"Expected exactly one sampled response in {path}:{line_number}, "
                    f"got {len(nbest) if isinstance(nbest, list) else type(nbest).__name__}"
                )
            order.append(text_id)
            rows[text_id] = row
    return order, rows


def merge_samples(
    input_jsonls: List[str], seeds: List[int], output_jsonl: str
) -> Dict[str, Any]:
    if len(input_jsonls) != len(seeds):
        raise ValueError(
            f"input_jsonls and seeds must have the same length, got "
            f"{len(input_jsonls)} and {len(seeds)}"
        )
    if not input_jsonls:
        raise ValueError("At least one seed file is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got: {seeds}")

    loaded = [load_seed_file(path) for path in input_jsonls]
    reference_order, reference_rows = loaded[0]
    reference_ids = set(reference_order)
    for path, (order, rows) in zip(input_jsonls[1:], loaded[1:]):
        current_ids = set(order)
        if current_ids != reference_ids:
            missing = sorted(reference_ids - current_ids)[:5]
            extra = sorted(current_ids - reference_ids)[:5]
            raise ValueError(
                f"Mismatched samples in {path}; missing={missing}, extra={extra}"
            )
        for text_id in reference_order:
            reference = reference_rows[text_id]
            current = rows[text_id]
            for field in IDENTITY_FIELDS:
                if current.get(field) != reference.get(field):
                    raise ValueError(
                        f"Field {field!r} differs across seed files for text_id={text_id!r}"
                    )

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    unique_distribution: Counter[int] = Counter()
    retained = 0
    dropped_all_identical = 0
    total_unique = 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for text_id in reference_order:
            responses = [
                str(rows[text_id]["nbest"][0])
                for _, rows in loaded
            ]
            unique_count = len(set(responses))
            unique_distribution[unique_count] += 1
            total_unique += unique_count
            if unique_count == 1:
                dropped_all_identical += 1
                continue

            output = dict(reference_rows[text_id])
            output["nbest"] = responses
            output["nbest_seeds"] = seeds
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            retained += 1

    total = len(reference_order)
    stats = {
        "input_files": input_jsonls,
        "seeds": seeds,
        "total_prompts": total,
        "retained_prompts": retained,
        "dropped_all_identical": dropped_all_identical,
        "retention_rate": retained / total if total else 0.0,
        "average_unique_responses": total_unique / total if total else 0.0,
        "unique_response_distribution": {
            str(count): frequency
            for count, frequency in sorted(unique_distribution.items())
        },
    }
    with open(output_jsonl + ".summary.json", "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independent seeded generations and remove all-identical prompts"
    )
    parser.add_argument("--input_jsonls", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    stats = merge_samples(args.input_jsonls, args.seeds, args.output_jsonl)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
