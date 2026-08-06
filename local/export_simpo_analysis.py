#!/usr/bin/env python3
"""Export sample-level n-best and SimPO pair information for analysis."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set


def load_pairs(path: str) -> DefaultDict[str, List[Dict[str, Any]]]:
    pairs_by_id: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            text_id = str(pair.get("text_id", ""))
            if not text_id:
                raise ValueError(f"Missing text_id in {path}:{line_number}")
            pairs_by_id[text_id].append(pair)
    return pairs_by_id


def find_candidate_rank(
    candidates: List[Dict[str, Any]],
    raw: Any,
    score: Any,
    excluded_ranks: Optional[Set[int]] = None,
) -> int:
    """Find the original rank of a pair endpoint, including duplicate raw outputs."""
    excluded_ranks = excluded_ranks or set()
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("raw", "") == raw
        and candidate.get("score", {}) == score
        and int(candidate.get("rank", -1)) not in excluded_ranks
    ]
    if not matches:
        matches = [
            candidate
            for candidate in candidates
            if candidate.get("raw", "") == raw
            and int(candidate.get("rank", -1)) not in excluded_ranks
        ]
    if not matches:
        raise ValueError("Unable to match SimPO pair target to a scored n-best candidate")
    return min(int(candidate["rank"]) for candidate in matches)


def export_analysis(
    input_jsonl: str, pairs_jsonl: str, output_jsonl: str
) -> Dict[str, int]:
    pairs_by_id = load_pairs(pairs_jsonl)
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    seen_ids: Set[str] = set()
    sample_count = 0
    pair_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(
        output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text_id = str(row.get("text_id", ""))
            if not text_id:
                raise ValueError(f"Missing text_id in {input_jsonl}:{line_number}")
            if text_id in seen_ids:
                raise ValueError(f"Duplicate text_id in {input_jsonl}: {text_id!r}")
            seen_ids.add(text_id)

            candidates = sorted(
                row.get("scored_nbest", []), key=lambda candidate: int(candidate.get("rank", -1))
            )
            chosen_targets: List[str] = []
            rejected_targets: List[str] = []
            chosen_ranks: List[int] = []
            rejected_ranks: List[int] = []
            used_rejected_ranks: Set[int] = set()

            for pair in pairs_by_id.get(text_id, []):
                chosen = str(pair.get("chosen", ""))
                rejected = str(pair.get("rejected", ""))
                if pair.get("chosen_source") == "ground_truth":
                    chosen_rank = -1
                else:
                    chosen_rank = find_candidate_rank(
                        candidates, chosen, pair.get("chosen_score", {})
                    )
                rejected_rank = find_candidate_rank(
                    candidates,
                    rejected,
                    pair.get("rejected_score", {}),
                    used_rejected_ranks,
                )
                used_rejected_ranks.add(rejected_rank)
                chosen_targets.append(chosen)
                rejected_targets.append(rejected)
                chosen_ranks.append(chosen_rank)
                rejected_ranks.append(rejected_rank)
                pair_count += 1

            output = {
                "id": text_id,
                "target": [str(row.get("text", ""))],
                "pred_targets": [str(candidate.get("raw", "")) for candidate in candidates],
                "ranks": [int(candidate.get("rank", -1)) for candidate in candidates],
                "emas": [
                    int(candidate.get("score", {}).get("oracle_ema", 0))
                    for candidate in candidates
                ],
                "preference_scores": [
                    float(candidate.get("preference_score", 0.0)) for candidate in candidates
                ],
                "chosen_targets": chosen_targets,
                "rejected_targets": rejected_targets,
                "chosen_ranks": chosen_ranks,
                "rejected_ranks": rejected_ranks,
            }
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            sample_count += 1

    unknown_pair_ids = set(pairs_by_id) - seen_ids
    if unknown_pair_ids:
        preview = ", ".join(repr(value) for value in sorted(unknown_pair_ids)[:5])
        raise ValueError(f"Pair text_id values missing from scored input: {preview}")
    return {"samples": sample_count, "pairs": pair_count}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aligned n-best targets, scores, and SimPO pair selections"
    )
    parser.add_argument("--input_jsonl", required=True, help="Stage 2 scored_nbest.jsonl")
    parser.add_argument("--pairs_jsonl", required=True, help="Stage 3 simpo_pairs.jsonl")
    parser.add_argument("--output_jsonl", required=True, help="Output analysis JSONL")
    args = parser.parse_args()
    stats = export_analysis(args.input_jsonl, args.pairs_jsonl, args.output_jsonl)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
