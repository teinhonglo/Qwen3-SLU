#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple


def is_plausible(item: Dict[str, Any]) -> bool:
    score = item.get("score", {})
    return bool(score.get("valid_json", 0)) and bool(item.get("raw", "").strip())


def is_oracle(item: Dict[str, Any]) -> bool:
    return bool(item.get("score", {}).get("oracle_ema", 0))


def oracle_balance_keep(row: Dict[str, Any], rank0_is_oracle: bool) -> Tuple[bool, str, float]:
    """Deterministically downsample easy (rank-0 oracle) examples by intent count."""
    if not rank0_is_oracle:
        return True, "rank0_error", 1.0

    semantics = row.get("semantics", [])
    intent_count = len(semantics) if isinstance(semantics, list) else 0
    if intent_count == 0:
        ratio = 0.1
    elif intent_count == 1:
        ratio = 0.05
    elif intent_count == 2:
        ratio = 0.20
    else:
        ratio = 1.00

    sample_key = json.dumps(
        [row.get("text_id", ""), row.get("audio", ""), row.get("query", "")],
        ensure_ascii=False,
        sort_keys=True,
    )
    bucket = int.from_bytes(hashlib.sha256(sample_key.encode("utf-8")).digest()[:8], "big") / 2**64
    return bucket < ratio, f"rank0_correct_{intent_count}_intents", ratio


def write_rank_preference_histogram(output_jsonl: str, values: List[Tuple[int, float]], bins: int = 10) -> None:
    ranks = sorted({rank for rank, _ in values})
    if not values:
        hist = {"bins": [], "ranks": [], "counts": {}}
    else:
        scores = [score for _, score in values]
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            edges = [min_score, max_score]
            counts = {str(rank): [0] for rank in ranks}
            for rank, _ in values:
                counts[str(rank)][0] += 1
        else:
            width = (max_score - min_score) / bins
            edges = [min_score + width * i for i in range(bins + 1)]
            counts = {str(rank): [0 for _ in range(bins)] for rank in ranks}
            for rank, score in values:
                idx = int((score - min_score) / width)
                if idx >= bins:
                    idx = bins - 1
                counts[str(rank)][idx] += 1
        hist = {"bins": edges, "ranks": ranks, "counts": counts}
    with open(output_jsonl + ".rank_preference_hist.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def build_pairs(input_jsonl: str, output_jsonl: str, min_score_margin: float, max_pairs_per_sample: int, pair_mode: str) -> Dict[str, Any]:
    if pair_mode not in {
        "nbest_only",
        "nbest_oracle",
        "oracle_balance",
        "sampled_highest_lowest",
        "oracle_sampled_highest_lowest",
    }:
        raise ValueError(f"Unsupported pair_mode: {pair_mode}")
    if max_pairs_per_sample < 1:
        raise ValueError("max_pairs_per_sample must be at least 1")
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    stats = {
        "samples": 0,
        "pairs": 0,
        "dropped_no_pair": 0,
        "dropped_tie": 0,
        "samples_with_nbest_oracle": 0,
        "samples_without_nbest_oracle": 0,
        "pairs_nbest_oracle_chosen": 0,
        "pairs_ground_truth_chosen": 0,
        "pairs_sampled_highest_chosen": 0,
        "pairs_sampled_oracle_chosen": 0,
        "dropped_oracle_balance": 0,
        "dropped_insufficient_candidates": 0,
        "dropped_no_sampled_oracle": 0,
        "dropped_no_nonoracle_rejected": 0,
        "oracle_balance_groups": {},
    }
    rank_preference_values: List[Tuple[int, float]] = []
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["samples"] += 1
            scored_nbest = row.get("scored_nbest", [])
            for candidate in scored_nbest:
                rank_preference_values.append(
                    (int(candidate.get("rank", -1)), float(candidate.get("preference_score", 0.0)))
                )

            if pair_mode in {"sampled_highest_lowest", "oracle_sampled_highest_lowest"}:
                candidates = list(scored_nbest)
                if len(candidates) < 2:
                    stats["dropped_insufficient_candidates"] += 1
                    stats["dropped_no_pair"] += 1
                    continue

                if pair_mode == "oracle_sampled_highest_lowest":
                    oracle_candidates = [candidate for candidate in candidates if is_oracle(candidate)]
                    nonoracle_candidates = [candidate for candidate in candidates if not is_oracle(candidate)]
                    if not oracle_candidates:
                        stats["samples_without_nbest_oracle"] += 1
                        stats["dropped_no_sampled_oracle"] += 1
                        stats["dropped_no_pair"] += 1
                        continue
                    stats["samples_with_nbest_oracle"] += 1
                    if not nonoracle_candidates:
                        stats["dropped_no_nonoracle_rejected"] += 1
                        stats["dropped_no_pair"] += 1
                        continue
                    chosen_candidates = oracle_candidates
                    rejected_candidates = nonoracle_candidates
                    chosen_source = "sampled_oracle"
                else:
                    chosen_candidates = candidates
                    rejected_candidates = candidates
                    chosen_source = "sampled_highest"

                chosen = max(
                    chosen_candidates,
                    key=lambda item: (
                        float(item.get("preference_score", 0.0)),
                        -int(item.get("rank", 999999)),
                    ),
                )
                rejected = min(
                    rejected_candidates,
                    key=lambda item: (
                        float(item.get("preference_score", 0.0)),
                        int(item.get("rank", 999999)),
                    ),
                )
                margin = float(chosen.get("preference_score", 0.0)) - float(
                    rejected.get("preference_score", 0.0)
                )
                if margin <= 0:
                    stats["dropped_tie"] += 1
                    stats["dropped_no_pair"] += 1
                    continue

                out = {
                    "text_id": row.get("text_id", ""),
                    "query": row.get("query", ""),
                    "audio": row.get("audio", ""),
                    "prompt": row.get("prompt", ""),
                    "semantics": row.get("semantics", []),
                    "chosen": chosen.get("raw", ""),
                    "rejected": rejected.get("raw", ""),
                    "chosen_score": chosen.get("score", {}),
                    "rejected_score": rejected.get("score", {}),
                    "pair_margin": margin,
                    "pair_mode": pair_mode,
                    "chosen_source": chosen_source,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["pairs"] += 1
                stats[f"pairs_{chosen_source}_chosen"] += 1
                continue

            candidates: List[Dict[str, Any]] = sorted(
                (c for c in scored_nbest if is_plausible(c)),
                key=lambda x: int(x.get("rank", 999999)),
            )
            oracle_candidates = [candidate for candidate in candidates if is_oracle(candidate)]
            rank0_is_oracle = bool(candidates and int(candidates[0].get("rank", -1)) == 0 and is_oracle(candidates[0]))
            if oracle_candidates:
                stats["samples_with_nbest_oracle"] += 1
                chosen = oracle_candidates[0]
                rejected_candidates = [candidate for candidate in candidates if not is_oracle(candidate)]
                chosen_source = "nbest_oracle"
            else:
                stats["samples_without_nbest_oracle"] += 1
                if pair_mode == "nbest_only":
                    stats["dropped_no_pair"] += 1
                    continue
                chosen = row.get("ground_truth_candidate")
                if not isinstance(chosen, dict) or not chosen.get("raw") or "preference_score" not in chosen:
                    raise ValueError(
                        "Missing scored ground_truth_candidate for "
                        f"text_id={row.get('text_id', '')!r}; rerun local/score_nbest_oracle.py"
                    )
                rejected_candidates = candidates
                chosen_source = "ground_truth"

            if pair_mode == "oracle_balance":
                keep, ratio_group, keep_ratio = oracle_balance_keep(row, rank0_is_oracle)
                group_stats = stats["oracle_balance_groups"].setdefault(
                    ratio_group, {"samples": 0, "kept": 0, "keep_ratio": keep_ratio}
                )
                group_stats["samples"] += 1
                if not keep:
                    stats["dropped_oracle_balance"] += 1
                    continue
                group_stats["kept"] += 1

            if not rejected_candidates:
                stats["dropped_no_pair"] += 1
                continue
            pairs = 0
            for rejected in rejected_candidates:
                margin = float(chosen.get("preference_score", 0.0)) - float(rejected.get("preference_score", 0.0))
                if margin <= 0:
                    stats["dropped_tie"] += 1
                    continue
                if margin < min_score_margin:
                    continue
                out = {
                    "text_id": row.get("text_id", ""),
                    "query": row.get("query", ""),
                    "audio": row.get("audio", ""),
                    "prompt": row.get("prompt", ""),
                    "semantics": row.get("semantics", []),
                    "chosen": chosen.get("raw", ""),
                    "rejected": rejected.get("raw", ""),
                    "chosen_score": chosen.get("score", {}),
                    "rejected_score": rejected.get("score", {}),
                    "pair_margin": margin,
                    "pair_mode": pair_mode,
                    "chosen_source": chosen_source,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["pairs"] += 1
                stats[f"pairs_{chosen_source}_chosen"] += 1
                pairs += 1
                if pairs >= max_pairs_per_sample:
                    break
            if pairs == 0:
                stats["dropped_no_pair"] += 1
    stats["pair_coverage"] = stats["pairs"] / stats["samples"] if stats["samples"] else 0.0
    with open(output_jsonl + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    write_rank_preference_histogram(output_jsonl, rank_preference_values)
    return stats


def main():
    p = argparse.ArgumentParser("Build SimPO chosen/rejected pairs from scored n-best")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--min_score_margin", type=float, default=0.1)
    p.add_argument("--max_pairs_per_sample", type=int, default=1)
    p.add_argument(
        "--pair_mode",
        choices=[
            "nbest_only",
            "nbest_oracle",
            "oracle_balance",
            "sampled_highest_lowest",
            "oracle_sampled_highest_lowest",
        ],
        default="nbest_only",
    )
    args = p.parse_args()
    stats = build_pairs(args.input_jsonl, args.output_jsonl, args.min_score_margin, args.max_pairs_per_sample, args.pair_mode)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
