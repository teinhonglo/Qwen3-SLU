#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def is_plausible(item: Dict[str, Any]) -> bool:
    score = item.get("score", {})
    return bool(score.get("valid_json", 0)) and bool(item.get("raw", "").strip())


def is_oracle(item: Dict[str, Any]) -> bool:
    return bool(item.get("score", {}).get("oracle_ema", 0))


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
    if pair_mode not in {"nbest_only", "nbest_oracle"}:
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
            candidates: List[Dict[str, Any]] = sorted(
                (c for c in scored_nbest if is_plausible(c)),
                key=lambda x: int(x.get("rank", 999999)),
            )
            oracle_candidates = [candidate for candidate in candidates if is_oracle(candidate)]
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
    p.add_argument("--pair_mode", choices=["nbest_only", "nbest_oracle"], default="nbest_only")
    args = p.parse_args()
    stats = build_pairs(args.input_jsonl, args.output_jsonl, args.min_score_margin, args.max_pairs_per_sample, args.pair_mode)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
