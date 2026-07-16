#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List


def is_plausible(item: Dict[str, Any]) -> bool:
    score = item.get("score", {})
    return bool(score.get("valid_json", 0)) and bool(item.get("raw", "").strip())


def build_pairs(input_jsonl: str, output_jsonl: str, min_score_margin: float, max_pairs_per_sample: int, pair_mode: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    stats = {"samples": 0, "pairs": 0, "dropped_no_pair": 0, "dropped_tie": 0}
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["samples"] += 1
            candidates: List[Dict[str, Any]] = [c for c in row.get("scored_nbest", []) if is_plausible(c)]
            candidates.sort(key=lambda x: (float(x.get("preference_score", 0.0)), -int(x.get("rank", 0))), reverse=True)
            if len(candidates) < 2:
                stats["dropped_no_pair"] += 1
                continue
            if pair_mode == "oracle_vs_top1":
                top1 = next((c for c in candidates if int(c.get("rank", -1)) == 0), None)
                oracle = max(
                    candidates,
                    key=lambda c: (
                        float(c.get("preference_score", 0.0)),
                        -int(c.get("rank", 999999)),
                    ),
                )
                if top1 is None or int(oracle.get("rank", -1)) == 0 or oracle.get("raw", "").strip() == top1.get("raw", "").strip():
                    stats["dropped_no_pair"] += 1
                    continue
                rejected_candidates = [top1]
            else:
                oracle = candidates[0]
                rejected_candidates = list(reversed(candidates[1:]))

            chosen = oracle
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
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["pairs"] += 1
                pairs += 1
                if pairs >= max_pairs_per_sample:
                    break
            if pairs == 0:
                stats["dropped_no_pair"] += 1
    stats["pair_coverage"] = stats["pairs"] / stats["samples"] if stats["samples"] else 0.0
    with open(output_jsonl + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    p = argparse.ArgumentParser("Build SimPO chosen/rejected pairs from scored n-best")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--min_score_margin", type=float, default=0.1)
    p.add_argument("--max_pairs_per_sample", type=int, default=1)
    p.add_argument("--pair_mode", choices=["nbest_only", "oracle_vs_top1"], default="nbest_only")
    args = p.parse_args()
    stats = build_pairs(args.input_jsonl, args.output_jsonl, args.min_score_margin, args.max_pairs_per_sample, args.pair_mode)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
