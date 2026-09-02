#!/usr/bin/env python3
"""Qwen3-Reranker control for clean MAC-SLU N-best evaluation.

This is intentionally independent from RobustGER: it consumes clean-test
N-best text only, performs no audio mixing, and never reads oracle candidates.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local.score_nbest_oracle import parse_hypothesis


SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
USER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return str(candidate.get("text", candidate.get("pred_raw", candidate.get("hyp", ""))))
    return str(candidate or "")


def rerank_scores(model, tokenizer, texts: List[str], device: torch.device) -> List[float]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]
    pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1)
    return torch.softmax(pair_logits.float(), dim=-1)[:, 1].cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean-test Qwen3-Reranker control")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=(
        "Select the most accurate MAC-SLU hypothesis for the user's spoken query."
    ))
    parser.add_argument("--n_best", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device).eval()

    rows = read_jsonl(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, 1):
            candidates = row.get("nbest", [])
            if len(candidates) != args.n_best:
                raise ValueError(
                    f"text_id={row.get('text_id')} has {len(candidates)} candidates; "
                    f"expected N={args.n_best}"
                )
            documents = [candidate_text(candidate) for candidate in candidates]
            formatted = [
                SYSTEM_PREFIX
                + f"Query: {args.instruction}\n"
                + f"Document: {document}\n"
                + USER_SUFFIX
                for document in documents
            ]
            scores = rerank_scores(model, tokenizer, formatted, device)
            selected_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            selected = parse_hypothesis(documents[selected_index])
            ranking = sorted(
                [
                    {"rank": rank + 1, "index": index, "score": float(scores[index])}
                    for rank, index in enumerate(
                        sorted(range(len(scores)), key=lambda index: (-scores[index], index))
                    )
                ],
                key=lambda item: item["rank"],
            )
            handle.write(json.dumps({
                "text_id": row["text_id"],
                "query": row.get("query", ""),
                "semantics": row.get("semantics", []),
                "pred_query": selected["pred_query"],
                "pred_semantics": selected["pred_semantics"],
                "pred_raw": selected["raw"],
                "reranker_selected_index": selected_index,
                "reranker_scores": scores,
                "reranker_ranking": ranking,
            }, ensure_ascii=False) + "\n")
            if count % 50 == 0 or count == len(rows):
                print(f"[INFO] reranked {count}/{len(rows)}")

    print(f"[INFO] saved clean reranker predictions: {output_path}")


if __name__ == "__main__":
    main()
