#!/usr/bin/env python3
"""Qwen3-Reranker control for clean MAC-SLU N-best evaluation.

This is intentionally independent from RobustGER: it consumes clean-test
N-best text only, performs no audio mixing, and never reads oracle candidates.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def rerank_scores(
    model,
    tokenizer,
    pairs: List[str],
    device: torch.device,
    max_length: int,
) -> List[float]:
    prefix_tokens = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(USER_SUFFIX, add_special_tokens=False)
    encoded = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    encoded["input_ids"] = [
        prefix_tokens + input_ids + suffix_tokens
        for input_ids in encoded["input_ids"]
    ]
    encoded = tokenizer.pad(
        encoded,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
        raise ValueError("Qwen3-Reranker tokenizer does not expose yes/no tokens")
    pair_logits = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1)
    return torch.softmax(pair_logits.float(), dim=-1)[:, 1].cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean-test Qwen3-Reranker control")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=(
        "Choose the candidate that best matches the user's spoken request "
        "and is a valid MAC-SLU JSON result."
    ))
    parser.add_argument("--query", default=(
        "Select the most accurate MAC-SLU hypothesis for this spoken utterance."
    ))
    parser.add_argument("--n_best", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    rows = read_jsonl(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        candidates = row.get("nbest", [])
        if len(candidates) != args.n_best:
            raise ValueError(
                f"text_id={row.get('text_id')} has {len(candidates)} candidates; "
                f"expected N={args.n_best}"
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, 1):
            documents = [candidate_text(candidate) for candidate in row["nbest"]]
            pairs = [
                f"<Instruct>: {args.instruction}\n"
                f"<Query>: {args.query}\n"
                f"<Document>: {document}"
                for document in documents
            ]
            scores = rerank_scores(
                model,
                tokenizer,
                pairs,
                device,
                args.max_length,
            )
            selected_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            selected = parse_hypothesis(documents[selected_index])
            ranking = [
                {"rank": rank + 1, "index": index, "score": float(scores[index])}
                for rank, index in enumerate(
                    sorted(range(len(scores)), key=lambda index: (-scores[index], index))
                )
            ]
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
