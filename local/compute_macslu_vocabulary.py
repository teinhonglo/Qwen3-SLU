#!/usr/bin/env python3
"""Compute MAC-SLU tokens plus the top-N frequent original Qwen tokens."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from qwen_asr import Qwen3ASRModel


TEXT_FIELDS = ("text", "prompt", "query", "semantics")


def iter_text(jsonl_paths):
    for path in jsonl_paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for field in TEXT_FIELDS:
                    value = row.get(field)
                    if value not in (None, ""):
                        yield value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--top_frequency_tokens", type=int, default=2000)
    parser.add_argument(
        "--qwen_frequency_file",
        default="",
        help="Optional JSON token->count/token-id->count mapping. Without it, tokenizer ID rank is used.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, device_map="cpu")
    tokenizer = wrapper.processor.tokenizer
    counts = Counter()
    macslu_ids = set()
    for text in iter_text(args.jsonl):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counts.update(token_ids)
        macslu_ids.update(token_ids)

    special_ids = set(tokenizer.all_special_ids)
    # Qwen3-ASR modality/control tokens may not all be listed as special tokens.
    asr_ids = {
        token_id for token, token_id in tokenizer.get_vocab().items()
        if any(marker in token.lower() for marker in ("audio", "asr", "im_start", "im_end"))
    }
    if args.qwen_frequency_file:
        frequency_payload = json.loads(Path(args.qwen_frequency_file).read_text(encoding="utf-8"))
        ranked = sorted(frequency_payload.items(), key=lambda item: (-int(item[1]), str(item[0])))
        original_frequency_ids = []
        vocabulary = tokenizer.get_vocab()
        for token_or_id, _ in ranked:
            token_id = int(token_or_id) if str(token_or_id).isdigit() else vocabulary.get(str(token_or_id))
            if token_id is not None:
                original_frequency_ids.append(token_id)
        frequency_source = args.qwen_frequency_file
    else:
        # Qwen does not publish raw tokenizer training counts with the model.
        # Token ID order is therefore the deterministic original-vocabulary rank fallback.
        original_frequency_ids = sorted(tokenizer.get_vocab().values())
        frequency_source = "tokenizer_id_rank"
    frequent_ids = set(original_frequency_ids[:args.top_frequency_tokens])
    retained = sorted(special_ids | asr_ids | macslu_ids | frequent_ids)
    payload = {
        "model_path": args.model_path,
        "source_jsonl": args.jsonl,
        "top_frequency_tokens": args.top_frequency_tokens,
        "qwen_frequency_source": frequency_source,
        "original_vocabulary_size": len(tokenizer),
        "macslu_token_count": len(macslu_ids),
        "qwen3_asr_control_token_count": len(special_ids | asr_ids),
        "retained_vocabulary_size": len(retained),
        "retained_token_ids": retained,
        "frequency": {str(token_id): count for token_id, count in counts.most_common()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("retained_token_ids", "frequency")}, indent=2))


if __name__ == "__main__":
    main()
