#!/usr/bin/env python3
"""Build prefix-conditioned next-token expert corpora from MAC-SLU jsonl."""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from slu_decoding.state_parser import (
    STATE_DOMAIN,
    STATE_INTENT,
    STATE_SLOTS_KEY,
    parse_state,
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"[warn] bad json {path}:{line_id}: {exc}")
    return rows


def build_target_text(row):
    return json.dumps(
        {
            "asr_text": row.get("query", ""),
            "semantics": json.dumps(row.get("semantics", []), ensure_ascii=False),
        },
        ensure_ascii=False,
    )


def rows_to_examples(rows, tokenizer):
    domain_intent_rows = []
    slot_key_rows = []

    for row_id, row in enumerate(rows):
        base_id = str(row.get("text_id", f"line{row_id + 1}"))
        target_text = build_target_text(row)
        token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        for tok_i in range(len(token_ids)):
            prefix_ids = token_ids[:tok_i]
            next_tid = token_ids[tok_i]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
            target_token_text = tokenizer.decode([next_tid], skip_special_tokens=False)
            state = parse_state(prefix_text)
            out = {
                "id": f"{base_id}#tok_{tok_i}",
                "state": state.state_name,
                "prefix_text": prefix_text,
                "target_token_text": target_token_text,
            }
            if state.state_name in (STATE_DOMAIN, STATE_INTENT):
                domain_intent_rows.append(out)
            elif state.state_name == STATE_SLOTS_KEY:
                slot_key_rows.append(out)

    return domain_intent_rows, slot_key_rows


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] wrote {len(rows)} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--output_dir", default="data-json/macslu_dexperts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    train_di, train_sk = rows_to_examples(load_jsonl(args.train_jsonl), tokenizer)
    dev_di, dev_sk = rows_to_examples(load_jsonl(args.dev_jsonl), tokenizer)

    dump_jsonl(os.path.join(args.output_dir, "domain_intent_train.jsonl"), train_di)
    dump_jsonl(os.path.join(args.output_dir, "domain_intent_dev.jsonl"), dev_di)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_train.jsonl"), train_sk)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_dev.jsonl"), dev_sk)


if __name__ == "__main__":
    main()
