#!/usr/bin/env python3
"""Build frame-level expert corpora from existing MAC-SLU jsonl."""

import argparse
import json
import os
import re

from slu_decoding.state_parser import (
    Q,
    STATE_DOMAIN,
    STATE_INTENT,
    STATE_SLOTS_KEY,
    key_re,
    parse_state,
)

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


def rows_to_examples_decode_prefix(rows):
    """Build state-aligned examples with continuation-span targets.

    For each prefix step routed to DI/SK states, target_text is the following
    text continuation until the next schema boundary.
    """
    domain_intent_rows = []
    slot_key_rows = []

    for row_id, row in enumerate(rows):
        base_id = str(row.get("text_id", f"line{row_id + 1}"))
        query = row.get("query", "")
        full_text = row.get("text", "")
        frames = row.get("semantics", []) or []

        if isinstance(frames, str):
            try:
                frames = json.loads(frames)
            except Exception:
                frames = []

        # Keep frame parse for basic quality guard.
        has_valid_frame = any(
            isinstance(frame, dict) and frame.get("domain", "") and frame.get("intent", "")
            for frame in frames
        )
        if not full_text or not has_valid_frame:
            continue

        if len(full_text) < 2:
            continue

        # state-aligned prefix examples from raw text slices (char-step)
        for t in range(1, len(full_text)):
            prefix_text = full_text[:t]
            state = parse_state(prefix_text)
            suffix_text = full_text[t:]
            if not suffix_text:
                continue

            if state.state_name in (STATE_DOMAIN, STATE_INTENT):
                # continuation until entering slots field.
                end_idx = _find_next_key_boundary(suffix_text, "slots")
                target_text = suffix_text if end_idx < 0 else suffix_text[:end_idx]
                if not target_text.strip():
                    continue
                domain_intent_rows.append(
                    {
                        "id": f"{base_id}#di_t{t}",
                        "text_id": base_id,
                        "query": query,
                        "state": state.state_name,
                        "step": t,
                        "input_text": prefix_text,
                        "target_text": target_text,
                    }
                )

            elif state.state_name == STATE_SLOTS_KEY:
                # continuation until key->value delimiter.
                end_idx = _find_key_value_delimiter(suffix_text)
                target_text = suffix_text if end_idx < 0 else suffix_text[:end_idx + 1]
                if not target_text.strip():
                    continue
                slot_key_rows.append(
                    {
                        "id": f"{base_id}#sk_t{t}",
                        "text_id": base_id,
                        "query": query,
                        "state": state.state_name,
                        "step": t,
                        "input_text": prefix_text,
                        "target_text": target_text,
                    }
                )

    return domain_intent_rows, slot_key_rows


def _find_next_key_boundary(text, key_name):
    m = re.search(rf"{key_re(key_name)}", text)
    return m.start() if m else -1


def _find_key_value_delimiter(text):
    m = re.search(rf"{Q}\s*:\s*{Q}?", text)
    return m.start() if m else -1


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] wrote {len(rows)} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--output_dir", default="data-json/macslu_dexperts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_di, train_sk = rows_to_examples_decode_prefix(load_jsonl(args.train_jsonl))
    dev_di, dev_sk = rows_to_examples_decode_prefix(load_jsonl(args.dev_jsonl))

    dump_jsonl(os.path.join(args.output_dir, "domain_intent_train.jsonl"), train_di)
    dump_jsonl(os.path.join(args.output_dir, "domain_intent_dev.jsonl"), dev_di)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_train.jsonl"), train_sk)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_dev.jsonl"), dev_sk)


if __name__ == "__main__":
    main()
