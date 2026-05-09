#!/usr/bin/env python3
"""Build frame-level expert corpora from existing MAC-SLU jsonl."""

import argparse
import json
import os


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


def rows_to_examples(rows):
    domain_intent_rows = []
    slot_key_rows = []

    for row_id, row in enumerate(rows):
        base_id = str(row.get("text_id", f"line{row_id + 1}"))
        query = row.get("query", "")
        frames = row.get("semantics", []) or []

        if isinstance(frames, str):
            try:
                frames = json.loads(frames)
            except Exception:
                frames = []

        for frame_id, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue

            domain = frame.get("domain", "")
            intent = frame.get("intent", "")
            if not domain or not intent:
                continue

            example_id = f"{base_id}#{frame_id}"
            domain_intent_rows.append(
                {
                    "id": example_id,
                    "query": query,
                    "domain": domain,
                    "intent": intent,
                    "input_text": f"query: {query}",
                    "target_text": json.dumps(
                        {"domain": domain, "intent": intent},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

            slot_keys = {k: "" for k in (frame.get("slots", {}) or {}).keys()}
            slot_key_rows.append(
                {
                    "id": example_id,
                    "query": query,
                    "domain": domain,
                    "intent": intent,
                    "slot_keys": list(slot_keys.keys()),
                    "input_text": f"query: {query}\ndomain: {domain}\nintent: {intent}",
                    "target_text": json.dumps(
                        {"slots": slot_keys}, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )

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
    parser.add_argument("--output_dir", default="data-json/macslu_dexperts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_di, train_sk = rows_to_examples(load_jsonl(args.train_jsonl))
    dev_di, dev_sk = rows_to_examples(load_jsonl(args.dev_jsonl))

    dump_jsonl(os.path.join(args.output_dir, "domain_intent_train.jsonl"), train_di)
    dump_jsonl(os.path.join(args.output_dir, "domain_intent_dev.jsonl"), dev_di)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_train.jsonl"), train_sk)
    dump_jsonl(os.path.join(args.output_dir, "slot_key_dev.jsonl"), dev_sk)


if __name__ == "__main__":
    main()
