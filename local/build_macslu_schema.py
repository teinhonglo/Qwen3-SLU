#!/usr/bin/env python3
"""Build MAC-SLU schema for optional DExperts decoding."""

import argparse
import json
import os
from collections import defaultdict

SEP = "|||"


def iter_rows(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as exc:
                    print(f"[warn] bad json {path}:{line_id}: {exc}")


def build_schema(paths):
    domains = set()
    domain2intents = defaultdict(set)
    di2slot = defaultdict(set)
    di2implicit = defaultdict(set)

    for row in iter_rows(paths):
        frames = row.get("semantics", []) or []
        if isinstance(frames, str):
            try:
                frames = json.loads(frames)
            except Exception:
                frames = []

        for frame in frames:
            if not isinstance(frame, dict):
                continue

            domain = frame.get("domain", "")
            intent = frame.get("intent", "")
            if not domain or not intent:
                continue

            domains.add(domain)
            domain2intents[domain].add(intent)
            di_key = f"{domain}{SEP}{intent}"

            for slot_key in (frame.get("slots", {}) or {}).keys():
                di2slot[di_key].add(slot_key)
            for implicit_key in (frame.get("implicit_slots", {}) or {}).keys():
                di2implicit[di_key].add(implicit_key)

    return {
        "domains": sorted(domains),
        "domain2intents": {k: sorted(v) for k, v in sorted(domain2intents.items())},
        "domain_intent2slot_keys": {k: sorted(v) for k, v in sorted(di2slot.items())},
        "domain_intent2implicit_slot_keys": {
            k: sorted(v) for k, v in sorted(di2implicit.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonls", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    for path in args.input_jsonls:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    schema = build_schema(args.input_jsonls)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"[info] saved schema: {args.output_json}")


if __name__ == "__main__":
    main()
