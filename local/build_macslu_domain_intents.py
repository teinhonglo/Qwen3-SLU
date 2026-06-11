#!/usr/bin/env python3
"""Write legal MAC-SLU joint domain-intent labels for prototype reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, Set

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.prototype_joint_utils import make_domain_intent_label  # noqa: E402
from slu_decoding.prototypes import MACSLULabelSchema  # noqa: E402


def iter_rows(paths: Iterable[str]):
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


def observed_domain_intents(paths: Iterable[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
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
            domain = str(frame.get("domain", "") or "").strip()
            intent = str(frame.get("intent", "") or "").strip()
            if domain and intent:
                out[domain].add(intent)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonls", nargs="*", default=[])
    parser.add_argument("--labels_path", default="")
    parser.add_argument("--output_txt", required=True)
    args = parser.parse_args()

    schema = MACSLULabelSchema(labels_path=args.labels_path) if args.labels_path else MACSLULabelSchema()
    for domain, intents in observed_domain_intents(args.input_jsonls).items():
        for intent in intents:
            schema.add_domain_intent(domain, intent)

    labels = [
        make_domain_intent_label(domain, intent)
        for domain in schema.valid_domains()
        for intent in schema.valid_intents(domain)
    ]
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")
    print(f"[info] saved {len(labels)} domain-intent labels: {args.output_txt}")


if __name__ == "__main__":
    main()
