#!/usr/bin/env python3
"""Materialize a runtime conf with vocabulary pruning enabled or disabled."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocabulary_pruning", choices=("true", "false"), required=True)
    parser.add_argument("--vocabulary_manifest", required=True)
    parser.add_argument("--teacher_exp_dir", default="")
    args = parser.parse_args()

    config = json.loads(Path(args.input).read_text(encoding="utf-8"))
    model = config[1]
    enabled = args.vocabulary_pruning == "true"
    model["vocabulary_pruning"] = {
        "enabled": enabled,
        "manifest": args.vocabulary_manifest,
    }
    if "teacher" in model:
        if args.teacher_exp_dir:
            model["teacher"]["exp_dir"] = args.teacher_exp_dir

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
