#!/usr/bin/env python3
"""Create the target-serialization variants used in the MAC-SLU auxiliary-task ablation."""

import argparse
import json
from pathlib import Path
from typing import Iterable


VARIANT_FIELDS = {
    "vanilla": (),
    "asr": ("asr_text",),
    "nsems": ("nsems",),
    "asr_nsems": ("asr_text", "nsems"),
    "nslots": ("nslots",),
    "asr_nslots": ("asr_text", "nslots"),
    "asr_nsems_nslots": ("asr_text", "nsems", "nslots"),
}
DEFAULT_VARIANTS = list(VARIANT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Figure 3 auxiliary-task JSONL variants from fixed MAC-SLU JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument(
        "--json-root-prefix",
        required=True,
        help="Output prefix; each variant is written to <prefix>_<variant>",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=DEFAULT_VARIANTS,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, row


def validate_frames(row: dict) -> list[dict]:
    semantics = row.get("semantics")
    if not isinstance(semantics, list):
        raise ValueError("'semantics' must be a JSON list")
    if not all(isinstance(frame, dict) for frame in semantics):
        raise ValueError("each semantic frame must be a JSON object")
    return semantics


def count_slots(frames: list[dict]) -> int:
    count = 0
    for frame in frames:
        for field in ("slots", "implicit_slots"):
            slots = frame.get(field, {})
            if not isinstance(slots, dict):
                raise ValueError(f"'{field}' must be a JSON object")
            count += len(slots)
    return count


def target_prefix(row: dict) -> str:
    text = row.get("text")
    marker = "<asr_text>"
    if not isinstance(text, str) or marker not in text:
        raise ValueError("'text' must contain the '<asr_text>' marker")
    return text.split(marker, maxsplit=1)[0] + marker


def convert_row(row: dict, variant: str) -> dict:
    frames = validate_frames(row)
    values = {
        "asr_text": str(row.get("query", "")),
        "nsems": len(frames),
        "nslots": count_slots(frames),
    }

    payload = {field: values[field] for field in VARIANT_FIELDS[variant]}
    # Keep semantics last and preserve the string-valued format used by
    # prepare_macslu_jsonl.py and qwen3_asr_test.py.
    payload["semantics"] = json.dumps(frames, ensure_ascii=False)

    result = dict(row)
    result["text"] = target_prefix(row) + json.dumps(payload, ensure_ascii=False)
    return result


def convert_split(src_path: Path, output_path: Path, variant: str) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in load_jsonl(src_path):
            try:
                result = convert_row(row, variant)
            except ValueError as error:
                raise ValueError(f"{src_path}:{line_number}: {error}") from error
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_json_root)
    for split in args.splits:
        src_path = src_root / f"{split}.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"Required source JSONL not found: {src_path}")

        for variant in args.variants:
            output_root = Path(f"{args.json_root_prefix}_{variant}")
            output_path = output_root / f"{split}.jsonl"
            count = convert_split(src_path, output_path, variant)
            print(f"[INFO] {variant}/{split}: wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
