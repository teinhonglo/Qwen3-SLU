#!/usr/bin/env python3
"""Filter MAC-SLU JSONL splits by semantic-frame count.

The MAC-SLU JSONL files store semantic frames in the ``semantics`` field.  This
utility keeps rows whose ``len(semantics)`` is less than or equal to a requested
threshold, which is useful for building curriculum-training subsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Filter MAC-SLU JSONL by len(semantics)")
    parser.add_argument("--jsonl-root", required=True, help="Directory containing input split JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory to write filtered split JSONL files")
    parser.add_argument("--max-semantics-len", type=int, required=True, help="Maximum allowed len(semantics)")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], help="Split names to filter")
    return parser.parse_args()


def semantics_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return 0
    if isinstance(value, list):
        return len(value)
    return 0


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}")
            yield line_no, row


def filter_split(jsonl_root: Path, output_dir: Path, split: str, max_semantics_len: int) -> None:
    in_path = jsonl_root / f"{split}.jsonl"
    out_path = output_dir / f"{split}.jsonl"

    if not in_path.is_file():
        raise FileNotFoundError(f"Missing split JSONL: {in_path}")

    kept = 0
    total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for _, row in read_jsonl(in_path):
            total += 1
            if semantics_len(row.get("semantics", [])) <= max_semantics_len:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    print(
        f"[filter] {split}: kept {kept}/{total} rows with "
        f"len(semantics) <= {max_semantics_len} -> {out_path}"
    )


def main() -> None:
    args = parse_args()
    jsonl_root = Path(args.jsonl_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        filter_split(jsonl_root, output_dir, split, args.max_semantics_len)


if __name__ == "__main__":
    main()
