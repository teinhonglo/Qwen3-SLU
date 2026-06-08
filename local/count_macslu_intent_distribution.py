#!/usr/bin/env python3
"""Count MAC-SLU semantic-frame / intent-cardinality distributions.

The prepared MAC-SLU JSONL stores one list item in ``semantics`` per semantic
frame.  This script reports the distribution of ``len(semantics)`` for each
split without collapsing higher-cardinality examples into a 3+ bucket.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Count MAC-SLU semantic frame counts")
    parser.add_argument("--jsonl-root", required=True, help="Directory containing split JSONL files")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Split names to count")
    parser.add_argument("--output-txt", default="", help="Optional text report path")
    parser.add_argument("--output-json", default="", help="Optional JSON report path")
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield line_no, obj


def semantic_frame_count(row: Dict[str, Any]) -> int:
    semantics = row.get("semantics", [])
    if semantics is None:
        return 0
    if isinstance(semantics, str):
        try:
            parsed = json.loads(semantics)
        except json.JSONDecodeError:
            return 0
        semantics = parsed
    if isinstance(semantics, list):
        return len(semantics)
    return 0


def count_split(path: str) -> Dict[str, Any]:
    counts: Counter[int] = Counter()
    total = 0
    for _, row in read_jsonl(path):
        counts[semantic_frame_count(row)] += 1
        total += 1
    return {
        "total": total,
        "frame_count": {str(k): counts[k] for k in sorted(counts)},
    }


def merge_counts(split_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter[int] = Counter()
    total = 0
    for report in split_reports.values():
        total += int(report.get("total", 0))
        for key, value in (report.get("frame_count", {}) or {}).items():
            counts[int(key)] += int(value)
    return {
        "total": total,
        "frame_count": {str(k): counts[k] for k in sorted(counts)},
    }


def format_section(name: str, report: Dict[str, Any]) -> List[str]:
    lines = [f"========== {name} ==========", f"total: {int(report.get('total', 0))}"]
    frame_count = report.get("frame_count", {}) or {}
    for key in sorted(frame_count, key=lambda x: int(x)):
        lines.append(f"{key}_intent: {frame_count[key]}")
    return lines


def format_report(report: Dict[str, Any], splits: List[str]) -> str:
    blocks: List[str] = []
    for split in splits:
        blocks.extend(format_section(split, report[split]))
        blocks.append("")
    blocks.extend(format_section("total", report["total"]))
    return "\n".join(blocks).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    split_reports: Dict[str, Dict[str, Any]] = {}
    for split in args.splits:
        path = os.path.join(args.jsonl_root, f"{split}.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing split JSONL: {path}")
        split_reports[split] = count_split(path)
    report: Dict[str, Any] = dict(split_reports)
    report["total"] = merge_counts(split_reports)
    text = format_report(report, list(args.splits))
    print(text, end="")
    if args.output_txt:
        os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
        with open(args.output_txt, "w", encoding="utf-8") as f:
            f.write(text)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
