#!/usr/bin/env python3
"""Expand prepared MAC-SLU JSONL into full-SLU and auxiliary prompt tasks."""

import argparse
import json
from pathlib import Path
from typing import Iterable


DOMAIN_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
请根据用户的语音查询，识别其中所有领域（Domain）。
只输出一个严格的 JSON List，列表元素为领域名称字符串，并按照语义出现顺序排列。
相同领域只保留一次；如果没有匹配的领域，输出 []。除了 JSON List 之外不要输出其他文字。"""

INTENT_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
请根据用户的语音查询，识别其中所有意图（Intent）。
只输出一个严格的 JSON List，列表元素为意图名称字符串，并按照语义出现顺序排列。
相同意图只保留一次；如果没有匹配的意图，输出 []。除了 JSON List 之外不要输出其他文字。"""

SLOT_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
请根据用户的语音查询，抽取其中所有槽位（Slot）。
只输出一个严格的 JSON List。每个元素必须包含且只包含 "slot"、"value"、"type" 三个字段；
type 必须是 "explicit"（槽值出现在语音文字中）或 "implicit"（隐含槽位）。
按照语义及槽位出现顺序排列；如果没有槽位，输出 []。除了 JSON List 之外不要输出其他文字。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multi-prompt MAC-SLU JSONL from prepared JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--expand-splits",
        nargs="+",
        default=["train", "dev"],
        help="Splits that receive domain, intent, and slot examples in addition to full SLU",
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
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield line_number, row


def unique_values(frames: list[dict], key: str) -> list[str]:
    values = []
    seen = set()
    for frame in frames:
        value = frame.get(key)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def slot_values(frames: list[dict]) -> list[dict[str, str]]:
    values = []
    for frame in frames:
        for field, slot_type in (("slots", "explicit"), ("implicit_slots", "implicit")):
            slots = frame.get(field, {})
            if not isinstance(slots, dict):
                continue
            for name, value in slots.items():
                values.append({"slot": str(name), "value": str(value), "type": slot_type})
    return values


def target_text(value: list) -> str:
    return "language None<asr_text>" + json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def task_row(row: dict, task: str, prompt: str, target: list) -> dict:
    result = dict(row)
    result["text_id"] = f"{row.get('text_id', '')}__{task}"
    result["task"] = task
    result["prompt"] = prompt
    result["text"] = target_text(target)
    return result


def expand_row(row: dict) -> list[dict]:
    semantics = row.get("semantics", [])
    if not isinstance(semantics, list):
        raise ValueError("'semantics' must be a JSON List")
    frames = [frame for frame in semantics if isinstance(frame, dict)]

    full_slu = dict(row)
    full_slu["text_id"] = f"{row.get('text_id', '')}__slu"
    full_slu["task"] = "slu"
    return [
        full_slu,
        task_row(row, "domain", DOMAIN_PROMPT, unique_values(frames, "domain")),
        task_row(row, "intent", INTENT_PROMPT, unique_values(frames, "intent")),
        task_row(row, "slot", SLOT_PROMPT, slot_values(frames)),
    ]


def convert_split(src_path: Path, output_path: Path, expand: bool) -> tuple[int, int]:
    input_count = 0
    output_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in load_jsonl(src_path):
            input_count += 1
            try:
                rows = expand_row(row) if expand else [row]
            except ValueError as error:
                raise ValueError(f"{src_path}:{line_number}: {error}") from error
            for result in rows:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_count += 1
    return input_count, output_count


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_json_root)
    output_root = Path(args.json_root)
    expand_splits = set(args.expand_splits)

    unknown_splits = expand_splits.difference(args.splits)
    if unknown_splits:
        raise ValueError(f"--expand-splits not present in --splits: {sorted(unknown_splits)}")

    for split in args.splits:
        src_path = src_root / f"{split}.jsonl"
        if not src_path.is_file():
            raise FileNotFoundError(f"Required source JSONL not found: {src_path}")
        output_path = output_root / f"{split}.jsonl"
        input_count, output_count = convert_split(
            src_path, output_path, split in expand_splits
        )
        print(f"[INFO] {split}: {input_count} source rows -> {output_count} output rows ({output_path})")


if __name__ == "__main__":
    main()
