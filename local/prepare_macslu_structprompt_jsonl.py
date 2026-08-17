#!/usr/bin/env python3
"""Create structure-aware multi-prompt data from prepared MAC-SLU JSONL."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


DOMAIN_INTENT_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
根据使用者语音，辨识其中的 Domain–Intent 关系。
每个 Domain 对应其在此语音中实际出现的 Intent。
按照 semantic frame 首次出现顺序输出，且相同关系只保留一次。
只输出指定 JSON。"""

DOMAIN_INTENT_SLOT_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
根据使用者语音，辨识 Domain、Intent 与 Slot 之间的阶层关系。
每个 Slot 必须归属于对应的 Domain–Intent semantic frame。
只输出 slot 名称，不输出 slot value。
按照 semantic frame 与 slot 首次出现顺序输出，只输出指定 JSON。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create structure-aware multi-prompt MAC-SLU JSONL"
    )
    parser.add_argument("--src-json-root", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--expand-splits",
        nargs="+",
        default=["train"],
        help="Splits that receive the two structural auxiliary tasks",
    )
    parser.add_argument(
        "--slu-repeat",
        type=int,
        default=2,
        help="Number of full-SLU rows per source row in expanded splits (default: 2)",
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


def domain_intent_target(frames: list[dict]) -> list[dict]:
    domains: OrderedDict[str, list[str]] = OrderedDict()
    for frame in frames:
        domain = frame.get("domain")
        intent = frame.get("intent")
        if not isinstance(domain, str) or not domain:
            continue
        intents = domains.setdefault(domain, [])
        if isinstance(intent, str) and intent and intent not in intents:
            intents.append(intent)
    return [{"domain": domain, "intents": intents} for domain, intents in domains.items()]


def domain_intent_slot_target(frames: list[dict]) -> list[dict]:
    relations = []
    for frame in frames:
        domain = frame.get("domain")
        intent = frame.get("intent")
        if not isinstance(domain, str) or not domain:
            continue
        if not isinstance(intent, str):
            intent = ""
        slot_names = []
        seen = set()
        for field in ("slots", "implicit_slots"):
            slots = frame.get(field, {})
            if not isinstance(slots, dict):
                continue
            for name in slots:
                name = str(name)
                if name not in seen:
                    seen.add(name)
                    slot_names.append(name)
        relations.append({"domain": domain, "intent": intent, "slots": slot_names})
    return relations


def target_text(value: list[dict]) -> str:
    return "language None<asr_text>" + json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    )


def slu_row(row: dict, suffix: str) -> dict:
    result = dict(row)
    result["text_id"] = f"{row.get('text_id', '')}{suffix}"
    result["task"] = "slu"
    return result


def auxiliary_row(row: dict, task: str, prompt: str, target: list[dict]) -> dict:
    result = dict(row)
    result["text_id"] = f"{row.get('text_id', '')}__{task}"
    result["task"] = task
    result["prompt"] = prompt
    result["text"] = target_text(target)
    return result


def expand_row(row: dict, slu_repeat: int) -> list[dict]:
    semantics = row.get("semantics", [])
    if not isinstance(semantics, list):
        raise ValueError("'semantics' must be a JSON List")
    frames = [frame for frame in semantics if isinstance(frame, dict)]
    slu_rows = [slu_row(row, f"__slu_{index}") for index in range(1, slu_repeat + 1)]
    return slu_rows + [
        auxiliary_row(
            row,
            "domain_intent",
            DOMAIN_INTENT_PROMPT,
            domain_intent_target(frames),
        ),
        auxiliary_row(
            row,
            "domain_intent_slot",
            DOMAIN_INTENT_SLOT_PROMPT,
            domain_intent_slot_target(frames),
        ),
    ]


def convert_split(
    src_path: Path, output_path: Path, expand: bool, slu_repeat: int
) -> dict[str, int]:
    counts = {
        "source_rows": 0,
        "slu": 0,
        "domain_intent": 0,
        "domain_intent_slot": 0,
        "total": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in load_jsonl(src_path):
            counts["source_rows"] += 1
            try:
                rows = expand_row(row, slu_repeat) if expand else [slu_row(row, "__slu")]
            except ValueError as error:
                raise ValueError(f"{src_path}:{line_number}: {error}") from error
            for result in rows:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["task"]] += 1
                counts["total"] += 1

    expected = counts["source_rows"] * (slu_repeat + 2 if expand else 1)
    if counts["total"] != expected:
        raise RuntimeError(
            f"Sanity check failed for {src_path}: total={counts['total']}, expected={expected}"
        )
    return counts


def main() -> None:
    args = parse_args()
    if args.slu_repeat < 1:
        raise ValueError("--slu-repeat must be at least 1")

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
        counts = convert_split(
            src_path,
            output_root / f"{split}.jsonl",
            split in expand_splits,
            args.slu_repeat,
        )
        print(f"[INFO] {split}:")
        print(f"source_rows={counts['source_rows']}")
        print(f"slu={counts['slu']}")
        print(f"domain_intent={counts['domain_intent']}")
        print(f"domain_intent_slot={counts['domain_intent_slot']}")
        print(f"total={counts['total']}")


if __name__ == "__main__":
    main()
