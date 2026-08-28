#!/usr/bin/env python3
"""Add Qwen3-ASR-generated transcripts to MAC-SLU JSONL records."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr import Qwen3ASRModel


TARGET_MARKER = "<asr_text>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ASR transcripts and add a text_asr field to MAC-SLU JSONL"
    )
    parser.add_argument("--train_conf", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def load_model_args(train_conf_path: str) -> Dict[str, Any]:
    with open(train_conf_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, list) or len(config) != 2 or not isinstance(config[1], dict):
        raise ValueError("train_conf must be [training_args, model_args]")
    return config[1]


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def replace_asr_text(target: str, generated_asr: str) -> str:
    if TARGET_MARKER not in target:
        raise ValueError(f"target does not contain {TARGET_MARKER!r}")
    header, payload_text = target.split(TARGET_MARKER, 1)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("target payload after <asr_text> is not valid JSON") from exc
    if not isinstance(payload, dict) or "asr_text" not in payload or "semantics" not in payload:
        raise ValueError("target payload must contain asr_text and semantics")
    payload["asr_text"] = generated_asr
    return header + TARGET_MARKER + json.dumps(payload, ensure_ascii=False)


def resolve_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def write_rows_atomic(rows: List[Dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_args = load_model_args(args.train_conf)
    model_path = model_args.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    rows = load_rows(args.input_jsonl)
    for index, row in enumerate(rows, start=1):
        if not row.get("audio"):
            raise ValueError(f"row {index} has no audio field")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"row {index} has no text field")

    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=resolve_dtype(args.device),
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        results = asr.transcribe(
            audio=[row["audio"] for row in batch],
            language=args.language or None,
        )
        if len(results) != len(batch):
            raise RuntimeError("ASR result count does not match input batch size")
        for row, result in zip(batch, results):
            generated_asr = (result.text or "").strip()
            row["generated_asr_text"] = generated_asr
            row["text_asr"] = replace_asr_text(row["text"], generated_asr)
        print(f"[ASR] {min(start + len(batch), len(rows))}/{len(rows)}")

    write_rows_atomic(rows, args.output_jsonl)
    print(f"[INFO] Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
