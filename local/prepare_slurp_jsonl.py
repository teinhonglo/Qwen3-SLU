#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_PROMPT = """You are an SLU expert.
Given spoken user audio, output ONLY a JSON object with exactly two keys:
1) \"asr_text\": ASR transcript string.
2) \"semantics\": a JSON string of a list of semantic frames.

Each semantic frame must contain exactly:
- \"domain\": scenario label
- \"intent\": action label
- \"slots\": key-value object from entities (empty object {} if no entities)

Return no extra words. Example output:
{"asr_text":"turn on the kitchen lights","semantics":"[{\"domain\":\"home\",\"intent\":\"switch_on\",\"slots\":{\"device\":\"lights\",\"room\":\"kitchen\"}}]"}
"""

SPLIT_TO_URL_NAME = {
    "train_real": "train",
    "train_synthetic": "train_synthetic",
    "devel": "devel",
    "test": "test",
}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare SLURP JSONL for qwen3_asr_sft.py")
    p.add_argument("--data-root", required=True, help="Root directory for downloaded/extracted SLURP data")
    p.add_argument("--jsonl-root", required=True, help="Output directory for train/dev/test JSONL")
    p.add_argument("--prompt-file", default="", help="Optional external prompt text file")
    p.add_argument("--skip-download", action="store_true", help="Skip download, use existing local files")
    return p.parse_args()


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"[download] {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def ensure_archives(data_root: Path, skip_download: bool = False):
    archives = {
        "slurp_real.tar.gz": "https://zenodo.org/record/4274930/files/slurp_real.tar.gz?download=1",
        "slurp_synth.tar.gz": "https://zenodo.org/record/4274930/files/slurp_synth.tar.gz?download=1",
    }
    for name, url in archives.items():
        path = data_root / name
        if not path.exists() and not skip_download:
            download(url, path)


def safe_extract(tar_path: Path, out_dir: Path):
    marker = out_dir / ".extract_done"
    if marker.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        out_abs = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_abs) + os.sep) and target != out_abs:
                raise RuntimeError(f"Unsafe member path in tar: {member.name}")
        tar.extractall(out_dir)
    marker.touch()


def ensure_audio_dirs(data_root: Path):
    real_dir = data_root / "slurp_real"
    synth_dir = data_root / "slurp_synth"

    if not real_dir.exists():
        safe_extract(data_root / "slurp_real.tar.gz", data_root)
    if not synth_dir.exists():
        safe_extract(data_root / "slurp_synth.tar.gz", data_root)


def ensure_split_jsonl(data_root: Path, split_name: str, skip_download: bool = False) -> Path:
    assert split_name in SPLIT_TO_URL_NAME
    url_name = SPLIT_TO_URL_NAME[split_name]
    dst = data_root / f"{split_name}.jsonl"
    if dst.exists():
        return dst

    if skip_download:
        raise FileNotFoundError(f"Missing split JSONL with --skip-download: {dst}")

    url = f"https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/{url_name}.jsonl"
    download(url, dst)
    return dst


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{i}") from e


def build_slots(entities: List[dict]) -> Dict[str, str]:
    slots = {}
    for ent in entities or []:
        et = str(ent.get("type", "")).strip()
        ev = str(ent.get("filler", "")).strip()
        if not et:
            continue
        if et in slots and slots[et] != ev:
            slots[et] = f"{slots[et]} | {ev}"
        else:
            slots[et] = ev
    return slots


def row_to_semantics(row: dict) -> List[dict]:
    return [{
        "domain": str(row.get("scenario", "")),
        "intent": str(row.get("action", "")),
        "slots": build_slots(row.get("entities", [])),
    }]


def recording_to_audio_path(data_root: Path, split_name: str, recording_file: str) -> Path:
    if split_name == "train_synthetic":
        return data_root / "slurp_synth" / recording_file
    return data_root / "slurp_real" / recording_file


def make_qwen_row(uid: str, sentence: str, audio_path: Path, prompt: str, semantics: List[dict]) -> dict:
    payload = {
        "asr_text": sentence,
        "semantics": json.dumps(semantics, ensure_ascii=False),
    }
    return {
        "text_id": uid,
        "query": sentence,
        "audio": str(audio_path.resolve()),
        "prompt": prompt,
        "text": f"language English<asr_text>{json.dumps(payload, ensure_ascii=False)}",
        "semantics": semantics,
    }


def write_output(rows: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] wrote {len(rows)} rows to {out_path}")


def process_split(data_root: Path, split_name: str, prompt: str, skip_download: bool = False) -> List[dict]:
    src_jsonl = ensure_split_jsonl(data_root, split_name, skip_download=skip_download)
    out_rows: List[dict] = []

    for item in load_jsonl(src_jsonl):
        sentence = str(item.get("sentence", ""))
        semantics = row_to_semantics(item)
        sid = str(item.get("slurp_id", item.get("id", "")))

        for rec in item.get("recordings", []):
            file_name = rec.get("file", "")
            if not file_name:
                continue
            audio_path = recording_to_audio_path(data_root, split_name, file_name)
            if not audio_path.exists():
                continue

            uid = f"{split_name}_{sid}_{Path(file_name).stem}"
            out_rows.append(make_qwen_row(uid, sentence, audio_path, prompt, semantics))

    return out_rows


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    data_root.mkdir(parents=True, exist_ok=True)
    ensure_archives(data_root, skip_download=args.skip_download)
    ensure_audio_dirs(data_root)

    train_real_rows = process_split(data_root, "train_real", prompt, skip_download=args.skip_download)
    train_syn_rows = process_split(data_root, "train_synthetic", prompt, skip_download=args.skip_download)
    dev_rows = process_split(data_root, "devel", prompt, skip_download=args.skip_download)
    test_rows = process_split(data_root, "test", prompt, skip_download=args.skip_download)

    train_rows = train_real_rows + train_syn_rows

    write_output(train_rows, jsonl_root / "train.jsonl")
    write_output(dev_rows, jsonl_root / "dev.jsonl")
    write_output(test_rows, jsonl_root / "test.jsonl")


if __name__ == "__main__":
    main()
