#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


DEFAULT_PROMPT = """你是一个专业的车载系统自然语言理解（NLU）专家。
你的任务是基于用户的查询（Query），同时完成两项任务：
1.  意图识别 (Intent Classification): 识别出查询中包含的所有领域（Domain）和意图（Intent）。
2.  槽位填充 (Slot Filling): 抽取出与每个意图相关的槽位（Slot）和槽位值（Value）。

你需要严格遵循以下规则：
1.  识别多个语义帧: 用户的单次查询可能包含多个独立的意图。你需要为每一个意图生成一个对应的语义结构。
2.  输出格式: 你的输出必须是一个严格的 JSON List (列表)。
3.  列表中的每一个 JSON 对象都必须包含且只包含这三个欄位："domain"、"intent"、"slots"。
4.  "slots" 必须是 JSON object；若该意图无槽位，請輸出空物件 {}。
5.  如果没有匹配到任何领域和意图，请返回空列表 []。
6.  最终回答中除了 JSON，不要包含其他文字。

输出格式范例：
- 单一语义帧：
[{"domain":"地图","intent":"导航","slots":{"终点目标":"广州塔"}}]

- 多语义帧：
[
  {"domain":"地图","intent":"导航","slots":{"终点目标":"公司"}},
  {"domain":"音乐","intent":"播放音乐","slots":{"歌曲名":"夜曲"}}
]

- 无槽位：
[{"domain":"播放控制","intent":"播放控制","slots":{}}]

- 无匹配：
[]
"""


def parse_args():
    p = argparse.ArgumentParser(description="Download Gatsby1984/MAC_SLU and prepare SFT JSONL")
    p.add_argument("--repo-id", default="Gatsby1984/MAC_SLU")
    p.add_argument("--download-dir", required=True)
    p.add_argument("--extract-root", required=True)
    p.add_argument("--jsonl-root", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=["train", "dev", "test"])
    p.add_argument("--prompt-file", default="")
    return p.parse_args()


def ensure_local_file(repo_id: str, filename: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    cached_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    shutil.copy2(cached_path, local_path)
    return local_path


def download_split_files(repo_id: str, split: str, download_dir: Path):
    label_relpath = f"label/{split}_set.jsonl"
    audio_relpath = f"audio_{split}.tar.gz"
    label_local_path = download_dir / "label" / f"{split}_set.jsonl"
    audio_local_path = download_dir / f"audio_{split}.tar.gz"
    label_path = ensure_local_file(repo_id, label_relpath, label_local_path)
    audio_tar_path = ensure_local_file(repo_id, audio_relpath, audio_local_path)
    return label_path, audio_tar_path


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extract_done"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        abs_out = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(abs_out) + os.sep) and target != abs_out:
                raise RuntimeError(f"Unsafe path in tar file: {member.name}")
        tar.extractall(out_dir)
    marker.touch()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}") from e
    return rows


def build_wav_index(audio_root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for wav in audio_root.rglob("*.wav"):
        idx.setdefault(wav.stem, []).append(wav.resolve())
    return idx


def resolve_wav(raw_id: str, wav_index: Dict[str, List[Path]]) -> Path:
    raw_id = str(raw_id)
    if raw_id in wav_index and len(wav_index[raw_id]) == 1:
        return wav_index[raw_id][0]
    suffix_matches = []
    for stem, paths in wav_index.items():
        if stem.endswith(raw_id):
            suffix_matches.extend(paths)
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise FileNotFoundError(f"Cannot uniquely resolve wav for id={raw_id}")


def to_semantics_text(record: dict) -> str:
    semantics = record.get("semantics", [])
    if isinstance(semantics, list):
        return json.dumps(semantics, ensure_ascii=False)
    return "[]"


def main():
    args = parse_args()
    download_dir = Path(args.download_dir).resolve()
    extract_root = Path(args.extract_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    for split in args.splits:
        label_path, audio_tar_path = download_split_files(args.repo_id, split, download_dir)
        split_audio_dir = extract_root / split
        safe_extract_tar(audio_tar_path, split_audio_dir)

        records = load_jsonl(label_path)
        wav_index = build_wav_index(split_audio_dir)
        if not wav_index:
            raise RuntimeError(f"No wav files found in {split_audio_dir}")

        out_path = jsonl_root / f"{split}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(records, start=1):
                rid = str(r.get("id", i))
                wav = resolve_wav(rid, wav_index)
                query = r.get("query", "")
                semantics_text = to_semantics_text(r)
                row = {
                    "text_id": rid,
                    "id": rid,
                    "query": query,
                    "audio": str(wav),
                    "prompt": prompt,
                    "text": semantics_text,
                    "semantics": r.get("semantics", []),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
