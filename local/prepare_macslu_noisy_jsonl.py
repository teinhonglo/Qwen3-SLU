#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
import subprocess
import tarfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    p = argparse.ArgumentParser(
        description="Download Gatsby1984/MAC_SLU and prepare noisy SFT JSONL"
    )
    p.add_argument("--repo-id", default="Gatsby1984/MAC_SLU")
    p.add_argument("--download-dir", required=True)
    p.add_argument("--extract-root", required=True)
    p.add_argument("--jsonl-root", required=True)
    p.add_argument(
        "--splits", nargs="+", default=["test"], choices=["train", "dev", "test"]
    )
    p.add_argument("--prompt-file", default="")

    # Keep augmentation args aligned with augment_data_dir.py
    p.add_argument("--fg-snrs", type=str, dest="fg_snr_str", default="20:10:0")
    p.add_argument("--bg-snrs", type=str, dest="bg_snr_str", default="20:10:0")
    p.add_argument("--num-bg-noises", type=str, dest="num_bg_noises", default="1")
    p.add_argument("--fg-interval", type=int, dest="fg_interval", default=0)
    p.add_argument("--random-seed", type=int, dest="random_seed", default=123)
    p.add_argument("--bg-noise-dir", type=str, dest="bg_noise_dir")
    p.add_argument("--fg-noise-dir", type=str, dest="fg_noise_dir")

    args = p.parse_args()
    return check_args(args)


def check_args(args):
    if args.fg_interval < 0:
        raise ValueError("--fg-interval must be 0 or greater")

    if args.bg_noise_dir is None and args.fg_noise_dir is None:
        raise ValueError("Either --fg-noise-dir or --bg-noise-dir must be specified")

    return args


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


def build_wav_index(audio_root: Path) -> Dict[str, Path]:
    idx = {}
    for wav in audio_root.rglob("*.wav"):
        idx[wav.stem] = wav.resolve()
    return idx


def resolve_wav(raw_id: str, wav_index: Dict[str, Path]) -> Optional[Path]:
    return wav_index.get(str(raw_id).strip())


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        fr = wf.getframerate()
    if fr == 0:
        raise ValueError(f"Invalid sample rate in wav: {wav_path}")
    return frames / fr


def get_noise_list(noise_wav_scp_filename: Path) -> Tuple[List[str], Dict[str, str]]:
    noise_wavs = {}
    noise_utts = []
    with noise_wav_scp_filename.open("r", encoding="utf-8") as f:
        for line in f:
            toks = line.rstrip().split(" ")
            if not toks:
                continue
            noise_utt = toks[0]
            wav = " ".join(toks[1:]).strip()
            if not wav:
                continue
            noise_utts.append(noise_utt)
            noise_wavs[noise_utt] = wav
    return noise_utts, noise_wavs


def compute_noise_durations(noise_wavs: Dict[str, str]) -> Dict[str, float]:
    noise2dur = {}
    for utt, wav_spec in noise_wavs.items():
        if wav_spec.endswith("|"):
            continue
        wav_path = Path(wav_spec)
        if wav_path.exists() and wav_path.suffix.lower() == ".wav":
            noise2dur[utt] = wav_duration_seconds(wav_path)
    return noise2dur


def augment_wav(
    wav: str,
    dur: float,
    fg_snr_opts: List[int],
    bg_snr_opts: List[int],
    fg_noise_utts: List[str],
    bg_noise_utts: List[str],
    noise_wavs: Dict[str, str],
    noise2dur: Dict[str, float],
    interval: int,
    num_opts: List[int],
) -> str:
    dur_str = str(dur)
    tot_noise_dur = 0.0
    snrs = []
    noises = []
    start_times = []

    if bg_noise_utts:
        num = random.choice(num_opts)
        for _ in range(num):
            noise_utt = random.choice(bg_noise_utts)
            noise = f'wav-reverberate --duration={dur_str} "{noise_wavs[noise_utt]}" - |'
            snr = random.choice(bg_snr_opts)
            snrs.append(snr)
            start_times.append(0)
            noises.append(noise)

    if fg_noise_utts:
        while tot_noise_dur < dur:
            noise_utt = random.choice(fg_noise_utts)
            noise = noise_wavs[noise_utt]
            if noise_utt not in noise2dur:
                raise ValueError(
                    f"Missing duration for foreground noise utt '{noise_utt}'. "
                    "Only direct .wav file paths in wav.scp are supported for foreground duration calculation."
                )
            snr = random.choice(fg_snr_opts)
            snrs.append(snr)
            start_times.append(tot_noise_dur)
            tot_noise_dur += noise2dur[noise_utt] + interval
            noises.append(noise)

    if not noises:
        return wav

    start_times_str = "--start-times='" + ",".join([str(i) for i in start_times]) + "'"
    snrs_str = "--snrs='" + ",".join([str(i) for i in snrs]) + "'"
    noises_str = "--additive-signals='" + ",".join(noises).strip() + "'"

    if wav.strip()[-1] != "|":
        return (
            "wav-reverberate --shift-output=true "
            + noises_str
            + " "
            + start_times_str
            + " "
            + snrs_str
            + " "
            + wav
            + " - |"
        )

    return (
        wav
        + " wav-reverberate --shift-output=true "
        + noises_str
        + " "
        + start_times_str
        + " "
        + snrs_str
        + " - - |"
    )

def materialize_noisy_audio(clean_wav_path: Path, noisy_cmd: str, output_wav_path: Path) -> Path:
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if output_wav_path.exists():
        return output_wav_path

    if noisy_cmd == str(clean_wav_path):
        shutil.copy2(clean_wav_path, output_wav_path)
        return output_wav_path

    cmd = f"{noisy_cmd} cat > {shlex_quote(str(output_wav_path))}"
    subprocess.run(["bash", "-lc", cmd], check=True)
    return output_wav_path


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

def to_semantics_text(ori_semantics):
    results = []

    slot_mapping = {
        "body": "对象",
        "object": "对象",
        "feature": "对象功能",
        "part": "调节内容",
        "action": "操作",
    }

    if not isinstance(ori_semantics, dict):
        return json.dumps(results, ensure_ascii=False), results

    for _, domain_dict in ori_semantics.items():
        if not isinstance(domain_dict, dict):
            continue

        for domain, items in domain_dict.items():
            if not isinstance(items, list):
                continue

            new_item = {"domain": domain, "intent": "", "slots": {}}

            for x in items:
                name = x.get("name")
                value = x.get("value")

                if name in slot_mapping:
                    name = slot_mapping[name]

                if name == "intent":
                    new_item["intent"] = value
                elif name is not None:
                    new_item["slots"][name] = value

            results.append(new_item)

    return json.dumps(results, ensure_ascii=False), results


def main():
    args = parse_args()
    download_dir = Path(args.download_dir).resolve()
    extract_root = Path(args.extract_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()

    fg_snrs = [int(i) for i in args.fg_snr_str.split(":")]
    bg_snrs = [int(i) for i in args.bg_snr_str.split(":")]
    num_bg_noises = [int(i) for i in args.num_bg_noises.split(":")]
    primary_snr = fg_snrs[0] if args.fg_noise_dir else bg_snrs[0]

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    noise_wavs: Dict[str, str] = {}
    noise_durations: Dict[str, float] = {}
    bg_noise_utts: List[str] = []
    fg_noise_utts: List[str] = []

    if args.bg_noise_dir:
        bg_noise_dir = Path(args.bg_noise_dir)
        bg_noise_wav_scp = bg_noise_dir / "wav.scp"
        bg_noise_utts, bg_noise_wavs = get_noise_list(bg_noise_wav_scp)
        noise_wavs.update(bg_noise_wavs)
        noise_durations.update(compute_noise_durations(bg_noise_wavs))

    if args.fg_noise_dir:
        fg_noise_dir = Path(args.fg_noise_dir)
        fg_noise_wav_scp = fg_noise_dir / "wav.scp"
        fg_noise_utts, fg_noise_wavs = get_noise_list(fg_noise_wav_scp)
        noise_wavs.update(fg_noise_wavs)
        noise_durations.update(compute_noise_durations(fg_noise_wavs))

    random.seed(args.random_seed)

    for split in args.splits:
        label_path, audio_tar_path = download_split_files(args.repo_id, split, download_dir)
        split_audio_dir = extract_root / split
        safe_extract_tar(audio_tar_path, split_audio_dir)

        records = load_jsonl(label_path)
        wav_index = build_wav_index(split_audio_dir)
        if not wav_index:
            raise RuntimeError(f"No wav files found in {split_audio_dir}")

        split_out_dir = jsonl_root / f"{split}_snr{primary_snr}"
        out_path = split_out_dir / f"{split}.jsonl"
        wav_out_dir = split_out_dir / "wavs"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        missing = []

        with out_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(records, start=1):
                rid = f"id_{str(r.get('id', i))}"

                wav_path = resolve_wav(rid, wav_index)
                if wav_path is None:
                    print(f"ID {rid} NOT found")
                    missing.append(rid)
                    continue

                wav_spec = str(wav_path)
                wav_dur = wav_duration_seconds(wav_path)
                noisy_audio = augment_wav(
                    wav=wav_spec,
                    dur=wav_dur,
                    fg_snr_opts=fg_snrs,
                    bg_snr_opts=bg_snrs,
                    fg_noise_utts=fg_noise_utts,
                    bg_noise_utts=bg_noise_utts,
                    noise_wavs=noise_wavs,
                    noise2dur=noise_durations,
                    interval=args.fg_interval,
                    num_opts=num_bg_noises,
                )
                
                output_wav_path = materialize_noisy_audio(
                    clean_wav_path=wav_path,
                    noisy_cmd=noisy_audio,
                    output_wav_path=(wav_out_dir / f"{rid}.wav"),
                )

                query = r.get("query", "")
                semantics_text, semantics = to_semantics_text(r.get("semantics", []))
                row = {
                    "text_id": rid,
                    "query": query,
                    "audio": str(output_wav_path.resolve()),
                    "prompt": prompt,
                    "text": f"language None<asr_text>{query}<slu>{semantics_text}",
                    "semantics": semantics,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if missing:
            miss_path = split_out_dir / f"{split}.missing_wavs.txt"
            miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
            print(f"[WARN] {split}: skipped {len(missing)} records without exact wav match. See {miss_path}")

        print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
