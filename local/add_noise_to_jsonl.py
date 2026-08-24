#!/usr/bin/env python3
"""Add single-channel background noise to every audio item in a JSONL file."""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio


NOISE_NAMES = tuple(f"noise_DX{i:02d}C01.wav" for i in range(1, 5))
ENERGY_EPS = 1.0e-12


def load_mono(path: Path, description: str) -> Tuple[torch.Tensor, int]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} audio does not exist: {path}")
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to read {description} audio {path}: {exc}") from exc
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise ValueError(f"{description} audio is empty or has an invalid shape: {path} ({tuple(waveform.shape)})")
    return waveform.to(torch.float32).mean(dim=0, keepdim=True), sample_rate


def find_scenarios(noise_dir: Path) -> List[Tuple[Path, List[Path]]]:
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"noise_dir does not exist or is not a directory: {noise_dir}")
    directories = sorted(path for path in noise_dir.iterdir() if path.is_dir())
    if not directories:
        raise ValueError(f"no scenario directories found under noise_dir: {noise_dir}")
    scenarios = []
    for directory in directories:
        channels = [directory / name for name in NOISE_NAMES if (directory / name).is_file()]
        if channels:
            scenarios.append((directory, channels))
    if not scenarios:
        expected = ", ".join(NOISE_NAMES)
        raise ValueError(f"scenario directories under {noise_dir} contain no valid noise WAV (expected: {expected})")
    return scenarios


def make_temp_path(destination: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    return Path(name)


def augment(args: argparse.Namespace) -> Dict[str, int]:
    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    scenarios = find_scenarios(Path(args.noise_dir))
    output_path = Path(args.output_jsonl)
    metadata_path = Path(str(output_path) + ".noise_meta.jsonl")
    audio_dir = Path(args.output_audio_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = audio_dir.resolve()
    rng = random.Random(args.seed)
    output_tmp, metadata_tmp = make_temp_path(output_path), make_temp_path(metadata_path)
    count = 0
    try:
        with input_path.open("r", encoding="utf-8") as source, output_tmp.open("w", encoding="utf-8") as output, metadata_tmp.open("w", encoding="utf-8") as metadata:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {input_path}: {exc}") from exc
                if "audio" not in row or not isinstance(row["audio"], str) or not row["audio"]:
                    raise ValueError(f"JSONL row index {count} (line {line_number}) is missing a non-empty 'audio' field")
                clean_path = Path(row["audio"]).expanduser()
                clean, clean_sr = load_mono(clean_path, "clean")
                scenario_path, channels = rng.choice(scenarios)
                noise_path = rng.choice(channels)
                noise, noise_sr = load_mono(noise_path, "noise")
                if noise_sr != clean_sr:
                    noise = torchaudio.functional.resample(noise, noise_sr, clean_sr)
                clean_length, noise_length = clean.shape[-1], noise.shape[-1]
                if noise_length < clean_length:
                    raise ValueError(
                        f"noise is shorter than speech after resampling: clean={clean_path} ({clean_length} samples), "
                        f"noise={noise_path} ({noise_length} samples)"
                    )
                start = rng.randint(0, noise_length - clean_length)
                segment = noise[..., start : start + clean_length]
                clean_energy = clean.square().mean().item()
                noise_energy = segment.square().mean().item()
                if clean_energy <= ENERGY_EPS:
                    raise ValueError(f"clean speech energy is near zero: {clean_path} (mean square={clean_energy:.3e})")
                if noise_energy <= ENERGY_EPS:
                    raise ValueError(f"cropped noise energy is near zero: {noise_path} (mean square={noise_energy:.3e})")
                mixed = torchaudio.functional.add_noise(
                    waveform=clean,
                    noise=segment,
                    snr=torch.tensor([args.snr_db], dtype=clean.dtype),
                )
                if mixed.shape[-1] != clean_length:
                    raise RuntimeError(f"mixed waveform length changed for {clean_path}: {mixed.shape[-1]} != {clean_length}")
                peak = mixed.abs().max().item()
                peak_scale = 0.99 / peak if peak > 0.99 else 1.0
                mixed = mixed * peak_scale
                stem = clean_path.stem or "audio"
                noisy_path = audio_dir / f"{count:08d}_{stem}.wav"
                torchaudio.save(str(noisy_path), mixed, clean_sr)
                original_audio = row["audio"]
                row["audio"] = str(noisy_path.resolve())
                meta = {
                    "index": count,
                    "original_audio": original_audio,
                    "noisy_audio": row["audio"],
                    "noise_audio": str(noise_path.resolve()),
                    "noise_scenario": scenario_path.name,
                    "noise_channel": noise_path.stem,
                    "noise_start_sample": start,
                    "noise_start_seconds": start / clean_sr,
                    "snr_db": args.snr_db,
                    "seed": args.seed,
                    "sample_rate": clean_sr,
                    "clean_num_samples": clean_length,
                    "peak_scale": peak_scale,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                metadata.write(json.dumps(meta, ensure_ascii=False) + "\n")
                count += 1
                if count == 1 or count % 100 == 0:
                    print(f"[add-noise] processed {count} utterance(s)", file=sys.stderr)
        if count == 0:
            raise ValueError(f"input JSONL is empty: {input_path}")
        os.replace(metadata_tmp, metadata_path)
        os.replace(output_tmp, output_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        raise
    print(
        f"[add-noise] complete: utterances={count}, scenarios={len(scenarios)}, "
        f"output={output_path}, metadata={metadata_path}",
        file=sys.stderr,
    )
    return {"utterances": count, "scenarios": len(scenarios)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard single-channel additive-noise augmentation for JSONL audio")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_audio_dir", required=True)
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--snr_db", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    augment(parser.parse_args())


if __name__ == "__main__":
    main()
