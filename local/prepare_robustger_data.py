#!/usr/bin/env python3
"""Prepare RobustGER's N-best language-noise and audio feature cache.

This script intentionally keeps reference text/semantics in the manifest only
for later evaluation and H2T supervision. They are never included in the H2T
input text or in the language-noise calculation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from qwen_asr import Qwen3ASRModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from local.score_nbest_oracle import parse_hypothesis  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def row_key(row: Dict[str, Any]) -> str:
    if "text_id" not in row:
        raise ValueError("Every JSONL row needs text_id")
    return str(row["text_id"])


def candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        return str(candidate.get("text", candidate.get("pred_raw", candidate.get("hyp", ""))))
    return str(candidate or "")


def build_h2t_input(candidates: Sequence[str]) -> str:
    sections = [
        "You are a MAC-SLU correction model.",
        "Given the N-best hypotheses produced by an ASR system, generate the clean MAC-SLU result.",
        "Use all hypotheses as evidence. Output only the target format used by the project.",
        "",
    ]
    for index, candidate in enumerate(candidates, 1):
        sections.extend([f"### Hypothesis {index}", candidate.strip(), ""])
    sections.append("### Response:")
    return "\n".join(sections)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def sentence_embedding(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    batch = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    return mean_pool(model(**batch).last_hidden_state, batch["attention_mask"])[0].float()


@torch.inference_mode()
def token_embeddings(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    dimension: int,
) -> Tuple[List[int], torch.Tensor]:
    batch = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"][0].to(device)
    if input_ids.numel() == 0:
        return [], torch.empty((0, dimension), dtype=torch.float32)
    attention_mask = batch["attention_mask"].to(device)
    hidden = model(input_ids=batch["input_ids"].to(device), attention_mask=attention_mask).last_hidden_state[0]
    return input_ids.detach().cpu().tolist(), hidden.float().detach().cpu()


def token_alignment_diff(
    left_ids: List[int],
    left_emb: torch.Tensor,
    right_ids: List[int],
    right_emb: torch.Tensor,
) -> torch.Tensor:
    """Global-align two token sequences and sum left-right vectors.

    A gap is represented by the zero vector (the paper's Ø token). Matching
    tokens have zero contribution; substitutions and insertions/deletions keep
    the aligned embedding difference.
    """
    n, m = len(left_ids), len(right_ids)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i][0] = "up"
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dp[i - 1, j - 1] + int(left_ids[i - 1] != right_ids[j - 1])
            up = dp[i - 1, j] + 1
            left = dp[i, j - 1] + 1
            best = min(diagonal, up, left)
            dp[i, j] = best
            # Prefer substitution/match, then deletion, then insertion for
            # deterministic alignment across runs.
            back[i][j] = "diag" if diagonal == best else ("up" if up == best else "left")

    pieces = []
    i, j = n, m
    zero_left = torch.zeros(left_emb.shape[-1] if left_emb.numel() else right_emb.shape[-1])
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            pieces.append(left_emb[i - 1] - right_emb[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            pieces.append(left_emb[i - 1] - zero_left)
            i -= 1
        else:
            pieces.append(zero_left - right_emb[j - 1])
            j -= 1

    if not pieces:
        return zero_left
    return torch.stack(list(reversed(pieces))).sum(dim=0)


def language_noise_embedding(
    candidates: Sequence[str],
    tokenizer,
    sbert,
    device: torch.device,
    dimension: int,
) -> torch.Tensor:
    parsed = [parse_hypothesis(candidate) for candidate in candidates]
    queries = [str(item.get("pred_query", "")) for item in parsed]

    utterance = [sentence_embedding(query, tokenizer, sbert, device) for query in queries]
    token_data = [
        token_embeddings(query, tokenizer, sbert, device, dimension)
        for query in queries
    ]

    pair_features = []
    for i in range(len(queries)):
        for j in range(i):
            pair_features.append(utterance[i] - utterance[j])
    for i in range(len(queries)):
        for j in range(i):
            left_ids, left_emb = token_data[i]
            right_ids, right_emb = token_data[j]
            pair_features.append(token_alignment_diff(left_ids, left_emb, right_ids, right_emb))

    expected = len(queries) * (len(queries) - 1)
    if len(pair_features) != expected:
        raise AssertionError(f"Expected {expected} language-noise vectors, got {len(pair_features)}")
    return torch.stack(pair_features).float()


def build_prefix_text(processor, prompt: str) -> str:
    messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    prefix = processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=False,
    )
    return prefix[0] if isinstance(prefix, list) else prefix


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(path, sr=sr, mono=True)
    return waveform


@torch.inference_mode()
def audio_embedding(asr_wrapper, audio_path: str, prompt: str) -> np.ndarray:
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    waveform = load_audio(audio_path)
    prefix = build_prefix_text(processor, prompt)
    inputs = processor(
        text=[prefix],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    model_dtype = getattr(model, "dtype", torch.float16)
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        moved[key] = value
    features = model.thinker.get_audio_features(
        input_features=moved["input_features"],
        feature_attention_mask=moved.get("feature_attention_mask"),
        audio_feature_lengths=None,
    )
    if features.ndim == 3:
        features = features[0]
    if features.ndim == 2:
        features = features.mean(dim=0)
    if features.ndim != 1:
        raise ValueError(f"Unexpected Qwen3 audio feature shape: {tuple(features.shape)}")
    return features.float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RobustGER feature caches")
    parser.add_argument("--nbest_jsonl", required=True)
    parser.add_argument("--clean_jsonl", required=True)
    parser.add_argument("--noisy_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    n_best = int(config["n_best"])
    noise_dim = int(config["noise_dim"])
    audio_dim = int(config["audio_dim"])
    expected_slots = n_best * (n_best - 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    arrays = [
        output_dir / "language_noise.npy",
        output_dir / "noisy_audio.npy",
        output_dir / "clean_audio.npy",
    ]
    if manifest_path.is_file() and all(path.is_file() for path in arrays) and not args.overwrite:
        print(f"[SKIP] RobustGER cache already exists: {output_dir}")
        return

    nbest_rows = read_jsonl(Path(args.nbest_jsonl))
    clean_rows = {row_key(row): row for row in read_jsonl(Path(args.clean_jsonl))}
    noisy_rows = {row_key(row): row for row in read_jsonl(Path(args.noisy_jsonl))}
    if len(nbest_rows) != len(clean_rows):
        raise ValueError("N-best and clean JSONL row counts differ")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    sbert = AutoModel.from_pretrained(config["sbert_model"]).to(device).eval()
    sbert_tokenizer = AutoTokenizer.from_pretrained(config["sbert_model"])
    discovered_dim = int(sbert.config.hidden_size)
    if discovered_dim != noise_dim:
        raise ValueError(
            f"Configured noise_dim={noise_dim}, but multilingual SBERT outputs {discovered_dim}"
        )

    asr_dtype = torch.float16 if device.type == "cuda" else "auto"
    asr = Qwen3ASRModel.from_pretrained(
        config["audio_model"],
        dtype=asr_dtype,
        device_map=str(device),
    )
    asr.model.eval()

    count = len(nbest_rows)
    language = np.lib.format.open_memmap(
        output_dir / "language_noise.npy",
        mode="w+",
        dtype="float32",
        shape=(count, expected_slots, noise_dim),
    )
    noisy_audio = np.lib.format.open_memmap(
        output_dir / "noisy_audio.npy",
        mode="w+",
        dtype="float32",
        shape=(count, audio_dim),
    )
    clean_audio = np.lib.format.open_memmap(
        output_dir / "clean_audio.npy",
        mode="w+",
        dtype="float32",
        shape=(count, audio_dim),
    )

    seen = set()
    manifest_handle = manifest_path.open("w", encoding="utf-8")
    with manifest_handle:
        for index, nbest_row in enumerate(nbest_rows):
            key = row_key(nbest_row)
            if key in seen:
                raise ValueError(f"Duplicate text_id in N-best JSONL: {key}")
            seen.add(key)
            if key not in clean_rows or key not in noisy_rows:
                raise ValueError(f"Missing clean/noisy row for text_id={key}")
            candidates = nbest_row.get("nbest", [])
            if len(candidates) != n_best:
                raise ValueError(
                    f"text_id={key} has {len(candidates)} candidates; expected exactly N={n_best}"
                )

            clean = clean_rows[key]
            noisy = noisy_rows[key]
            parsed = [parse_hypothesis(item) for item in candidates]
            candidate_texts = [item["raw"] for item in parsed]
            input_text = build_h2t_input(candidate_texts)
            language[index] = language_noise_embedding(
                candidate_texts,
                sbert_tokenizer,
                sbert,
                device,
                noise_dim,
            ).numpy()

            if not noisy.get("audio") or not clean.get("audio"):
                raise ValueError(f"Missing clean/noisy audio path at text_id={key}")
            # Match build_audio_topk_semantics.py default: audio-only embedding;
            # the row-level SLU prompt must not become an audio signal shortcut.
            noisy_vector = audio_embedding(asr, str(noisy["audio"]), "")
            clean_vector = audio_embedding(asr, str(clean["audio"]), "")
            if noisy_vector.shape[0] != audio_dim or clean_vector.shape[0] != audio_dim:
                raise ValueError(
                    f"Qwen3 audio dimension mismatch at text_id={key}: "
                    f"noisy={noisy_vector.shape[0]}, clean={clean_vector.shape[0]}, expected={audio_dim}"
                )
            noisy_audio[index] = noisy_vector
            clean_audio[index] = clean_vector

            manifest_handle.write(json.dumps({
                "index": index,
                "text_id": clean["text_id"],
                "input_text": input_text,
                "target_text": str(clean.get("text", "")),
                "query": clean.get("query", ""),
                "semantics": clean.get("semantics", []),
                "noisy_audio": noisy.get("audio", ""),
                "clean_audio": clean.get("audio", ""),
                "candidate_count": n_best,
            }, ensure_ascii=False) + "\n")

            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f"[INFO] prepared {index + 1}/{count}")

    language.flush()
    noisy_audio.flush()
    clean_audio.flush()
    meta = {
        "n_best": n_best,
        "language_noise_slots": expected_slots,
        "noise_dim": noise_dim,
        "audio_dim": audio_dim,
        "sbert_model": config["sbert_model"],
        "audio_model": config["audio_model"],
    }
    with (output_dir / "feature_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] RobustGER cache written to {output_dir}")


if __name__ == "__main__":
    main()
