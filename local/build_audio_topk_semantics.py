#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import librosa
from qwen_asr import Qwen3ASRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-K semantic neighbors by audio embedding similarity from a JSONL file."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input preprocessed JSONL path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--topk", type=int, required=True, help="Number of nearest neighbors")
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3-ASR-0.6B",
        help="ASR model used as audio encoder",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--use-prompt", action="store_true", help="Use row-level prompt field during embedding extraction")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {err}") from err
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def _to_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return None


@torch.inference_mode()
def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_msgs = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(prefix_text, list):
        prefix_text = prefix_text[0]
    return prefix_text


def move_inputs_to_device(inputs: Dict[str, object], device: str, model_dtype: torch.dtype):
    new_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        new_inputs[key] = value
    return new_inputs


@torch.inference_mode()
def extract_embedding(asr_wrapper, audio_path: str, prompt: str = "") -> np.ndarray:
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    wav = load_audio(audio_path, sr=16000)
    prefix_text = build_prefix_text(processor, prompt)

    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    model_dtype = getattr(model, "dtype", torch.float16)
    inputs = move_inputs_to_device(inputs, device=str(device), model_dtype=model_dtype)

    audio_features = model.thinker.get_audio_features(
        input_features=inputs["input_features"],
        feature_attention_mask=inputs.get("feature_attention_mask"),
        audio_feature_lengths=None,
    )

    pooled = audio_features.mean(dim=1).squeeze(0)
    return pooled.detach().cpu().to(torch.float32).numpy()


def build_topk(rows: List[dict], embeddings: np.ndarray, topk: int) -> List[dict]:
    x = torch.from_numpy(embeddings)
    x = torch.nn.functional.normalize(x, dim=1)
    sims = x @ x.T

    n = sims.size(0)
    topk = min(topk, max(n - 1, 0))

    output = []
    for i, row in enumerate(rows):
        out_row = dict(row)
        if topk <= 0:
            out_row["topK"] = []
            output.append(out_row)
            continue

        values, indices = torch.topk(sims[i], k=topk + 1, largest=True)
        items = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            if idx == i:
                continue
            target = rows[idx]
            items.append(
                {
                    "text_id": target.get("text_id", ""),
                    "semantics": target.get("semantics", []),
                    "similarity": float(score),
                }
            )
            if len(items) >= topk:
                break

        out_row["topK"] = items
        output.append(out_row)

    return output


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_path = output_dir / f"{input_jsonl.stem}.topk{args.topk}.jsonl"

    rows = read_jsonl(input_jsonl)
    if not rows:
        raise ValueError(f"No valid rows found in {input_jsonl}")

    dtype = _to_torch_dtype(args.dtype)

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_name_or_path,
        dtype=dtype if dtype is not None else "auto",
        device_map=args.device,
    )
    asr_wrapper.model.eval()

    embeddings = []
    for i, row in enumerate(rows, start=1):
        audio_path = row.get("audio")
        if not audio_path:
            raise ValueError(f"Missing 'audio' field at row index {i}")
        if args.use_prompt:
            prompt = row.get("prompt", "")
        else:
            prompt = ""
        emb = extract_embedding(asr_wrapper, audio_path, prompt=prompt)
        embeddings.append(emb)

    emb_mat = np.stack(embeddings, axis=0)
    out_rows = build_topk(rows, emb_mat, args.topk)
    write_jsonl(output_path, out_rows)

    print(f"[INFO] rows: {len(rows)}")
    print(f"[INFO] topK: {args.topk}")
    print(f"[INFO] output: {output_path}")


if __name__ == "__main__":
    main()
