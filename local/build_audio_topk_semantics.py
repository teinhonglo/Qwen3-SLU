#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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
def extract_embedding(model, processor, audio_path: str, device: str) -> np.ndarray:
    inputs = processor(audio=[audio_path], text=[""], return_tensors="pt", padding=True)
    input_features = inputs["input_features"].to(device)
    feature_attention_mask = inputs.get("feature_attention_mask")
    if feature_attention_mask is not None:
        feature_attention_mask = feature_attention_mask.to(device)

    audio_features = model.get_audio_features(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=None,
    )

    pooled = audio_features.mean(dim=0)
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
    model_kwargs: Dict[str, object] = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = model.to(args.device)
    model.eval()

    embeddings = []
    for i, row in enumerate(rows, start=1):
        audio_path = row.get("audio")
        if not audio_path:
            raise ValueError(f"Missing 'audio' field at row index {i}")
        emb = extract_embedding(model, processor, audio_path, args.device)
        embeddings.append(emb)

    emb_mat = np.stack(embeddings, axis=0)
    out_rows = build_topk(rows, emb_mat, args.topk)
    write_jsonl(output_path, out_rows)

    print(f"[INFO] rows: {len(rows)}")
    print(f"[INFO] topK: {args.topk}")
    print(f"[INFO] output: {output_path}")


if __name__ == "__main__":
    main()
