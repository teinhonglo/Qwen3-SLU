#!/usr/bin/env python3
"""Run RobustGER H2T decoding and emit the project's metrics JSONL format."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

from finetuning.qwen3_robustger_model import RobustGERForCausalLM
from local.score_nbest_oracle import parse_hypothesis


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_model(config: Dict[str, Any], checkpoint: Path, device: torch.device):
    base_config = __import__("transformers").AutoConfig.from_pretrained(config["base_model"])
    base_config.robustger_noise_dim = int(config["noise_dim"])
    base_config.robustger_adapter_prompt_length = int(config["adapter_prompt_length"])
    base_config.robustger_adapter_start_layer = int(config["adapter_start_layer"])
    base_config._attn_implementation = "eager"
    dtype = torch.float16 if device.type == "cuda" and config["dtype"] == "float16" else torch.float32
    model = RobustGERForCausalLM.from_pretrained(
        config["base_model"],
        config=base_config,
        torch_dtype=dtype,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_adapter_state_dict(state.get("model", state))
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RobustGER adapter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(config, Path(args.checkpoint), device)

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    language_noise = np.load(manifest_path.parent / "language_noise.npy", mmap_mode="r")
    max_new_tokens = int(
        config["max_new_tokens"] if args.max_new_tokens is None else args.max_new_tokens
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for count, row in enumerate(rows, 1):
            encoded = tokenizer(
                row["input_text"],
                return_tensors="pt",
                truncation=True,
                max_length=int(config["max_length"]),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            feature = torch.from_numpy(
                np.asarray(language_noise[int(row["index"])], dtype=np.float32)
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    noise_embedding=feature,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            generated_text = tokenizer.decode(
                generated[0, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            parsed = parse_hypothesis(generated_text)
            handle.write(json.dumps({
                "text_id": row["text_id"],
                "query": row.get("query", ""),
                "semantics": row.get("semantics", []),
                "pred_query": parsed["pred_query"],
                "pred_semantics": parsed["pred_semantics"],
                "pred_raw": generated_text,
            }, ensure_ascii=False) + "\n")
            if count % 50 == 0 or count == len(rows):
                print(f"[INFO] decoded {count}/{len(rows)}")

    print(f"[INFO] saved RobustGER predictions: {output_path}")


if __name__ == "__main__":
    main()
