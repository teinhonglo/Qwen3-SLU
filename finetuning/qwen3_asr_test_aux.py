#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-output inference for StructSFT auxiliary tasks such as PII and CDI."""

import argparse
import json
import os

import numpy as np
import torch
from peft.peft_model import PeftModelForCausalLM
from qwen_asr import Qwen3ASRModel

from qwen3_asr_test import (
    build_output_subdir_name,
    find_latest_checkpoint,
    infer_one,
    load_decoding_conf,
    load_jsonl,
    load_train_conf_from_exp_dir,
    resolve_decoding_conf,
    resolve_dtype,
    save_resolved_decoding_conf,
    validate_decoding_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR raw-output inference for PII/CDI auxiliary tasks"
    )
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--auto_latest_checkpoint", action="store_true")
    parser.add_argument("--auto_best_checkpoint", action="store_true")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--decoding_conf",
        type=str,
        default="conf/decoding/basic_decoding.json",
    )
    parser.add_argument("--num_return_sequences", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def write_aux_prediction_jsonl(rows_out, output_root: str, jsonl_name: str) -> None:
    """Write raw generations without forcing Full-SLU JSON parsing."""
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as output:
        for row in rows_out:
            # Keep the evaluation metadata and, critically, pred_raw/nbest.
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] saved auxiliary predictions: {out_path}")


def main() -> None:
    args = parse_args()
    if args.auto_latest_checkpoint and args.auto_best_checkpoint:
        raise ValueError(
            "Use only one of --auto_latest_checkpoint or --auto_best_checkpoint"
        )

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    if train_conf is None:
        raise ValueError("Unable to load train_conf from exp_dir")
    _, model_args_conf = train_conf

    sr = int(model_args_conf.get("sr", 16000))
    dtype_str = str(model_args_conf.get("dtype", "auto"))

    decoding_conf = load_decoding_conf(args.decoding_conf)
    resolved_decoding = resolve_decoding_conf(model_args_conf, decoding_conf)
    effective_mode = validate_decoding_mode(resolved_decoding)
    gen_cfg = resolved_decoding["generation"]
    if args.num_return_sequences is not None:
        gen_cfg["num_return_sequences"] = max(1, int(args.num_return_sequences))
    if args.beam_size is not None:
        gen_cfg["beam_size"] = max(1, int(args.beam_size))
    gen_cfg["beam_size"] = max(
        int(gen_cfg.get("beam_size", 1)),
        int(gen_cfg.get("num_return_sequences", 1)),
    )

    max_new_tokens = int(gen_cfg["max_new_tokens"])
    do_sample = bool(gen_cfg["do_sample"])
    temperature = float(gen_cfg["temperature"])
    top_p = float(gen_cfg["top_p"])
    top_k = int(gen_cfg.get("top_k", 0))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.0))
    num_return_sequences = int(gen_cfg.get("num_return_sequences", 1))
    beam_size = int(gen_cfg.get("beam_size", num_return_sequences))

    checkpoint_path = args.exp_dir
    if args.auto_best_checkpoint:
        checkpoint_path = os.path.join(args.exp_dir, "checkpoint-best")
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    elif args.auto_latest_checkpoint:
        checkpoint_path = find_latest_checkpoint(args.exp_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint-* found under: {args.exp_dir}")

    print(f"[info] use checkpoint: {checkpoint_path}")

    dtype = resolve_dtype(dtype_str, args.device)
    output_name = build_output_subdir_name(
        args.input_jsonl, effective_mode, args.decoding_conf
    )
    resolved_decoding["effective_mode"] = effective_mode
    save_resolved_decoding_conf(resolved_decoding, args.output_root, output_name)

    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        lora_type = model_args_conf.get("lora_type", "default")
        print(f"LoRA Finetuning: {lora_type} and {effective_mode}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_args_conf["model_path"],
            dtype=dtype,
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        if effective_mode == "layer_lmhead":
            layer_cfg = resolved_decoding.get("layer_lmhead", {})
            asr_wrapper.model.set_layer_lmhead_index(
                int(layer_cfg.get("layer_index", -1))
            )
        asr_wrapper.model = PeftModelForCausalLM.from_pretrained(
            asr_wrapper.model,
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        print(f"Full Finetuning: {effective_mode}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            checkpoint_path,
            dtype=dtype,
            device_map=args.device,
        )
        if effective_mode == "layer_lmhead":
            layer_cfg = resolved_decoding.get("layer_lmhead", {})
            asr_wrapper.model.set_layer_lmhead_index(
                int(layer_cfg.get("layer_index", -1))
            )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[info] decoding seed: {args.seed}")

    rows = load_jsonl(args.input_jsonl)
    rows_out = []
    for index, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{index}")).strip()
        audio_path = row.get("audio", "")
        prompt = row.get("prompt", "")
        input_mode = str(row.get("input_mode", "audio") or "audio").strip().lower()
        if input_mode not in {"audio", "text"}:
            raise ValueError(f"Unsupported input_mode for {text_id}: {input_mode}")
        use_audio = input_mode == "audio"
        input_text = str(row.get("query", "") or "").strip() if not use_audio else ""
        if use_audio and not audio_path:
            print(f"[skip] line {index}: no audio field")
            continue
        if not use_audio and not input_text:
            print(f"[skip] line {index}: no query field for text-only input")
            continue

        generated = infer_one(
            asr_wrapper=asr_wrapper,
            audio_path=audio_path,
            prompt=prompt,
            input_text=input_text,
            use_audio=use_audio,
            sr=sr,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            output_root=args.output_root,
            decoding_mode=effective_mode,
            dola_conf=resolved_decoding.get("dola", {}),
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            beam_size=beam_size,
        )
        nbest = generated if isinstance(generated, list) else [generated]
        pred_raw = nbest[0] if nbest else ""

        result = dict(row)
        result["pred_raw"] = pred_raw
        result["nbest"] = nbest
        rows_out.append(result)
        print(f"[{index}/{len(rows)}] done: {text_id}")

    write_aux_prediction_jsonl(
        rows_out=rows_out,
        output_root=args.output_root,
        jsonl_name=output_name,
    )


if __name__ == "__main__":
    main()
