#!/usr/bin/env python3
"""Silence and real-minus-silence decoding experiments for Qwen3-SLU.

The contrast decoder intentionally supports greedy decoding only. It keeps two
independent KV caches, feeds both branches the same generated token history, and
contrasts their vocabulary logits at each decoding step.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft.peft_model import PeftModelForCausalLM
from qwen_asr import Qwen3ASRModel

from qwen3_asr_test import (
    batch_decode_text,
    build_prefix_text,
    find_latest_checkpoint,
    load_audio,
    load_jsonl,
    load_train_conf_from_exp_dir,
    move_inputs_to_device,
    resolve_dtype,
    try_parse_score_dict,
)


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR silence contrast test")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--auto_latest_checkpoint", action="store_true")
    p.add_argument("--auto_best_checkpoint", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--contrast_scope", choices=["first_step", "all_steps", "both"], default="both")
    p.add_argument("--silence_value", type=float, default=0.0)
    p.add_argument("--plausibility_alpha", type=float, default=0.0,
                   help="Keep tokens with p_real >= plausibility_alpha * max(p_real). 0 disables filtering.")
    return p.parse_args()


def load_model(args, model_args_conf):
    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, "checkpoint-best")
    elif args.auto_latest_checkpoint:
        model_path = find_latest_checkpoint(model_path)
        if model_path is None:
            raise ValueError(f"No checkpoint-* found under: {args.exp_dir}")

    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    lora_config = model_args_conf.get("lora_config")
    if lora_config:
        wrapper = Qwen3ASRModel.from_pretrained(
            model_args_conf["model_path"], dtype=dtype, device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        wrapper.model = PeftModelForCausalLM.from_pretrained(
            wrapper.model, model_path, torch_dtype=dtype,
        )
    else:
        wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=args.device)
    wrapper.model.eval()
    print(f"[info] use checkpoint: {model_path}")
    return wrapper


def get_thinker(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "thinker"):
        raise TypeError(f"Cannot locate Qwen3-ASR thinker under {type(model)}")
    return base.thinker


def prepare_pair(wrapper, wav: np.ndarray, prompt: str, device, dtype, silence_value: float):
    prefix = build_prefix_text(wrapper.processor, prompt)
    silence = np.full_like(wav, fill_value=silence_value)
    common = dict(text=[prefix], return_tensors="pt", padding=True, truncation=False)
    real_inputs = wrapper.processor(audio=[wav], **common)
    silence_inputs = wrapper.processor(audio=[silence], **common)
    for key in ("input_ids", "attention_mask", "feature_attention_mask"):
        if key in real_inputs and key in silence_inputs and real_inputs[key].shape != silence_inputs[key].shape:
            raise ValueError(f"Real/silence {key} shape mismatch: {real_inputs[key].shape} vs {silence_inputs[key].shape}")
    return (
        move_inputs_to_device(real_inputs, device, dtype),
        move_inputs_to_device(silence_inputs, device, dtype),
    )


def forward_step(thinker, step_inputs: Dict[str, Any], past=None):
    kwargs = dict(step_inputs)
    past_len = past.get_seq_length() if past is not None else 0
    seq_len = kwargs["input_ids"].shape[1]
    kwargs["cache_position"] = torch.arange(
        past_len, past_len + seq_len, device=kwargs["input_ids"].device,
    )
    kwargs.update(
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    return thinker(**kwargs)


def greedy_dual_decode(
    wrapper,
    real_inputs: Dict[str, Any],
    silence_inputs: Dict[str, Any],
    max_new_tokens: int,
    alpha: float,
    contrast_scope: str,
    plausibility_alpha: float,
) -> Tuple[torch.Tensor, List[float]]:
    """Decode with (1 + alpha) * real_logits - alpha * silence_logits."""
    thinker = get_thinker(wrapper.model)
    eos = {151645, 151643}
    generated: List[torch.Tensor] = []
    contrast_norms: List[float] = []
    real_past = silence_past = None
    real_step = dict(real_inputs)
    silence_step = dict(silence_inputs)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            real_out = forward_step(thinker, real_step, real_past)
            silence_out = forward_step(thinker, silence_step, silence_past)
            real_logits = real_out.logits[:, -1, :].float()
            silence_logits = silence_out.logits[:, -1, :].float()
            use_contrast = contrast_scope == "all_steps" or (contrast_scope == "first_step" and step == 0)
            decode_logits = (
                (1.0 + alpha) * real_logits - alpha * silence_logits
                if use_contrast else real_logits
            )
            contrast_norms.append(float((real_logits - silence_logits).norm(dim=-1).mean().cpu()))

            if use_contrast and plausibility_alpha > 0.0:
                if not 0.0 < plausibility_alpha <= 1.0:
                    raise ValueError("plausibility_alpha must be in (0, 1]")
                cutoff = real_logits.max(dim=-1, keepdim=True).values + math.log(plausibility_alpha)
                decode_logits = decode_logits.masked_fill(real_logits < cutoff, -torch.inf)

            next_token = decode_logits.argmax(dim=-1)
            generated.append(next_token)
            real_past, silence_past = real_out.past_key_values, silence_out.past_key_values
            if int(next_token[0]) in eos:
                break

            # Both branches must receive the identical decoded history. Their
            # caches remain separate because their audio-conditioned prefixes differ.
            real_mask = torch.cat([real_step["attention_mask"], torch.ones_like(next_token[:, None])], dim=1)
            silence_mask = torch.cat([silence_step["attention_mask"], torch.ones_like(next_token[:, None])], dim=1)
            real_step = {"input_ids": next_token[:, None], "attention_mask": real_mask}
            silence_step = {"input_ids": next_token[:, None], "attention_mask": silence_mask}

    if not generated:
        return torch.empty((1, 0), dtype=torch.long, device=real_inputs["input_ids"].device), contrast_norms
    return torch.stack(generated, dim=1), contrast_norms


def silence_generate(wrapper, silence_inputs, max_new_tokens: int):
    with torch.inference_mode():
        out = wrapper.model.generate(
            **silence_inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1,
        )
    prefix_len = silence_inputs["input_ids"].shape[1]
    return out.sequences[:, prefix_len:]


def decode(wrapper, ids):
    texts = batch_decode_text(wrapper.processor, ids)
    return texts[0].strip() if texts else ""


def prediction_fields(text: str):
    obj = try_parse_score_dict(text)
    semantics = obj.get("semantics", [])
    if isinstance(semantics, str):
        try:
            semantics = json.loads(semantics)
        except json.JSONDecodeError:
            semantics = [{"FAILED": obj}]
    return obj.get("asr_text", "FAILED"), semantics, obj


def main():
    args = parse_args()
    _, model_args_conf = load_train_conf_from_exp_dir(args.exp_dir)
    sr = int(model_args_conf.get("sr", 16000))
    wrapper = load_model(args, model_args_conf)
    device = next(wrapper.model.parameters()).device
    dtype = getattr(wrapper.model, "dtype", resolve_dtype("auto", str(device)))
    scopes = ["first_step", "all_steps"] if args.contrast_scope == "both" else [args.contrast_scope]
    os.makedirs(args.output_dir, exist_ok=True)
    handles = {
        name: open(os.path.join(args.output_dir, f"predictions_{name}.jsonl"), "w", encoding="utf-8")
        for name in ["silence"] + scopes
    }
    analysis_f = open(os.path.join(args.output_dir, "contrast_analysis.jsonl"), "w", encoding="utf-8")
    try:
        rows = load_jsonl(args.input_jsonl)
        for i, row in enumerate(rows, 1):
            wav = load_audio(row["audio"], sr=sr)
            real_inputs, silence_inputs = prepare_pair(
                wrapper, wav, row.get("prompt", ""), device, dtype, args.silence_value,
            )
            results = {"silence": decode(wrapper, silence_generate(wrapper, silence_inputs, args.max_new_tokens))}
            norms = {}
            for scope in scopes:
                ids, step_norms = greedy_dual_decode(
                    wrapper, real_inputs, silence_inputs, args.max_new_tokens,
                    args.alpha, scope, args.plausibility_alpha,
                )
                results[scope] = decode(wrapper, ids)
                norms[scope] = step_norms

            text_id = str(row.get("text_id", f"line{i}"))
            for name, text in results.items():
                pred_query, pred_semantics, _ = prediction_fields(text)
                item = {
                    "text_id": text_id, "query": row.get("query", ""),
                    "semantics": row.get("semantics", []), "pred_query": pred_query,
                    "pred_semantics": pred_semantics, "pred_raw": text,
                }
                handles[name].write(json.dumps(item, ensure_ascii=False) + "\n")
            analysis_f.write(json.dumps({
                "text_id": text_id, "audio": row["audio"], "num_samples": len(wav),
                "duration_sec": len(wav) / sr, "alpha": args.alpha,
                "plausibility_alpha": args.plausibility_alpha, "outputs": results,
                "logit_contrast_l2_norm_by_step": norms,
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] done: {text_id}")
    finally:
        for f in handles.values():
            f.close()
        analysis_f.close()
    print(f"[info] saved silence/contrast results under: {args.output_dir}")


if __name__ == "__main__":
    main()
