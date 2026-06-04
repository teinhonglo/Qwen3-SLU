#!/usr/bin/env python3
"""Qwen3-ASR SLU test with pre-generation domain/intent prototype prompting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import librosa
import torch
from transformers import AutoProcessor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetuning.prototype_prompt_utils import extract_gold_domain_intents, format_domain_intent_candidates, get_prompt_template  # noqa: E402

_CKPT_RE = __import__("re").compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        match = _CKPT_RE.match(name)
        if not match:
            continue
        step = int(match.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_text = processor.apply_chat_template([build_prefix_messages(prompt, None)], add_generation_prompt=True, tokenize=False)
    return prefix_text[0] if isinstance(prefix_text, list) else prefix_text


def batch_decode_text(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def unwrap_generate_output(gen_out):
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    return gen_out


def try_parse_score_dict(text: str) -> Dict[str, Any]:
    import re

    payload = (text or "").strip()
    match = re.match(r"^language\s+.+?<asr_text>(.*)$", payload, flags=re.DOTALL)
    if match:
        payload = match.group(1).strip()
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}: {exc}") from exc
    return rows


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            major = torch.cuda.get_device_capability(device=device)[0]
        except Exception:
            major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def get_jsonl_name(input_jsonl: str) -> str:
    return os.path.splitext(os.path.basename(input_jsonl))[0]


def build_output_subdir_name(input_jsonl: str, decoding_mode: str, decoding_conf_path: str) -> str:
    jsonl_name = get_jsonl_name(input_jsonl)
    if decoding_mode == "basic":
        return jsonl_name
    conf_name = os.path.splitext(os.path.basename(decoding_conf_path))[0] if decoding_conf_path else "decoding"
    return f"{jsonl_name}_{conf_name}"


def write_slu_prediction_jsonl(rows_out: List[Dict[str, Any]], output_root: str, jsonl_name: str):
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            item = {
                "text_id": row["text_id"],
                "query": row.get("query", ""),
                "semantics": row.get("semantics", []),
                "pred_query": row.get("pred_query", ""),
                "pred_semantics": row.get("pred_semantics", []),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[info] saved: {out_path}")


def write_prototype_prediction_jsonl(rows_out: List[Dict[str, Any]], output_root: str, jsonl_name: str):
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "prototype_predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[info] saved prototype predictions: {out_path}")

def build_corrected_prompt(corrected_prompt: str, pred_json: Dict[str, Any]) -> str:
    return f"""{corrected_prompt}\n\npred_json:\n{json.dumps(pred_json, ensure_ascii=False)}\n"""


def write_corrected_jsonl(rows_out: List[Dict[str, Any]], output_jsonl_dir: str, corrected_prompt: str, jsonl_name: str):
    save_dir = os.path.join(output_jsonl_dir, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "corrected.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            item = {
                "text_id": row["text_id"],
                "query": row.get("query", ""),
                "audio": row.get("audio", ""),
                "prompt": build_corrected_prompt(corrected_prompt, row.get("pred_json", {})),
                "text": row.get("text", ""),
                "semantics": row.get("semantics", []),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[info] saved corrected jsonl: {out_path}")


def load_corrected_prompt(corrected_prompt_file: str) -> str:
    if not corrected_prompt_file:
        return ""
    with open(corrected_prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_decoding_conf(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("decoding config must be a JSON object")
    return cfg


def resolve_decoding_conf(model_args_conf: Dict[str, Any], decoding_conf: Dict[str, Any]) -> Dict[str, Any]:
    dec = (decoding_conf or {}).get("decoding", {})
    gen = dec.get("generation", {})
    return {
        "mode": str(dec.get("mode", "basic")),
        "strict": bool(dec.get("strict", False)),
        "generation": {
            "max_new_tokens": int(gen.get("max_new_tokens", model_args_conf.get("max_new_tokens", 256))),
            "do_sample": bool(gen.get("do_sample", model_args_conf.get("do_sample", False))),
            "temperature": float(gen.get("temperature", model_args_conf.get("temperature", 0.0))),
            "top_p": float(gen.get("top_p", model_args_conf.get("top_p", 1.0))),
            "top_k": int(gen.get("top_k", 0)),
            "repetition_penalty": float(gen.get("repetition_penalty", 1.0)),
        },
        "dola": dec.get("dola", {}),
        "layer_lmhead": dec.get("layer_lmhead", {}),
    }


def validate_decoding_mode(resolved: Dict[str, Any]) -> str:
    mode = resolved.get("mode", "basic")
    if mode not in {"basic", "dola", "layer_lmhead"}:
        raise ValueError(f"Unsupported decoding mode: {mode}")
    return mode


def save_resolved_decoding_conf(resolved: Dict[str, Any], output_root: str, jsonl_name: str):
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "resolved_decoding_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    print(f"[info] saved decoding config: {out_path}")


def load_train_conf_from_exp_dir(exp_dir: str) -> List[Dict[str, Any]]:
    train_conf_path = os.path.join(exp_dir, "train_conf.json")
    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list) or len(cfg) != 2 or not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("train_conf.json must be [training_args, model_args]")
    return cfg


from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig  # noqa: E402
from qwen_asr.core.transformers_backend.modeling_qwen3_asr_prototype import (  # noqa: E402
    Qwen3ASRPrototypeForConditionalGeneration,
)


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SLU test with domain/intent prototypes")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--auto_latest_checkpoint", action="store_true")
    p.add_argument("--auto_best_checkpoint", action="store_true")
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_root", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_jsonl_dir", type=str, default="")
    p.add_argument("--corrected_prompt_file", type=str, default="data/macslu/corrected_prompt.txt")
    p.add_argument("--decoding_conf", type=str, default="conf/decoding/basic_decoding.json")
    p.add_argument("--prototype_top_k", type=int, default=0, help="Override prototype.k from train_conf when > 0")
    return p.parse_args()


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype):
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        out[key] = value
    return out


def parse_semantics(pred_json: Dict[str, Any]):
    try:
        if isinstance(pred_json.get("semantics"), list):
            return pred_json["semantics"]
        return json.loads(pred_json.get("semantics", "[]"))
    except Exception:
        return [{"FAILED": pred_json}]


def resolve_checkpoint_path(args) -> str:
    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        return os.path.join(model_path, "checkpoint-best")
    if args.auto_latest_checkpoint:
        latest = find_latest_checkpoint(model_path)
        if latest is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        return latest
    return model_path


def load_prototype_model(args, model_args_conf: Dict[str, Any], dtype: torch.dtype):
    ckpt_path = resolve_checkpoint_path(args)
    print(f"[info] use checkpoint: {ckpt_path}")
    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    if not prototype_conf.get("enabled", False):
        raise ValueError("train_conf model_args.prototype.enabled must be true for prototype testing")

    if model_args_conf.get("lora_config", None):
        from peft.peft_model import PeftModelForCausalLM

        base_path = model_args_conf["model_path"]
        config = Qwen3ASRConfig.from_pretrained(base_path)
        config.thinker_config.prototype_config = prototype_conf
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(
            base_path,
            config=config,
            dtype=dtype,
            device_map=args.device,
            attn_implementation="flash_attention_2",
        )
        model = PeftModelForCausalLM.from_pretrained(model, ckpt_path, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(base_path, fix_mistral_regex=True)
    else:
        model = Qwen3ASRPrototypeForConditionalGeneration.from_pretrained(ckpt_path, dtype=dtype, device_map=args.device)
        processor = AutoProcessor.from_pretrained(ckpt_path, fix_mistral_regex=True)
    model.eval()
    return model, processor, ckpt_path


def get_predict_model(model):
    if hasattr(model, "predict_prototypes"):
        return model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if hasattr(base, "predict_prototypes"):
            return base
    base_model = getattr(model, "base_model", None)
    inner = getattr(base_model, "model", None) if base_model is not None else None
    if inner is not None and hasattr(inner, "predict_prototypes"):
        return inner
    raise RuntimeError("Unable to locate prototype-aware base model for predict_prototypes")


def infer_one(
    model,
    processor,
    audio_path: str,
    prompt: str,
    sr: int,
    prototype_top_k: int,
    prompt_template: Dict[str, str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    decoding_mode: str,
    dola_conf: Optional[Dict[str, Any]],
):
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)
    wav = load_audio(audio_path, sr=sr)

    proto_prefix_text = build_prefix_text(processor, prompt)
    proto_inputs = processor(text=[proto_prefix_text], audio=[wav], return_tensors="pt", padding=True, truncation=False)
    proto_prefix_len = int(proto_inputs["attention_mask"][0].sum().item())
    proto_inputs["prototype_prefix_lengths"] = torch.tensor([proto_prefix_len], dtype=torch.long)
    proto_inputs = move_inputs_to_device(proto_inputs, device=device, model_dtype=model_dtype)
    proto_model = get_predict_model(model)
    proto_hits = proto_model.predict_prototypes(top_k=prototype_top_k, **proto_inputs)
    domains = [item["label"] for item in proto_hits["domains"][0]]
    intents = [item["label"] for item in proto_hits["intents"][0]]
    augmented_prompt = format_domain_intent_candidates(prompt, domains, intents, **prompt_template)

    prefix_text = build_prefix_text(processor, augmented_prompt)
    inputs = processor(text=[prefix_text], audio=[wav], return_tensors="pt", padding=True, truncation=False)
    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "repetition_penalty": repetition_penalty}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    if decoding_mode == "dola":
        dola_conf = dola_conf or {}
        gen_kwargs["repetition_penalty"] = dola_conf["repetition_penalty"]
        gen_kwargs["dola_layers"] = dola_conf.get("layers", "high")
        gen_kwargs["trust_remote_code"] = True
        gen_kwargs["output_hidden_states"] = True
    elif decoding_mode not in {"basic", "layer_lmhead"}:
        raise ValueError(f"Unsupported decoding mode: {decoding_mode}")

    with torch.inference_mode():
        gen_out = model.generate(**inputs, **gen_kwargs)
    output_ids = unwrap_generate_output(gen_out)
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    gen_only_ids = output_ids[:, prefix_len:] if output_ids.size(1) > prefix_len else output_ids
    decoded = batch_decode_text(processor, gen_only_ids)[0].strip()
    return decoded, {
        "prototype_domains": proto_hits["domains"][0],
        "prototype_intents": proto_hits["intents"][0],
        "prototype_augmented_prompt": augmented_prompt,
    }


def main():
    args = parse_args()
    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    _, model_args_conf = train_conf
    sr = int(model_args_conf.get("sr", 16000))
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    decoding_conf = load_decoding_conf(args.decoding_conf)
    resolved_decoding = resolve_decoding_conf(model_args_conf, decoding_conf)
    effective_mode = validate_decoding_mode(resolved_decoding)
    gen_cfg = resolved_decoding["generation"]
    max_new_tokens = int(gen_cfg["max_new_tokens"])
    do_sample = bool(gen_cfg["do_sample"])
    temperature = float(gen_cfg["temperature"])
    top_p = float(gen_cfg["top_p"])
    top_k = int(gen_cfg.get("top_k", 0))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.0))

    prototype_conf = dict(model_args_conf.get("prototype", {}) or {})
    prototype_top_k = int(args.prototype_top_k or prototype_conf.get("k", 5))
    prompt_template = get_prompt_template(prototype_conf)
    model, processor, _ = load_prototype_model(args, model_args_conf, dtype)
    if effective_mode == "layer_lmhead":
        layer_cfg = resolved_decoding.get("layer_lmhead", {})
        layer_index = int(layer_cfg.get("layer_index", -1))
        target_model = get_predict_model(model)
        target_model.set_layer_lmhead_index(layer_index)

    jsonl_name = build_output_subdir_name(args.input_jsonl, effective_mode, args.decoding_conf)
    resolved_decoding["effective_mode"] = effective_mode
    resolved_decoding["prototype_top_k"] = prototype_top_k
    save_resolved_decoding_conf(resolved_decoding, args.output_root, jsonl_name)

    rows = load_jsonl(args.input_jsonl)
    rows_out: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    prototype_pred_rows: List[Dict[str, Any]] = []
    
    for i, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{i}")).strip()
        audio_path = row.get("audio", "")
        if not audio_path:
            print(f"[skip] line {i}: no audio field")
            continue
        pred_raw, trace = infer_one(
            model=model,
            processor=processor,
            audio_path=audio_path,
            prompt=row.get("prompt", ""),
            sr=sr,
            prototype_top_k=prototype_top_k,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            decoding_mode=effective_mode,
            dola_conf=resolved_decoding.get("dola", {}),
        )
        pred_json = try_parse_score_dict(pred_raw)
        pred_query = pred_json.get("asr_text", "FAILED")
        pred_semantics = parse_semantics(pred_json)
        out = {
            "text_id": text_id,
            "query": row.get("query", ""),
            "audio": audio_path,
            "text": row.get("text", ""),
            "semantics": row.get("semantics", []),
            "pred_json": pred_json,
            "pred_query": pred_query,
            "pred_raw": pred_raw,
            "pred_semantics": pred_semantics,
        }
        rows_out.append(out)
        trace_rows.append({"text_id": text_id, **trace, "pred_raw": pred_raw, "pred_json": pred_json})
        
        gold_domains, gold_intents = extract_gold_domain_intents(row)
        prototype_pred_rows.append(
            {
                "id": text_id,
                "pred_domains": [x.get("label", "") for x in trace.get("prototype_domains", [])],
                "pred_intents": [x.get("label", "") for x in trace.get("prototype_intents", [])],
                "gold_domains": gold_domains,
                "gold_intents": gold_intents,
            }
        )
        print(f"[{i}/{len(rows)}] done: {text_id}")

    write_slu_prediction_jsonl(rows_out=rows_out, output_root=args.output_root, jsonl_name=jsonl_name)
    write_prototype_prediction_jsonl(prototype_pred_rows, args.output_root, jsonl_name)
    
    save_dir = os.path.join(args.output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    trace_path = os.path.join(save_dir, "domain_intent_prototype_trace.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for item in trace_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[info] saved prototype trace: {trace_path}")

    if args.output_jsonl_dir:
        corrected_prompt = load_corrected_prompt(args.corrected_prompt_file)
        write_corrected_jsonl(rows_out=rows_out, output_jsonl_dir=args.output_jsonl_dir, corrected_prompt=corrected_prompt, jsonl_name=jsonl_name)


if __name__ == "__main__":
    main()
