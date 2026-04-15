#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional

import librosa
import torch
from qwen_asr import Qwen3ASRModel
from peft.peft_model import PeftModelForCausalLM

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
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


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype):
    new_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        new_inputs[k] = v
    return new_inputs


def batch_decode_text(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return processor.tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def unwrap_generate_output(gen_out):
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    return gen_out



def extract_payload_text(raw_text: str) -> str:
    """
    Example:
        language English<asr_text>{"content": 8, "vocabulary":4,"pronunciation":2}
    -> returns:
        {"content": 8, "vocabulary":4,"pronunciation":2}
    """
    raw_text = (raw_text or "").strip()
    m = re.match(r"^language\s+.+?<asr_text>(.*)$", raw_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text

def try_parse_score_dict(text: str) -> Dict[str, Any]:
    """
    Robustly parse score json from model output / label text.
    """
    payload = extract_payload_text(text)

    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}

def infer_one(
    asr_wrapper,
    audio_path: str,
    prompt: str = "",
    sr: int = 16000,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)

    wav = load_audio(audio_path, sr=sr)
    prefix_text = build_prefix_text(processor, prompt)

    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    
    model.eval()
    with torch.inference_mode():
        gen_out = model.generate(**inputs, **gen_kwargs)

    output_ids = unwrap_generate_output(gen_out)

    if not torch.is_tensor(output_ids):
        raise TypeError(f"generate() returned unsupported type: {type(output_ids)}")

    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    if output_ids.size(1) > prefix_len:
        gen_only_ids = output_ids[:, prefix_len:]
    else:
        gen_only_ids = output_ids

    decoded = batch_decode_text(processor, gen_only_ids)[0].strip()
    return decoded


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}: {e}")
    return data


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
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_jsonl_name(input_jsonl: str) -> str:
    base = os.path.basename(input_jsonl)
    name, _ = os.path.splitext(base)
    return name


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


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SLU test script")

    p.add_argument("--exp_dir", type=str, required=True,
                   help="Experiment directory. Will load train_conf.json from this directory")
    
    p.add_argument("--auto_latest_checkpoint", action="store_true",
                   help="If exp_dir contains checkpoints, automatically use latest checkpoint")

    p.add_argument("--auto_best_checkpoint", action="store_true",
                   help="If exp_dir contains checkpoints, automatically use best checkpoint")

    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL with fields like text_id, query, audio, prompt")

    p.add_argument("--output_root", type=str, default="checkpoints",
                   help='Root output dir. Default: "checkpoints"')

    p.add_argument("--device", type=str, default="cuda:0",
                   help='e.g. "cuda:0", "cuda:1", "cpu"')
    return p.parse_args()


def load_train_conf_from_exp_dir(exp_dir: str) -> Optional[List[Dict[str, Any]]]:
    if not exp_dir:
        return None

    train_conf_path = os.path.join(exp_dir, "train_conf.json")
    if not os.path.isfile(train_conf_path):
        raise FileNotFoundError(f"train_conf.json not found under exp_dir: {train_conf_path}")

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf.json must be [training_args, model_args]")
    if not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("Both train_conf entries must be dictionaries")
    return cfg


def main():
    args = parse_args()

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    if train_conf is None:
        raise ValueError("Unable to load train_conf from exp_dir")

    training_args_conf, model_args_conf = train_conf
    sr = int(model_args_conf.get("sr", 16000))
    max_new_tokens = int(model_args_conf.get("max_new_tokens", 256))
    do_sample = bool(model_args_conf.get("do_sample", False))
    temperature = float(model_args_conf.get("temperature", 0.0))
    top_p = float(model_args_conf.get("top_p", 1.0))
    dtype_str = str(model_args_conf.get("dtype", "auto"))

    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, "checkpoint-best")
    elif args.auto_latest_checkpoint:
        latest_ckpt = find_latest_checkpoint(model_path)
        if latest_ckpt is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = latest_ckpt
    
    print(f"[info] use checkpoint: {model_path}")

    dtype = resolve_dtype(dtype_str, args.device)
    jsonl_name = get_jsonl_name(args.input_jsonl)

    # LoRA
    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        lora_type = model_args_conf.get("lora_type", "default")
        print(f"LoRA Finetuning {lora_type}")
        lora_path = model_path
        model_path = model_args_conf["model_path"]

        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=args.device,
        )
        asr_wrapper.model = PeftModelForCausalLM.from_pretrained(
            asr_wrapper.model,
            lora_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        print("Full Finetuning")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=args.device,
        )

    rows = load_jsonl(args.input_jsonl)
    rows_out = []

    for i, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{i}")).strip()
        audio_path = row.get("audio", "")
        prompt = row.get("prompt", "")
        #print(prompt)
        query = row.get("query", "")
        semantics = row.get("semantics", "")

        if not audio_path:
            print(f"[skip] line {i}: no audio field")
            continue

        pred_raw = infer_one(
            asr_wrapper=asr_wrapper,
            audio_path=audio_path,
            prompt=prompt,
            sr=sr,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        '''
        if "<slu>" in pred_raw:
            pred_query = pred_raw.split("<slu>")[0].split("<asr_text>")[1]
            pred_raw = pred_raw.split("<slu>")[1]
        else:
            pred_query = ""

        rows_out.append({
            "text_id": text_id,
            "query": query,
            "pred_query": pred_query,
            "pred_raw": pred_raw,
            "pred_semantics": try_parse_semantics_list(pred_raw),
        })
        '''
        pred_json = try_parse_score_dict(pred_raw)
        pred_query = pred_json["asr_text"]
        
        try:
            pred_semantics = json.loads(pred_json["semantics"])
        except:
            print(f"[WARNING] Processing failed for {text_id}: {pred_json['semantics']}")
            pred_semantics = [{"Failed": pred_json["semantics"]}]

        rows_out.append({
            "text_id": text_id,
            "query": query,
            "semantics": semantics,
            "pred_query": pred_query,
            "pred_raw": pred_raw,
            "pred_semantics": pred_semantics
        })

        print(f"[{i}/{len(rows)}] done: {text_id}")

    write_slu_prediction_jsonl(rows_out=rows_out, output_root=args.output_root, jsonl_name=jsonl_name)


if __name__ == "__main__":
    main()
