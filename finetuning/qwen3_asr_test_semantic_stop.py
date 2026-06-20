#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
from peft.peft_model import PeftModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetuning.qwen3_asr_test import (  # noqa: E402
    batch_decode_text,
    build_prefix_text,
    find_latest_checkpoint,
    load_decoding_conf,
    load_jsonl,
    load_train_conf_from_exp_dir,
    move_inputs_to_device,
    resolve_decoding_conf,
    resolve_dtype,
    try_parse_score_dict,
    unwrap_generate_output,
    validate_decoding_mode,
)
from local.metrics import normalize_semantics  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402


def canonicalize_semantic_frame(frame):
    """Return a hashable semantic frame using local.metrics normalization."""
    normalized = normalize_semantics([frame])
    if not normalized:
        return None
    item = normalized[0]
    slots = item.get("slots", {})
    implicit_slots = item.get("implicit_slots", {})
    if not isinstance(slots, dict):
        slots = {}
    if not isinstance(implicit_slots, dict):
        implicit_slots = {}
    return (
        item.get("domain", ""),
        item.get("intent", ""),
        tuple(sorted(slots.items())),
        tuple(sorted(implicit_slots.items())),
    )


def semantic_frame_counter(semantics_list):
    """Duplicate-aware Counter over canonical semantic frames."""
    counter = Counter()
    if not isinstance(semantics_list, list):
        return counter
    for frame in semantics_list:
        canonical = canonicalize_semantic_frame(frame)
        if canonical is not None:
            counter[canonical] += 1
    return counter


def canonical_frame_to_dict(frame):
    domain, intent, slots, implicit_slots = frame
    return {
        "domain": domain,
        "intent": intent,
        "slots": dict(slots),
        "implicit_slots": dict(implicit_slots),
    }


def expand_semantic_counter(counter):
    rows = []
    for frame, count in counter.items():
        rows.extend([canonical_frame_to_dict(frame)] * count)
    return rows


def semantics_full_exact_multiset(pred_semantics, gold_semantics):
    return semantic_frame_counter(pred_semantics) == semantic_frame_counter(gold_semantics)


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def decode_ids(processor, ids: List[int]) -> str:
    if not ids:
        return ""
    return batch_decode_text(processor, torch.tensor([ids], dtype=torch.long))[0]


def decode_token(processor, token_id: int) -> str:
    tok = getattr(processor, "tokenizer", processor)
    return tok.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _find_semantics_value_start(text: str) -> Optional[int]:
    in_str = False
    esc = False
    start = None
    key = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                raw = text[start:i]
                try:
                    key = json.loads('"' + raw + '"')
                except Exception:
                    key = raw
                in_str = False
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if key == "semantics" and j < len(text) and text[j] == ":":
                    j += 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    return j if j < len(text) else None
        else:
            if ch == '"':
                in_str = True
                esc = False
                start = i + 1
        i += 1
    return None


def _list_started_closed(fragment: str) -> Tuple[bool, bool]:
    in_str = False
    esc = False
    depth = 0
    started = False
    for ch in fragment:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            esc = False
        elif ch == '[':
            depth += 1
            started = True
        elif ch == ']' and started:
            depth -= 1
            if depth <= 0:
                return True, True
    return started, False


def _decode_partial_json_string(fragment: str) -> str:
    if not fragment or fragment[0] != '"':
        return ""
    out = []
    esc = False
    for ch in fragment[1:]:
        if esc:
            if ch in ['"', "\\", "/"]:
                out.append(ch)
            elif ch == "n":
                out.append("\n")
            elif ch == "t":
                out.append("\t")
            elif ch == "r":
                out.append("\r")
            else:
                out.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            break
        else:
            out.append(ch)
    return "".join(out)


def semantics_list_status(text: str) -> Tuple[bool, bool]:
    pos = _find_semantics_value_start(text)
    if pos is None:
        return False, False
    frag = text[pos:]
    frag_l = frag.lstrip()
    if not frag_l:
        return False, False
    if frag_l[0] == '[':
        return _list_started_closed(frag_l)
    if frag_l[0] == '"':
        return _list_started_closed(_decode_partial_json_string(frag_l))
    return False, False


def would_close_semantics_list(processor, prev_ids: List[int], next_id: int) -> bool:
    before = decode_ids(processor, prev_ids)
    after = decode_ids(processor, prev_ids + [int(next_id)])
    before_started, before_closed = semantics_list_status(before)
    after_started, after_closed = semantics_list_status(after)
    return after_started and after_closed and not (before_started and before_closed)


def find_first_semantics_list_closure(processor, generated_ids: List[int]) -> Dict[str, Any]:
    prefix_ids: List[int] = []
    for step, token_id in enumerate(generated_ids):
        before = decode_ids(processor, prefix_ids)
        prefix_ids.append(int(token_id))
        after = decode_ids(processor, prefix_ids)
        b_started, b_closed = semantics_list_status(before)
        a_started, a_closed = semantics_list_status(after)
        if a_started and a_closed and not (b_started and b_closed):
            return {
                "stop_step": step,
                "stop_token_id": int(token_id),
                "stop_token_text": decode_token(processor, int(token_id)),
                "prefix_before_stop": before,
                "decoded_after_stop": after,
            }
    return {
        "stop_step": None,
        "stop_token_id": None,
        "stop_token_text": None,
        "prefix_before_stop": decode_ids(processor, generated_ids),
        "decoded_after_stop": decode_ids(processor, generated_ids),
    }


class FirstSemanticsListStopSuppressor(LogitsProcessor):
    def __init__(self, processor, prompt_len: int):
        self.processor = processor
        self.prompt_len = int(prompt_len)
        self.applied = False
        self.suppressed_step = None
        self.suppressed_token_id = None
        self.suppressed_token_text = None
        self.forced_first_token_id = None
        self.forced_first_token_text = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.applied or input_ids.size(0) != 1:
            return scores
        gen_ids = input_ids[0, self.prompt_len:].detach().cpu().tolist()
        top1_id = int(torch.argmax(scores[0]).item())
        if would_close_semantics_list(self.processor, gen_ids, top1_id):
            self.applied = True
            self.suppressed_step = len(gen_ids)
            self.suppressed_token_id = top1_id
            self.suppressed_token_text = decode_token(self.processor, top1_id)
            scores = scores.clone()
            scores[0, top1_id] = -float("inf")
            forced_id = int(torch.argmax(scores[0]).item())
            self.forced_first_token_id = forced_id
            self.forced_first_token_text = decode_token(self.processor, forced_id)
        return scores


def parse_pred(raw: str) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    pred_json = try_parse_score_dict(raw)
    pred_query = pred_json.get("asr_text", "FAILED") if isinstance(pred_json, dict) else "FAILED"
    try:
        semantics = pred_json.get("semantics", [])
        if isinstance(semantics, list):
            pred_semantics = semantics
        elif isinstance(semantics, str):
            pred_semantics = json.loads(semantics)
        else:
            pred_semantics = []
        if not isinstance(pred_semantics, list):
            pred_semantics = []
    except Exception:
        pred_semantics = [{"FAILED": pred_json}]
    return pred_json, pred_query, pred_semantics


def prepare_inputs(asr_wrapper, audio_path: str, prompt: str, sr: int, device: str):
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    model_dtype = getattr(model, "dtype", torch.float16)
    wav = load_audio(audio_path, sr=sr)
    prefix_text = build_prefix_text(processor, prompt)
    inputs = processor(text=[prefix_text], audio=[wav], return_tensors="pt", padding=True, truncation=False)
    prefix_len = int(inputs["attention_mask"][0].sum().item())
    return move_inputs_to_device(inputs, device=device, model_dtype=model_dtype), prefix_len


def generate_once(asr_wrapper, inputs, prefix_len: int, gen_kwargs: Dict[str, Any], logits_processor=None):
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    kwargs = dict(gen_kwargs)
    
    kwargs.update({"do_sample": False, "num_beams": 1, "output_scores": True})
  
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    model.eval()
    with torch.inference_mode():
        gen_out = model.generate(**inputs, **kwargs)
    output_ids = unwrap_generate_output(gen_out)
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    gen_ids = output_ids[:, prefix_len:] if output_ids.size(1) > prefix_len else output_ids
    raw = batch_decode_text(processor, gen_ids)[0].strip()
    return gen_out, gen_ids[0].detach().cpu().tolist(), raw


def stop_stats(gen_out, stop_step: Optional[int], stop_token_id: Optional[int]) -> Dict[str, Any]:
    if stop_step is None or stop_token_id is None or not hasattr(gen_out, "scores") or stop_step >= len(gen_out.scores):
        return {k: None for k in ["stop_logprob", "stop_probability", "stop_rank", "continue_token_id", "continue_logprob", "continue_probability", "stop_margin"]}
    logits = gen_out.scores[stop_step][0].float().detach().cpu()
    log_probs = torch.log_softmax(logits, dim=-1)
    stop_lp = float(log_probs[int(stop_token_id)].item())
    non_stop = log_probs.clone()
    non_stop[int(stop_token_id)] = -float("inf")
    cont_id = int(torch.argmax(non_stop).item())
    cont_lp = float(non_stop[cont_id].item())
    rank = int((log_probs > log_probs[int(stop_token_id)]).sum().item()) + 1
    return {
        "stop_logprob": stop_lp,
        "stop_probability": float(math.exp(stop_lp)),
        "stop_rank": rank,
        "continue_token_id": cont_id,
        "continue_logprob": cont_lp,
        "continue_probability": float(math.exp(cont_lp)),
        "stop_margin": float(stop_lp - cont_lp),
    }


def count_status(gold_count: int, greedy_count: int) -> str:
    if gold_count == 0:
        return "no_intent"
    if greedy_count == gold_count:
        return "correct_count"
    if greedy_count == gold_count - 1:
        return "under_by_1"
    if greedy_count < gold_count - 1:
        return "under_by_2plus"
    return "overprediction"


def compare_frames(gold, greedy, forced):
    gold_c = semantic_frame_counter(gold)
    greedy_c = semantic_frame_counter(greedy)
    forced_c = semantic_frame_counter(forced)
    missing = gold_c - greedy_c
    added = forced_c - greedy_c
    duplicate = added.copy()
    for frame in list(duplicate):
        if greedy_c[frame] <= 0:
            del duplicate[frame]
    if not added:
        outcome = "no_complete_extra_frame"
    elif sum(gold_c.values()) == 0:
        outcome = "wrong_extra_frame"
    elif added and not (added - missing):
        outcome = "exact_missing_frame_recovered"
    elif duplicate and not (added - duplicate):
        outcome = "duplicate_existing_frame"
    else:
        outcome = "wrong_extra_frame"
    return missing, added, outcome


def load_model(args, model_args_conf, dtype, effective_mode, resolved_decoding):
    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, "checkpoint-best")
    elif args.auto_latest_checkpoint:
        latest_ckpt = find_latest_checkpoint(model_path)
        if latest_ckpt is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = latest_ckpt
    print(f"[info] use checkpoint: {model_path}")
    lora_config = model_args_conf.get("lora_config", None)
    if lora_config:
        print(f"LoRA Finetuning: {model_args_conf.get('lora_type', 'default')} and {effective_mode}")
        lora_path = model_path
        base_path = model_args_conf["model_path"]
        asr_wrapper = Qwen3ASRModel.from_pretrained(base_path, dtype=dtype, device_map=args.device, attn_implementation="flash_attention_2")
        if effective_mode == "layer_lmhead":
            asr_wrapper.model.set_layer_lmhead_index(int(resolved_decoding.get("layer_lmhead", {}).get("layer_index", -1)))
        asr_wrapper.model = PeftModelForCausalLM.from_pretrained(asr_wrapper.model, lora_path, torch_dtype=torch.bfloat16)
    else:
        print(f"Full Finetuning: {effective_mode}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=args.device)
        if effective_mode == "layer_lmhead":
            asr_wrapper.model.set_layer_lmhead_index(int(resolved_decoding.get("layer_lmhead", {}).get("layer_index", -1)))
    return asr_wrapper


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR MAC-SLU semantic STOP diagnostic")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--auto_latest_checkpoint", action="store_true")
    p.add_argument("--auto_best_checkpoint", action="store_true")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decoding_conf", default="conf/decoding/basic_decoding.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--smoke_print", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    training_args_conf, model_args_conf = train_conf
    sr = int(model_args_conf.get("sr", 16000))
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    resolved_decoding = resolve_decoding_conf(model_args_conf, load_decoding_conf(args.decoding_conf))
    effective_mode = validate_decoding_mode(resolved_decoding)
    if effective_mode not in {"basic", "layer_lmhead"}:
        print(f"[warning] forcing greedy diagnostic with mode={effective_mode}; generation extras are ignored")
    gen_cfg = resolved_decoding["generation"]
    gen_kwargs = {"max_new_tokens": int(gen_cfg["max_new_tokens"]), "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.0))}
    asr_wrapper = load_model(args, model_args_conf, dtype, effective_mode, resolved_decoding)
    rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "records.jsonl")
    split = args.split or os.path.splitext(os.path.basename(args.input_jsonl))[0]
    debug_printed = 0
    under_printed = 0
    greedy_pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    forced_pred_path = os.path.join(args.output_dir, "forced_predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f, \
         open(greedy_pred_path, "w", encoding="utf-8") as greedy_f, \
         open(forced_pred_path, "w", encoding="utf-8") as forced_f:
        for i, row in enumerate(rows, start=1):
            text_id = str(row.get("text_id", f"line{i}")).strip()
            audio_path = row.get("audio", "")
            prompt = row.get("prompt", "")
            gold = row.get("semantics", [])
            if isinstance(gold, str):
                try:
                    gold = json.loads(gold)
                except Exception:
                    gold = []
            inputs, prefix_len = prepare_inputs(asr_wrapper, audio_path, prompt, sr, args.device)
            greedy_out, greedy_ids, greedy_raw = generate_once(asr_wrapper, inputs, prefix_len, gen_kwargs)
            _, greedy_query, greedy_sem = parse_pred(greedy_raw)
            closure = find_first_semantics_list_closure(asr_wrapper.processor, greedy_ids)
            stats = stop_stats(greedy_out, closure["stop_step"], closure["stop_token_id"])
            cont_text = decode_token(asr_wrapper.processor, stats["continue_token_id"]) if stats.get("continue_token_id") is not None else None
            suppressor = FirstSemanticsListStopSuppressor(asr_wrapper.processor, prefix_len)
            _, forced_ids, forced_raw = generate_once(asr_wrapper, inputs, prefix_len, gen_kwargs, LogitsProcessorList([suppressor]))
            _, forced_query, forced_sem = parse_pred(forced_raw)
            missing, added, outcome = compare_frames(gold, greedy_sem, forced_sem)
            gold_count = len(normalize_semantics(gold))
            greedy_count = len(normalize_semantics(greedy_sem))
            forced_count = len(normalize_semantics(forced_sem))
            rec = {
                "split": split, "text_id": text_id, "audio": audio_path,
                "gold_frame_count": gold_count, "greedy_frame_count": greedy_count, "forced_frame_count": forced_count,
                "count_status": count_status(gold_count, greedy_count),
                "stop_step": closure["stop_step"], "stop_token_id": closure["stop_token_id"], "stop_token_text": closure["stop_token_text"],
                "stop_logprob": stats["stop_logprob"], "stop_probability": stats["stop_probability"], "stop_rank": stats["stop_rank"],
                "continue_token_id": stats["continue_token_id"], "continue_token_text": cont_text,
                "continue_logprob": stats["continue_logprob"], "continue_probability": stats["continue_probability"], "stop_margin": stats["stop_margin"],
                "greedy_raw": greedy_raw, "greedy_semantics": greedy_sem,
                "forced_raw": forced_raw, "forced_semantics": forced_sem,
                "suppression_applied": suppressor.applied, "suppressed_step": suppressor.suppressed_step,
                "suppressed_token_id": suppressor.suppressed_token_id, "suppressed_token_text": suppressor.suppressed_token_text,
                "forced_first_token_id": suppressor.forced_first_token_id, "forced_first_token_text": suppressor.forced_first_token_text,
                "missing_gold_frames": expand_semantic_counter(missing), "added_frames": expand_semantic_counter(added),
                "forced_outcome": outcome,
                "greedy_full_exact": semantics_full_exact_multiset(greedy_sem, gold),
                "forced_full_exact": semantics_full_exact_multiset(forced_sem, gold),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            greedy_f.write(json.dumps({
                "text_id": text_id,
                "query": row.get("query", ""),
                "semantics": gold,
                "pred_query": greedy_query,
                "pred_semantics": greedy_sem,
            }, ensure_ascii=False) + "\n")
            forced_f.write(json.dumps({
                "text_id": text_id,
                "query": row.get("query", ""),
                "semantics": gold,
                "pred_query": forced_query,
                "pred_semantics": forced_sem,
            }, ensure_ascii=False) + "\n")
            if debug_printed < args.smoke_print:
                print("=" * 20, "STOP DEBUG", text_id)
                print("prefix before STOP:", closure["prefix_before_stop"])
                print("STOP token ID:", closure["stop_token_id"])
                print("STOP token text:", closure["stop_token_text"])
                print("decoded output after STOP:", closure["decoded_after_stop"])
                print("parsed greedy semantics:", json.dumps(greedy_sem, ensure_ascii=False))
                debug_printed += 1
            if rec["count_status"].startswith("under") and under_printed < 5:
                print("=" * 20, "UNDERPREDICTION", text_id)
                print("greedy semantics:", json.dumps(greedy_sem, ensure_ascii=False))
                print("forced semantics:", json.dumps(forced_sem, ensure_ascii=False))
                print("missing gold frames:", json.dumps(rec["missing_gold_frames"], ensure_ascii=False))
                print("added frames:", json.dumps(rec["added_frames"], ensure_ascii=False))
                print("forced outcome:", outcome)
                under_printed += 1
            print(f"[{i}/{len(rows)}] done: {text_id}")
    print(f"[info] saved: {out_path}")
    print(f"[info] saved greedy predictions: {greedy_pred_path}")
    print(f"[info] saved forced predictions: {forced_pred_path}")


if __name__ == "__main__":
    main()
