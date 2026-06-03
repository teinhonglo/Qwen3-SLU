#!/usr/bin/env python3
"""Prototype-guided second-pass inference for MAC-SLU."""

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SLU test + prototype tracker")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--auto_latest_checkpoint", action="store_true")
    p.add_argument("--auto_best_checkpoint", action="store_true")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_root", default="checkpoints")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decoding_conf", default="conf/decoding/basic_decoding.json")
    p.add_argument("--prototype_json", required=True)
    p.add_argument("--labels_path", default="data/macslu/labels.txt")
    p.add_argument("--schema_path", default="")
    p.add_argument("--prototype_top_k", type=int, default=5)
    p.add_argument("--domain_threshold", type=float, default=0.35)
    p.add_argument("--intent_threshold", type=float, default=0.35)
    p.add_argument("--slot_key_threshold", type=float, default=0.35)
    p.add_argument("--replacement_margin", type=float, default=0.05)
    p.add_argument("--disable_replacement", action="store_true")
    return p.parse_args()


def _load_asr(args, model_args_conf, dtype):
    import torch
    from peft.peft_model import PeftModelForCausalLM
    from qwen_asr import Qwen3ASRModel
    from qwen3_asr_test import find_latest_checkpoint

    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, "checkpoint-best")
    elif args.auto_latest_checkpoint:
        ck = find_latest_checkpoint(model_path)
        if ck is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = ck
    print(f"[info] use checkpoint: {model_path}")
    if model_args_conf.get("lora_config", None):
        wrapper = Qwen3ASRModel.from_pretrained(
            model_args_conf["model_path"], dtype=dtype, device_map=args.device, attn_implementation="flash_attention_2"
        )
        wrapper.model = PeftModelForCausalLM.from_pretrained(wrapper.model, model_path, torch_dtype=torch.bfloat16)
    else:
        wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=args.device)
    wrapper.model.eval()
    return wrapper


def _compress_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"domain": [], "intent": [], "slot_key": []}
    last_kind = None
    current = None
    for rec in records:
        kind = rec.get("kind")
        if kind not in grouped:
            continue
        top = rec.get("top") or []
        if not top:
            continue
        if kind != last_kind:
            current = copy.deepcopy(rec)
            grouped[kind].append(current)
            last_kind = kind
            continue
        # Keep the strongest record from a consecutive state span.
        if current is not None and top[0].get("score", -999) > (current.get("top") or [{}])[0].get("score", -999):
            current.update(copy.deepcopy(rec))
    return grouped


def _hits_from_record(rec: Dict[str, Any]):
    from slu_decoding.prototypes import PrototypeHit

    hits = []
    for item in rec.get("top", []) or []:
        hits.append(PrototypeHit(label=item.get("label", ""), score=float(item.get("score", 0.0)), count=int(item.get("count", 0) or 0), meta=item.get("meta", {}) or {}))
    return hits


def apply_replacements(pred_semantics, records, label_schema, thresholds):
    from slu_decoding.prototypes import choose_replacement

    frames = copy.deepcopy(pred_semantics) if isinstance(pred_semantics, list) else []
    grouped = _compress_records(records)
    trace = {"events": grouped, "changes": []}
    slot_event_idx = 0
    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        domain_rec = grouped["domain"][frame_idx] if frame_idx < len(grouped["domain"]) else None
        if domain_rec:
            old = str(frame.get("domain", "") or "")
            new, reason = choose_replacement(
                old,
                _hits_from_record(domain_rec),
                thresholds["domain"],
                thresholds["margin"],
                label_schema.is_valid_domain(old),
            )
            if new != old:
                frame["domain"] = new
            trace["changes"].append({"frame": frame_idx, "field": "domain", "old": old, "new": new, "reason": reason})

        intent_rec = grouped["intent"][frame_idx] if frame_idx < len(grouped["intent"]) else None
        if intent_rec:
            domain = str(frame.get("domain", "") or "")
            old = str(frame.get("intent", "") or "")
            new, reason = choose_replacement(
                old,
                _hits_from_record(intent_rec),
                thresholds["intent"],
                thresholds["margin"],
                label_schema.is_valid_intent(domain, old),
            )
            if new != old:
                frame["intent"] = new
            trace["changes"].append({"frame": frame_idx, "field": "intent", "old": old, "new": new, "reason": reason})

        slots = frame.get("slots", {}) or {}
        if isinstance(slots, dict):
            new_slots = {}
            for old_key, value in slots.items():
                slot_rec = grouped["slot_key"][slot_event_idx] if slot_event_idx < len(grouped["slot_key"]) else None
                slot_event_idx += 1
                new_key = old_key
                reason = "no_record"
                if slot_rec:
                    domain = str(frame.get("domain", "") or "")
                    intent = str(frame.get("intent", "") or "")
                    new_key, reason = choose_replacement(
                        str(old_key),
                        _hits_from_record(slot_rec),
                        thresholds["slot_key"],
                        thresholds["margin"],
                        label_schema.is_valid_slot_key(domain, intent, str(old_key)),
                    )
                new_slots[new_key] = value
                trace["changes"].append({"frame": frame_idx, "field": "slot_key", "old": old_key, "new": new_key, "reason": reason})
            frame["slots"] = new_slots
    return frames, trace


def main():
    args = parse_args()

    import librosa
    import torch
    from transformers import LogitsProcessorList
    from qwen3_asr_test import (
        batch_decode_text,
        build_output_subdir_name,
        build_prefix_text,
        load_decoding_conf,
        load_jsonl,
        load_train_conf_from_exp_dir,
        move_inputs_to_device,
        resolve_decoding_conf,
        resolve_dtype,
        try_parse_score_dict,
        unwrap_generate_output,
        validate_decoding_mode,
        write_slu_prediction_jsonl,
        save_resolved_decoding_conf,
    )
    from slu_decoding.logits_processors import StateAwarePrototypeTrackerLogitsProcessor
    from slu_decoding.prototypes import MACSLULabelSchema, PrototypeIndex, AudioStatsPrefixEmbedder, TokenEmbeddingPrefixEmbedder, parse_semantics_field

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    _, model_args_conf = train_conf
    sr = int(model_args_conf.get("sr", 16000))
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    decoding_conf = load_decoding_conf(args.decoding_conf)
    resolved_decoding = resolve_decoding_conf(model_args_conf, decoding_conf)
    effective_mode = validate_decoding_mode(resolved_decoding)
    if effective_mode != "basic":
        print(f"[warning] prototype script currently uses basic generate kwargs; requested mode={effective_mode}")
    gen_cfg = resolved_decoding["generation"]
    max_new_tokens = int(gen_cfg["max_new_tokens"])
    do_sample = bool(gen_cfg["do_sample"])
    temperature = float(gen_cfg["temperature"])
    top_p = float(gen_cfg["top_p"])
    top_k = int(gen_cfg.get("top_k", 0))
    repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.0))

    asr_wrapper = _load_asr(args, model_args_conf, dtype)
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    prototype_index = PrototypeIndex.load(args.prototype_json)
    label_schema = MACSLULabelSchema(labels_path=args.labels_path, schema_path=args.schema_path)
    proto_schema = prototype_index.data.get("label_schema", {}) or {}
    # Merge prototype schema as a JSON-like temporary schema.
    for d in proto_schema.get("domains", []) or []:
        label_schema.domains.add(str(d))
    for d, intents in (proto_schema.get("domain2intents", {}) or {}).items():
        for intent in intents or []:
            label_schema.add_domain_intent(str(d), str(intent))
    for di, slots in (proto_schema.get("domain_intent2slot_keys", {}) or {}).items():
        parts = str(di).split("|||")
        if len(parts) >= 2:
            for slot in slots or []:
                label_schema.add_slot_key(parts[0], parts[1], str(slot))

    text_embedder = TokenEmbeddingPrefixEmbedder(
        tok,
        model,
        processor=processor,
        device=args.device,
        pooling=prototype_index.data.get("prototype_pooling", "mean_pooling"),
        sample_rate=sr,
    )
    current_audio = {"path": "", "prompt": ""}
    if prototype_index.prototype_source == "audio_prefix":
        embedder = AudioStatsPrefixEmbedder(text_embedder, sample_rate=sr)
        embed_fn = lambda text: embedder(text, audio_path=current_audio["path"], prompt=current_audio["prompt"])
    else:
        embedder = text_embedder
        embed_fn = lambda text: embedder(text)
    logits_processor = StateAwarePrototypeTrackerLogitsProcessor(
        tok,
        prototype_index=prototype_index,
        embed_text_fn=embed_fn,
        label_schema=label_schema,
        top_k=args.prototype_top_k,
    )

    def load_audio(path: str):
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav

    def infer_one(row):
        current_audio["path"] = row.get("audio", "")
        current_audio["prompt"] = row.get("prompt", "")
        wav = load_audio(row.get("audio", ""))
        prefix_text = build_prefix_text(processor, row.get("prompt", ""))
        inputs = processor(text=[prefix_text], audio=[wav], return_tensors="pt", padding=True, truncation=False)
        prefix_len = int(inputs["attention_mask"][0].sum().item())
        inputs = move_inputs_to_device(inputs, device=args.device, model_dtype=getattr(model, "dtype", torch.float16))
        logits_processor.base_prefix_len = prefix_len
        logits_processor.reset()
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "repetition_penalty": repetition_penalty, "logits_processor": LogitsProcessorList([logits_processor])}
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        with torch.inference_mode():
            gen_out = model.generate(**inputs, **gen_kwargs)
        output_ids = unwrap_generate_output(gen_out)
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)
        gen_only = output_ids[:, prefix_len:] if output_ids.size(1) > prefix_len else output_ids
        return batch_decode_text(processor, gen_only)[0].strip(), logits_processor.get_records(), logits_processor.get_debug_stats()

    rows = load_jsonl(args.input_jsonl)
    rows_out = []
    trace_rows = []
    thresholds = {"domain": args.domain_threshold, "intent": args.intent_threshold, "slot_key": args.slot_key_threshold, "margin": args.replacement_margin}
    for i, row in enumerate(rows, 1):
        text_id = str(row.get("text_id", f"line{i}"))
        pred_raw, records, dbg = infer_one(row)
        pred_json = try_parse_score_dict(pred_raw)
        pred_query = pred_json.get("asr_text", "FAILED")
        pred_semantics = parse_semantics_field(pred_json.get("semantics", []))
        trace = {"events": [], "changes": []}
        if not args.disable_replacement:
            pred_semantics, trace = apply_replacements(pred_semantics, records, label_schema, thresholds)
            pred_json["semantics"] = json.dumps(pred_semantics, ensure_ascii=False)
        rows_out.append({"text_id": text_id, "query": row.get("query", ""), "semantics": row.get("semantics", []), "pred_query": pred_query, "pred_semantics": pred_semantics})
        trace_rows.append({"text_id": text_id, "pred_raw": pred_raw, "pred_json_after": pred_json, "debug_stats": dbg, "prototype_records": records, "replacement_trace": trace})
        print(f"[{i}/{len(rows)}] done: {text_id}")

    jsonl_name = build_output_subdir_name(args.input_jsonl, effective_mode, args.decoding_conf)
    resolved_decoding["effective_mode"] = effective_mode
    save_resolved_decoding_conf(resolved_decoding, args.output_root, jsonl_name)
    write_slu_prediction_jsonl(rows_out, args.output_root, jsonl_name)
    save_dir = os.path.join(args.output_root, jsonl_name)
    trace_path = os.path.join(save_dir, "prototype_trace.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for item in trace_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[info] saved prototype trace: {trace_path}")


if __name__ == "__main__":
    main()
