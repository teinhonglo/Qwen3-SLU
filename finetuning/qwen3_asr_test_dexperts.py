#!/usr/bin/env python3
"""Opt-in DExperts-aware inference entrypoint for MAC-SLU."""
import argparse
import json
import os
import sys
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser('Qwen3-ASR SLU test + optional DExperts')
    p.add_argument('--exp_dir', type=str, required=True)
    p.add_argument('--auto_latest_checkpoint', action='store_true')
    p.add_argument('--auto_best_checkpoint', action='store_true')
    p.add_argument('--input_jsonl', type=str, required=True)
    p.add_argument('--output_root', type=str, default='checkpoints')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--output_jsonl_dir', type=str, default='')
    p.add_argument('--corrected_prompt_file', type=str, default='data/macslu/corrected_prompt.txt')
    p.add_argument('--use_dexperts', action='store_true')
    p.add_argument('--dexperts_config', type=str, default='')
    p.add_argument('--schema_path', type=str, default='')
    p.add_argument('--domain_intent_expert_path', type=str, default='')
    p.add_argument('--slot_key_expert_path', type=str, default='')
    p.add_argument('--slot_grounding_mode', type=str, default='copy_bias')
    p.add_argument('--disable_schema_mask', action='store_true')
    p.add_argument('--disable_grounding', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    import librosa
    import torch
    from transformers import LogitsProcessorList
    from qwen_asr import Qwen3ASRModel
    from peft.peft_model import PeftModelForCausalLM
    from qwen3_asr_test import (
        load_train_conf_from_exp_dir, resolve_dtype, get_jsonl_name, find_latest_checkpoint,
        load_jsonl, move_inputs_to_device, unwrap_generate_output, batch_decode_text,
        write_slu_prediction_jsonl, write_corrected_jsonl, load_corrected_prompt, try_parse_score_dict,
    )
    from slu_decoding.config import load_dexperts_config
    from slu_decoding.schema import SLUSchema
    from slu_decoding.experts import ExpertLM
    from slu_decoding.logits_processors import StateAwareDExpertsLogitsProcessor

    def load_audio(path: str, sr: int = 16000):
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav

    def build_prefix_text(processor, prompt: str) -> str:
        txt = processor.apply_chat_template([[{"role": "system", "content": prompt or ""}, {"role": "user", "content": [{"type": "audio", "audio": None}]}]], add_generation_prompt=True, tokenize=False)
        if isinstance(txt, list):
            return txt[0]
        return txt

    def infer_one(asr_wrapper, audio_path, prompt, sr, max_new_tokens, do_sample, temperature, top_p, logits_processor=None):
        processor = asr_wrapper.processor
        model = asr_wrapper.model
        device = next(model.parameters()).device
        model_dtype = getattr(model, 'dtype', torch.float16)
        wav = load_audio(audio_path, sr=sr)
        prefix_text = build_prefix_text(processor, prompt)
        inputs = processor(text=[prefix_text], audio=[wav], return_tensors='pt', padding=True, truncation=False)
        prefix_len = int(inputs['attention_mask'][0].sum().item())
        inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}

        if logits_processor is not None and hasattr(logits_processor, 'base_prefix_len'):
            logits_processor.base_prefix_len = prefix_len

        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})
        if logits_processor is not None:
            gen_kwargs['logits_processor'] = LogitsProcessorList([logits_processor])

        with torch.inference_mode():
            gen_out = model.generate(**inputs, **gen_kwargs)

        output_ids = unwrap_generate_output(gen_out)
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)
        gen_only_ids = output_ids[:, prefix_len:] if output_ids.size(1) > prefix_len else output_ids

        return batch_decode_text(processor, gen_only_ids)[0].strip()

    cfg = load_dexperts_config(args.dexperts_config) if args.dexperts_config else {}
    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    
    if train_conf is None:
        raise ValueError("Unable to load train_conf from exp_dir")

    training_args_conf, model_args_conf = train_conf
    sr = int(model_args_conf.get('sr', 16000))
    max_new_tokens = int(model_args_conf.get('max_new_tokens', 256))
    do_sample = bool(model_args_conf.get('do_sample', False))
    temperature = float(model_args_conf.get('temperature', 0.0))
    top_p = float(model_args_conf.get('top_p', 1.0))

    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, 'checkpoint-best')
    elif args.auto_latest_checkpoint:
        ck = find_latest_checkpoint(model_path)
        if ck is None:
            raise ValueError(f'No checkpoint-* found under: {model_path}')
        model_path = ck
    dtype = resolve_dtype(str(model_args_conf.get('dtype', 'auto')), args.device)
    base_path = model_args_conf['model_path']
    if model_args_conf.get('lora_config', None):
        lora_path = model_path
        asr_wrapper = Qwen3ASRModel.from_pretrained(base_path, dtype=dtype, device_map=args.device)
        asr_wrapper.model = PeftModelForCausalLM.from_pretrained(asr_wrapper.model, lora_path)
    else:
        asr_wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=args.device)

    schema_path = args.schema_path or cfg.get('schema_path', '')
    schema = SLUSchema(schema_path) if (args.use_dexperts and schema_path and os.path.isfile(schema_path)) else None
    if args.use_dexperts and schema is None:
        warnings.warn('schema missing; continue without schema mask')

    exp_cfg = cfg.get('experts', {}) if isinstance(cfg, dict) else {}
    di_path = args.domain_intent_expert_path or exp_cfg.get('domain_intent', {}).get('path', '')
    sk_path = args.slot_key_expert_path or exp_cfg.get('slot_key', {}).get('path', '')
    di_alpha = float(exp_cfg.get('domain_intent', {}).get('alpha', 1.0)); sk_alpha = float(exp_cfg.get('slot_key', {}).get('alpha', 1.0))
    gr = float(cfg.get('grounding', {}).get('strength', 1.0)) if isinstance(cfg, dict) else 1.0
    di_expert = ExpertLM(path=di_path, device=args.device) if args.use_dexperts else None
    sk_expert = ExpertLM(path=sk_path, device=args.device) if args.use_dexperts else None
    tok = asr_wrapper.processor.tokenizer if hasattr(asr_wrapper.processor, 'tokenizer') else asr_wrapper.processor

    if args.use_dexperts:
        print(f"[DExperts] domain_intent expert path: {di_path or '<empty>'}")
        print(f"[DExperts] slot_key expert path: {sk_path or '<empty>'}")
        print(f"[DExperts] schema path: {schema_path or '<empty>'}")
        print(f"[DExperts] domain_intent loaded: {bool(di_expert and di_expert.model is not None)}")
        print(f"[DExperts] slot_key loaded: {bool(sk_expert and sk_expert.model is not None)}")
        try:
            base_vocab = int(getattr(tok, "vocab_size", 0) or 0)
            di_vocab = int(getattr(getattr(di_expert, "tokenizer", None), "vocab_size", 0) or 0)
            sk_vocab = int(getattr(getattr(sk_expert, "tokenizer", None), "vocab_size", 0) or 0)
            if di_expert and di_expert.model is not None and di_vocab != base_vocab:
                warnings.warn(
                    f"domain_intent vocab mismatch: base={base_vocab}, expert={di_vocab}. "
                    "Expert logits may be skipped due to shape mismatch."
                )
            if sk_expert and sk_expert.model is not None and sk_vocab != base_vocab:
                warnings.warn(
                    f"slot_key vocab mismatch: base={base_vocab}, expert={sk_vocab}. "
                    "Expert logits may be skipped due to shape mismatch."
                )
        except Exception as exc:
            warnings.warn(f"Unable to validate tokenizer vocab alignment: {exc}")

    logits_processor = StateAwareDExpertsLogitsProcessor(tok, schema=schema, domain_intent_expert=di_expert, slot_key_expert=sk_expert, alpha_domain_intent=di_alpha, alpha_slot_key=sk_alpha, grounding_strength=gr, enable_schema_mask=not args.disable_schema_mask, enable_grounding=not args.disable_grounding) if args.use_dexperts else None

    rows = load_jsonl(args.input_jsonl)
    rows_out = []
    jsonl_name = get_jsonl_name(args.input_jsonl)
    for i, row in enumerate(rows, 1):
        pred_raw = infer_one(asr_wrapper, row.get('audio', ''), row.get('prompt', ''), sr, max_new_tokens, do_sample, temperature, top_p, logits_processor=logits_processor)
        pred_json = try_parse_score_dict(pred_raw)
        pred_query = pred_json.get('asr_text', 'FAILED')
        try:
            pred_semantics = json.loads(pred_json['semantics'])
        except Exception:
            print(f"[WARNING] Processing failed for {row.get('text_id', f'line{i}')}: {pred_json}")
            pred_semantics = [{'FAILED': pred_json}]
        rows_out.append({'text_id': str(row.get('text_id', f'line{i}')), 'query': row.get('query', ''), 'audio': row.get('audio', ''), 'text': row.get('text', ''), 'semantics': row.get('semantics', []), 'pred_json': pred_json, 'pred_query': pred_query, 'pred_raw': pred_raw, 'pred_semantics': pred_semantics})

        if args.use_dexperts and logits_processor is not None and hasattr(logits_processor, "get_debug_stats"):
            dbg = logits_processor.get_debug_stats()
            print("[DExperts] decode summary:")
            print(
                "[DExperts] steps={steps}, state_domain={state_domain}, state_intent={state_intent}, "
                "state_slots_key={state_slots_key}, state_slots_value={state_slots_value}".format(**dbg)
            )
            print(
                "[DExperts] di_applied={di_applied}, di_skipped_shape={di_skipped_shape}, "
                "sk_applied={sk_applied}, sk_skipped_shape={sk_skipped_shape}, changed_max={changed_max}".format(**dbg)
            )


    write_slu_prediction_jsonl(rows_out, args.output_root, jsonl_name)
    if args.output_jsonl_dir:
        write_corrected_jsonl(rows_out, args.output_jsonl_dir, load_corrected_prompt(args.corrected_prompt_file), jsonl_name)


if __name__ == '__main__':
    main()
