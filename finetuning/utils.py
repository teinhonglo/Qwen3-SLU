import json
import re
from typing import Any, Dict, List

import torch
from transformers import TrainerCallback, TrainingArguments


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
    raw_text = (raw_text or "").strip()
    m = re.match(r"^language\s+.+?<asr_text>(.*)$", raw_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text


def parse_nlu_payload(raw_text: str) -> Dict[str, Any]:
    payload = extract_payload_text(raw_text)
    obj = {}
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            obj = parsed
    except Exception:
        m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                pass

    query = obj.get("query", "") if isinstance(obj, dict) else ""
    semantics = obj.get("semantics", []) if isinstance(obj, dict) else []
    if not isinstance(query, str):
        query = str(query)
    if not isinstance(semantics, list):
        semantics = []
    return {"query": query, "semantics": semantics}


def normalize_metric_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def normalize_metric_semantics(semantics_list: Any) -> List[Dict[str, Any]]:
    if not isinstance(semantics_list, list):
        return []
    normalized = []
    for item in semantics_list:
        if not isinstance(item, dict):
            continue
        cur = {
            "domain": normalize_metric_text(item.get("domain", "")),
            "intent": normalize_metric_text(item.get("intent", "")),
            "slots": {},
            "implicit_slots": {},
        }
        slots = item.get("slots", {})
        if isinstance(slots, dict):
            cur["slots"] = {normalize_metric_text(k): normalize_metric_text(v) for k, v in slots.items()}
        implicit_slots = item.get("implicit_slots", {})
        if isinstance(implicit_slots, dict):
            cur["implicit_slots"] = {
                normalize_metric_text(k): normalize_metric_text(v) for k, v in implicit_slots.items()
            }
        normalized.append(cur)
    return normalized


def safe_prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_metrics_from_parsed_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    total_count = len(pairs)
    success_count = 0
    overall_match_count = 0
    intent_match_count = 0
    slot_tp = slot_fp = slot_fn = 0
    implicit_tp = implicit_fp = implicit_fn = 0
    slot_match_counts = valid_slots_total = 0

    for pair in pairs:
        pred_sem = normalize_metric_semantics(pair["pred"]["semantics"])
        gt_sem = normalize_metric_semantics(pair["gt"]["semantics"])

        if pred_sem == gt_sem:
            overall_match_count += 1

        pred_intents = sorted((x.get("domain", ""), x.get("intent", "")) for x in pred_sem)
        gt_intents = sorted((x.get("domain", ""), x.get("intent", "")) for x in gt_sem)
        if pred_intents == gt_intents:
            intent_match_count += 1

        pred_slot_set, gt_slot_set = set(), set()
        pred_implicit_set, gt_implicit_set = set(), set()

        for s in pred_sem:
            for k, v in s.get("slots", {}).items():
                pred_slot_set.add((k, v))
            for k, v in s.get("implicit_slots", {}).items():
                pred_implicit_set.add((k, v))
        for s in gt_sem:
            for k, v in s.get("slots", {}).items():
                gt_slot_set.add((k, v))
            for k, v in s.get("implicit_slots", {}).items():
                gt_implicit_set.add((k, v))

        slot_tp += len(pred_slot_set & gt_slot_set)
        slot_fp += len(pred_slot_set - gt_slot_set)
        slot_fn += len(gt_slot_set - pred_slot_set)
        implicit_tp += len(pred_implicit_set & gt_implicit_set)
        implicit_fp += len(pred_implicit_set - gt_implicit_set)
        implicit_fn += len(gt_implicit_set - pred_implicit_set)

        pred_slot_values_ori = []
        for s in pair["pred"]["semantics"]:
            if isinstance(s, dict) and isinstance(s.get("slots", {}), dict):
                pred_slot_values_ori.extend(list(s["slots"].values()))
        gt_semantics_ori_text = str(pair["gt"]["semantics"])
        pred_query_ori = str(pair["pred"]["query"])
        valid_slots = len(pred_slot_values_ori)
        slot_match = 0
        for slot_val in pred_slot_values_ori:
            sv = str(slot_val)
            if sv in pred_query_ori:
                slot_match += 1
            elif sv in gt_semantics_ori_text:
                valid_slots -= 1
        slot_match_counts += slot_match
        valid_slots_total += valid_slots
        success_count += 1

    slot_p, slot_r, slot_f1 = safe_prf(slot_tp, slot_fp, slot_fn)
    implicit_p, implicit_r, implicit_f1 = safe_prf(implicit_tp, implicit_fp, implicit_fn)
    return {
        "total_count": float(total_count),
        "success_count": float(success_count),
        "overall_accuracy": overall_match_count / total_count if total_count else 0.0,
        "intent_accuracy": intent_match_count / total_count if total_count else 0.0,
        "slot_precision": slot_p,
        "slot_recall": slot_r,
        "slot_f1": slot_f1,
        "implicit_slot_precision": implicit_p,
        "implicit_slot_recall": implicit_r,
        "implicit_slot_f1": implicit_f1,
        "slot_match_accs": slot_match_counts / valid_slots_total if valid_slots_total else 0.0,
    }


class OverallEvalMetricsCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, load_audio_fn, sampling_rate: int = 16000, max_new_tokens: int = 256):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.load_audio_fn = load_audio_fn
        self.sampling_rate = sampling_rate
        self.max_new_tokens = max_new_tokens

    def _run_generation_eval(self, model):
        device = next(model.parameters()).device
        model_dtype = getattr(model, "dtype", torch.float16)
        parsed_pairs = []

        model.eval()
        with torch.inference_mode():
            for ex in self.eval_dataset:
                wav = self.load_audio_fn(ex["audio"], sr=self.sampling_rate)
                inputs = self.processor(
                    text=[ex["prefix_text"]],
                    audio=[wav],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                prefix_len = int(inputs["attention_mask"][0].sum().item())
                inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                output_ids = unwrap_generate_output(gen_out)
                if output_ids.dim() == 1:
                    output_ids = output_ids.unsqueeze(0)
                gen_only_ids = output_ids[:, prefix_len:] if output_ids.size(1) > prefix_len else output_ids
                pred_text = batch_decode_text(self.processor, gen_only_ids)[0].strip()

                pred_obj = parse_nlu_payload(pred_text)
                gt_obj = parse_nlu_payload(ex["target"])
                parsed_pairs.append({"pred": pred_obj, "gt": gt_obj})

        return compute_metrics_from_parsed_pairs(parsed_pairs)

    def on_evaluate(self, args: TrainingArguments, state, control, model=None, metrics=None, **kwargs):
        if args.process_index != 0 or model is None:
            return control

        extra_metrics = self._run_generation_eval(model)
        mapped = {f"eval_{k}": v for k, v in extra_metrics.items()}
        if metrics is not None:
            metrics.update(mapped)

        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.log(mapped)
        return control
