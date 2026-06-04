#!/usr/bin/env python3
# Build MAC-SLU domain/intent/slot-key prototypes without training expert LMs.
#
# Prototype source modes (--prototype_source):
#   audio_only   : default. Use the utterance audio only. The audio still goes
#                  through the Qwen3-ASR processor/audio tower/thinker; no row
#                  prompt and no decoded semantic prefix are appended.
#   audio_prompt : use utterance audio plus the row prompt. No decoded semantic
#                  prefix is appended.
#   audio_prefix : use utterance audio plus the row prompt plus the decoded
#                  semantic prefix before the current domain/intent/slot label.
#                  This is closest to state-aware generation-time prototype
#                  lookup.
#   text_prefix  : legacy text-only mode. Use only the decoded semantic prefix;
#                  no audio and no row prompt are used.
#
# All audio modes produce hidden-state prototypes from the Qwen3-ASR thinker
# after audio features are merged; they do not store raw waveform or raw audio
# tower outputs as prototype vectors.
"""Build MAC-SLU domain/intent/slot-key prototypes without training expert LMs."""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slu_decoding.prototypes import (  # noqa: E402
    MACSLULabelSchema,
    SEP,
    AudioStatsPrefixEmbedder,
    TokenEmbeddingPrefixEmbedder,
    dump_json,
    l2_normalize,
    parse_semantics_field,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_id}: {exc}") from exc
    return rows


def _find_after(text: str, needle: str, start: int) -> int:
    idx = text.find(needle, max(0, start))
    if idx >= 0:
        return idx
    # Values inside the nested semantics string may be escaped JSON.
    escaped = json.dumps(needle, ensure_ascii=False)[1:-1]
    return text.find(escaped, max(0, start))


def _prefix_before_value(full_text: str, marker: str, value: str, start: int = 0) -> Tuple[str, int]:
    marker_idx = _find_after(full_text, marker, start)
    search_from = marker_idx if marker_idx >= 0 else start
    value_idx = _find_after(full_text, value, search_from)
    if value_idx < 0:
        return full_text[: max(1, min(len(full_text), search_from))], search_from
    return full_text[:value_idx], value_idx + len(value)


def _prefix_before_slot_key(full_text: str, slot_key: str, start: int = 0) -> Tuple[str, int]:
    candidates = [f'"{slot_key}"', f'\\"{slot_key}\\"', slot_key]
    best = -1
    for cand in candidates:
        idx = full_text.find(cand, max(0, start))
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    if best < 0:
        return full_text[: max(1, min(len(full_text), start))], start
    return full_text[:best], best + len(slot_key)


def iter_prefix_examples(rows: Iterable[Dict[str, Any]], label_schema: MACSLULabelSchema):
    for row in rows:
        full_text = row.get("text", "") or ""
        if not full_text:
            continue
        frames = parse_semantics_field(row.get("semantics", []))
        cursor = 0
        for frame_idx, frame in enumerate(frames):
            domain = str(frame.get("domain", "") or "")
            intent = str(frame.get("intent", "") or "")
            if not domain or not intent:
                continue
            label_schema.add_domain_intent(domain, intent)

            prefix, cursor = _prefix_before_value(full_text, "domain", domain, cursor)
            yield {
                "kind": "domain",
                "key": domain,
                "label": domain,
                "prefix": prefix,
                "audio": row.get("audio", ""),
                "prompt": row.get("prompt", ""),
                "text_id": row.get("text_id", ""),
                "meta": {"label": domain, "domain": domain, "frame_index": frame_idx},
            }

            prefix, cursor = _prefix_before_value(full_text, "intent", intent, cursor)
            yield {
                "kind": "intent",
                "key": f"{domain}{SEP}{intent}",
                "label": intent,
                "prefix": prefix,
                "audio": row.get("audio", ""),
                "prompt": row.get("prompt", ""),
                "text_id": row.get("text_id", ""),
                "meta": {"label": intent, "domain": domain, "intent": intent, "frame_index": frame_idx},
            }

            slots = frame.get("slots", {}) or {}
            if not isinstance(slots, dict):
                slots = {}
            for slot_key in slots.keys():
                slot_key = str(slot_key)
                label_schema.add_slot_key(domain, intent, slot_key)
                prefix, cursor = _prefix_before_slot_key(full_text, slot_key, cursor)
                yield {
                    "kind": "slot_key",
                    "key": f"{domain}{SEP}{intent}{SEP}{slot_key}",
                    "label": slot_key,
                    "prefix": prefix,
                    "audio": row.get("audio", ""),
                    "prompt": row.get("prompt", ""),
                    "text_id": row.get("text_id", ""),
                    "meta": {"label": slot_key, "domain": domain, "intent": intent, "slot_key": slot_key, "frame_index": frame_idx},
                }


def resolve_model(args):
    import torch
    from peft.peft_model import PeftModelForCausalLM
    from qwen_asr import Qwen3ASRModel
    from finetuning.qwen3_asr_test import find_latest_checkpoint, load_train_conf_from_exp_dir, resolve_dtype

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    _, model_args_conf = train_conf
    model_path = args.exp_dir
    if args.auto_best_checkpoint:
        model_path = os.path.join(model_path, "checkpoint-best")
    elif args.auto_latest_checkpoint:
        ckpt = find_latest_checkpoint(model_path)
        if ckpt is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = ckpt
    dtype = resolve_dtype(str(model_args_conf.get("dtype", "auto")), args.device)
    if model_args_conf.get("lora_config", None):
        wrapper = Qwen3ASRModel.from_pretrained(model_args_conf["model_path"], dtype=dtype, device_map=args.device)
        wrapper.model = PeftModelForCausalLM.from_pretrained(wrapper.model, model_path, torch_dtype=torch.bfloat16)
    else:
        wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=args.device)
    wrapper.model.eval()
    return wrapper


def _example_record(ex: Dict[str, Any], vec: List[float], split: str) -> Dict[str, Any]:
    meta = dict(ex.get("meta", {}) or {})
    return {
        "split": split,
        "kind": ex.get("kind", ""),
        "key": ex.get("key", ""),
        "label": ex.get("label", meta.get("label", "")),
        "domain": meta.get("domain", ""),
        "intent": meta.get("intent", ""),
        "slot_key": meta.get("slot_key", ""),
        "text_id": ex.get("text_id", ""),
        "frame_index": meta.get("frame_index", -1),
        "audio": ex.get("audio", ""),
        "prefix_tail": (ex.get("prefix", "") or "")[-240:],
        "vector": vec,
    }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"[info] saved {count} prototype instance examples: {path}")


class PrototypeSourceEmbedder:
    """Route prototype examples to the hidden-state embedder by source mode.

    Source modes:
    - audio_only: audio chat prefix only; ignores row prompt and decoded prefix.
    - audio_prompt: audio chat prefix with the row prompt; ignores decoded prefix.
    - audio_prefix: audio chat prefix with row prompt plus decoded semantic prefix.
    - text_prefix: legacy text-only decoded semantic prefix.
    """

    def __init__(self, embedder, source: str):
        self.embedder = embedder
        self.source = source

    def __call__(self, text: str, audio_path: str = "", prompt: str = ""):
        if self.source == "audio_only":
            return self.embedder("", audio_path=audio_path, prompt="")
        if self.source == "audio_prompt":
            return self.embedder("", audio_path=audio_path, prompt=prompt)
        if self.source == "audio_prefix":
            return self.embedder(text, audio_path=audio_path, prompt=prompt)
        if self.source == "text_prefix":
            return self.embedder(text)
        raise ValueError(f"Unsupported prototype_source: {self.source}")


def embed_instance_examples(examples, embedder, split: str, max_examples_per_label: int = 0):
    counts: Dict[str, int] = defaultdict(int)
    rows = []
    for n, ex in enumerate(examples, 1):
        key = f"{ex.get('kind', '')}::{ex.get('key', '')}"
        if max_examples_per_label > 0 and counts[key] >= max_examples_per_label:
            continue
        vec = embedder(ex.get("prefix", ""), audio_path=ex.get("audio", ""), prompt=ex.get("prompt", ""))
        if not vec:
            continue
        rows.append(_example_record(ex, vec, split))
        counts[key] += 1
        if n % 500 == 0:
            print(f"[info] embedded {split} instance {n} prefix examples", flush=True)
    return rows


def sample_embedded_examples(rows: Iterable[Dict[str, Any]], max_examples_per_label: int = 0) -> List[Dict[str, Any]]:
    if max_examples_per_label <= 0:
        return list(rows)
    counts: Dict[str, int] = defaultdict(int)
    sampled = []
    for row in rows:
        key = f"{row.get('kind', '')}::{row.get('key', '')}"
        if counts[key] >= max_examples_per_label:
            continue
        sampled.append(row)
        counts[key] += 1
    return sampled

def aggregate(embedded_examples, max_examples_per_label: int):
    sums: Dict[str, Dict[str, List[float]]] = {"domain": {}, "intent": {}, "slot_key": {}}
    counts: Dict[str, Dict[str, int]] = {"domain": defaultdict(int), "intent": defaultdict(int), "slot_key": defaultdict(int)}
    metas: Dict[str, Dict[str, Dict[str, Any]]] = {"domain": {}, "intent": {}, "slot_key": {}}

    for ex in embedded_examples:
        kind = ex["kind"]
        key = ex["key"]
        if kind not in sums:
            continue
        if max_examples_per_label > 0 and counts[kind][key] >= max_examples_per_label:
            continue
        vec = ex.get("vector", [])
        if not vec:
            continue
        if key not in sums[kind]:
            sums[kind][key] = [0.0] * len(vec)
            metas[kind][key] = {
                "label": ex.get("label", ""),
                "domain": ex.get("domain", ""),
                "intent": ex.get("intent", ""),
                "slot_key": ex.get("slot_key", ""),
            }
        for i, val in enumerate(vec):
            sums[kind][key][i] += float(val)
        counts[kind][key] += 1

    out = {"domain": {}, "intent": {}, "slot_key": {}}
    for kind in out:
        for key, vec_sum in sums[kind].items():
            count = counts[kind][key]
            if count <= 0:
                continue
            mean = [x / count for x in vec_sum]
            out[kind][key] = {"vector": l2_normalize(mean), "count": count, "meta": metas[kind][key]}
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--test_jsonl", default="")
    p.add_argument("--labels_path", default="data/macslu/labels.txt")
    p.add_argument("--schema_path", default="")
    p.add_argument("--output_json", required=True)
    p.add_argument("--train_examples_jsonl", default="")
    p.add_argument("--test_examples_jsonl", default="")
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--auto_latest_checkpoint", action="store_true")
    p.add_argument("--auto_best_checkpoint", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--prototype_source",
        choices=["audio_only", "audio_prompt", "audio_prefix", "text_prefix"],
        default="audio_only",
        help="Prototype input source. Default audio_only uses only audio; audio_prompt adds the row prompt; audio_prefix adds prompt plus decoded prefix; text_prefix is legacy text-only.",
    )
    p.add_argument("--prototype_pooling", choices=["mean_pooling", "last_hidden_state"], default="mean_pooling")
    p.add_argument("--max_examples_per_label", type=int, default=0, help="Deprecated compatibility option; prototypes are aggregated from all train examples")
    p.add_argument("--max_instance_examples_per_label", type=int, default=200, help="Max train/test instance embeddings per label saved for visualization")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.train_jsonl)
    test_rows = load_jsonl(args.test_jsonl) if args.test_jsonl else []
    label_schema = MACSLULabelSchema(labels_path=args.labels_path, schema_path=args.schema_path)
    from finetuning.qwen3_asr_test import load_train_conf_from_exp_dir

    _, model_args_conf = load_train_conf_from_exp_dir(args.exp_dir)
    wrapper = resolve_model(args)
    tok = wrapper.processor.tokenizer if hasattr(wrapper.processor, "tokenizer") else wrapper.processor
    
    text_embedder = TokenEmbeddingPrefixEmbedder(
        tok,
        wrapper.model,
        processor=wrapper.processor,
        device=args.device,
        pooling=args.prototype_pooling,
        sample_rate=int(model_args_conf.get("sr", 16000)),
    )

    audio_embedder = AudioStatsPrefixEmbedder(text_embedder, sample_rate=int(model_args_conf.get("sr", 16000)))
    embedder = PrototypeSourceEmbedder(
        audio_embedder if args.prototype_source in {"audio_only", "audio_prompt", "audio_prefix"} else text_embedder,
        args.prototype_source,
    )
    examples = list(iter_prefix_examples(rows, label_schema))
    test_examples = list(iter_prefix_examples(test_rows, label_schema)) if test_rows else []
    print(f"[info] collected {len(examples)} train prefix examples")
    if test_rows:
        print(f"[info] collected {len(test_examples)} test prefix examples")
    
    train_instance_rows = embed_instance_examples(examples, embedder, "train", 0)
    test_instance_rows = embed_instance_examples(test_examples, embedder, "test", 0) if test_examples else []
    sections = aggregate(train_instance_rows, 0)
    write_jsonl(args.train_examples_jsonl, sample_embedded_examples(train_instance_rows, args.max_instance_examples_per_label))
    write_jsonl(args.test_examples_jsonl, sample_embedded_examples(test_instance_rows, args.max_instance_examples_per_label))
    obj = {
        "prototype_source": args.prototype_source,
        "prototype_pooling": args.prototype_pooling,
        "embedding_backend": "hidden_state",
        "label_schema": label_schema.to_dict(),
        **sections,
    }
    dump_json(args.output_json, obj)
    print(f"[info] saved prototypes: {args.output_json}")
    print(
        "[info] prototype counts: domain={d}, intent={i}, slot_key={s}".format(
            d=len(sections["domain"]), i=len(sections["intent"]), s=len(sections["slot_key"])
        )
    )


if __name__ == "__main__":
    main()
