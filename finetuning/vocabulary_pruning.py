"""Structural vocabulary pruning for Qwen3-ASR embeddings and LM heads.

The processor keeps the original tokenizer so BPE tokenization remains valid.
Immediately after tokenization, original token IDs are mapped to compact IDs;
generated compact IDs are mapped back before decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch import nn


@dataclass(frozen=True)
class VocabularyMapping:
    retained_token_ids: tuple[int, ...]
    old_to_new: Dict[int, int]
    new_to_old: Dict[int, int]


def load_vocabulary_mapping(manifest_path: str) -> VocabularyMapping:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    retained = tuple(sorted({int(token_id) for token_id in payload["retained_token_ids"]}))
    if not retained:
        raise ValueError(f"Vocabulary manifest has no retained token IDs: {manifest_path}")
    old_to_new = {old: new for new, old in enumerate(retained)}
    return VocabularyMapping(retained, old_to_new, {new: old for old, new in old_to_new.items()})


def _thinker(model):
    current = model
    for _ in range(8):
        if hasattr(current, "thinker"):
            return current.thinker
        if hasattr(current, "base_model"):
            current = current.base_model
        elif hasattr(current, "model") and current.model is not current:
            current = current.model
        else:
            break
    raise RuntimeError(f"Cannot locate Qwen3-ASR thinker under {type(model).__name__}")


def _remap_special_ids(config, mapping: VocabularyMapping):
    names = {"bos_token_id", "eos_token_id", "pad_token_id"}
    names.update(name for name in vars(config) if name.endswith("_token_id"))
    for name in names:
        value = getattr(config, name, None)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        remapped = [mapping.old_to_new[int(item)] for item in values if int(item) in mapping.old_to_new]
        if remapped:
            setattr(config, name, remapped if isinstance(value, list) else remapped[0])


@torch.no_grad()
def apply_structural_vocabulary_pruning(model, mapping: VocabularyMapping):
    """Shrink input embeddings and output head to the retained vocabulary rows."""
    thinker = _thinker(model)
    embedding = thinker.model.embed_tokens
    lm_head = thinker.lm_head
    weights_were_tied = embedding.weight.data_ptr() == lm_head.weight.data_ptr()
    target_size = len(mapping.retained_token_ids)
    if embedding.num_embeddings == target_size and lm_head.out_features == target_size:
        model._macslu_vocabulary_mapping = mapping
        return model
    original_size = embedding.num_embeddings
    if mapping.retained_token_ids[-1] >= original_size:
        raise ValueError(
            f"Vocabulary manifest token {mapping.retained_token_ids[-1]} exceeds embedding size {original_size}"
        )
    rows = torch.tensor(mapping.retained_token_ids, device=embedding.weight.device)
    new_embedding = nn.Embedding(
        target_size, embedding.embedding_dim,
        padding_idx=mapping.old_to_new.get(embedding.padding_idx),
        device=embedding.weight.device, dtype=embedding.weight.dtype,
    )
    new_embedding.weight.copy_(embedding.weight.index_select(0, rows))
    new_head = nn.Linear(
        lm_head.in_features, target_size, bias=lm_head.bias is not None,
        device=lm_head.weight.device, dtype=lm_head.weight.dtype,
    )
    new_head.weight.copy_(lm_head.weight.index_select(0, rows.to(lm_head.weight.device)))
    if lm_head.bias is not None:
        new_head.bias.copy_(lm_head.bias.index_select(0, rows.to(lm_head.bias.device)))
    thinker.model.embed_tokens = new_embedding
    if weights_were_tied:
        new_head.weight = new_embedding.weight
    thinker.lm_head = new_head
    thinker.config.text_config.vocab_size = target_size
    if hasattr(thinker.config, "vocab_size"):
        thinker.config.vocab_size = target_size
    _remap_special_ids(thinker.config, mapping)
    _remap_special_ids(thinker.config.text_config, mapping)
    outer_config = getattr(model, "config", None)
    if outer_config is not None:
        _remap_special_ids(outer_config, mapping)
        thinker_config = getattr(outer_config, "thinker_config", None)
        if thinker_config is not None:
            thinker_config.text_config.vocab_size = target_size
            _remap_special_ids(thinker_config, mapping)
            _remap_special_ids(thinker_config.text_config, mapping)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _remap_special_ids(generation_config, mapping)
    model._macslu_vocabulary_mapping = mapping
    model._macslu_original_vocabulary_size = original_size
    return model


def remap_token_tensor(token_ids: torch.Tensor, mapping: VocabularyMapping,
                       ignore_index: int | None = None) -> torch.Tensor:
    result = torch.full_like(token_ids, ignore_index if ignore_index is not None else -1)
    for old, new in mapping.old_to_new.items():
        result[token_ids == old] = new
    invalid = result.eq(-1) if ignore_index is None else (result.eq(ignore_index) & token_ids.ne(ignore_index))
    if invalid.any():
        missing = torch.unique(token_ids[invalid]).tolist()
        raise ValueError(f"Token IDs are absent from pruned vocabulary: {missing[:20]}")
    return result


def restore_token_tensor(token_ids: torch.Tensor, mapping: VocabularyMapping) -> torch.Tensor:
    result = torch.full_like(token_ids, -1)
    for new, old in mapping.new_to_old.items():
        result[token_ids == new] = old
    if result.eq(-1).any():
        raise ValueError(f"Generated compact token IDs exceed pruned vocabulary: {torch.unique(token_ids[result.eq(-1)]).tolist()}")
    return result


class VocabularyRemappingCollator:
    """Wrap an existing collator and compact its input IDs and labels."""
    def __init__(self, collator, mapping: VocabularyMapping):
        self.collator = collator
        self.mapping = mapping

    def __call__(self, features: List[Dict[str, Any]]):
        batch = self.collator(features)
        batch["input_ids"] = remap_token_tensor(batch["input_ids"], self.mapping)
        if "labels" in batch:
            batch["labels"] = remap_token_tensor(batch["labels"], self.mapping, ignore_index=-100)
        return batch
