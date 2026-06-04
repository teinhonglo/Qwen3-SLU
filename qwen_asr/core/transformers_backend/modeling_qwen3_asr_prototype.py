# coding=utf-8
"""Prototype-aware Qwen3-ASR models for domain/intent prediction."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRThinkerForConditionalGeneration,
)


class DomainIntentPrototypeHead(nn.Module):
    """Domain/intent prototype scorer backed by trainable embedding tables."""

    def __init__(
        self,
        hidden_size: int,
        num_domains: int,
        num_intents: int,
        temperature: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.domain_prototypes = nn.Embedding(int(num_domains), int(hidden_size))
        self.intent_prototypes = nn.Embedding(int(num_intents), int(hidden_size))
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def forward(self, pooled_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = pooled_hidden.float()
        domain_weight = self.domain_prototypes.weight.float()
        intent_weight = self.intent_prototypes.weight.float()
        if self.normalize:
            query = F.normalize(query, dim=-1)
            domain_weight = F.normalize(domain_weight, dim=-1)
            intent_weight = F.normalize(intent_weight, dim=-1)
        temperature = max(self.temperature, 1e-6)
        domain_logits = torch.matmul(query, domain_weight.transpose(0, 1)) / temperature
        intent_logits = torch.matmul(query, intent_weight.transpose(0, 1)) / temperature
        return domain_logits, intent_logits


class Qwen3ASRPrototypeThinkerForConditionalGeneration(Qwen3ASRThinkerForConditionalGeneration):
    """Qwen3-ASR thinker with an auxiliary domain/intent prototype loss."""

    def __init__(self, config):
        super().__init__(config)
        self.prototype_config = dict(getattr(config, "prototype_config", {}) or {})
        if self.prototype_config.get("enabled", False):
            self.prototype_head = DomainIntentPrototypeHead(
                hidden_size=config.text_config.hidden_size,
                num_domains=self.prototype_config.get("num_domains", 0),
                num_intents=self.prototype_config.get("num_intents", 0),
                temperature=self.prototype_config.get("temperature", 1.0),
                normalize=self.prototype_config.get("normalize", True),
            )
        else:
            self.prototype_head = None

    @staticmethod
    def _pool_for_prototype(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prototype_prefix_lengths: Optional[torch.Tensor],
        pooling: str,
    ) -> torch.Tensor:
        if pooling == "last_hidden_state":
            if prototype_prefix_lengths is not None:
                idx = prototype_prefix_lengths.to(hidden_states.device).long().clamp(min=1) - 1
            elif attention_mask is not None:
                idx = attention_mask.to(hidden_states.device).long().sum(dim=1).clamp(min=1) - 1
            else:
                idx = torch.full((hidden_states.size(0),), hidden_states.size(1) - 1, device=hidden_states.device, dtype=torch.long)
            return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idx]

        if attention_mask is None:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.to(hidden_states.device).bool()
        if prototype_prefix_lengths is not None:
            prefix_lengths = prototype_prefix_lengths.to(hidden_states.device).long().clamp(min=0, max=hidden_states.size(1))
            positions = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
            mask = mask & (positions < prefix_lengths.unsqueeze(1))
        mask_f = mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask_f).sum(dim=1) / denom

    def forward(
        self,
        *args,
        domain_labels: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        prototype_prefix_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        need_proto = self.prototype_head is not None and (domain_labels is not None or intent_labels is not None)
        if need_proto:
            kwargs["output_hidden_states"] = True
        attention_mask = kwargs.get("attention_mask", None)
        outputs = super().forward(*args, **kwargs)
        if not need_proto:
            return outputs

        hidden_states = outputs.hidden_states[-1]
        pooled_hidden = self._pool_for_prototype(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            prototype_prefix_lengths=prototype_prefix_lengths,
            pooling=str(self.prototype_config.get("pooling", "mean_pooling")),
        )
        domain_logits, intent_logits = self.prototype_head(pooled_hidden)

        losses = []
        if domain_labels is not None:
            domain_targets = domain_labels.to(domain_logits.device, dtype=domain_logits.dtype)
            if domain_targets.dim() != 2:
                raise ValueError("domain_labels must be a multi-hot tensor with shape (batch, num_domains)")
            valid = domain_targets.sum(dim=-1) > 0
            if valid.any():
                losses.append(
                    float(self.prototype_config.get("domain_loss_weight", 1.0))
                    * F.binary_cross_entropy_with_logits(domain_logits[valid], domain_targets[valid])
                )
        if intent_labels is not None:
            intent_targets = intent_labels.to(intent_logits.device, dtype=intent_logits.dtype)
            if intent_targets.dim() != 2:
                raise ValueError("intent_labels must be a multi-hot tensor with shape (batch, num_intents)")
            valid = intent_targets.sum(dim=-1) > 0
            if valid.any():
                losses.append(
                    float(self.prototype_config.get("intent_loss_weight", 1.0))
                    * F.binary_cross_entropy_with_logits(intent_logits[valid], intent_targets[valid])
                )
        if losses:
            proto_loss = sum(losses)
            total_weight = float(self.prototype_config.get("loss_weight", 1.0))
            outputs.loss = proto_loss * total_weight if outputs.loss is None else outputs.loss + proto_loss * total_weight
        return outputs

    def prototype_logits(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        prototype_prefix_lengths=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prototype_head is None:
            raise RuntimeError("prototype_head is not enabled on this model")
        outputs = super().forward(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        pooled_hidden = self._pool_for_prototype(
            hidden_states=outputs.hidden_states[-1],
            attention_mask=attention_mask,
            prototype_prefix_lengths=prototype_prefix_lengths,
            pooling=str(self.prototype_config.get("pooling", "mean_pooling")),
        )
        return self.prototype_head(pooled_hidden)


class Qwen3ASRPrototypeForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Outer Qwen3-ASR model whose thinker includes domain/intent prototypes."""

    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.thinker = Qwen3ASRPrototypeThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        domain_labels=None,
        intent_labels=None,
        prototype_prefix_lengths=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            domain_labels=domain_labels,
            intent_labels=intent_labels,
            prototype_prefix_lengths=prototype_prefix_lengths,
            **kwargs,
        )

    @torch.inference_mode()
    def predict_prototypes(self, top_k: int = 5, **inputs) -> Dict[str, List[List[Dict[str, float]]]]:
        domain_logits, intent_logits = self.thinker.prototype_logits(**inputs)
        proto_cfg = dict(getattr(self.thinker, "prototype_config", {}) or {})
        domain_labels = list(proto_cfg.get("domain_labels", []) or [])
        intent_labels = list(proto_cfg.get("intent_labels", []) or [])
        k_domain = min(int(top_k), domain_logits.size(-1))
        k_intent = min(int(top_k), intent_logits.size(-1))
        domain_scores = torch.softmax(domain_logits.float(), dim=-1)
        intent_scores = torch.softmax(intent_logits.float(), dim=-1)
        domain_top = torch.topk(domain_scores, k=k_domain, dim=-1)
        intent_top = torch.topk(intent_scores, k=k_intent, dim=-1)

        def pack(top, labels):
            rows = []
            for row_scores, row_indices in zip(top.values, top.indices):
                packed = []
                for score, idx in zip(row_scores.tolist(), row_indices.tolist()):
                    label = labels[idx] if idx < len(labels) else str(idx)
                    packed.append({"label": label, "score": float(score), "index": int(idx)})
                rows.append(packed)
            return rows

        return {"domains": pack(domain_top, domain_labels), "intents": pack(intent_top, intent_labels)}


__all__ = [
    "DomainIntentPrototypeHead",
    "Qwen3ASRPrototypeForConditionalGeneration",
    "Qwen3ASRPrototypeThinkerForConditionalGeneration",
]
