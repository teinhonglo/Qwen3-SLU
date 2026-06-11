# coding=utf-8
"""Joint domain-intent prototype-aware Qwen3-ASR models."""

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


class JointDomainIntentPrototypeHead(nn.Module):
    """Joint domain-intent prototype scorer backed by one embedding table."""

    def __init__(
        self,
        hidden_size: int,
        num_domain_intents: int,
        temperature: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        # The previous independent domain_prototypes/intent_prototypes heads are
        # intentionally disabled: prototype training now predicts legal
        # domain-intent pairs directly.
        self.domain_intent_prototypes = nn.Embedding(int(num_domain_intents), int(hidden_size))
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        query = pooled_hidden.float()
        domain_intent_weight = self.domain_intent_prototypes.weight.float()
        if self.normalize:
            query = F.normalize(query, dim=-1)
            domain_intent_weight = F.normalize(domain_intent_weight, dim=-1)
        temperature = max(self.temperature, 1e-6)
        return torch.matmul(query, domain_intent_weight.transpose(0, 1)) / temperature


class Qwen3ASRJointPrototypeThinkerForConditionalGeneration(Qwen3ASRThinkerForConditionalGeneration):
    """Qwen3-ASR thinker with an auxiliary domain/intent prototype loss."""

    def __init__(self, config):
        super().__init__(config)
        self.prototype_config = dict(getattr(config, "prototype_config", {}) or {})
        if self.prototype_config.get("enabled", False):
            self.prototype_head = JointDomainIntentPrototypeHead(
                hidden_size=config.text_config.hidden_size,
                num_domain_intents=self.prototype_config.get("num_domain_intents", 0),
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
        domain_intent_labels: Optional[torch.Tensor] = None,
        # Legacy independent labels are intentionally unused; the independent
        # domain/intent prediction heads have been replaced by a joint head.
        domain_labels: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        prototype_prefix_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        need_proto = self.prototype_head is not None and domain_intent_labels is not None
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
        domain_intent_logits = self.prototype_head(pooled_hidden)
        domain_intent_targets = domain_intent_labels.to(domain_intent_logits.device, dtype=domain_intent_logits.dtype)
        if domain_intent_targets.dim() != 2:
            raise ValueError("domain_intent_labels must be a multi-hot tensor with shape (batch, num_domain_intents)")
        proto_loss = (
            float(self.prototype_config.get("domain_intent_loss_weight", 1.0))
            * F.binary_cross_entropy_with_logits(domain_intent_logits, domain_intent_targets)
            * float(self.prototype_config.get("loss_weight", 1.0))
        )
        total_loss = proto_loss if outputs.loss is None else outputs.loss + proto_loss
        return outputs.__class__(
            loss=total_loss,
            aux_loss=getattr(outputs, "aux_loss", None),
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )

    def prototype_logits(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        prototype_prefix_lengths=None,
        **kwargs,
    ) -> torch.Tensor:
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


class Qwen3ASRJointPrototypeForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Outer Qwen3-ASR model whose thinker includes domain/intent prototypes."""

    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.thinker = Qwen3ASRJointPrototypeThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        domain_intent_labels=None,
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
            domain_intent_labels=domain_intent_labels,
            domain_labels=domain_labels,
            intent_labels=intent_labels,
            prototype_prefix_lengths=prototype_prefix_lengths,
            **kwargs,
        )

    @torch.inference_mode()
    def predict_prototypes(self, top_k: int = 5, **inputs) -> Dict[str, List[List[Dict[str, float]]]]:
        domain_intent_logits = self.thinker.prototype_logits(**inputs)
        proto_cfg = dict(getattr(self.thinker, "prototype_config", {}) or {})
        domain_intent_labels = list(proto_cfg.get("domain_intent_labels", []) or [])
        k_domain_intent = min(int(top_k), domain_intent_logits.size(-1))
        temperature = max(float(proto_cfg.get("temperature", 1.0)), 1e-6)
        domain_intent_scores = torch.softmax(domain_intent_logits.float(), dim=-1)
        # ``prototype_logits`` returns dot-product prototype scores divided by
        # temperature.  Multiplying by temperature recovers the raw prototype
        # similarity used for ranking/threshold analysis.  When the prototype
        # head is normalized, this value is cosine similarity.
        domain_intent_similarities = domain_intent_logits.float() * temperature
        domain_intent_top = torch.topk(domain_intent_scores, k=k_domain_intent, dim=-1)

        def pack(top, labels, similarities):
            rows = []
            for batch_idx, (row_scores, row_indices) in enumerate(zip(top.values, top.indices)):
                packed = []
                for score, idx in zip(row_scores.tolist(), row_indices.tolist()):
                    label = labels[idx] if idx < len(labels) else str(idx)
                    similarity = similarities[batch_idx, int(idx)].item()
                    packed.append({"label": label, "score": float(score), "similarity": float(similarity), "index": int(idx)})
                rows.append(packed)
            return rows

        return {
            "domain_intents": pack(domain_intent_top, domain_intent_labels, domain_intent_similarities),
        }


__all__ = [
    "JointDomainIntentPrototypeHead",
    "Qwen3ASRJointPrototypeForConditionalGeneration",
    "Qwen3ASRJointPrototypeThinkerForConditionalGeneration",
]
