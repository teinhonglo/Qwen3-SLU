"""Shared losses for prototype-based multi-label retrieval."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class LearnableScaledCosine(nn.Module):
    """Convert bounded cosine similarities into trainable BCE logits."""

    def __init__(
        self,
        num_labels: int,
        scale_init: float = 10.0,
        scale_max: float = 100.0,
        bias_init: float | None = None,
    ) -> None:
        super().__init__()
        num_labels = int(num_labels)
        scale_init = float(scale_init)
        scale_max = float(scale_max)
        if num_labels <= 0:
            raise ValueError(f"num_labels must be positive; got {num_labels}")
        if scale_init <= 0.0 or scale_max < scale_init:
            raise ValueError(
                "scaled-cosine BCE requires 0 < logit_scale_init <= logit_scale_max; "
                f"got {scale_init} and {scale_max}"
            )

        # Without a dataset-specific prior, initialize from a one-positive-label
        # assumption. The learnable bias then adapts to observed cardinality.
        if bias_init is None:
            positive_prior = 1.0 / num_labels
            positive_prior = min(max(positive_prior, 1e-6), 1.0 - 1e-6)
            bias_init = math.log(positive_prior / (1.0 - positive_prior))

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(scale_init), dtype=torch.float32)
        )
        self.logit_bias = nn.Parameter(torch.tensor(float(bias_init), dtype=torch.float32))
        self.scale_max = scale_max

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.scale_max)

    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        return (
            similarities * self.scale().to(similarities.dtype)
            + self.logit_bias.to(similarities.dtype)
        )

    def recover_similarities(self, logits: torch.Tensor) -> torch.Tensor:
        scale = self.scale().to(logits.dtype)
        return (logits - self.logit_bias.to(logits.dtype)) / scale


def clipped_dynamic_margin_loss(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    gamma_min: float = 0.1,
    gamma_max: float = 0.3,
) -> torch.Tensor:
    """PRIME clipped dynamic-margin loss over all positive-negative pairs.

    ``similarities`` must contain cosine similarities. The dynamic margin is
    detached exactly as in Dahiya et al. (NAACL 2025). With the small MAC-SLU
    label spaces, using every positive-negative pair is inexpensive and avoids
    introducing a separate negative-mining path.
    """

    if similarities.dim() != 2 or targets.dim() != 2:
        raise ValueError("similarities and targets must both have shape (batch, num_labels)")
    if similarities.shape != targets.shape:
        raise ValueError(
            f"similarities and targets must have the same shape: "
            f"{tuple(similarities.shape)} != {tuple(targets.shape)}"
        )
    gamma_min = float(gamma_min)
    gamma_max = float(gamma_max)
    if gamma_min < 0.0 or gamma_max < gamma_min:
        raise ValueError(
            f"dynamic margins must satisfy 0 <= gamma_min <= gamma_max; "
            f"got {gamma_min} and {gamma_max}"
        )

    positive_mask = targets.ge(0.5)
    negative_mask = ~positive_mask
    pair_mask = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)

    # pair_gap[b, p, n] = s_positive - s_negative.
    pair_gap = similarities.unsqueeze(2) - similarities.unsqueeze(1)
    clipped_margin = pair_gap.abs().clamp(min=gamma_min, max=gamma_max).detach()

    # PRIME deliberately reverses the gradient when a positive is only
    # marginally ahead of a negative (0 < gap < gamma_min), treating that area
    # as an ambiguity/missing-label region. Hard pairs retain the usual triplet
    # gradient; sufficiently separated pairs contribute zero loss.
    easy = pair_gap.ge(gamma_min)
    positive_ahead = pair_gap.gt(0.0)
    ambiguous_loss = pair_gap + gamma_min
    hard_loss = -pair_gap + clipped_margin
    pair_loss = torch.where(
        easy,
        torch.zeros_like(pair_gap),
        torch.where(positive_ahead, ambiguous_loss, hard_loss),
    )
    pair_weight = pair_mask.to(pair_loss.dtype)
    return (pair_loss * pair_weight).sum() / pair_weight.sum().clamp(min=1.0)


def prototype_multi_label_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prototype_config: Mapping[str, Any],
) -> torch.Tensor:
    """Select the configured prototype objective while preserving BCE support."""

    if targets.dim() != 2:
        raise ValueError("prototype targets must be a multi-hot tensor with shape (batch, num_labels)")
    targets = targets.to(logits.device, dtype=logits.dtype)
    loss_type = str(prototype_config.get("loss_type", "bce")).lower()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if loss_type != "clipped_dynamic_margin":
        raise ValueError(
            f"unsupported prototype loss_type: {loss_type}; "
            "expected 'bce' or 'clipped_dynamic_margin'"
        )
    if not bool(prototype_config.get("normalize", True)):
        raise ValueError("clipped_dynamic_margin requires prototype.normalize=true")

    # Prototype heads retain temperature-scaled logits for backward-compatible
    # inference output. Recover the raw cosine similarity required by PRIME.
    temperature = max(float(prototype_config.get("temperature", 1.0)), 1e-6)
    similarities = logits * temperature
    return clipped_dynamic_margin_loss(
        similarities,
        targets,
        gamma_min=float(prototype_config.get("dynamic_margin_min", 0.1)),
        gamma_max=float(prototype_config.get("dynamic_margin_max", 0.3)),
    )


__all__ = [
    "LearnableScaledCosine",
    "clipped_dynamic_margin_loss",
    "prototype_multi_label_loss",
]
