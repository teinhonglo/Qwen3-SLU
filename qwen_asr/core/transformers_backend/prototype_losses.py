"""Shared calibration and loss for prototype-based multi-label retrieval."""

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

        self.scale_init = scale_init
        self.bias_init = float(bias_init)
        self.scale_max = scale_max
        self.logit_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the configured calibration initialization."""

        with torch.no_grad():
            self.logit_scale.fill_(math.log(self.scale_init))
            self.logit_bias.fill_(self.bias_init)

    def scale(self) -> torch.Tensor:
        # Clamp in log space so exp() cannot overflow before the upper bound is
        # applied; exp(inf).clamp(...) can otherwise yield a NaN gradient.
        return self.logit_scale.clamp(max=math.log(self.scale_max)).exp()

    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        return (
            similarities * self.scale().to(similarities.dtype)
            + self.logit_bias.to(similarities.dtype)
        )

    def recover_similarities(self, logits: torch.Tensor) -> torch.Tensor:
        scale = self.scale().to(logits.dtype)
        return (logits - self.logit_bias.to(logits.dtype)) / scale


def prototype_multi_label_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prototype_config: Mapping[str, Any],
) -> torch.Tensor:
    """Compute multi-label BCE over scaled-cosine prototype logits."""

    if targets.dim() != 2:
        raise ValueError("prototype targets must be a multi-hot tensor with shape (batch, num_labels)")
    targets = targets.to(logits.device, dtype=logits.dtype)
    loss_type = str(prototype_config.get("loss_type", "bce")).lower()
    if loss_type != "bce":
        raise ValueError(
            f"unsupported prototype loss_type: {loss_type}; "
            "expected 'bce'"
        )
    return F.binary_cross_entropy_with_logits(logits, targets)


__all__ = [
    "LearnableScaledCosine",
    "prototype_multi_label_loss",
]
