"""Loss primitives shared by the MAC-SLU distillation experiments."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def validate_logit_compatibility(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor) -> None:
    if student_logits.size(-1) != teacher_logits.size(-1):
        raise ValueError("Teacher/student vocabulary sizes differ: "
                         f"teacher={teacher_logits.size(-1)}, student={student_logits.size(-1)}")
    expected = tuple(labels.shape)
    if tuple(student_logits.shape[:2]) != expected or tuple(teacher_logits.shape[:2]) != expected:
        raise ValueError("Logits and labels sequence shapes differ: "
                         f"teacher={tuple(teacher_logits.shape)}, student={tuple(student_logits.shape)}, "
                         f"labels={tuple(labels.shape)}")


def masked_token_kl(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor,
                    temperature: float) -> Tensor:
    """Temperature-scaled KL(teacher || student), averaged over target tokens only."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    validate_logit_compatibility(student_logits, teacher_logits, labels)
    mask = labels.ne(-100)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("KL loss has no valid target tokens (all labels are -100)")
    teacher_probability = torch.softmax(teacher_logits / temperature, dim=-1)
    student_log_probability = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probability = torch.log_softmax(teacher_logits / temperature, dim=-1)
    elementwise = teacher_probability * (teacher_log_probability - student_log_probability)
    per_token = elementwise.sum(dim=-1)
    return (per_token * mask).sum() / count * temperature**2


def masked_mean_pool(hidden: Tensor, labels: Tensor) -> Tensor:
    if hidden.ndim != 3 or labels.ndim != 2 or tuple(hidden.shape[:2]) != tuple(labels.shape):
        raise ValueError(f"Hidden states and labels are not token-aligned: hidden={tuple(hidden.shape)}, "
                         f"labels={tuple(labels.shape)}")
    mask = labels.ne(-100)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        bad = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Samples contain no target tokens: batch indices {bad}")
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


class RepresentationProjector(nn.Module):
    """Layer-pair projections used only while training the student."""
    def __init__(self, student_dims: Sequence[int], teacher_dims: Sequence[int],
                 projection_dimension: Optional[int] = None):
        super().__init__()
        if len(student_dims) != len(teacher_dims) or not student_dims:
            raise ValueError("student_dims and teacher_dims must be non-empty and equally sized")
        self.student = nn.ModuleList()
        self.teacher = nn.ModuleList()
        for sd, td in zip(student_dims, teacher_dims):
            out = td if projection_dimension is None else projection_dimension
            self.student.append(nn.Identity() if sd == out else nn.Linear(sd, out, bias=False))
            # With null projection only the student is projected to the frozen teacher space.
            self.teacher.append(nn.Identity() if projection_dimension is None or td == out
                                else nn.Linear(td, out, bias=False))

    def forward(self, student: Tensor, teacher: Tensor, pair: int) -> Tuple[Tensor, Tensor]:
        return self.student[pair](student), self.teacher[pair](teacher)


def representation_contrastive_loss(student_hidden_states: Sequence[Tensor],
                                    teacher_hidden_states: Sequence[Tensor], labels: Tensor,
                                    student_layers: Sequence[int], teacher_layers: Sequence[int],
                                    projector: RepresentationProjector, temperature: float,
                                    allow_batch_size_one: bool = False,
                                    return_similarities: bool = False):
    if len(student_layers) != len(teacher_layers) or not student_layers:
        raise ValueError("student_layers and teacher_layers must be non-empty and equally sized")
    if temperature <= 0:
        raise ValueError("contrastive_temperature must be positive")
    batch = labels.size(0)
    if batch == 1 and not allow_batch_size_one:
        raise ValueError("Contrastive loss requires actual forward batch size > 1; gradient accumulation is not enough")
    losses, matrices = [], []
    target = torch.arange(batch, device=labels.device)
    for pair, (sl, tl) in enumerate(zip(student_layers, teacher_layers)):
        s = masked_mean_pool(student_hidden_states[sl], labels)
        t = masked_mean_pool(teacher_hidden_states[tl], labels)
        s, t = projector(s, t, pair)
        similarity = F.normalize(s, dim=-1) @ F.normalize(t, dim=-1).transpose(0, 1)
        similarity = similarity / temperature
        if tuple(similarity.shape) != (batch, batch):
            raise RuntimeError(f"Expected [B,B] similarity, got {tuple(similarity.shape)}")
        matrices.append(similarity)
        losses.append(F.cross_entropy(similarity, target))
    loss = torch.stack(losses).mean()
    return (loss, matrices) if return_similarities else loss


def combine_distillation_losses(ce: Tensor, kl: Tensor, contrastive: Tensor,
                                *, ce_weight: float, kl_weight: float,
                                contrastive_weight: float) -> Tensor:
    return ce_weight * ce + kl_weight * kl + contrastive_weight * contrastive
