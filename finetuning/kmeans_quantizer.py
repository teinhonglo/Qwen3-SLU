"""Deterministic scalar K-means weight sharing and index packing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional
import torch
from torch import Tensor, nn


def pack_indices(indices: Tensor, bit_width: int) -> bytes:
    values = indices.detach().to(torch.uint8).cpu().flatten().tolist()
    if bit_width == 8:
        return bytes(values)
    if bit_width != 4:
        raise ValueError("Packed artifacts support bit_width 4 or 8")
    if any(v > 15 for v in values):
        raise ValueError("4-bit index exceeds 15")
    return bytes(values[i] | ((values[i + 1] if i + 1 < len(values) else 0) << 4)
                 for i in range(0, len(values), 2))


def unpack_indices(data: bytes, bit_width: int, count: int) -> Tensor:
    if bit_width == 8:
        values = list(data[:count])
    elif bit_width == 4:
        values = [v for byte in data for v in (byte & 15, byte >> 4)][:count]
    else:
        raise ValueError("Packed artifacts support bit_width 4 or 8")
    return torch.tensor(values, dtype=torch.long)


@dataclass
class QuantizedTensor:
    centroids: Tensor
    assignments: Tensor
    shape: tuple


class ScalarKMeansQuantizer:
    def __init__(self, model: nn.Module, bit_width: int = 8,
                 include_patterns: Optional[Iterable[str]] = None,
                 exclude_patterns: Optional[Iterable[str]] = None, seed: int = 66,
                 initialize: bool = True):
        if bit_width not in (4, 8):
            raise ValueError("bit_width must be 4 or 8")
        self.model, self.bit_width, self.seed = model, bit_width, seed
        self.include_patterns = list(include_patterns or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"])
        self.exclude_patterns = list(exclude_patterns or ["embed_tokens", "lm_head", "norm"])
        self.tensors: Dict[str, QuantizedTensor] = {}
        self.step_counter = 0
        named = list(model.named_parameters())
        self.selected_names = [n for n, p in named if self._selected(n, p)]
        self.excluded_names = [n for n, p in named if n not in self.selected_names]
        if not self.selected_names:
            raise ValueError("No parameters match quantization scope. Available parameter names: " +
                             ", ".join(n for n, _ in named))
        total = sum(p.numel() for _, p in named)
        selected = sum(dict(named)[n].numel() for n in self.selected_names)
        print("Selected parameter names:", *self.selected_names, sep="\n  ")
        print("Excluded parameter names:", *self.excluded_names, sep="\n  ")
        print(f"Quantized parameter count: {selected}\nTotal parameter count: {total}\n"
              f"Quantized parameter percentage: {100 * selected / total:.4f}%")
        if initialize:
            self.initialize()

    def _selected(self, name: str, parameter: Tensor) -> bool:
        lname = name.lower()
        return (parameter.ndim >= 2 and parameter.is_floating_point() and not lname.endswith("bias")
                and "audio" not in lname and any(x in name for x in self.include_patterns)
                and not any(x in name for x in self.exclude_patterns))

    def _cluster(self, values: Tensor) -> QuantizedTensor:
        flat = values.detach().float().cpu().flatten()
        unique = torch.unique(flat, sorted=True)
        k = min(2**self.bit_width, flat.numel(), unique.numel())
        if k == unique.numel():
            centroids = unique
        else:
            # Quantile initialization plus exact Lloyd updates is deterministic.
            positions = torch.linspace(0, unique.numel() - 1, k).round().long()
            centroids = unique[positions].clone()
            for _ in range(30):
                assignment = (flat[:, None] - centroids[None, :]).abs().argmin(1)
                updated = centroids.clone()
                for i in range(k):
                    members = flat[assignment == i]
                    if members.numel():
                        updated[i] = members.mean()
                if torch.equal(updated, centroids) or torch.allclose(updated, centroids):
                    centroids = updated; break
                centroids = updated
        assignment = (flat[:, None] - centroids[None, :]).abs().argmin(1)
        return QuantizedTensor(centroids, assignment, tuple(values.shape))

    def initialize(self):
        params = dict(self.model.named_parameters())
        self.tensors = {name: self._cluster(params[name]) for name in self.selected_names}

    @torch.no_grad()
    def update_and_project(self, update_centroids=True, update_assignments=False):
        params = dict(self.model.named_parameters())
        for name, qt in self.tensors.items():
            flat = params[name].detach().float().cpu().flatten()
            if update_assignments:
                qt.assignments = (flat[:, None] - qt.centroids[None, :]).abs().argmin(1)
            if update_centroids:
                for i in range(qt.centroids.numel()):
                    members = flat[qt.assignments == i]
                    if members.numel():
                        qt.centroids[i] = members.mean()
            params[name].copy_(qt.centroids[qt.assignments].reshape(qt.shape).to(params[name]))
        self.step_counter += 1

    def state_dict(self):
        return {"configuration": {"bit_width": self.bit_width, "include_patterns": self.include_patterns,
                                  "exclude_patterns": self.exclude_patterns, "seed": self.seed},
                "step_counter": self.step_counter,
                "tensors": {n: {"centroids": q.centroids, "assignments": q.assignments,
                                "shape": q.shape} for n, q in self.tensors.items()}}

    def load_state_dict(self, state):
        self.step_counter = int(state["step_counter"])
        self.tensors = {n: QuantizedTensor(v["centroids"], v["assignments"], tuple(v["shape"]))
                        for n, v in state["tensors"].items()}
