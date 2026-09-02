#!/usr/bin/env python3
"""Qwen3 implementation of the RobustGER adapter.

The base Qwen3 model stays frozen. Selected decoder layers receive the
official RobustGER adapter: a learned prompt, language-noise key/value
perturbations, per-head key/value gates, and a zero-initialized attention gate.
"""

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class RobustGERRMSNorm(nn.Module):
    """RMSNorm used for the official adapter key/value projections."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class RobustGERQwen3Attention(Qwen3Attention):
    """Qwen3 attention with the official RobustGER adapter."""

    def __init__(
        self,
        config,
        layer_idx: int,
        noise_dim: int,
        adapter_prompt_length: int,
        adapter_start_layer: int,
    ):
        super().__init__(config, layer_idx)
        self.robustger_noise_dim = int(noise_dim)
        self.adapter_prompt_length = int(adapter_prompt_length)
        self.adapter_start_layer = int(adapter_start_layer)
        self.last_noise_states: Optional[List[torch.Tensor]] = None

        if self.layer_idx >= self.adapter_start_layer:
            self.adapter_wte = nn.Embedding(
                self.adapter_prompt_length, config.hidden_size
            )
            self.gating_factor = nn.Parameter(
                torch.zeros(1, 1, self.num_heads, 1)
            )

            # Official RobustGER maps the 384-dim language-noise embedding
            # into the LLM hidden dimension separately for adapter keys/values.
            self.ef_key = nn.Linear(
                self.robustger_noise_dim, config.hidden_size, bias=False
            )
            self.ef_value = nn.Linear(
                self.robustger_noise_dim, config.hidden_size
            )
            self.projection_rms_key = RobustGERRMSNorm(
                config.hidden_size,
                eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            )
            self.projection_rms_value = RobustGERRMSNorm(
                config.hidden_size,
                eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            )
            self.key_gating_factor = nn.Parameter(
                torch.zeros(
                    1,
                    self.num_heads,
                    self.adapter_prompt_length,
                    self.head_dim,
                )
            )
            self.value_gating_factor = nn.Parameter(
                torch.zeros(
                    1,
                    self.num_heads,
                    self.adapter_prompt_length,
                    self.head_dim,
                )
            )

    @property
    def has_robustger_adapter(self) -> bool:
        return self.layer_idx >= self.adapter_start_layer

    def _adapter_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """Attention over already-expanded adapter K/V states.

        Qwen3 uses GQA, whereas the official RobustGER backbone uses MHA.
        The K/V states are expanded before entering this helper so the
        per-head official key/value gates retain their intended semantics.
        """
        attention_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling
        attention_weights = F.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attention_weights = F.dropout(
            attention_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attention_output = torch.matmul(attention_weights, value_states)
        return attention_output.transpose(1, 2).contiguous(), attention_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        noise_embedding = kwargs.pop("noise_embedding", None)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        self.last_noise_states = None
        if self.has_robustger_adapter:
            if noise_embedding is None:
                raise ValueError(
                    "RobustGER attention requires noise_embedding for every forward pass"
                )
            if noise_embedding.ndim != 3:
                raise ValueError(
                    "noise_embedding must have shape [batch, N*(N-1), noise_dim]"
                )
            if noise_embedding.shape[1] != self.adapter_prompt_length:
                raise ValueError(
                    f"Expected {self.adapter_prompt_length} language-noise slots, "
                    f"got {noise_embedding.shape[1]}"
                )
            if noise_embedding.shape[2] != self.robustger_noise_dim:
                raise ValueError(
                    f"Expected language-noise dimension {self.robustger_noise_dim}, "
                    f"got {noise_embedding.shape[2]}"
                )

            adapter_dtype = self.adapter_wte.weight.dtype
            language_noise = noise_embedding.to(adapter_dtype)
            ek = self.ef_key(language_noise)
            ev = self.ef_value(language_noise)

            # Keep both intermediate key/value projections for stage-2 MINE,
            # exactly as the official implementation does.
            self.last_noise_states = [ek.clone(), ev.clone()]

            adapter_states = self.adapter_wte.weight.unsqueeze(0).expand(
                hidden_states.shape[0], -1, -1
            ).to(adapter_dtype)
            adapter_shape = (
                hidden_states.shape[0],
                self.adapter_prompt_length,
                -1,
                self.head_dim,
            )
            adapter_key = self.k_norm(
                self.k_proj(adapter_states).view(adapter_shape)
            ).transpose(1, 2)
            adapter_value = self.v_proj(adapter_states).view(adapter_shape).transpose(1, 2)

            if self.num_key_value_groups > 1:
                adapter_key = adapter_key.repeat_interleave(
                    self.num_key_value_groups, dim=1
                )
                adapter_value = adapter_value.repeat_interleave(
                    self.num_key_value_groups, dim=1
                )

            ek_norm = self.projection_rms_key(ek).view(
                hidden_states.shape[0],
                self.adapter_prompt_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            ev_norm = self.projection_rms_value(ev).view(
                hidden_states.shape[0],
                self.adapter_prompt_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)
            adapter_key = adapter_key + ek_norm * self.key_gating_factor.to(ek_norm.dtype)
            adapter_value = adapter_value + ev_norm * self.value_gating_factor.to(ev_norm.dtype)

            adapter_output, _ = self._adapter_attention(
                query_states, adapter_key, adapter_value
            )
            attn_output = attn_output + self.gating_factor.to(
                attn_output.dtype
            ) * adapter_output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class RobustGERForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM with the full RobustGER adapter insertion."""

    def __init__(self, config):
        config._attn_implementation = "eager"
        super().__init__(config)

        noise_dim = int(getattr(config, "robustger_noise_dim", 384))
        prompt_length = int(
            getattr(config, "robustger_adapter_prompt_length", 90)
        )
        start_layer = int(getattr(config, "robustger_adapter_start_layer", 2))

        for layer in self.model.layers:
            original_attention = layer.self_attn
            replacement = RobustGERQwen3Attention(
                config=config,
                layer_idx=original_attention.layer_idx,
                noise_dim=noise_dim,
                adapter_prompt_length=prompt_length,
                adapter_start_layer=start_layer,
            )
            replacement.load_state_dict(original_attention.state_dict(), strict=False)
            layer.self_attn = replacement

        self._last_noise_states: List[torch.Tensor] = []

    def forward(self, *args, noise_embedding=None, **kwargs):
        self._last_noise_states = []
        outputs = super().forward(
            *args,
            noise_embedding=noise_embedding,
            **kwargs,
        )
        for layer in self.model.layers:
            if getattr(layer.self_attn, "has_robustger_adapter", False):
                states = layer.self_attn.last_noise_states
                if states is not None:
                    self._last_noise_states.extend(states)
        return outputs

    def freeze_base_parameters(self) -> None:
        adapter_markers = (
            "adapter_wte",
            "gating_factor",
            "ef_key",
            "ef_value",
            "projection_rms_key",
            "projection_rms_value",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(
                marker in name for marker in adapter_markers
            )

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_adapter_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if "robustger" not in key]
        if unexpected:
            raise RuntimeError(f"Unexpected RobustGER checkpoint keys: {unexpected}")

        adapter_markers = (
            "adapter_wte",
            "gating_factor",
            "ef_key",
            "ef_value",
            "projection_rms_key",
            "projection_rms_value",
        )
        missing_adapter = [
            key for key in missing
            if any(marker in key for marker in adapter_markers)
        ]
        if missing_adapter:
            raise RuntimeError(
                "RobustGER checkpoint is missing adapter keys: "
                f"{missing_adapter[:8]}"
            )
