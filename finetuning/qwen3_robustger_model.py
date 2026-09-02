#!/usr/bin/env python3
"""Qwen3 implementation of the RobustGER adapter and language-noise tuner.

The base Qwen3 model stays frozen. Each selected decoder layer receives the
paper's N-best language-noise embedding through a zero-initialized denoising
gate and a zero-initialized LLaMA-Adapter-style attention gate.
"""

from typing import Dict, Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class RobustGERQwen3Attention(Qwen3Attention):
    """Qwen3 attention with the RobustGER denoising adapter."""

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
        self.last_noise_states: Optional[torch.Tensor] = None

        if self.layer_idx >= self.adapter_start_layer:
            self.adapter_wte = nn.Embedding(self.adapter_prompt_length, config.hidden_size)
            self.gating_factor = nn.Parameter(
                torch.zeros(1, 1, config.num_attention_heads, 1)
            )
            self.language_noise_tuner = nn.Linear(
                self.robustger_noise_dim, config.hidden_size, bias=False
            )
            self.denoise_gate = nn.Parameter(torch.zeros(1, 1, 1))

    @property
    def has_robustger_adapter(self) -> bool:
        return self.layer_idx >= self.adapter_start_layer

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

        # Keep the Qwen3 attention path unchanged for the original sequence.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
            noise_embedding = kwargs.get("noise_embedding")
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

            # T_omega(E_LN), then G_l - g_l^dn*T_omega(E_LN).
            tuner_dtype = self.language_noise_tuner.weight.dtype
            tuned_noise = self.language_noise_tuner(noise_embedding.to(tuner_dtype))
            self.last_noise_states = tuned_noise

            adapter_states = self.adapter_wte.weight.unsqueeze(0).expand(
                hidden_states.shape[0], -1, -1
            ).to(tuner_dtype)
            adapter_states = adapter_states - self.denoise_gate.to(tuner_dtype) * tuned_noise

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

            # eager_attention_forward performs the QKV-group repetition itself.
            adapter_output, _ = eager_attention_forward(
                self,
                query_states,
                adapter_key,
                adapter_value,
                None,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=None,
            )
            attn_output = attn_output + self.gating_factor.to(attn_output.dtype) * adapter_output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class RobustGERForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM with full RobustGER adapter insertion."""

    def __init__(self, config):
        config._attn_implementation = "eager"
        super().__init__(config)

        noise_dim = int(getattr(config, "robustger_noise_dim", 384))
        prompt_length = int(getattr(config, "robustger_adapter_prompt_length", 90))
        start_layer = int(getattr(config, "robustger_adapter_start_layer", 1))

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

        self._last_noise_states = []

    def forward(self, *args, noise_embedding=None, **kwargs):
        self._last_noise_states = []
        outputs = super().forward(
            *args,
            noise_embedding=noise_embedding,
            **kwargs,
        )
        self._last_noise_states = [
            layer.self_attn.last_noise_states
            for layer in self.model.layers
            if getattr(layer.self_attn, "has_robustger_adapter", False)
            and layer.self_attn.last_noise_states is not None
        ]
        return outputs

    def freeze_base_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(
                marker in name
                for marker in (
                    "adapter_wte",
                    "gating_factor",
                    "language_noise_tuner",
                    "denoise_gate",
                )
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
        if missing:
            missing_adapter = [
                key for key in missing
                if any(marker in key for marker in (
                    "adapter_wte",
                    "gating_factor",
                    "language_noise_tuner",
                    "denoise_gate",
                ))
            ]
            if missing_adapter:
                raise RuntimeError(
                    f"RobustGER checkpoint is missing adapter keys: {missing_adapter[:8]}"
                )
