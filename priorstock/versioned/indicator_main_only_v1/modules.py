"""Reusable modules for the pure indicator-main only PriorStock variant."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerBlockOutput:
    """One pure transformer block output bundle."""

    hidden_states: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class TransformerBlockIndicatorMainOnly(nn.Module):
    """Pre-LN transformer block without any auxiliary modality injection."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_probability: float,
    ) -> None:
        """Create one self-attention + feed-forward block for the main indicator stream."""

        super().__init__()
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_probability,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout_probability)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_probability),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
        collect_trace_tensors: bool,
    ) -> TransformerBlockOutput:
        """Run one pure self-attention block and return the updated hidden states."""

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        attention_input = self.attention_layer_norm(hidden_states)
        if collect_trace_tensors:
            attention_output, attention_weights = self.self_attention(
                attention_input,
                attention_input,
                attention_input,
                attn_mask=temporal_attention_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attention_output, _ = self.self_attention(
                attention_input,
                attention_input,
                attention_input,
                attn_mask=temporal_attention_mask,
                need_weights=False,
            )
            attention_weights = None

        if attention_weights is not None:
            attention_weights_cpu = attention_weights.detach().to(device="cpu", dtype=torch.float32)
            del attention_weights
        else:
            attention_weights_cpu = None

        hidden_states_after_attention = hidden_states + self.attention_dropout(attention_output)
        feed_forward_input = self.feed_forward_layer_norm(hidden_states_after_attention)
        feed_forward_output = self.feed_forward(feed_forward_input)
        hidden_states_after_block = hidden_states_after_attention + feed_forward_output

        diagnostics = {
            "hidden_shape": list(hidden_states_after_block.shape),
        }
        if attention_weights_cpu is not None:
            attention_weight_floor = torch.finfo(attention_weights_cpu.dtype).tiny
            stable_attention_weights = attention_weights_cpu.clamp_min(attention_weight_floor)
            attention_entropy = -(stable_attention_weights * stable_attention_weights.log()).sum(dim=-1)
            diagonal_attention_mass = torch.diagonal(attention_weights_cpu, dim1=-2, dim2=-1)
            diagnostics.update(
                {
                    "attention_shape": list(attention_weights_cpu.shape),
                    "attention_mean": float(attention_weights_cpu.mean().item()),
                    "attention_std": float(attention_weights_cpu.std(unbiased=False).item()),
                    "attention_min": float(attention_weights_cpu.min().item()),
                    "attention_max": float(attention_weights_cpu.max().item()),
                    "attention_entropy_mean": float(attention_entropy.mean().item()),
                    "diagonal_attention_mass_mean": float(diagonal_attention_mass.mean().item()),
                }
            )

        trace_tensors: dict[str, torch.Tensor] = {}
        if collect_trace_tensors:
            trace_tensors = {
                "attention_input": _trace_to_cpu(attention_input),
                "attention_output": _trace_to_cpu(attention_output),
                "hidden_states_after_attention": _trace_to_cpu(hidden_states_after_attention),
                "feed_forward_input": _trace_to_cpu(feed_forward_input),
                "feed_forward_output": _trace_to_cpu(feed_forward_output),
                "hidden_states_after_block": _trace_to_cpu(hidden_states_after_block),
            }
            if attention_weights_cpu is not None:
                trace_tensors["attention_weights"] = attention_weights_cpu

        return TransformerBlockOutput(
            hidden_states=hidden_states_after_block,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
