"""Reusable model modules for PriorStock."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerBlockOutput:
    """One gated transformer block's output bundle."""

    hidden_states: torch.Tensor
    tech_gate_regularization_loss: torch.Tensor
    news_gate_regularization_loss: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class ModalityGate(nn.Module):
    """Time-step gate that decides how much auxiliary modality information to inject."""

    def __init__(
        self,
        d_model: int,
        gate_hidden_dim: int,
        gate_bias_initial_value: float,
        has_presence_flag: bool,
    ) -> None:
        """Construct one gating MLP with an explicit optional presence flag."""

        super().__init__()
        input_dim = (2 * d_model) + (1 if has_presence_flag else 0)
        self._gate_network = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        nn.init.constant_(self._gate_network[-1].bias, gate_bias_initial_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_features: torch.Tensor,
        presence_flag: torch.Tensor | None,
    ) -> torch.Tensor:
        """Produce one sigmoid gate value per batch item and per time step."""

        gate_input = torch.cat([hidden_states, modality_features], dim=-1)
        if presence_flag is not None:
            gate_input = torch.cat([gate_input, presence_flag], dim=-1)
        return torch.sigmoid(self._gate_network(gate_input))


class TransformerBlockWithGating(nn.Module):
    """Pre-LN transformer block followed by dual-modality gated residual injection."""

    def __init__(
        self,
        technical_input_dim: int,
        news_input_dim: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_probability: float,
        gate_hidden_dim: int,
        tech_gate_bias_initial_value: float,
        news_gate_bias_initial_value: float,
        use_post_injection_layer_norm: bool,
        use_layer_specific_text_projections: bool,
        trace_layer_specific_projected_text_features: bool,
        use_block_level_news_hard_mask: bool,
    ) -> None:
        """Create one complete transformer block plus two modality gates."""

        super().__init__()
        self._use_layer_specific_text_projections = use_layer_specific_text_projections
        self._trace_layer_specific_projected_text_features = trace_layer_specific_projected_text_features
        self._use_block_level_news_hard_mask = use_block_level_news_hard_mask
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
        if use_layer_specific_text_projections:
            self.layer_tech_proj = nn.Linear(technical_input_dim, d_model)
            self.layer_news_proj = nn.Linear(news_input_dim, d_model)
            self.layer_tech_layer_norm = nn.LayerNorm(d_model)
            self.layer_news_layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_tech_proj = None
            self.layer_news_proj = None
            self.layer_tech_layer_norm = None
            self.layer_news_layer_norm = None
        self.tech_gate = ModalityGate(d_model, gate_hidden_dim, tech_gate_bias_initial_value, False)
        self.news_gate = ModalityGate(d_model, gate_hidden_dim, news_gate_bias_initial_value, True)
        self.post_injection_layer_norm = nn.LayerNorm(d_model) if use_post_injection_layer_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        technical_indicator_inputs: torch.Tensor,
        news_inputs: torch.Tensor,
        has_news: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
        collect_trace_tensors: bool,
    ) -> TransformerBlockOutput:
        """Run the block and return fused hidden states plus diagnostics."""

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        if self._use_layer_specific_text_projections:
            if self.layer_tech_proj is None or self.layer_news_proj is None:
                raise RuntimeError("Layer-specific text projections are not initialized.")
            if self.layer_tech_layer_norm is None or self.layer_news_layer_norm is None:
                raise RuntimeError("Layer-specific text layer norms are not initialized.")
            layer_technical_indicator_features = self.layer_tech_layer_norm(
                self.layer_tech_proj(technical_indicator_inputs)
            )
            layer_news_features = self.layer_news_layer_norm(self.layer_news_proj(news_inputs))
        else:
            layer_technical_indicator_features = technical_indicator_inputs
            layer_news_features = news_inputs

        if self._use_block_level_news_hard_mask:
            layer_news_features = layer_news_features * has_news

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
        hidden_states_before_fusion = hidden_states_after_attention + feed_forward_output

        tech_gate_values = self.tech_gate(
            hidden_states_before_fusion,
            layer_technical_indicator_features,
            None,
        )
        news_gate_values = self.news_gate(hidden_states_before_fusion, layer_news_features, has_news)
        fused_hidden_states = (
            hidden_states_before_fusion
            + (tech_gate_values * layer_technical_indicator_features)
            + (news_gate_values * layer_news_features)
        )

        if self.post_injection_layer_norm is not None:
            fused_hidden_states = self.post_injection_layer_norm(fused_hidden_states)

        tech_gate_regularization_loss = tech_gate_values.abs().mean()
        news_gate_regularization_loss = news_gate_values.abs().mean()
        diagnostics = {
            "hidden_shape": list(fused_hidden_states.shape),
            "tech_gate_mean": float(tech_gate_values.mean().detach().cpu()),
            "news_gate_mean": float(news_gate_values.mean().detach().cpu()),
            "tech_gate_max": float(tech_gate_values.max().detach().cpu()),
            "news_gate_max": float(news_gate_values.max().detach().cpu()),
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
                "attention_weights": attention_weights_cpu,
                "hidden_states_after_attention": _trace_to_cpu(hidden_states_after_attention),
                "feed_forward_input": _trace_to_cpu(feed_forward_input),
                "feed_forward_output": _trace_to_cpu(feed_forward_output),
                "hidden_states_before_fusion": _trace_to_cpu(hidden_states_before_fusion),
                "tech_gate_values": _trace_to_cpu(tech_gate_values),
                "news_gate_values": _trace_to_cpu(news_gate_values),
                "fused_hidden_states": _trace_to_cpu(fused_hidden_states),
            }
            if self._trace_layer_specific_projected_text_features:
                trace_tensors["layer_technical_indicator_features"] = _trace_to_cpu(
                    layer_technical_indicator_features
                )
                trace_tensors["layer_news_features"] = _trace_to_cpu(layer_news_features)
        return TransformerBlockOutput(
            hidden_states=fused_hidden_states,
            tech_gate_regularization_loss=tech_gate_regularization_loss,
            news_gate_regularization_loss=news_gate_regularization_loss,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
