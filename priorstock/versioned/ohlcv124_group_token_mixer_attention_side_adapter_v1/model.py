"""OHLCV-124 group-token mixer with an A-conditioned attention-side adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from priorstock.config import ExperimentConfig
from priorstock.versioned.ohlcv124_group_fusion_v1.feature_schema import (
    B_TO_J_GROUP_NAMES,
)
from priorstock.versioned.ohlcv124_group_token_mixer_v1.model import (
    PriorStockV3OHLCV124GroupTokenMixer,
    PriorStockModelOutput,
    GroupTokenTransformerOutput,
)


@dataclass(frozen=True)
class AttentionSideAdapterConfig:
    """Script-local hyperparameters for the attention-side residual adapter."""

    adapter_rank: int
    adapter_dropout_probability: float
    alpha_initial_value: float
    should_replace_attention_update_with_adapter_delta: bool
    condition_projection_initialization_std: float
    up_projection_initialization_std: float
    scale_control_target_ratio: float
    scale_control_max_scale: float
    scale_control_epsilon: float


@dataclass(frozen=True)
class AttentionSideAdapterOutput:
    """Intermediate tensors emitted by one A-conditioned residual adapter pass."""

    scaled_delta: torch.Tensor
    raw_delta: torch.Tensor
    condition_gamma: torch.Tensor
    condition_beta: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class AConditionedLowRankFiLMResidualAdapter(nn.Module):
    """Low-rank FiLM adapter that conditions cross-group attention writes on A context."""

    def __init__(
        self,
        group_token_dim: int,
        a_context_dim: int,
        adapter_config: AttentionSideAdapterConfig,
        group_names: tuple[str, ...],
    ) -> None:
        """Create the attention-side adapter and initialize it near a low-impact state."""

        super().__init__()
        self._adapter_config = adapter_config
        self._group_names = group_names
        adapter_input_dim = group_token_dim * 3
        condition_input_dim = a_context_dim + group_token_dim
        condition_output_dim = adapter_config.adapter_rank * 2
        self.down_projection = nn.Linear(adapter_input_dim, adapter_config.adapter_rank)
        self.condition_projection = nn.Linear(condition_input_dim, condition_output_dim)
        self.up_projection = nn.Linear(adapter_config.adapter_rank, group_token_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(adapter_config.adapter_dropout_probability)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Apply explicit near-baseline initialization to all adapter projections."""

        nn.init.kaiming_normal_(self.down_projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.down_projection.bias)
        nn.init.normal_(
            self.condition_projection.weight,
            mean=0.0,
            std=self._adapter_config.condition_projection_initialization_std,
        )
        nn.init.zeros_(self.condition_projection.bias)
        nn.init.normal_(
            self.up_projection.weight,
            mean=0.0,
            std=self._adapter_config.up_projection_initialization_std,
        )
        nn.init.zeros_(self.up_projection.bias)

    def _scale_delta_to_attention_reference(
        self,
        raw_delta: torch.Tensor,
        attention_update: torch.Tensor,
    ) -> torch.Tensor:
        """Scale raw adapter output against attention-update RMS to keep residual size controlled."""

        raw_delta_rms = torch.sqrt(
            raw_delta.pow(2).mean(dim=-1, keepdim=True)
            + self._adapter_config.scale_control_epsilon
        )
        attention_update_rms = torch.sqrt(
            attention_update.pow(2).mean(dim=-1, keepdim=True)
            + self._adapter_config.scale_control_epsilon
        )
        relative_scale = (
            attention_update_rms / raw_delta_rms
        ).clamp(max=self._adapter_config.scale_control_max_scale)
        return raw_delta * relative_scale * self._adapter_config.scale_control_target_ratio

    def _compute_cosine_mean(
        self,
        source_tensor: torch.Tensor,
        reference_tensor: torch.Tensor,
    ) -> float:
        """Compute mean cosine similarity between two token grids over their feature axis."""

        cosine_values = F.cosine_similarity(
            source_tensor.detach().reshape(-1, source_tensor.shape[-1]),
            reference_tensor.detach().reshape(-1, reference_tensor.shape[-1]),
            dim=-1,
        )
        return float(cosine_values.mean().cpu().item())

    def _build_per_group_diagnostics(
        self,
        scaled_delta: torch.Tensor,
        effective_scaled_delta: torch.Tensor,
        group_tokens_before: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """Summarize adapter delta norms separately for each B-J semantic group."""

        group_diagnostics: dict[str, dict[str, float]] = {}
        hidden_floor = torch.finfo(group_tokens_before.dtype).tiny
        for group_index, group_name in enumerate(self._group_names):
            group_delta_norm = scaled_delta[:, group_index, :].detach().norm(dim=-1)
            group_effective_delta_norm = effective_scaled_delta[:, group_index, :].detach().norm(
                dim=-1
            )
            group_hidden_norm = group_tokens_before[:, group_index, :].detach().norm(dim=-1)
            group_effective_delta_ratio = (
                group_effective_delta_norm.mean()
                / group_hidden_norm.mean().clamp_min(hidden_floor)
            )
            group_diagnostics[group_name] = {
                "delta_norm_mean": float(group_delta_norm.mean().cpu().item()),
                "delta_norm_std": float(
                    group_delta_norm.std(unbiased=False).cpu().item()
                ),
                "effective_delta_ratio": float(group_effective_delta_ratio.cpu().item()),
            }
        return group_diagnostics

    def forward(
        self,
        group_tokens_before: torch.Tensor,
        attention_update: torch.Tensor,
        a_context_hidden_states: torch.Tensor,
        group_identity_features: torch.Tensor,
        adapter_alpha: torch.Tensor,
        collect_trace_tensors: bool,
    ) -> AttentionSideAdapterOutput:
        """Return a scaled A-conditioned residual delta for one attention update."""

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        expanded_group_identity_features = group_identity_features.expand(
            group_tokens_before.shape[0],
            -1,
            -1,
        )
        adapter_input = torch.cat(
            [group_tokens_before, attention_update, expanded_group_identity_features],
            dim=-1,
        )
        hidden_states = self.activation(self.down_projection(adapter_input))
        condition_input = torch.cat(
            [a_context_hidden_states, expanded_group_identity_features],
            dim=-1,
        )
        condition_values = self.condition_projection(condition_input)
        condition_gamma, condition_beta = torch.chunk(condition_values, chunks=2, dim=-1)
        modulated_hidden_states = (
            hidden_states * (1.0 + condition_gamma)
        ) + condition_beta
        dropped_hidden_states = self.dropout(modulated_hidden_states)
        raw_delta = self.up_projection(dropped_hidden_states)
        scaled_delta = self._scale_delta_to_attention_reference(
            raw_delta=raw_delta,
            attention_update=attention_update,
        )
        effective_scaled_delta = adapter_alpha * scaled_delta

        attention_abs_mean = attention_update.detach().abs().mean()
        hidden_abs_mean = group_tokens_before.detach().abs().mean()
        scaled_delta_abs_mean = scaled_delta.detach().abs().mean()
        effective_scaled_delta_abs_mean = effective_scaled_delta.detach().abs().mean()
        numeric_floor = torch.finfo(attention_abs_mean.dtype).tiny
        diagnostics = {
            "adapter_raw_delta_std": float(
                raw_delta.detach().std(unbiased=False).cpu().item()
            ),
            "adapter_scaled_delta_std": float(
                scaled_delta.detach().std(unbiased=False).cpu().item()
            ),
            "effective_adapter_delta_std": float(
                effective_scaled_delta.detach().std(unbiased=False).cpu().item()
            ),
            "adapter_delta_to_attention_update_ratio": float(
                (
                    scaled_delta_abs_mean
                    / attention_abs_mean.clamp_min(numeric_floor)
                )
                .cpu()
                .item()
            ),
            "effective_adapter_delta_to_attention_update_ratio": float(
                (
                    effective_scaled_delta_abs_mean
                    / attention_abs_mean.clamp_min(numeric_floor)
                )
                .cpu()
                .item()
            ),
            "effective_adapter_delta_to_hidden_ratio": float(
                (
                    effective_scaled_delta_abs_mean
                    / hidden_abs_mean.clamp_min(torch.finfo(hidden_abs_mean.dtype).tiny)
                )
                .cpu()
                .item()
            ),
            "adapter_delta_cosine_with_attention_update": self._compute_cosine_mean(
                scaled_delta,
                attention_update,
            ),
            "adapter_delta_cosine_with_group_tokens": self._compute_cosine_mean(
                scaled_delta,
                group_tokens_before,
            ),
            "condition_gamma_mean": float(condition_gamma.detach().mean().cpu().item()),
            "condition_gamma_std": float(
                condition_gamma.detach().std(unbiased=False).cpu().item()
            ),
            "condition_beta_mean": float(condition_beta.detach().mean().cpu().item()),
            "condition_beta_std": float(
                condition_beta.detach().std(unbiased=False).cpu().item()
            ),
            "per_group_delta_statistics": self._build_per_group_diagnostics(
                scaled_delta=scaled_delta,
                effective_scaled_delta=effective_scaled_delta,
                group_tokens_before=group_tokens_before,
            ),
        }

        trace_tensors: dict[str, torch.Tensor] = {}
        if collect_trace_tensors:
            trace_tensors = {
                "adapter_input": _trace_to_cpu(adapter_input),
                "adapter_hidden_states": _trace_to_cpu(hidden_states),
                "condition_gamma": _trace_to_cpu(condition_gamma),
                "condition_beta": _trace_to_cpu(condition_beta),
                "adapter_raw_delta": _trace_to_cpu(raw_delta),
                "adapter_scaled_delta": _trace_to_cpu(scaled_delta),
                "effective_adapter_delta": _trace_to_cpu(effective_scaled_delta),
            }

        return AttentionSideAdapterOutput(
            scaled_delta=scaled_delta,
            raw_delta=raw_delta,
            condition_gamma=condition_gamma,
            condition_beta=condition_beta,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )


class GroupTokenTransformerBlockWithAttentionSideAdapter(nn.Module):
    """Pre-LN B-J group-token block with an A-conditioned residual side adapter."""

    def __init__(
        self,
        token_dim: int,
        a_context_dim: int,
        feed_forward_dim: int,
        head_count: int,
        dropout_probability: float,
        adapter_config: AttentionSideAdapterConfig,
    ) -> None:
        """Create one group-token block and its attention-side adapter."""

        super().__init__()
        self._adapter_config = adapter_config
        self.attention_layer_norm = nn.LayerNorm(token_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=head_count,
            dropout=dropout_probability,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout_probability)
        self.attention_side_adapter = AConditionedLowRankFiLMResidualAdapter(
            group_token_dim=token_dim,
            a_context_dim=a_context_dim,
            adapter_config=adapter_config,
            group_names=tuple(B_TO_J_GROUP_NAMES),
        )
        if adapter_config.should_replace_attention_update_with_adapter_delta:
            self.register_buffer("attention_side_adapter_alpha", torch.ones(()))
        else:
            self.attention_side_adapter_alpha = nn.Parameter(
                torch.full((), float(adapter_config.alpha_initial_value))
            )
        self.feed_forward_layer_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(feed_forward_dim, token_dim),
            nn.Dropout(dropout_probability),
        )

    def forward(
        self,
        group_token_hidden_states: torch.Tensor,
        a_context_hidden_states: torch.Tensor,
        group_identity_features: torch.Tensor,
        collect_trace_tensors: bool,
    ) -> GroupTokenTransformerOutput:
        """Mix group tokens, add the A-conditioned residual delta, then run the FFN."""

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        attention_input = self.attention_layer_norm(group_token_hidden_states)
        if collect_trace_tensors:
            attention_output, attention_weights = self.self_attention(
                attention_input,
                attention_input,
                attention_input,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attention_output, _ = self.self_attention(
                attention_input,
                attention_input,
                attention_input,
                need_weights=False,
            )
            attention_weights = None

        if attention_weights is not None:
            attention_weights_cpu = attention_weights.detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            del attention_weights
        else:
            attention_weights_cpu = None

        attention_update = self.attention_dropout(attention_output)
        adapter_output = self.attention_side_adapter(
            group_tokens_before=group_token_hidden_states,
            attention_update=attention_update,
            a_context_hidden_states=a_context_hidden_states,
            group_identity_features=group_identity_features,
            adapter_alpha=self.attention_side_adapter_alpha,
            collect_trace_tensors=collect_trace_tensors,
        )
        if self._adapter_config.should_replace_attention_update_with_adapter_delta:
            hidden_states_after_attention = group_token_hidden_states + adapter_output.scaled_delta
        else:
            hidden_states_after_attention = (
                group_token_hidden_states
                + attention_update
                + (self.attention_side_adapter_alpha * adapter_output.scaled_delta)
            )
        feed_forward_input = self.feed_forward_layer_norm(hidden_states_after_attention)
        feed_forward_output = self.feed_forward(feed_forward_input)
        hidden_states_after_block = hidden_states_after_attention + feed_forward_output

        diagnostics = {
            "hidden_shape": list(hidden_states_after_block.shape),
            "should_replace_attention_update_with_adapter_delta": float(
                self._adapter_config.should_replace_attention_update_with_adapter_delta
            ),
            "adapter_alpha": float(
                self.attention_side_adapter_alpha.detach().cpu().item()
            ),
            "attention_side_adapter": adapter_output.diagnostics,
        }
        if attention_weights_cpu is not None:
            attention_weight_floor = torch.finfo(attention_weights_cpu.dtype).tiny
            stable_attention_weights = attention_weights_cpu.clamp_min(attention_weight_floor)
            attention_entropy = -(
                stable_attention_weights * stable_attention_weights.log()
            ).sum(dim=-1)
            diagonal_attention_mass = torch.diagonal(
                attention_weights_cpu,
                dim1=-2,
                dim2=-1,
            )
            diagnostics.update(
                {
                    "attention_shape": list(attention_weights_cpu.shape),
                    "attention_mean": float(attention_weights_cpu.mean().item()),
                    "attention_std": float(
                        attention_weights_cpu.std(unbiased=False).item()
                    ),
                    "attention_min": float(attention_weights_cpu.min().item()),
                    "attention_max": float(attention_weights_cpu.max().item()),
                    "attention_entropy_mean": float(attention_entropy.mean().item()),
                    "diagonal_attention_mass_mean": float(
                        diagonal_attention_mass.mean().item()
                    ),
                }
            )

        trace_tensors: dict[str, torch.Tensor] = {}
        if collect_trace_tensors:
            trace_tensors = {
                "attention_input": _trace_to_cpu(attention_input),
                "attention_output": _trace_to_cpu(attention_output),
                "attention_update": _trace_to_cpu(attention_update),
                "hidden_states_after_attention": _trace_to_cpu(hidden_states_after_attention),
                "feed_forward_input": _trace_to_cpu(feed_forward_input),
                "feed_forward_output": _trace_to_cpu(feed_forward_output),
                "hidden_states_after_block": _trace_to_cpu(hidden_states_after_block),
                "attention_side_adapter_alpha": _trace_to_cpu(
                    self.attention_side_adapter_alpha
                ),
            }
            trace_tensors.update(adapter_output.trace_tensors)
            if attention_weights_cpu is not None:
                trace_tensors["attention_weights"] = attention_weights_cpu

        return GroupTokenTransformerOutput(
            hidden_states=hidden_states_after_block,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )


class PriorStockV3OHLCV124GroupTokenMixerAttentionSideAdapter(
    PriorStockV3OHLCV124GroupTokenMixer
):
    """Group-token mixer with A-conditioned side adapters after cross-group attention."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        attention_side_adapter_config: AttentionSideAdapterConfig,
    ) -> None:
        """Construct the attention-side adapter variant from config values."""

        super().__init__(experiment_config)
        self._attention_side_adapter_config = attention_side_adapter_config
        group_token_dim = experiment_config.model.technical_indicator_group_token_dim
        self.group_token_transformer_layers = nn.ModuleList(
            [
                GroupTokenTransformerBlockWithAttentionSideAdapter(
                    token_dim=group_token_dim,
                    a_context_dim=experiment_config.model.d_model,
                    feed_forward_dim=(
                        experiment_config.model.technical_indicator_group_token_mixer_feed_forward_dim
                    ),
                    head_count=(
                        experiment_config.model.technical_indicator_group_token_mixer_head_count
                    ),
                    dropout_probability=(
                        experiment_config.model.technical_indicator_group_token_mixer_dropout_probability
                    ),
                    adapter_config=attention_side_adapter_config,
                )
                for _ in range(
                    experiment_config.model.technical_indicator_group_token_mixer_layer_count
                )
            ]
        )

    def forward(
        self,
        price_features: torch.Tensor,
        technical_indicator_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        has_news: torch.Tensor,
        collect_trace_tensors: bool = False,
    ) -> PriorStockModelOutput:
        """Run one forward pass with attention-side adapters inside the B-J mixer."""

        del news_embeddings
        del has_news

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        batch_size, sequence_length, feature_count = technical_indicator_features.shape
        if sequence_length != self._experiment_config.data.lookback_window_size:
            raise ValueError("technical_indicator_features has an unexpected sequence length.")
        if feature_count != self._experiment_config.model.technical_indicator_feature_count:
            raise ValueError("technical_indicator_features has an unexpected final dimension.")

        grouped_features = self._split_feature_groups(technical_indicator_features)
        base_branch_hidden_states = self.a_branch_encoder(grouped_features["A"])
        group_token_hidden_states = torch.stack(
            [
                self.incremental_group_token_projectors[group_name](grouped_features[group_name])
                for group_name in B_TO_J_GROUP_NAMES
            ],
            dim=2,
        )
        group_token_hidden_states_with_identity = (
            group_token_hidden_states + self.group_identity_embedding
        )
        flattened_batch_time_group_tokens = group_token_hidden_states_with_identity.reshape(
            batch_size * sequence_length,
            self._incremental_group_count,
            self._experiment_config.model.technical_indicator_group_token_dim,
        )
        flattened_a_context_hidden_states = base_branch_hidden_states.reshape(
            batch_size * sequence_length,
            self._experiment_config.model.d_model,
        )
        expanded_a_context_hidden_states = flattened_a_context_hidden_states.unsqueeze(1).expand(
            -1,
            self._incremental_group_count,
            -1,
        )
        flattened_group_identity_features = self.group_identity_embedding.reshape(
            1,
            self._incremental_group_count,
            self._experiment_config.model.technical_indicator_group_token_dim,
        )

        group_token_layer_diagnostics: dict[str, dict] = {}
        trace_tensors: dict[str, torch.Tensor] = {}
        group_token_mixer_hidden_states = flattened_batch_time_group_tokens
        for layer_index, group_token_transformer_layer in enumerate(
            self.group_token_transformer_layers
        ):
            group_token_layer_output = group_token_transformer_layer(
                group_token_hidden_states=group_token_mixer_hidden_states,
                a_context_hidden_states=expanded_a_context_hidden_states,
                group_identity_features=flattened_group_identity_features,
                collect_trace_tensors=collect_trace_tensors,
            )
            group_token_mixer_hidden_states = group_token_layer_output.hidden_states
            group_token_layer_diagnostics[
                f"group_token_mixer_layer_{layer_index + 1}"
            ] = group_token_layer_output.diagnostics
            if collect_trace_tensors:
                for trace_name, trace_tensor in group_token_layer_output.trace_tensors.items():
                    trace_tensors[
                        f"group_token_mixer_layer_{layer_index + 1}/{trace_name}"
                    ] = trace_tensor

        group_token_mixer_hidden_states = group_token_mixer_hidden_states.reshape(
            batch_size,
            sequence_length,
            self._incremental_group_count,
            self._experiment_config.model.technical_indicator_group_token_dim,
        )
        flattened_group_token_hidden_states = group_token_mixer_hidden_states.reshape(
            batch_size,
            sequence_length,
            self._incremental_group_count
            * self._experiment_config.model.technical_indicator_group_token_dim,
        )
        incremental_branch_hidden_states = self.incremental_group_token_readout(
            flattened_group_token_hidden_states
        )
        fusion_gate_values = self.fusion_gate(
            torch.cat([base_branch_hidden_states, incremental_branch_hidden_states], dim=-1)
        )
        fused_indicator_hidden_states = base_branch_hidden_states + (
            fusion_gate_values * incremental_branch_hidden_states
        )
        hidden_states = self.input_layer_norm(
            fused_indicator_hidden_states + self.positional_embedding
        )
        temporal_attention_mask = self._build_temporal_attention_mask(
            sequence_length,
            technical_indicator_features.device,
        )

        if collect_trace_tensors:
            trace_tensors["price_features"] = _trace_to_cpu(price_features)
            trace_tensors["ohlcv124_features"] = _trace_to_cpu(technical_indicator_features)
            trace_tensors["a_branch_hidden_states"] = _trace_to_cpu(base_branch_hidden_states)
            trace_tensors["expanded_a_context_hidden_states"] = _trace_to_cpu(
                expanded_a_context_hidden_states
            )
            trace_tensors["group_token_hidden_states"] = _trace_to_cpu(group_token_hidden_states)
            trace_tensors["group_identity_embedding"] = _trace_to_cpu(
                self.group_identity_embedding
            )
            trace_tensors["group_token_hidden_states_with_identity"] = _trace_to_cpu(
                group_token_hidden_states_with_identity
            )
            trace_tensors["group_token_mixer_hidden_states"] = _trace_to_cpu(
                group_token_mixer_hidden_states
            )
            trace_tensors["flattened_group_token_hidden_states"] = _trace_to_cpu(
                flattened_group_token_hidden_states
            )
            trace_tensors["incremental_branch_hidden_states"] = _trace_to_cpu(
                incremental_branch_hidden_states
            )
            trace_tensors["fusion_gate_values"] = _trace_to_cpu(fusion_gate_values)
            trace_tensors["fused_indicator_hidden_states"] = _trace_to_cpu(
                fused_indicator_hidden_states
            )
            trace_tensors["main_indicator_hidden_states"] = _trace_to_cpu(hidden_states)

        temporal_layer_diagnostics: dict[str, dict] = {}
        for layer_index, transformer_layer in enumerate(self.transformer_layers):
            layer_output = transformer_layer(
                hidden_states=hidden_states,
                temporal_attention_mask=temporal_attention_mask,
                collect_trace_tensors=collect_trace_tensors,
            )
            hidden_states = layer_output.hidden_states
            temporal_layer_diagnostics[f"layer_{layer_index + 1}"] = layer_output.diagnostics
            if collect_trace_tensors:
                for trace_name, trace_tensor in layer_output.trace_tensors.items():
                    trace_tensors[f"layer_{layer_index + 1}/{trace_name}"] = trace_tensor

        final_hidden_states = self.final_layer_norm(hidden_states)
        technical_indicator_representation = final_hidden_states[:, -1, :]
        logits = self.classifier(technical_indicator_representation)

        if collect_trace_tensors:
            trace_tensors["final_hidden_states"] = _trace_to_cpu(final_hidden_states)
            trace_tensors["technical_indicator_representation"] = _trace_to_cpu(
                technical_indicator_representation
            )
            trace_tensors["logits"] = _trace_to_cpu(logits)

        diagnostics = {
            "price_features_shape": list(price_features.shape),
            "ohlcv124_features_shape": list(technical_indicator_features.shape),
            "a_branch_hidden_states_shape": list(base_branch_hidden_states.shape),
            "expanded_a_context_hidden_states_shape": list(
                expanded_a_context_hidden_states.shape
            ),
            "group_token_hidden_states_shape": list(group_token_hidden_states.shape),
            "group_identity_embedding_shape": list(self.group_identity_embedding.shape),
            "group_token_mixer_hidden_states_shape": list(group_token_mixer_hidden_states.shape),
            "flattened_group_token_hidden_states_shape": list(
                flattened_group_token_hidden_states.shape
            ),
            "incremental_branch_hidden_states_shape": list(incremental_branch_hidden_states.shape),
            "fusion_gate_values_shape": list(fusion_gate_values.shape),
            "fusion_gate_mean": float(fusion_gate_values.detach().mean().cpu().item()),
            "fusion_gate_std": float(
                fusion_gate_values.detach().std(unbiased=False).cpu().item()
            ),
            "main_indicator_hidden_states_shape": list(hidden_states.shape),
            "technical_indicator_representation_shape": list(
                technical_indicator_representation.shape
            ),
            "final_hidden_states_shape": list(final_hidden_states.shape),
            "logits_shape": list(logits.shape),
            "group_token_mixer_layers": group_token_layer_diagnostics,
            "layers": temporal_layer_diagnostics,
        }
        return PriorStockModelOutput(
            logits=logits,
            tech_gate_regularization_loss=torch.zeros(
                (),
                device=technical_indicator_features.device,
            ),
            news_gate_regularization_loss=torch.zeros(
                (),
                device=technical_indicator_features.device,
            ),
            technical_indicator_representation=technical_indicator_representation,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
