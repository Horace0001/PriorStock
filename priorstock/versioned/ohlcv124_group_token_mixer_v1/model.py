"""OHLCV-124 grouped-fusion model with a B-J semantic group-token mixer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from priorstock.config import ExperimentConfig
from priorstock.versioned.indicator_main_only_v1.modules import TransformerBlockIndicatorMainOnly
from priorstock.versioned.ohlcv124_group_fusion_v1.feature_schema import (
    B_TO_J_GROUP_NAMES,
    GROUP_NAME_TO_FEATURE_NAMES,
)


@dataclass(frozen=True)
class PriorStockModelOutput:
    """One forward pass output bundle."""

    logits: torch.Tensor
    tech_gate_regularization_loss: torch.Tensor
    news_gate_regularization_loss: torch.Tensor
    technical_indicator_representation: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class GroupTokenTransformerOutput:
    """Output bundle for one B-J group-token transformer block."""

    hidden_states: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class FeatureGroupTokenProjection(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear projection for one B-J group token."""

    def __init__(self, input_dim: int, hidden_dim: int, token_dim: int) -> None:
        """Construct one group-local token projector."""

        super().__init__()
        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, token_dim)

    def forward(self, group_features: torch.Tensor) -> torch.Tensor:
        """Project one feature group into one semantic token per time step."""

        hidden_features = self.input_projection(self.input_layer_norm(group_features))
        return self.output_projection(self.activation(hidden_features))


class GroupTokenTransformerBlock(nn.Module):
    """Pre-LN transformer block over the fixed B-J semantic group-token axis."""

    def __init__(
        self,
        token_dim: int,
        feed_forward_dim: int,
        head_count: int,
        dropout_probability: float,
    ) -> None:
        """Create one self-attention block for cross-group token mixing."""

        super().__init__()
        self.attention_layer_norm = nn.LayerNorm(token_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=head_count,
            dropout=dropout_probability,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout_probability)
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
        collect_trace_tensors: bool,
    ) -> GroupTokenTransformerOutput:
        """Mix semantic group tokens and return the updated token states."""

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
            attention_weights_cpu = attention_weights.detach().to(device="cpu", dtype=torch.float32)
            del attention_weights
        else:
            attention_weights_cpu = None

        hidden_states_after_attention = group_token_hidden_states + self.attention_dropout(
            attention_output
        )
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

        return GroupTokenTransformerOutput(
            hidden_states=hidden_states_after_block,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )


class PriorStockV3OHLCV124GroupTokenMixer(nn.Module):
    """PriorStock variant with A anchoring and B-J semantic group-token mixing."""

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        """Construct the group-token mixer OHLCV-124 architecture from config values."""

        super().__init__()
        self._experiment_config = experiment_config
        self._group_feature_dims = {
            group_name: len(feature_names)
            for group_name, feature_names in GROUP_NAME_TO_FEATURE_NAMES.items()
        }
        self._group_slices = self._build_group_slices()
        self._incremental_group_count = len(B_TO_J_GROUP_NAMES)
        group_token_dim = experiment_config.model.technical_indicator_group_token_dim

        a_group_dim = self._group_feature_dims["A"]
        self.a_branch_encoder = nn.Sequential(
            nn.LayerNorm(a_group_dim),
            nn.Linear(a_group_dim, experiment_config.model.technical_indicator_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(
                experiment_config.model.technical_indicator_mlp_hidden_dim,
                experiment_config.model.d_model,
            ),
            nn.LayerNorm(experiment_config.model.d_model),
        )

        self.incremental_group_token_projectors = nn.ModuleDict(
            {
                group_name: FeatureGroupTokenProjection(
                    input_dim=self._group_feature_dims[group_name],
                    hidden_dim=experiment_config.model.technical_indicator_mlp_hidden_dim,
                    token_dim=group_token_dim,
                )
                for group_name in B_TO_J_GROUP_NAMES
            }
        )
        if experiment_config.model.use_technical_indicator_group_identity_embedding:
            self.group_identity_embedding = nn.Parameter(
                torch.randn(1, 1, self._incremental_group_count, group_token_dim)
                * experiment_config.model.positional_embedding_initialization_std
            )
        else:
            self.register_buffer(
                "group_identity_embedding",
                torch.zeros(1, 1, self._incremental_group_count, group_token_dim),
                persistent=False,
            )
        self.group_token_transformer_layers = nn.ModuleList(
            [
                GroupTokenTransformerBlock(
                    token_dim=group_token_dim,
                    feed_forward_dim=(
                        experiment_config.model.technical_indicator_group_token_mixer_feed_forward_dim
                    ),
                    head_count=(
                        experiment_config.model.technical_indicator_group_token_mixer_head_count
                    ),
                    dropout_probability=(
                        experiment_config.model.technical_indicator_group_token_mixer_dropout_probability
                    ),
                )
                for _ in range(
                    experiment_config.model.technical_indicator_group_token_mixer_layer_count
                )
            ]
        )
        self.incremental_group_token_readout = nn.Sequential(
            nn.Linear(self._incremental_group_count * group_token_dim, experiment_config.model.d_model),
            nn.LayerNorm(experiment_config.model.d_model),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(experiment_config.model.d_model * 2, experiment_config.model.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(experiment_config.model.gate_hidden_dim, experiment_config.model.d_model),
            nn.Sigmoid(),
        )

        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                experiment_config.data.lookback_window_size,
                experiment_config.model.d_model,
            )
            * experiment_config.model.positional_embedding_initialization_std
        )
        self.input_layer_norm = nn.LayerNorm(experiment_config.model.d_model)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlockIndicatorMainOnly(
                    d_model=experiment_config.model.d_model,
                    d_ff=experiment_config.model.d_ff,
                    num_heads=experiment_config.model.num_heads,
                    dropout_probability=experiment_config.model.dropout_probability,
                )
                for _ in range(experiment_config.model.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(experiment_config.model.d_model)
        self.classifier = nn.Linear(experiment_config.model.d_model, experiment_config.model.num_classes)

    def _build_group_slices(self) -> dict[str, slice]:
        """Build deterministic feature slices from the shared OHLCV-124 schema order."""

        group_slices: dict[str, slice] = {}
        start_index = 0
        for group_name, group_feature_names in GROUP_NAME_TO_FEATURE_NAMES.items():
            end_index = start_index + len(group_feature_names)
            group_slices[group_name] = slice(start_index, end_index)
            start_index = end_index
        if start_index != self._experiment_config.model.technical_indicator_feature_count:
            raise ValueError("OHLCV-124 group schema does not match configured feature count.")
        return group_slices

    def _build_temporal_attention_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Build the boolean upper-triangular mask expected by PyTorch attention."""

        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _split_feature_groups(self, technical_indicator_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split one OHLCV-124 tensor into its named A-J feature groups."""

        return {
            group_name: technical_indicator_features[:, :, group_slice]
            for group_name, group_slice in self._group_slices.items()
        }

    def forward(
        self,
        price_features: torch.Tensor,
        technical_indicator_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        has_news: torch.Tensor,
        collect_trace_tensors: bool = False,
    ) -> PriorStockModelOutput:
        """Run one forward pass through the group-token mixer grouped-fusion model."""

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
        group_token_hidden_states_with_identity = group_token_hidden_states + self.group_identity_embedding
        flattened_batch_time_group_tokens = group_token_hidden_states_with_identity.reshape(
            batch_size * sequence_length,
            self._incremental_group_count,
            self._experiment_config.model.technical_indicator_group_token_dim,
        )

        group_token_layer_diagnostics: dict[str, dict] = {}
        trace_tensors: dict[str, torch.Tensor] = {}
        group_token_mixer_hidden_states = flattened_batch_time_group_tokens
        for layer_index, group_token_transformer_layer in enumerate(self.group_token_transformer_layers):
            group_token_layer_output = group_token_transformer_layer(
                group_token_hidden_states=group_token_mixer_hidden_states,
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
        hidden_states = self.input_layer_norm(fused_indicator_hidden_states + self.positional_embedding)
        temporal_attention_mask = self._build_temporal_attention_mask(
            sequence_length,
            technical_indicator_features.device,
        )

        if collect_trace_tensors:
            trace_tensors["price_features"] = _trace_to_cpu(price_features)
            trace_tensors["ohlcv124_features"] = _trace_to_cpu(technical_indicator_features)
            trace_tensors["a_branch_hidden_states"] = _trace_to_cpu(base_branch_hidden_states)
            trace_tensors["group_token_hidden_states"] = _trace_to_cpu(group_token_hidden_states)
            trace_tensors["group_identity_embedding"] = _trace_to_cpu(self.group_identity_embedding)
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
            trace_tensors["fused_indicator_hidden_states"] = _trace_to_cpu(fused_indicator_hidden_states)
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
            "group_token_hidden_states_shape": list(group_token_hidden_states.shape),
            "group_identity_embedding_shape": list(self.group_identity_embedding.shape),
            "group_token_mixer_hidden_states_shape": list(group_token_mixer_hidden_states.shape),
            "flattened_group_token_hidden_states_shape": list(
                flattened_group_token_hidden_states.shape
            ),
            "incremental_branch_hidden_states_shape": list(incremental_branch_hidden_states.shape),
            "fusion_gate_values_shape": list(fusion_gate_values.shape),
            "fusion_gate_mean": float(fusion_gate_values.detach().mean().cpu().item()),
            "fusion_gate_std": float(fusion_gate_values.detach().std(unbiased=False).cpu().item()),
            "main_indicator_hidden_states_shape": list(hidden_states.shape),
            "technical_indicator_representation_shape": list(technical_indicator_representation.shape),
            "final_hidden_states_shape": list(final_hidden_states.shape),
            "logits_shape": list(logits.shape),
            "group_token_mixer_layers": group_token_layer_diagnostics,
            "layers": temporal_layer_diagnostics,
        }
        return PriorStockModelOutput(
            logits=logits,
            tech_gate_regularization_loss=torch.zeros((), device=technical_indicator_features.device),
            news_gate_regularization_loss=torch.zeros((), device=technical_indicator_features.device),
            technical_indicator_representation=technical_indicator_representation,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
