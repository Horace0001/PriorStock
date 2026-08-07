"""Factor-attention concat classifier over a frozen wide-adapter base model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from priorstock.config import ExperimentConfig
from priorstock.versioned.ohlcv124_group_token_mixer_attention_side_adapter_v1.model import (
    AttentionSideAdapterConfig,
    PriorStockV3OHLCV124GroupTokenMixerAttentionSideAdapter,
)
from priorstock.versioned.ohlcv124_group_token_mixer_v1.model import PriorStockModelOutput


BASE_ATTENTION_SIDE_ADAPTER_VARIANT_NAME = "ohlcv124_group_token_mixer_attention_side_adapter_v1"
FACTOR_PROJECTION_MODE_MLP_THEN_ADD_RANK = "mlp_then_add_rank"
FACTOR_PROJECTION_MODE_RANK_CONCAT_LINEAR = "rank_concat_linear"


@dataclass(frozen=True)
class FactorConcatLogitClassifierConfig:
    """Hyperparameters for the factor concat-logit classifier head."""

    base_checkpoint_file_path: str
    factor_embedding_cache_directory: str
    factor_count: int
    factor_input_dim: int
    factor_projector_hidden_dim: int
    classifier_hidden_dimensions: tuple[int, ...]
    attention_head_count: int
    dropout_probability: float
    base_framework_variant_name: str
    should_use_factor_transformer_ffn: bool
    factor_transformer_ffn_hidden_dim: int
    factor_transformer_ffn_dropout_probability: float
    should_train_base_model: bool = False
    base_frozen_epoch_count: int = 0
    factor_learning_rate: float = 0.0
    base_tail_learning_rate: float = 0.0
    base_classifier_learning_rate: float = 0.0
    base_trainable_scope: str = "tail"
    should_freeze_fusion_gate_during_joint_training: bool = True
    allow_partial_base_checkpoint_loading: bool = False
    soft_mcc_loss_weight: float = 0.0
    soft_metric_loss_epsilon: float = 1.0e-8
    factor_projection_mode: str = FACTOR_PROJECTION_MODE_MLP_THEN_ADD_RANK


class FactorPostAttentionFeedForwardBlock(nn.Module):
    """Pre-LN residual FFN block for refining the factor attention context."""

    def __init__(
        self,
        technical_dimension: int,
        feed_forward_hidden_dimension: int,
        dropout_probability: float,
    ) -> None:
        """Build a transformer-style feed-forward refinement block."""

        super().__init__()
        if technical_dimension <= 0:
            raise ValueError("technical_dimension must be positive.")
        if feed_forward_hidden_dimension <= 0:
            raise ValueError("feed_forward_hidden_dimension must be positive.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1).")
        self.layer_norm = nn.LayerNorm(technical_dimension)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(technical_dimension, feed_forward_hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(feed_forward_hidden_dimension, technical_dimension),
            nn.Dropout(dropout_probability),
        )

    def forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Return the residual-refined factor attention output."""

        return attention_output + self.feed_forward_network(self.layer_norm(attention_output))


class FactorConcatLogitClassifierHead(nn.Module):
    """Classify from normalized technical representation, factor context, and base logit."""

    def __init__(
        self,
        technical_dimension: int,
        hidden_dimensions: tuple[int, ...],
        dropout_probability: float,
    ) -> None:
        """Build the concat classifier MLP."""

        super().__init__()
        if not hidden_dimensions:
            raise ValueError("hidden_dimensions must contain at least one layer size.")
        self.technical_layer_norm = nn.LayerNorm(technical_dimension)
        self.factor_layer_norm = nn.LayerNorm(technical_dimension)
        classifier_input_dimension = (technical_dimension * 2) + 1
        classifier_layers: list[nn.Module] = []
        previous_dimension = classifier_input_dimension
        for hidden_dimension in hidden_dimensions:
            classifier_layers.extend(
                [
                    nn.Linear(previous_dimension, hidden_dimension),
                    nn.GELU(),
                    nn.Dropout(dropout_probability),
                ]
            )
            previous_dimension = hidden_dimension
        classifier_layers.append(nn.Linear(previous_dimension, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(
        self,
        technical_representation: torch.Tensor,
        factor_context: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits from concatenated normalized features and frozen base logits."""

        classifier_input = torch.cat(
            [
                self.technical_layer_norm(technical_representation),
                self.factor_layer_norm(factor_context),
                base_logits,
            ],
            dim=1,
        )
        return self.classifier(classifier_input)


class FactorConcatLogitClassifierV1(nn.Module):
    """Use factor attention plus frozen base logits as inputs to a small classifier."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        attention_side_adapter_config: AttentionSideAdapterConfig,
        factor_classifier_config: FactorConcatLogitClassifierConfig,
    ) -> None:
        """Build the frozen base model and trainable factor concat-logit classifier."""

        super().__init__()
        self.config = factor_classifier_config
        self.base_model = self._build_base_model(
            experiment_config=experiment_config,
            attention_side_adapter_config=attention_side_adapter_config,
            factor_classifier_config=factor_classifier_config,
        )
        technical_dimension = experiment_config.model.d_model
        if factor_classifier_config.factor_projection_mode == FACTOR_PROJECTION_MODE_RANK_CONCAT_LINEAR:
            self.factor_projector = nn.Sequential(
                nn.Linear(factor_classifier_config.factor_input_dim + technical_dimension, technical_dimension),
                nn.LayerNorm(technical_dimension),
            )
        else:
            self.factor_projector = nn.Sequential(
                nn.Linear(
                    factor_classifier_config.factor_input_dim,
                    factor_classifier_config.factor_projector_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(factor_classifier_config.dropout_probability),
                nn.Linear(factor_classifier_config.factor_projector_hidden_dim, technical_dimension),
                nn.LayerNorm(technical_dimension),
            )
        self.rank_embedding = nn.Parameter(torch.zeros(factor_classifier_config.factor_count, technical_dimension))
        self.query_layer_norm = nn.LayerNorm(technical_dimension)
        self.factor_attention = nn.MultiheadAttention(
            embed_dim=technical_dimension,
            num_heads=factor_classifier_config.attention_head_count,
            dropout=factor_classifier_config.dropout_probability,
            batch_first=True,
        )
        self.factor_transformer_ffn = (
            FactorPostAttentionFeedForwardBlock(
                technical_dimension=technical_dimension,
                feed_forward_hidden_dimension=factor_classifier_config.factor_transformer_ffn_hidden_dim,
                dropout_probability=factor_classifier_config.factor_transformer_ffn_dropout_probability,
            )
            if factor_classifier_config.should_use_factor_transformer_ffn
            else None
        )
        self.classifier_head = FactorConcatLogitClassifierHead(
            technical_dimension=technical_dimension,
            hidden_dimensions=factor_classifier_config.classifier_hidden_dimensions,
            dropout_probability=factor_classifier_config.dropout_probability,
        )

    def _build_base_model(
        self,
        experiment_config: ExperimentConfig,
        attention_side_adapter_config: AttentionSideAdapterConfig,
        factor_classifier_config: FactorConcatLogitClassifierConfig,
    ) -> nn.Module:
        """Instantiate the requested frozen base architecture for factor classification."""

        if factor_classifier_config.base_framework_variant_name != BASE_ATTENTION_SIDE_ADAPTER_VARIANT_NAME:
            raise ValueError(
                "base_framework_variant_name must be "
                f"'{BASE_ATTENTION_SIDE_ADAPTER_VARIANT_NAME}'."
            )
        return PriorStockV3OHLCV124GroupTokenMixerAttentionSideAdapter(
            experiment_config=experiment_config,
            attention_side_adapter_config=attention_side_adapter_config,
        )

    def freeze_base_model(self) -> None:
        """Freeze all wide-adapter base-model parameters."""

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        price_features: torch.Tensor,
        technical_indicator_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        has_news: torch.Tensor,
        factor_embeddings: torch.Tensor,
        has_factors: torch.Tensor,
        collect_trace_tensors: bool,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> PriorStockModelOutput:
        """Return concat-classifier logits with no-factor samples falling back to base logits."""

        del news_embeddings
        del has_news
        should_use_base_output_cache = (
            not self.config.should_train_base_model and sample_ids is not None
        )
        base_output = (
            self._load_cached_base_output(sample_ids, price_features.device)
            if should_use_base_output_cache and sample_ids is not None
            else None
        )
        if base_output is None and self.config.should_train_base_model:
            base_output = self.base_model(
                price_features=price_features,
                technical_indicator_features=technical_indicator_features,
                news_embeddings=torch.empty(
                    price_features.shape[0],
                    price_features.shape[1],
                    1,
                    device=price_features.device,
                ),
                has_news=torch.zeros(
                    price_features.shape[0],
                    price_features.shape[1],
                    1,
                    device=price_features.device,
                ),
                collect_trace_tensors=collect_trace_tensors,
            )
        elif base_output is None:
            self.base_model.eval()
            with torch.no_grad():
                base_output = self.base_model(
                    price_features=price_features,
                    technical_indicator_features=technical_indicator_features,
                    news_embeddings=torch.empty(
                        price_features.shape[0],
                        price_features.shape[1],
                        1,
                        device=price_features.device,
                    ),
                    has_news=torch.zeros(
                        price_features.shape[0],
                        price_features.shape[1],
                        1,
                        device=price_features.device,
                    ),
                    collect_trace_tensors=collect_trace_tensors,
                )
        if should_use_base_output_cache and sample_ids is not None:
            self._store_cached_base_output(sample_ids, base_output)
        technical_representation = base_output.technical_indicator_representation
        has_factors_bool = has_factors.to(dtype=torch.bool)
        has_any_factor = has_factors_bool.any(dim=1, keepdim=True)
        factor_tokens = self._project_factor_tokens(factor_embeddings)
        key_padding_mask = ~has_factors_bool
        safe_key_padding_mask = torch.where(
            has_any_factor,
            key_padding_mask,
            torch.zeros_like(key_padding_mask),
        )
        query = self.query_layer_norm(technical_representation).unsqueeze(1)
        attention_output, attention_weights = self.factor_attention(
            query=query,
            key=factor_tokens,
            value=factor_tokens,
            key_padding_mask=safe_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attention_context = attention_output.squeeze(1)
        factor_context = (
            self.factor_transformer_ffn(attention_context)
            if self.factor_transformer_ffn is not None
            else attention_context
        )
        classifier_logits = self.classifier_head(
            technical_representation=technical_representation,
            factor_context=factor_context,
            base_logits=base_output.logits,
        )
        final_logits = torch.where(has_any_factor, classifier_logits, base_output.logits)
        delta_logit = final_logits - base_output.logits
        diagnostics = self._build_diagnostics(
            base_output=base_output,
            final_logits=final_logits,
            delta_logit=delta_logit,
            attention_weights=attention_weights,
            attention_context=attention_context,
            factor_context=factor_context,
            has_factors=has_factors_bool,
            has_any_factor=has_any_factor,
        )
        trace_tensors = dict(base_output.trace_tensors)
        trace_tensors.update(
            {
                "factor_enhanced/base_logits": base_output.logits.detach().cpu(),
                "factor_enhanced/logits": final_logits.detach().cpu(),
                "factor_enhanced/delta_logit": delta_logit.detach().cpu(),
                "factor_enhanced/attention_weights": attention_weights.detach().cpu(),
                "factor_enhanced/attention_context": attention_context.detach().cpu(),
                "factor_enhanced/factor_context": factor_context.detach().cpu(),
                "factor_enhanced/has_any_factor": has_any_factor.detach().cpu(),
            }
        )
        return PriorStockModelOutput(
            logits=final_logits,
            tech_gate_regularization_loss=base_output.tech_gate_regularization_loss,
            news_gate_regularization_loss=base_output.news_gate_regularization_loss,
            technical_indicator_representation=technical_representation,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )

    def _load_cached_base_output(
        self,
        sample_ids: list[str] | tuple[str, ...],
        device: torch.device,
    ) -> PriorStockModelOutput | None:
        """Load frozen base representations when every sample is already cached."""

        cache = getattr(self, "_technical_only_base_output_cache", {})
        normalized_sample_ids = [str(sample_id) for sample_id in sample_ids]
        if any(sample_id not in cache for sample_id in normalized_sample_ids):
            return None
        technical_representation = torch.stack(
            [cache[sample_id][0] for sample_id in normalized_sample_ids]
        ).to(device)
        base_logits = torch.stack(
            [cache[sample_id][1] for sample_id in normalized_sample_ids]
        ).to(device)
        zero_loss = technical_representation.new_zeros(())
        return PriorStockModelOutput(
            logits=base_logits,
            tech_gate_regularization_loss=zero_loss,
            news_gate_regularization_loss=zero_loss,
            technical_indicator_representation=technical_representation,
            diagnostics={},
            trace_tensors={},
        )

    def _store_cached_base_output(
        self,
        sample_ids: list[str] | tuple[str, ...],
        base_output: PriorStockModelOutput,
    ) -> None:
        """Cache deterministic frozen-base outputs on CPU by stable sample ID."""

        cache = getattr(self, "_technical_only_base_output_cache", None)
        if cache is None:
            cache = {}
            self._technical_only_base_output_cache = cache
        for sample_index, sample_id in enumerate(sample_ids):
            normalized_sample_id = str(sample_id)
            if normalized_sample_id in cache:
                continue
            cache[normalized_sample_id] = (
                base_output.technical_indicator_representation[sample_index].detach().cpu(),
                base_output.logits[sample_index].detach().cpu(),
            )

    def _project_factor_tokens(self, factor_embeddings: torch.Tensor) -> torch.Tensor:
        """Project factor embeddings while applying the configured rank treatment."""

        rank_tokens = self.rank_embedding.unsqueeze(0)
        if self.config.factor_projection_mode == FACTOR_PROJECTION_MODE_RANK_CONCAT_LINEAR:
            expanded_rank_tokens = rank_tokens.expand(factor_embeddings.shape[0], -1, -1)
            return self.factor_projector(torch.cat([factor_embeddings, expanded_rank_tokens], dim=-1))
        return self.factor_projector(factor_embeddings) + rank_tokens

    def _build_diagnostics(
        self,
        base_output: PriorStockModelOutput,
        final_logits: torch.Tensor,
        delta_logit: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_context: torch.Tensor,
        factor_context: torch.Tensor,
        has_factors: torch.Tensor,
        has_any_factor: torch.Tensor,
    ) -> dict[str, Any]:
        """Aggregate scalar factor-attention diagnostics for logging."""

        averaged_attention_weights = attention_weights.mean(dim=1).squeeze(1)
        attention_floor = torch.finfo(averaged_attention_weights.dtype).tiny
        attention_entropy = -(
            averaged_attention_weights.clamp_min(attention_floor)
            * averaged_attention_weights.clamp_min(attention_floor).log()
        ).sum(dim=1)
        valid_sample_mask = has_any_factor.squeeze(1)
        no_factor_sample_mask = ~valid_sample_mask
        factor_context_delta = factor_context - attention_context
        diagnostics: dict[str, Any] = dict(base_output.diagnostics)
        diagnostics["factor_enhanced_classifier"] = {
            "base_logit_mean": _masked_mean(base_output.logits.squeeze(1), valid_sample_mask),
            "logit_mean": _masked_mean(final_logits.squeeze(1), valid_sample_mask),
            "delta_logit_mean": _masked_mean(delta_logit.squeeze(1), valid_sample_mask),
            "delta_logit_abs_mean": _masked_mean(delta_logit.squeeze(1).abs(), valid_sample_mask),
            "delta_logit_no_factor_abs_mean": _masked_mean(delta_logit.squeeze(1).abs(), no_factor_sample_mask),
            "alpha": 0.0,
            "has_factor_ratio": float(valid_sample_mask.to(dtype=torch.float32).mean().detach().cpu().item()),
            "attention_entropy_mean": _masked_mean(attention_entropy, valid_sample_mask),
            "attention_max_mean": _masked_mean(averaged_attention_weights.max(dim=1).values, valid_sample_mask),
            "attention_context_std": _masked_std(attention_context, valid_sample_mask),
            "factor_context_std": _masked_std(factor_context, valid_sample_mask),
            "factor_context_ffn_delta_abs_mean": _masked_mean(
                factor_context_delta.abs().mean(dim=1),
                valid_sample_mask,
            ),
            "attention_weights_per_factor": [
                _masked_mean(averaged_attention_weights[:, factor_index], valid_sample_mask)
                for factor_index in range(self.config.factor_count)
            ],
            "has_factors_per_factor": [
                float(has_factors[:, factor_index].to(dtype=torch.float32).mean().detach().cpu().item())
                for factor_index in range(self.config.factor_count)
            ],
        }
        return diagnostics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return a CPU float mean over a boolean mask, or zero when empty."""

    selected_values = values.detach()[mask.detach()]
    if selected_values.numel() == 0:
        return 0.0
    return float(selected_values.mean().cpu().item())


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return a CPU float standard deviation over rows selected by a boolean mask."""

    selected_values = values.detach()[mask.detach()]
    if selected_values.numel() == 0:
        return 0.0
    return float(selected_values.std(unbiased=False).cpu().item())
