"""PriorStock v3 model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from priorstock.config import ExperimentConfig
from priorstock.models.modules import TransformerBlockWithGating


@dataclass(frozen=True)
class PriorStockModelOutput:
    """One forward pass output bundle."""

    logits: torch.Tensor
    tech_gate_regularization_loss: torch.Tensor
    news_gate_regularization_loss: torch.Tensor
    technical_indicator_representation: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class PriorStockV3(nn.Module):
    """Document-faithful gated multimodal transformer for stock movement classification."""

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        """Construct the full PriorStock architecture from validated config values."""

        super().__init__()
        self._experiment_config = experiment_config
        self._use_raw_technical_indicator_branch = experiment_config.model.use_raw_technical_indicator_branch
        if not self._use_raw_technical_indicator_branch:
            raise ValueError("PriorStockV3 expects model.use_raw_technical_indicator_branch to be enabled.")
        news_embedding_dim = experiment_config.news_encoder.output_embedding_dim
        self._use_layer_specific_text_projections = experiment_config.model.use_layer_specific_text_projections
        self._disable_outer_no_news_placeholder_embedding = (
            experiment_config.model.disable_outer_no_news_placeholder_embedding
        )
        self._use_shared_text_bottlenecks = experiment_config.model.use_shared_text_bottlenecks
        self.price_projection = nn.Linear(
            experiment_config.model.input_price_feature_count,
            experiment_config.model.d_model,
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
        self.technical_indicator_pre_layer_norm = (
            nn.LayerNorm(experiment_config.model.technical_indicator_feature_count)
            if experiment_config.model.use_technical_indicator_pre_layer_norm
            else nn.Identity()
        )
        self.technical_indicator_encoder = nn.Sequential(
            nn.Linear(
                experiment_config.model.technical_indicator_feature_count,
                experiment_config.model.technical_indicator_mlp_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                experiment_config.model.technical_indicator_mlp_hidden_dim,
                experiment_config.model.technical_indicator_representation_dim,
            ),
        )
        self.technical_indicator_representation_layer_norm = nn.LayerNorm(
            experiment_config.model.technical_indicator_representation_dim
        )
        if experiment_config.model.use_pre_projection_input_dropout:
            self.input_dropout = nn.Dropout(experiment_config.model.input_dropout_probability)
        else:
            self.input_dropout = nn.Identity()
        if self._use_shared_text_bottlenecks:
            self.news_bottleneck = nn.Sequential(
                nn.Linear(news_embedding_dim, experiment_config.model.shared_text_bottleneck_dim),
                nn.LayerNorm(experiment_config.model.shared_text_bottleneck_dim),
                nn.GELU(),
            )
            news_block_input_dim = experiment_config.model.shared_text_bottleneck_dim
        else:
            self.news_bottleneck = None
            news_block_input_dim = news_embedding_dim
        technical_block_input_dim = experiment_config.model.technical_indicator_representation_dim
        if self._use_layer_specific_text_projections:
            self.news_projection = None
            self.news_layer_norm = None
        else:
            self.news_projection = nn.Linear(
                news_block_input_dim,
                experiment_config.model.d_model,
            )
            self.news_layer_norm = nn.LayerNorm(experiment_config.model.d_model)
        if self._disable_outer_no_news_placeholder_embedding:
            self.no_news_embedding = None
        else:
            self.no_news_embedding = nn.Parameter(
                torch.randn(news_embedding_dim)
                * experiment_config.model.no_news_embedding_initialization_std
            )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlockWithGating(
                    technical_input_dim=(
                        technical_block_input_dim
                        if self._use_layer_specific_text_projections
                        else experiment_config.model.d_model
                    ),
                    news_input_dim=(
                        news_block_input_dim
                        if self._use_layer_specific_text_projections
                        else experiment_config.model.d_model
                    ),
                    d_model=experiment_config.model.d_model,
                    d_ff=experiment_config.model.d_ff,
                    num_heads=experiment_config.model.num_heads,
                    dropout_probability=experiment_config.model.dropout_probability,
                    gate_hidden_dim=experiment_config.model.gate_hidden_dim,
                    tech_gate_bias_initial_value=experiment_config.model.tech_gate_bias_initial_value,
                    news_gate_bias_initial_value=experiment_config.model.news_gate_bias_initial_value,
                    use_post_injection_layer_norm=experiment_config.model.use_post_injection_layer_norm,
                    use_layer_specific_text_projections=experiment_config.model.use_layer_specific_text_projections,
                    trace_layer_specific_projected_text_features=experiment_config.model.trace_layer_specific_projected_text_features,
                    use_block_level_news_hard_mask=experiment_config.model.use_block_level_news_hard_mask,
                )
                for _ in range(experiment_config.model.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(experiment_config.model.d_model)
        self.classifier = nn.Linear(experiment_config.model.d_model, experiment_config.model.num_classes)

    def _build_temporal_attention_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """Build the boolean upper-triangular mask expected by PyTorch attention."""

        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        price_features: torch.Tensor,
        technical_indicator_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        has_news: torch.Tensor,
        collect_trace_tensors: bool = False,
    ) -> PriorStockModelOutput:
        """Run one forward pass and return logits, gate penalties, and tensor diagnostics."""

        def _trace_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            """Detach one top-level trace tensor and immediately offload it to CPU."""

            return tensor.detach().to(device="cpu")

        batch_size, sequence_length, feature_count = price_features.shape
        if feature_count != self._experiment_config.model.input_price_feature_count:
            raise ValueError("price_features has an unexpected final dimension.")
        if sequence_length != self._experiment_config.data.lookback_window_size:
            raise ValueError("price_features has an unexpected sequence length.")
        if technical_indicator_features.shape[-1] != self._experiment_config.model.technical_indicator_feature_count:
            raise ValueError("technical_indicator_features has an unexpected final dimension.")
        if news_embeddings.shape[-1] != self._experiment_config.news_encoder.output_embedding_dim:
            raise ValueError("news_embeddings has an unexpected final dimension.")

        hidden_states = self.input_layer_norm(self.price_projection(price_features) + self.positional_embedding)
        raw_technical_indicator_inputs = self.input_dropout(technical_indicator_features)
        technical_indicator_representation = self.technical_indicator_representation_layer_norm(
            self.technical_indicator_encoder(
                self.technical_indicator_pre_layer_norm(raw_technical_indicator_inputs)
            )
        )
        news_inputs = self.input_dropout(news_embeddings)

        if self._use_shared_text_bottlenecks:
            if self.news_bottleneck is None:
                raise RuntimeError("Shared news bottleneck is not initialized.")
            news_inputs = self.news_bottleneck(news_inputs)

        if not self._disable_outer_no_news_placeholder_embedding:
            if self.no_news_embedding is None:
                raise RuntimeError("Outer no-news placeholder embedding is not initialized.")
            no_news_embedding = self.no_news_embedding.view(1, 1, -1)
            news_inputs = (has_news * news_embeddings) + ((1.0 - has_news) * no_news_embedding)

        if not self._use_layer_specific_text_projections:
            if self.news_projection is None or self.news_layer_norm is None:
                raise RuntimeError("Global news projections are not initialized.")
            news_inputs = self.news_layer_norm(self.news_projection(news_inputs))

        temporal_attention_mask = self._build_temporal_attention_mask(sequence_length, price_features.device)
        total_tech_gate_regularization_loss = torch.zeros((), device=price_features.device)
        total_news_gate_regularization_loss = torch.zeros((), device=price_features.device)
        layer_diagnostics: dict[str, dict] = {}
        trace_tensors: dict[str, torch.Tensor] = {}
        if collect_trace_tensors:
            trace_tensors["price_features"] = _trace_to_cpu(price_features)
            trace_tensors["technical_indicator_features"] = _trace_to_cpu(technical_indicator_features)
            trace_tensors["input_hidden_states"] = _trace_to_cpu(hidden_states)
            trace_tensors["technical_indicator_representation"] = _trace_to_cpu(
                technical_indicator_representation
            )
            if self._use_shared_text_bottlenecks:
                trace_tensors["news_bottleneck_features"] = _trace_to_cpu(news_inputs)

        for layer_index, transformer_layer in enumerate(self.transformer_layers):
            layer_output = transformer_layer(
                hidden_states=hidden_states,
                technical_indicator_inputs=technical_indicator_representation,
                news_inputs=news_inputs,
                has_news=has_news,
                temporal_attention_mask=temporal_attention_mask,
                collect_trace_tensors=collect_trace_tensors,
            )
            hidden_states = layer_output.hidden_states
            total_tech_gate_regularization_loss = (
                total_tech_gate_regularization_loss + layer_output.tech_gate_regularization_loss
            )
            total_news_gate_regularization_loss = (
                total_news_gate_regularization_loss + layer_output.news_gate_regularization_loss
            )
            layer_diagnostics[f"layer_{layer_index + 1}"] = layer_output.diagnostics
            if collect_trace_tensors:
                for trace_name, trace_tensor in layer_output.trace_tensors.items():
                    trace_tensors[f"layer_{layer_index + 1}/{trace_name}"] = trace_tensor

        final_hidden_states = self.final_layer_norm(hidden_states)
        last_hidden_state = final_hidden_states[:, -1, :]
        logits = self.classifier(last_hidden_state)
        if collect_trace_tensors:
            trace_tensors["final_hidden_states"] = _trace_to_cpu(final_hidden_states)
            trace_tensors["last_hidden_state"] = _trace_to_cpu(last_hidden_state)
            trace_tensors["logits"] = _trace_to_cpu(logits)

        diagnostics = {
            "price_features_shape": [batch_size, sequence_length, feature_count],
            "raw_technical_indicator_shape": list(technical_indicator_features.shape),
            "raw_news_embedding_shape": list(news_embeddings.shape),
            "technical_indicator_representation_shape": list(technical_indicator_representation.shape),
            "news_input_shape": list(news_inputs.shape),
            "final_hidden_states_shape": list(final_hidden_states.shape),
            "logits_shape": list(logits.shape),
            "layers": layer_diagnostics,
        }
        return PriorStockModelOutput(
            logits=logits,
            tech_gate_regularization_loss=total_tech_gate_regularization_loss,
            news_gate_regularization_loss=total_news_gate_regularization_loss,
            technical_indicator_representation=technical_indicator_representation[:, -1, :],
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
