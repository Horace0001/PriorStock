"""A lightweight MLP probe for technical-text embedding quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from priorstock.probe_config import ProbeExperimentConfig


@dataclass(frozen=True)
class TechnicalTextProbeOutput:
    """One forward-pass output bundle for the lightweight text probe."""

    logits: torch.Tensor
    projection_representation: torch.Tensor
    diagnostics: dict
    trace_tensors: dict[str, torch.Tensor]


class TechnicalTextProbeModel(nn.Module):
    """A minimal MLP that probes whether frozen E5 vectors contain predictive signal."""

    def __init__(self, experiment_config: ProbeExperimentConfig) -> None:
        """Construct the probe architecture from the validated YAML config."""

        super().__init__()
        self._experiment_config = experiment_config
        self.input_layer_norm = nn.LayerNorm(experiment_config.model.input_embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(experiment_config.model.input_embedding_dim, experiment_config.model.projection_dim),
            nn.LayerNorm(experiment_config.model.projection_dim),
            nn.GELU(),
            nn.Dropout(experiment_config.model.hidden_dropout_probability),
        )
        self.classifier = nn.Linear(experiment_config.model.projection_dim, experiment_config.model.num_classes)

    def forward(
        self,
        tech_embedding: torch.Tensor,
        collect_trace_tensors: bool = False,
    ) -> TechnicalTextProbeOutput:
        """Project one batch of E5 vectors and classify them into the configured label space."""

        if tech_embedding.shape[-1] != self._experiment_config.model.input_embedding_dim:
            raise ValueError("tech_embedding has an unexpected final dimension.")

        normalized_embedding = self.input_layer_norm(tech_embedding)
        projection_representation = self.projection(normalized_embedding)
        logits = self.classifier(projection_representation)
        trace_tensors: dict[str, torch.Tensor] = {}
        if collect_trace_tensors:
            trace_tensors["input_embedding"] = tech_embedding.detach().to(device="cpu")
            trace_tensors["normalized_embedding"] = normalized_embedding.detach().to(device="cpu")
            trace_tensors["projection_representation"] = projection_representation.detach().to(device="cpu")
            trace_tensors["logits"] = logits.detach().to(device="cpu")

        diagnostics = {
            "input_embedding_shape": list(tech_embedding.shape),
            "projection_representation_shape": list(projection_representation.shape),
            "logits_shape": list(logits.shape),
        }
        return TechnicalTextProbeOutput(
            logits=logits,
            projection_representation=projection_representation,
            diagnostics=diagnostics,
            trace_tensors=trace_tensors,
        )
