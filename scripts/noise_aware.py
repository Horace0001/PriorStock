"""Noise-aware fine-tuning for the factor concat-logit classifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from priorstock.exceptions import ConfigurationError
from priorstock.training.engine import _safe_wandb_log
from priorstock.training.single_logit_binary_engine import (
    compute_positive_class_weight,
    compute_single_logit_binary_metrics,
    compute_single_logit_checkpoint_selection_score,
    compute_single_logit_soft_balanced_accuracy_loss,
    compute_single_logit_soft_mcc_loss,
    compute_up_probability_and_margin,
)
from priorstock.utils.environment import load_project_local_environment_files
from priorstock.utils.io import write_json_file
from priorstock.utils.logging_utils import configure_logger
from priorstock.utils.seed import set_global_seed
from priorstock.versioned.ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_v1 import (
    FactorConcatLogitClassifierV1,
)
from scripts.factor import (
    FACTOR_CONCAT_CLASSIFIER_SECTION_NAME,
    _build_data_loader,
    _build_model,
    _extract_dataset_label_values,
    _flatten_factor_diagnostics,
    _iter_base_classifier_parameters,
    _iter_base_joint_parameters,
    _iter_base_tail_parameters,
    _load_checkpoint,
    _load_frozen_base_checkpoint,
    _move_batch_to_device,
    _parse_factor_concat_classifier_config,
    _set_requires_grad,
)
from scripts.base import (
    ATTENTION_SIDE_ADAPTER_SECTION_NAME,
    BINARY_CLASSIFICATION_SECTION_NAME,
    SINGLE_LOGIT_SECTION_NAME,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _load_merged_raw_config,
    _parse_attention_side_adapter_config,
    _parse_single_logit_objective_config,
    _write_single_logit_labeling_contract,
)
from priorstock.config import load_experiment_config


EXPECTED_FRAMEWORK_VARIANT_NAME = (
    "ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_v1"
)
NOISE_AWARE_SECTION_NAME = "factor_noise_aware_training"
VALIDATION_SIGNIFICANT_SPLIT_NAME = "validation_significant"
VALIDATION_FULL_SPLIT_NAME = "validation_full"
MCC_ACCURACY_SELECTION_SCORE_NAME = "mcc_accuracy_selection_score"
LOGGER = configure_logger("priorstock.noise_aware")


@dataclass(frozen=True)
class FactorNoiseAwareTrainingConfig:
    """Hyperparameters for noise-aware low-return fine-tuning."""

    fine_tune_checkpoint_file_path: str | None
    significant_return_absolute_threshold: float
    near_boundary_return_absolute_threshold: float
    soft_target_temperature: float
    subsignificant_loss_weight: float
    boundary_margin_loss_weight: float
    factor_learning_rate: float
    classifier_learning_rate: float
    checkpoint_selection_split_name: str | None = None
    checkpoint_selection_mcc_weight: float = 0.7
    checkpoint_selection_accuracy_weight: float = 0.3
    checkpoint_selection_metric_name: str = "auto"
    base_checkpoint_file_path: str | None = None
    should_evaluate_full_splits_each_epoch: bool = False
    significant_soft_mcc_loss_weight: float = 0.0
    mid_return_soft_mcc_loss_weight: float = 0.0
    mid_return_soft_balanced_accuracy_loss_weight: float = 0.0
    soft_metric_loss_epsilon: float = 1.0e-8
    omitted_evaluation_iterator_count: int = 0


@dataclass(frozen=True)
class NoiseAwareLossOutput:
    """Partitioned noise-aware loss components."""

    total_loss: torch.Tensor
    significant_loss: torch.Tensor
    subsignificant_loss: torch.Tensor
    boundary_margin_loss: torch.Tensor
    significant_soft_mcc_loss: torch.Tensor
    mid_return_soft_mcc_loss: torch.Tensor
    mid_return_soft_balanced_accuracy_loss: torch.Tensor
    significant_sample_count: int
    subsignificant_sample_count: int
    near_boundary_sample_count: int
    mid_return_sample_count: int


def compute_noise_aware_partitioned_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    target_returns: torch.Tensor,
    positive_class_weight: torch.Tensor,
    noise_aware_config: FactorNoiseAwareTrainingConfig,
) -> NoiseAwareLossOutput:
    """Compute significant BCE plus low-return soft-label calibration losses."""

    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError("noise-aware loss expects logits with shape [batch_size, 1].")
    logit_values = logits.squeeze(dim=-1)
    hard_targets = hard_labels.to(device=logit_values.device, dtype=logit_values.dtype)
    returns = target_returns.to(device=logit_values.device, dtype=logit_values.dtype)
    absolute_returns = returns.abs()
    significant_mask = absolute_returns >= noise_aware_config.significant_return_absolute_threshold
    subsignificant_mask = ~significant_mask
    near_boundary_mask = absolute_returns < noise_aware_config.near_boundary_return_absolute_threshold
    mid_return_mask = (
        (absolute_returns >= noise_aware_config.near_boundary_return_absolute_threshold)
        & (absolute_returns < noise_aware_config.significant_return_absolute_threshold)
    )
    zero_loss = logit_values.sum() * 0.0

    if significant_mask.any():
        significant_loss = F.binary_cross_entropy_with_logits(
            logit_values[significant_mask],
            hard_targets[significant_mask],
            pos_weight=positive_class_weight.to(
                device=logit_values.device,
                dtype=logit_values.dtype,
            ),
        )
        significant_soft_mcc_loss = compute_single_logit_soft_mcc_loss(
            logits=logit_values[significant_mask].unsqueeze(-1),
            hard_labels=hard_targets[significant_mask],
            epsilon=noise_aware_config.soft_metric_loss_epsilon,
        )
    else:
        significant_loss = zero_loss
        significant_soft_mcc_loss = zero_loss

    if subsignificant_mask.any():
        soft_targets = 0.5 + (
            0.5
            * torch.tanh(
                returns[subsignificant_mask] / noise_aware_config.soft_target_temperature
            )
        )
        sample_weights = (
            0.05
            + 0.45
            * (
                absolute_returns[subsignificant_mask]
                / noise_aware_config.significant_return_absolute_threshold
            ).pow(2)
        ).clamp(max=0.50)
        raw_subsignificant_loss = F.binary_cross_entropy_with_logits(
            logit_values[subsignificant_mask],
            soft_targets,
            reduction="none",
        )
        subsignificant_loss = (
            raw_subsignificant_loss * sample_weights
        ).sum() / sample_weights.sum().clamp_min(torch.finfo(logit_values.dtype).tiny)
    else:
        subsignificant_loss = zero_loss

    if mid_return_mask.any():
        mid_return_soft_mcc_loss = compute_single_logit_soft_mcc_loss(
            logits=logit_values[mid_return_mask].unsqueeze(-1),
            hard_labels=hard_targets[mid_return_mask],
            epsilon=noise_aware_config.soft_metric_loss_epsilon,
        )
        mid_return_soft_balanced_accuracy_loss = compute_single_logit_soft_balanced_accuracy_loss(
            logits=logit_values[mid_return_mask].unsqueeze(-1),
            hard_labels=hard_targets[mid_return_mask],
            epsilon=noise_aware_config.soft_metric_loss_epsilon,
        )
    else:
        mid_return_soft_mcc_loss = zero_loss
        mid_return_soft_balanced_accuracy_loss = zero_loss

    if near_boundary_mask.any():
        boundary_margin_loss = logit_values[near_boundary_mask].pow(2).mean()
    else:
        boundary_margin_loss = zero_loss

    total_loss = (
        significant_loss
        + (noise_aware_config.significant_soft_mcc_loss_weight * significant_soft_mcc_loss)
        + (noise_aware_config.subsignificant_loss_weight * subsignificant_loss)
        + (noise_aware_config.boundary_margin_loss_weight * boundary_margin_loss)
        + (noise_aware_config.mid_return_soft_mcc_loss_weight * mid_return_soft_mcc_loss)
        + (
            noise_aware_config.mid_return_soft_balanced_accuracy_loss_weight
            * mid_return_soft_balanced_accuracy_loss
        )
    )
    return NoiseAwareLossOutput(
        total_loss=total_loss,
        significant_loss=significant_loss,
        subsignificant_loss=subsignificant_loss,
        boundary_margin_loss=boundary_margin_loss,
        significant_soft_mcc_loss=significant_soft_mcc_loss,
        mid_return_soft_mcc_loss=mid_return_soft_mcc_loss,
        mid_return_soft_balanced_accuracy_loss=mid_return_soft_balanced_accuracy_loss,
        significant_sample_count=int(significant_mask.sum().detach().cpu().item()),
        subsignificant_sample_count=int(subsignificant_mask.sum().detach().cpu().item()),
        near_boundary_sample_count=int(near_boundary_mask.sum().detach().cpu().item()),
        mid_return_sample_count=int(mid_return_mask.sum().detach().cpu().item()),
    )


def _parse_noise_aware_training_config(raw_section: object) -> FactorNoiseAwareTrainingConfig:
    """Parse the noise-aware training YAML section."""

    if not isinstance(raw_section, dict):
        raise ConfigurationError("factor_noise_aware_training must be a YAML mapping.")
    checkpoint_field_names = {
        "fine_tune_checkpoint_file_path",
        "base_checkpoint_file_path",
    }
    required_field_names = {
        "significant_return_absolute_threshold",
        "near_boundary_return_absolute_threshold",
        "soft_target_temperature",
        "subsignificant_loss_weight",
        "boundary_margin_loss_weight",
        "factor_learning_rate",
        "classifier_learning_rate",
    }
    optional_field_names = {
        "checkpoint_selection_split_name",
        "checkpoint_selection_mcc_weight",
        "checkpoint_selection_accuracy_weight",
        "checkpoint_selection_metric_name",
        "should_evaluate_full_splits_each_epoch",
        "significant_soft_mcc_loss_weight",
        "mid_return_soft_mcc_loss_weight",
        "mid_return_soft_balanced_accuracy_loss_weight",
        "soft_metric_loss_epsilon",
        "omitted_evaluation_iterator_count",
    }
    provided_field_names = set(raw_section.keys())
    missing_field_names = sorted(required_field_names - provided_field_names)
    unexpected_field_names = sorted(
        provided_field_names - required_field_names - optional_field_names - checkpoint_field_names
    )
    provided_checkpoint_field_names = sorted(provided_field_names & checkpoint_field_names)
    if missing_field_names:
        raise ConfigurationError(
            "factor_noise_aware_training missing fields: " + ", ".join(missing_field_names)
        )
    if len(provided_checkpoint_field_names) != 1:
        raise ConfigurationError(
            "factor_noise_aware_training must provide exactly one of "
            "fine_tune_checkpoint_file_path or base_checkpoint_file_path."
        )
    if unexpected_field_names:
        raise ConfigurationError(
            "factor_noise_aware_training unexpected fields: " + ", ".join(unexpected_field_names)
        )
    config = FactorNoiseAwareTrainingConfig(
        fine_tune_checkpoint_file_path=(
            str(raw_section["fine_tune_checkpoint_file_path"])
            if "fine_tune_checkpoint_file_path" in raw_section
            else None
        ),
        significant_return_absolute_threshold=_coerce_float(
            raw_section["significant_return_absolute_threshold"],
            "significant_return_absolute_threshold",
        ),
        near_boundary_return_absolute_threshold=_coerce_float(
            raw_section["near_boundary_return_absolute_threshold"],
            "near_boundary_return_absolute_threshold",
        ),
        soft_target_temperature=_coerce_float(raw_section["soft_target_temperature"], "soft_target_temperature"),
        subsignificant_loss_weight=_coerce_float(
            raw_section["subsignificant_loss_weight"],
            "subsignificant_loss_weight",
        ),
        boundary_margin_loss_weight=_coerce_float(
            raw_section["boundary_margin_loss_weight"],
            "boundary_margin_loss_weight",
        ),
        factor_learning_rate=_coerce_float(raw_section["factor_learning_rate"], "factor_learning_rate"),
        classifier_learning_rate=_coerce_float(
            raw_section["classifier_learning_rate"],
            "classifier_learning_rate",
        ),
        checkpoint_selection_split_name=(
            str(raw_section["checkpoint_selection_split_name"])
            if "checkpoint_selection_split_name" in raw_section
            else None
        ),
        checkpoint_selection_mcc_weight=_coerce_float(
            raw_section.get("checkpoint_selection_mcc_weight", 0.7),
            "checkpoint_selection_mcc_weight",
        ),
        checkpoint_selection_accuracy_weight=_coerce_float(
            raw_section.get("checkpoint_selection_accuracy_weight", 0.3),
            "checkpoint_selection_accuracy_weight",
        ),
        checkpoint_selection_metric_name=str(raw_section.get("checkpoint_selection_metric_name", "auto")),
        base_checkpoint_file_path=(
            str(raw_section["base_checkpoint_file_path"])
            if "base_checkpoint_file_path" in raw_section
            else None
        ),
        should_evaluate_full_splits_each_epoch=_coerce_bool(
            raw_section.get("should_evaluate_full_splits_each_epoch", False),
            "should_evaluate_full_splits_each_epoch",
        ),
        significant_soft_mcc_loss_weight=_coerce_float(
            raw_section.get("significant_soft_mcc_loss_weight", 0.0),
            "significant_soft_mcc_loss_weight",
        ),
        mid_return_soft_mcc_loss_weight=_coerce_float(
            raw_section.get("mid_return_soft_mcc_loss_weight", 0.0),
            "mid_return_soft_mcc_loss_weight",
        ),
        mid_return_soft_balanced_accuracy_loss_weight=_coerce_float(
            raw_section.get("mid_return_soft_balanced_accuracy_loss_weight", 0.0),
            "mid_return_soft_balanced_accuracy_loss_weight",
        ),
        soft_metric_loss_epsilon=_coerce_float(
            raw_section.get("soft_metric_loss_epsilon", 1.0e-8),
            "soft_metric_loss_epsilon",
        ),
        omitted_evaluation_iterator_count=_coerce_int(
            raw_section.get("omitted_evaluation_iterator_count", 0),
            "omitted_evaluation_iterator_count",
        ),
    )
    if config.significant_return_absolute_threshold <= 0.0:
        raise ConfigurationError("significant_return_absolute_threshold must be positive.")
    if not 0.0 < config.near_boundary_return_absolute_threshold < config.significant_return_absolute_threshold:
        raise ConfigurationError(
            "near_boundary_return_absolute_threshold must be in (0, significant threshold)."
        )
    if config.soft_target_temperature <= 0.0:
        raise ConfigurationError("soft_target_temperature must be positive.")
    if config.subsignificant_loss_weight < 0.0 or config.boundary_margin_loss_weight < 0.0:
        raise ConfigurationError("noise-aware loss weights must be non-negative.")
    if (
        config.significant_soft_mcc_loss_weight < 0.0
        or config.mid_return_soft_mcc_loss_weight < 0.0
        or config.mid_return_soft_balanced_accuracy_loss_weight < 0.0
    ):
        raise ConfigurationError("noise-aware soft metric loss weights must be non-negative.")
    if config.soft_metric_loss_epsilon <= 0.0:
        raise ConfigurationError("soft_metric_loss_epsilon must be positive.")
    if config.factor_learning_rate <= 0.0 or config.classifier_learning_rate <= 0.0:
        raise ConfigurationError("noise-aware learning rates must be positive.")
    supported_selection_split_names = {
        VALIDATION_SIGNIFICANT_SPLIT_NAME,
        VALIDATION_FULL_SPLIT_NAME,
    }
    if (
        config.checkpoint_selection_split_name is not None
        and config.checkpoint_selection_split_name not in supported_selection_split_names
    ):
        raise ConfigurationError(
            "checkpoint_selection_split_name must be one of "
            + ", ".join(sorted(supported_selection_split_names))
            + "."
        )
    if config.checkpoint_selection_mcc_weight < 0.0 or config.checkpoint_selection_accuracy_weight < 0.0:
        raise ConfigurationError("checkpoint selection weights must be non-negative.")
    if config.checkpoint_selection_mcc_weight + config.checkpoint_selection_accuracy_weight <= 0.0:
        raise ConfigurationError("at least one checkpoint selection weight must be positive.")
    supported_selection_metric_names = {
        "auto",
        "selection_score",
        MCC_ACCURACY_SELECTION_SCORE_NAME,
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "roc_auc",
    }
    if config.checkpoint_selection_metric_name not in supported_selection_metric_names:
        raise ConfigurationError(
            "checkpoint_selection_metric_name must be one of "
            + ", ".join(sorted(supported_selection_metric_names))
            + "."
        )
    return config


def compute_mcc_accuracy_selection_score(
    metrics: dict[str, float],
    mcc_weight: float,
    accuracy_weight: float,
) -> float:
    """Compute a compact MCC/accuracy checkpoint score."""

    return float((mcc_weight * metrics["mcc"]) + (accuracy_weight * metrics["accuracy"]))


def _compute_noise_aware_checkpoint_selection_score(
    metrics: dict[str, float],
    noise_aware_config: FactorNoiseAwareTrainingConfig,
) -> float:
    """Compute the configured noise-aware checkpoint selection score."""

    metric_name = _resolve_noise_aware_checkpoint_selection_metric_name(noise_aware_config)
    return float(metrics[metric_name])


def _resolve_noise_aware_checkpoint_selection_metric_name(
    noise_aware_config: FactorNoiseAwareTrainingConfig,
) -> str:
    """Resolve the concrete noise-aware checkpoint selection metric name."""

    if noise_aware_config.checkpoint_selection_metric_name != "auto":
        return noise_aware_config.checkpoint_selection_metric_name
    if noise_aware_config.checkpoint_selection_split_name is None:
        return "selection_score"
    return "selection_score"


def _load_config_with_noise_aware_extensions(config_file_path: Path, run_directory: Path):
    """Load ExperimentConfig while accepting factor concat and noise-aware script sections."""

    raw_config = _load_merged_raw_config(config_file_path)
    sanitized_config = dict(raw_config)
    attention_side_adapter_config = _parse_attention_side_adapter_config(
        sanitized_config.pop(ATTENTION_SIDE_ADAPTER_SECTION_NAME)
    )
    objective_config = _parse_single_logit_objective_config(
        sanitized_config.pop(SINGLE_LOGIT_SECTION_NAME)
    )
    factor_classifier_config = _parse_factor_concat_classifier_config(
        sanitized_config.pop(FACTOR_CONCAT_CLASSIFIER_SECTION_NAME)
    )
    noise_aware_config = _parse_noise_aware_training_config(
        sanitized_config.pop(NOISE_AWARE_SECTION_NAME)
    )
    sanitized_config.pop(BINARY_CLASSIFICATION_SECTION_NAME, None)
    run_directory.mkdir(parents=True, exist_ok=True)
    sanitized_config_file_path = run_directory / "experiment_config_without_script_extensions.yaml"
    with sanitized_config_file_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(sanitized_config, file_handle, sort_keys=False, allow_unicode=False)
    experiment_config = load_experiment_config(sanitized_config_file_path)
    return (
        experiment_config,
        attention_side_adapter_config,
        objective_config,
        factor_classifier_config,
        noise_aware_config,
    )


def _freeze_base_and_enable_noise_aware_parameters(model: FactorConcatLogitClassifierV1) -> None:
    """Freeze the base model and train only factor modules plus classifier head."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _iter_factor_side_parameters(model):
        parameter.requires_grad = True
    for parameter in model.classifier_head.parameters():
        parameter.requires_grad = True


def _apply_noise_aware_training_epoch_state(
    model: FactorConcatLogitClassifierV1,
    epoch_index: int,
) -> None:
    """Apply the noise-aware freeze schedule for the current epoch."""

    _freeze_base_and_enable_noise_aware_parameters(model)
    if not model.config.should_train_base_model:
        return
    if epoch_index <= model.config.base_frozen_epoch_count:
        return
    _set_requires_grad(_iter_base_joint_parameters(model), True)
    _set_requires_grad(_iter_base_classifier_parameters(model), True)
    if model.config.should_freeze_fusion_gate_during_joint_training and hasattr(
        model.base_model,
        "fusion_gate",
    ):
        for parameter in model.base_model.fusion_gate.parameters():
            parameter.requires_grad = False


def _iter_factor_side_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return trainable factor attention/projection parameters excluding the classifier head."""

    factor_modules: list[nn.Module] = [
        model.factor_projector,
        model.query_layer_norm,
        model.factor_attention,
    ]
    if model.factor_transformer_ffn is not None:
        factor_modules.append(model.factor_transformer_ffn)
    parameters: list[nn.Parameter] = [model.rank_embedding]
    for module in factor_modules:
        parameters.extend(list(module.parameters()))
    return parameters


def _iter_noise_aware_base_joint_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return base-model parameters that may be unfrozen during noise-aware joint training."""

    if not model.config.should_train_base_model:
        return []
    return _iter_base_joint_parameters(model) + _iter_base_classifier_parameters(model)


def _build_noise_aware_optimizer(
    model: FactorConcatLogitClassifierV1,
    experiment_config,
    noise_aware_config: FactorNoiseAwareTrainingConfig,
) -> torch.optim.Optimizer:
    """Build AdamW with separate factor and classifier learning rates."""

    parameter_groups: list[dict[str, Any]] = [
        {
            "params": _iter_factor_side_parameters(model),
            "lr": noise_aware_config.factor_learning_rate,
            "name": "factor_modules",
        },
        {
            "params": list(model.classifier_head.parameters()),
            "lr": noise_aware_config.classifier_learning_rate,
            "name": "classifier_head",
        },
    ]
    if model.config.should_train_base_model:
        parameter_groups.extend(
            [
                {
                    "params": _iter_base_joint_parameters(model),
                    "lr": model.config.base_tail_learning_rate,
                    "name": f"base_{model.config.base_trainable_scope}",
                },
                {
                    "params": _iter_base_classifier_parameters(model),
                    "lr": model.config.base_classifier_learning_rate,
                    "name": "base_classifier",
                },
            ]
        )
    return torch.optim.AdamW(
        parameter_groups,
        betas=(experiment_config.training.adam_beta_one, experiment_config.training.adam_beta_two),
        eps=experiment_config.training.adam_epsilon,
        weight_decay=experiment_config.training.weight_decay,
    )


def _compute_parameter_gradient_norm(parameters: list[nn.Parameter]) -> float:
    """Compute the L2 norm over all available parameter gradients."""

    squared_norm_sum = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient_norm = float(parameter.grad.detach().norm(2).cpu().item())
        squared_norm_sum += gradient_norm * gradient_norm
    return squared_norm_sum ** 0.5


def _build_noise_bucket_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    up_probabilities: list[float],
    logit_margins: list[float],
    target_returns: list[float],
    significant_threshold: float,
) -> dict[str, float]:
    """Add explicit low-return and significant-return diagnostics."""

    true_label_array = np.asarray(true_labels, dtype=np.int64)
    prediction_array = np.asarray(predicted_labels, dtype=np.int64)
    probability_array = np.asarray(up_probabilities, dtype=np.float64)
    margin_array = np.asarray(logit_margins, dtype=np.float64)
    return_array = np.asarray(target_returns, dtype=np.float64)
    absolute_returns = np.abs(return_array)
    bucket_masks = {
        "noise_bucket_abs_lt_0_001": absolute_returns < 0.001,
        "noise_bucket_abs_lt_0_002": absolute_returns < 0.002,
        "noise_bucket_abs_lt_0_005": absolute_returns < 0.005,
        "noise_bucket_abs_0_005_to_0_010": (
            (absolute_returns >= 0.005) & (absolute_returns < significant_threshold)
        ),
        "significant_abs_ge_0_010": absolute_returns >= significant_threshold,
        "subsignificant_abs_lt_0_010": absolute_returns < significant_threshold,
    }
    metrics: dict[str, float] = {}
    for prefix, mask in bucket_masks.items():
        metrics.update(
            _compute_subset_metrics(
                prefix=prefix,
                true_label_array=true_label_array[mask],
                prediction_array=prediction_array[mask],
                probability_array=probability_array[mask],
                margin_array=margin_array[mask],
            )
        )
    metrics["significant_pred_up_ratio"] = metrics["significant_abs_ge_0_010/pred_up_ratio"]
    metrics["subsignificant_pred_up_ratio"] = metrics["subsignificant_abs_lt_0_010/pred_up_ratio"]
    metrics["near_boundary_probability_mean"] = metrics["noise_bucket_abs_lt_0_005/probability_mean"]
    metrics["near_boundary_logit_abs_mean"] = metrics["noise_bucket_abs_lt_0_005/logit_abs_mean"]
    return metrics


def _compute_subset_metrics(
    prefix: str,
    true_label_array: np.ndarray,
    prediction_array: np.ndarray,
    probability_array: np.ndarray,
    margin_array: np.ndarray,
) -> dict[str, float]:
    """Compute compact binary diagnostics for one subset."""

    if true_label_array.size == 0:
        return {
            f"{prefix}/count": 0.0,
            f"{prefix}/accuracy": 0.0,
            f"{prefix}/balanced_accuracy": 0.0,
            f"{prefix}/macro_f1": 0.0,
            f"{prefix}/mcc": 0.0,
            f"{prefix}/auc": 0.5,
            f"{prefix}/pred_up_ratio": 0.0,
            f"{prefix}/probability_mean": 0.0,
            f"{prefix}/logit_abs_mean": 0.0,
        }
    auc_value = 0.5
    if len(np.unique(true_label_array)) >= 2:
        auc_value = float(roc_auc_score(true_label_array, probability_array))
    return {
        f"{prefix}/count": float(true_label_array.size),
        f"{prefix}/accuracy": float(accuracy_score(true_label_array, prediction_array)),
        f"{prefix}/balanced_accuracy": float(balanced_accuracy_score(true_label_array, prediction_array)),
        f"{prefix}/macro_f1": float(f1_score(true_label_array, prediction_array, average="macro", zero_division=0)),
        f"{prefix}/mcc": float(matthews_corrcoef(true_label_array, prediction_array)),
        f"{prefix}/auc": auc_value,
        f"{prefix}/pred_up_ratio": float(prediction_array.mean()),
        f"{prefix}/probability_mean": float(probability_array.mean()),
        f"{prefix}/logit_abs_mean": float(np.abs(margin_array).mean()),
    }


def _run_noise_aware_epoch(
    model: FactorConcatLogitClassifierV1,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    experiment_config,
    objective_config,
    noise_aware_config: FactorNoiseAwareTrainingConfig,
    positive_class_weight: torch.Tensor,
    split_name: str,
    should_use_noise_aware_loss: bool,
) -> dict[str, float]:
    """Run one epoch using either partitioned training loss or plain evaluation BCE diagnostics."""

    is_training = optimizer is not None
    device = torch.device(experiment_config.experiment.device)
    model.train(mode=is_training)
    base_parameters = _iter_noise_aware_base_joint_parameters(model)
    has_trainable_base_parameters = any(parameter.requires_grad for parameter in base_parameters)
    model.base_model.train(mode=is_training and has_trainable_base_parameters)
    total_loss = 0.0
    significant_loss_sum = 0.0
    subsignificant_loss_sum = 0.0
    boundary_margin_loss_sum = 0.0
    significant_soft_mcc_loss_sum = 0.0
    mid_return_soft_mcc_loss_sum = 0.0
    mid_return_soft_balanced_accuracy_loss_sum = 0.0
    significant_count = 0
    subsignificant_count = 0
    near_boundary_count = 0
    mid_return_count = 0
    total_samples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    up_probabilities: list[float] = []
    logit_margins: list[float] = []
    target_returns: list[float] = []
    soft_up_targets: list[float] = []
    diagnostic_weighted_sums: dict[str, float] = {}
    factor_gradient_norm_sum = 0.0
    classifier_gradient_norm_sum = 0.0
    base_gradient_norm_sum = 0.0
    gradient_batch_count = 0
    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        for batch in data_loader:
            batch = _move_batch_to_device(batch=batch, device=device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            model_output = model(
                price_features=batch["price_features"],
                technical_indicator_features=batch["technical_indicator_features"],
                news_embeddings=batch["news_embeddings"],
                has_news=batch["has_news"],
                factor_embeddings=batch["factor_embeddings"],
                has_factors=batch["has_factors"],
                collect_trace_tensors=False,
                sample_ids=batch["sample_id"],
            )
            if should_use_noise_aware_loss:
                loss_output = compute_noise_aware_partitioned_loss(
                    logits=model_output.logits,
                    hard_labels=batch["label"].to(dtype=torch.float32),
                    target_returns=batch["target_return"],
                    positive_class_weight=positive_class_weight,
                    noise_aware_config=noise_aware_config,
                )
                loss = loss_output.total_loss
            else:
                loss_output = compute_noise_aware_partitioned_loss(
                    logits=model_output.logits,
                    hard_labels=batch["label"].to(dtype=torch.float32),
                    target_returns=batch["target_return"],
                    positive_class_weight=positive_class_weight,
                    noise_aware_config=noise_aware_config,
                )
                loss = loss_output.significant_loss
            if is_training:
                loss.backward()
                factor_gradient_norm_sum += _compute_parameter_gradient_norm(
                    _iter_factor_side_parameters(model)
                )
                classifier_gradient_norm_sum += _compute_parameter_gradient_norm(
                    list(model.classifier_head.parameters())
                )
                base_gradient_norm_sum += _compute_parameter_gradient_norm(base_parameters)
                gradient_batch_count += 1
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    experiment_config.training.gradient_clip_max_norm,
                )
                optimizer.step()
            batch_size = int(batch["label"].shape[0])
            total_samples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            significant_loss_sum += float(loss_output.significant_loss.detach().cpu()) * max(
                loss_output.significant_sample_count,
                1,
            )
            subsignificant_loss_sum += float(loss_output.subsignificant_loss.detach().cpu()) * max(
                loss_output.subsignificant_sample_count,
                1,
            )
            boundary_margin_loss_sum += float(loss_output.boundary_margin_loss.detach().cpu()) * max(
                loss_output.near_boundary_sample_count,
                1,
            )
            significant_soft_mcc_loss_sum += float(
                loss_output.significant_soft_mcc_loss.detach().cpu()
            ) * max(
                loss_output.significant_sample_count,
                1,
            )
            mid_return_soft_mcc_loss_sum += float(
                loss_output.mid_return_soft_mcc_loss.detach().cpu()
            ) * max(
                loss_output.mid_return_sample_count,
                1,
            )
            mid_return_soft_balanced_accuracy_loss_sum += float(
                loss_output.mid_return_soft_balanced_accuracy_loss.detach().cpu()
            ) * max(
                loss_output.mid_return_sample_count,
                1,
            )
            significant_count += loss_output.significant_sample_count
            subsignificant_count += loss_output.subsignificant_sample_count
            near_boundary_count += loss_output.near_boundary_sample_count
            mid_return_count += loss_output.mid_return_sample_count
            up_probability, logit_margin = compute_up_probability_and_margin(model_output.logits)
            predicted_label = (up_probability >= 0.5).to(dtype=torch.long)
            true_labels.extend([int(value) for value in batch["label"].detach().cpu().tolist()])
            predicted_labels.extend([int(value) for value in predicted_label.detach().cpu().tolist()])
            up_probabilities.extend([float(value) for value in up_probability.detach().cpu().tolist()])
            logit_margins.extend([float(value) for value in logit_margin.detach().cpu().tolist()])
            target_returns.extend([float(value) for value in batch["target_return"].detach().cpu().tolist()])
            soft_up_targets.extend([float(value) for value in batch["label"].detach().cpu().tolist()])
            factor_diagnostics = _flatten_factor_diagnostics(
                model_output.diagnostics["factor_enhanced_classifier"]
            )
            for metric_name, metric_value in factor_diagnostics.items():
                diagnostic_weighted_sums[metric_name] = diagnostic_weighted_sums.get(
                    metric_name,
                    0.0,
                ) + (metric_value * batch_size)
    metrics = compute_single_logit_binary_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        up_probabilities=up_probabilities,
        logit_margins=logit_margins,
        target_returns=target_returns,
        soft_up_targets=soft_up_targets,
        objective_config=objective_config,
    )
    metrics["total_loss"] = total_loss / float(max(total_samples, 1))
    metrics["significant_loss"] = significant_loss_sum / float(max(significant_count, 1))
    metrics["subsignificant_loss"] = subsignificant_loss_sum / float(max(subsignificant_count, 1))
    metrics["boundary_margin_loss"] = boundary_margin_loss_sum / float(max(near_boundary_count, 1))
    metrics["significant_soft_mcc_loss"] = significant_soft_mcc_loss_sum / float(max(significant_count, 1))
    metrics["mid_return_soft_mcc_loss"] = mid_return_soft_mcc_loss_sum / float(max(mid_return_count, 1))
    metrics["mid_return_soft_balanced_accuracy_loss"] = (
        mid_return_soft_balanced_accuracy_loss_sum / float(max(mid_return_count, 1))
    )
    metrics["significant_sample_count"] = float(significant_count)
    metrics["subsignificant_sample_count"] = float(subsignificant_count)
    metrics["near_boundary_sample_count"] = float(near_boundary_count)
    metrics["mid_return_sample_count"] = float(mid_return_count)
    metrics.update(
        _build_noise_bucket_metrics(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            up_probabilities=up_probabilities,
            logit_margins=logit_margins,
            target_returns=target_returns,
            significant_threshold=noise_aware_config.significant_return_absolute_threshold,
        )
    )
    metrics.update(
        {
            f"factor_enhanced/{metric_name}": metric_sum / float(max(total_samples, 1))
            for metric_name, metric_sum in diagnostic_weighted_sums.items()
        }
    )
    metrics["training/factor_grad_norm"] = factor_gradient_norm_sum / float(max(gradient_batch_count, 1))
    metrics["training/classifier_grad_norm"] = classifier_gradient_norm_sum / float(max(gradient_batch_count, 1))
    metrics["training/base_grad_norm"] = base_gradient_norm_sum / float(max(gradient_batch_count, 1))
    metrics[MCC_ACCURACY_SELECTION_SCORE_NAME] = compute_mcc_accuracy_selection_score(
        metrics=metrics,
        mcc_weight=noise_aware_config.checkpoint_selection_mcc_weight,
        accuracy_weight=noise_aware_config.checkpoint_selection_accuracy_weight,
    )
    metrics["selection_score"] = compute_single_logit_checkpoint_selection_score(
        metrics,
        objective_config,
    )
    LOGGER.info(
        "%s loss=%.6f sig_loss=%.6f sig_soft_mcc=%.6f sub_loss=%.6f boundary=%.6f "
        "mid_soft_mcc=%.6f mid_soft_ba=%.6f score=%.6f mcc_acc=%.6f "
        "mcc=%.6f auc=%.6f acc=%.6f pred_up=%.6f sig_pred_up=%.6f sub_pred_up=%.6f "
        "near_prob=%.6f near_abs_logit=%.6f base_grad=%.6f delta_abs=%.8f",
        split_name,
        metrics["total_loss"],
        metrics["significant_loss"],
        metrics["significant_soft_mcc_loss"],
        metrics["subsignificant_loss"],
        metrics["boundary_margin_loss"],
        metrics["mid_return_soft_mcc_loss"],
        metrics["mid_return_soft_balanced_accuracy_loss"],
        metrics["selection_score"],
        metrics[MCC_ACCURACY_SELECTION_SCORE_NAME],
        metrics["mcc"],
        metrics["roc_auc"],
        metrics["accuracy"],
        metrics["prediction_up_ratio"],
        metrics["significant_pred_up_ratio"],
        metrics["subsignificant_pred_up_ratio"],
        metrics["near_boundary_probability_mean"],
        metrics["near_boundary_logit_abs_mean"],
        metrics["training/base_grad_norm"],
        metrics.get("factor_enhanced/delta_logit_abs_mean", 0.0),
    )
    return metrics


def _save_checkpoint(
    checkpoint_file_path: Path,
    model: FactorConcatLogitClassifierV1,
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    validation_metrics: dict[str, float],
    checkpoint_selection_score: float | None = None,
    checkpoint_selection_metric_name: str = "selection_score",
    checkpoint_selection_split_name: str | None = VALIDATION_SIGNIFICANT_SPLIT_NAME,
) -> None:
    """Save a noise-aware fine-tuned checkpoint."""

    resolved_selection_score = (
        float(validation_metrics["selection_score"])
        if checkpoint_selection_score is None
        else float(checkpoint_selection_score)
    )
    checkpoint_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "selection_metric_name": checkpoint_selection_metric_name,
            "selection_split_name": checkpoint_selection_split_name,
            "selection_score": resolved_selection_score,
        },
        checkpoint_file_path,
    )


def _fit_model(
    model: FactorConcatLogitClassifierV1,
    train_full_loader: DataLoader,
    train_significant_loader: DataLoader,
    validation_significant_loader: DataLoader,
    validation_full_loader: DataLoader,
    experiment_config,
    objective_config,
    full_objective_config,
    noise_aware_config: FactorNoiseAwareTrainingConfig,
    run_directory: Path,
    wandb_run,
) -> dict[str, Any]:
    """Fine-tune with full train samples and configurable checkpoint selection."""

    device = torch.device(experiment_config.experiment.device)
    model.to(device)
    model.freeze_base_model()
    _apply_noise_aware_training_epoch_state(model, epoch_index=1)
    positive_class_weight = compute_positive_class_weight(
        _extract_dataset_label_values(train_significant_loader)
    )
    positive_class_weight = positive_class_weight.to(device)
    optimizer = _build_noise_aware_optimizer(
        model=model,
        experiment_config=experiment_config,
        noise_aware_config=noise_aware_config,
    )
    checkpoint_directory = run_directory / experiment_config.paths.checkpoint_subdirectory
    best_checkpoint_file_path = checkpoint_directory / "best_model.pt"
    latest_checkpoint_file_path = checkpoint_directory / experiment_config.training.latest_checkpoint_file_name
    history: list[dict[str, Any]] = []
    best_checkpoint_selection_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch_index in range(1, experiment_config.training.num_epochs + 1):
        _apply_noise_aware_training_epoch_state(model, epoch_index=epoch_index)
        train_metrics = _run_noise_aware_epoch(
            model=model,
            data_loader=train_full_loader,
            optimizer=optimizer,
            experiment_config=experiment_config,
            objective_config=full_objective_config,
            noise_aware_config=noise_aware_config,
            positive_class_weight=positive_class_weight,
            split_name=f"train_full epoch={epoch_index:03d}",
            should_use_noise_aware_loss=True,
        )
        validation_metrics = _run_noise_aware_epoch(
            model=model,
            data_loader=validation_significant_loader,
            optimizer=None,
            experiment_config=experiment_config,
            objective_config=objective_config,
            noise_aware_config=noise_aware_config,
            positive_class_weight=positive_class_weight,
            split_name=f"validation_significant epoch={epoch_index:03d}",
            should_use_noise_aware_loss=False,
        )
        validation_full_metrics = None
        should_evaluate_validation_full = (
            noise_aware_config.should_evaluate_full_splits_each_epoch
            or noise_aware_config.checkpoint_selection_split_name == VALIDATION_FULL_SPLIT_NAME
        )
        if should_evaluate_validation_full:
            validation_full_metrics = _run_noise_aware_epoch(
                model=model,
                data_loader=validation_full_loader,
                optimizer=None,
                experiment_config=experiment_config,
                objective_config=full_objective_config,
                noise_aware_config=noise_aware_config,
                positive_class_weight=positive_class_weight,
                split_name=f"validation_full epoch={epoch_index:03d}",
                should_use_noise_aware_loss=False,
            )
        omitted_iterator_count = noise_aware_config.omitted_evaluation_iterator_count
        for _ in range(omitted_iterator_count):
            torch.empty((), dtype=torch.int64).random_().item()
        selection_metrics_by_split = {
            VALIDATION_SIGNIFICANT_SPLIT_NAME: validation_metrics,
        }
        if validation_full_metrics is not None:
            selection_metrics_by_split[VALIDATION_FULL_SPLIT_NAME] = validation_full_metrics
        checkpoint_selection_metrics = (
            validation_metrics
            if noise_aware_config.checkpoint_selection_split_name is None
            else selection_metrics_by_split[noise_aware_config.checkpoint_selection_split_name]
        )
        checkpoint_selection_score = _compute_noise_aware_checkpoint_selection_score(
            checkpoint_selection_metrics,
            noise_aware_config,
        )
        checkpoint_selection_metric_name = _resolve_noise_aware_checkpoint_selection_metric_name(
            noise_aware_config
        )
        is_base_frozen = not any(
            parameter.requires_grad
            for parameter in _iter_noise_aware_base_joint_parameters(model)
        )
        epoch_record = {
            "epoch": epoch_index,
            "train_full": train_metrics,
            "validation_significant": validation_metrics,
            "learning_rates": {
                str(parameter_group.get("name", f"group_{group_index}")): float(parameter_group["lr"])
                for group_index, parameter_group in enumerate(optimizer.param_groups)
            },
        }
        if noise_aware_config.checkpoint_selection_split_name is not None:
            epoch_record["checkpoint_selection"] = {
                "split_name": noise_aware_config.checkpoint_selection_split_name,
                "score_name": checkpoint_selection_metric_name,
                "score": checkpoint_selection_score,
                "is_base_frozen": is_base_frozen,
            }
        if validation_full_metrics is not None:
            epoch_record["validation_full"] = validation_full_metrics
        history.append(epoch_record)
        _save_checkpoint(
            latest_checkpoint_file_path,
            model,
            optimizer,
            epoch_index,
            validation_metrics,
            checkpoint_selection_score=checkpoint_selection_score,
            checkpoint_selection_metric_name=checkpoint_selection_metric_name,
            checkpoint_selection_split_name=noise_aware_config.checkpoint_selection_split_name,
        )
        should_save_checkpoint = (
            True
            if noise_aware_config.checkpoint_selection_split_name is None
            else checkpoint_selection_score > best_checkpoint_selection_score
        )
        if should_save_checkpoint:
            best_checkpoint_selection_score = float(checkpoint_selection_score)
            best_epoch = epoch_index
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_file_path,
                model,
                optimizer,
                epoch_index,
                validation_metrics,
                checkpoint_selection_score=checkpoint_selection_score,
                checkpoint_selection_metric_name=checkpoint_selection_metric_name,
                checkpoint_selection_split_name=noise_aware_config.checkpoint_selection_split_name,
            )
        elif noise_aware_config.checkpoint_selection_split_name is not None:
            epochs_without_improvement += 1
        epoch_log_payload = {
            **{f"train_full/{key}": value for key, value in train_metrics.items()},
            **{f"validation_significant/{key}": value for key, value in validation_metrics.items()},
            "epoch": epoch_index,
            **{
                f"learning_rate/{parameter_group.get('name', f'group_{group_index}')}": float(
                    parameter_group["lr"]
                )
                for group_index, parameter_group in enumerate(optimizer.param_groups)
            },
            "checkpoint_selection/score": checkpoint_selection_score,
            "checkpoint_selection/best_score": best_checkpoint_selection_score,
            "checkpoint_selection/is_base_frozen": float(is_base_frozen),
        }
        _safe_wandb_log(
            wandb_run,
            epoch_log_payload,
            log_context=f"noise-aware epoch {epoch_index}",
        )
        if validation_full_metrics is not None:
            full_split_log_payload = {}
            full_split_log_payload.update(
                {f"validation_full/{key}": value for key, value in validation_full_metrics.items()}
            )
            _safe_wandb_log(
                wandb_run,
                full_split_log_payload,
                log_context=f"noise-aware full-split epoch {epoch_index}",
            )
        if noise_aware_config.checkpoint_selection_split_name is None:
            LOGGER.info(
                "Epoch %03d final_epoch_checkpoint=%03d validation_score=%.6f base_frozen=%s",
                epoch_index,
                best_epoch,
                checkpoint_selection_score,
                is_base_frozen,
            )
        else:
            LOGGER.info(
                "Epoch %03d best_epoch=%03d best_%s_%s=%.6f current=%.6f no_improve=%d base_frozen=%s",
                epoch_index,
                best_epoch,
                noise_aware_config.checkpoint_selection_split_name,
                checkpoint_selection_metric_name,
                best_checkpoint_selection_score,
                checkpoint_selection_score,
                epochs_without_improvement,
                is_base_frozen,
            )
        if (
            noise_aware_config.checkpoint_selection_split_name is not None
            and epochs_without_improvement >= experiment_config.training.early_stopping_patience
        ):
            LOGGER.info("Early stopping triggered at epoch %03d.", epoch_index)
            break
    if best_epoch == 0:
        raise RuntimeError("No eligible checkpoint was selected.")
    write_json_file(run_directory / "training_history.json", history)
    return {
        "best_epoch": best_epoch,
        "best_validation_score": best_checkpoint_selection_score,
        "best_checkpoint_selection_score": (
            best_checkpoint_selection_score
            if noise_aware_config.checkpoint_selection_split_name is not None
            else None
        ),
        "checkpoint_selection_split_name": noise_aware_config.checkpoint_selection_split_name,
        "checkpoint_selection_metric_name": checkpoint_selection_metric_name,
        "history": history,
        "best_checkpoint_file_path": str(best_checkpoint_file_path),
        "positive_class_weight_from_significant_train": float(positive_class_weight.item()),
    }


def _evaluate_checkpoint(
    checkpoint_file_path: Path,
    model: FactorConcatLogitClassifierV1,
    data_loader: DataLoader,
    train_significant_loader: DataLoader,
    experiment_config,
    objective_config,
    noise_aware_config: FactorNoiseAwareTrainingConfig,
    split_name: str,
    run_directory: Path,
    wandb_run,
) -> dict[str, float]:
    """Evaluate one saved noise-aware checkpoint."""

    device = torch.device(experiment_config.experiment.device)
    _load_checkpoint(model=model, checkpoint_file_path=checkpoint_file_path, device=device)
    model.to(device)
    model.freeze_base_model()
    _freeze_base_and_enable_noise_aware_parameters(model)
    positive_class_weight = compute_positive_class_weight(
        _extract_dataset_label_values(train_significant_loader)
    ).to(device)
    metrics = _run_noise_aware_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=None,
        experiment_config=experiment_config,
        objective_config=objective_config,
        noise_aware_config=noise_aware_config,
        positive_class_weight=positive_class_weight,
        split_name=split_name,
        should_use_noise_aware_loss=False,
    )
    write_json_file(run_directory / f"{split_name}_metrics.json", metrics)
    _safe_wandb_log(
        wandb_run,
        {f"{split_name}/{key}": value for key, value in metrics.items()},
        log_context=f"{split_name} final evaluation",
    )
    return metrics


def main() -> None:
    """Run noise-aware factor concat-logit fine-tuning and evaluation."""

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--config-file", required=True, type=Path)
    argument_parser.add_argument("--run-directory", required=True, type=Path)
    parsed_arguments = argument_parser.parse_args()

    load_project_local_environment_files(PROJECT_ROOT)
    run_directory = parsed_arguments.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)
    (
        experiment_config,
        attention_side_adapter_config,
        objective_config,
        factor_classifier_config,
        noise_aware_config,
    ) = _load_config_with_noise_aware_extensions(parsed_arguments.config_file, run_directory)
    if experiment_config.model.framework_variant_name != EXPECTED_FRAMEWORK_VARIANT_NAME:
        raise ValueError(
            "Noise-aware factor concat runner expects model.framework_variant_name to equal "
            f"{EXPECTED_FRAMEWORK_VARIANT_NAME}."
        )
    if experiment_config.model.num_classes != 1:
        raise ValueError("Noise-aware factor concat runner requires model.num_classes to equal 1.")
    set_global_seed(experiment_config.experiment.random_seed)
    write_json_file(run_directory / "attention_side_adapter_config_snapshot.json", asdict(attention_side_adapter_config))
    write_json_file(run_directory / "single_logit_objective_config_snapshot.json", asdict(objective_config))
    write_json_file(run_directory / "factor_concat_classifier_config_snapshot.json", asdict(factor_classifier_config))
    write_json_file(run_directory / "noise_aware_training_config_snapshot.json", asdict(noise_aware_config))
    _write_single_logit_labeling_contract(experiment_config, objective_config, run_directory)

    wandb_run = None
    if experiment_config.logging.use_wandb:
        wandb_run = wandb.init(
            project=experiment_config.logging.wandb_project_name,
            entity=experiment_config.logging.wandb_entity_name,
            mode=experiment_config.logging.wandb_mode,
            dir=str(run_directory),
            name=run_directory.name,
            config={
                **asdict(experiment_config),
                ATTENTION_SIDE_ADAPTER_SECTION_NAME: asdict(attention_side_adapter_config),
                SINGLE_LOGIT_SECTION_NAME: asdict(objective_config),
                FACTOR_CONCAT_CLASSIFIER_SECTION_NAME: asdict(factor_classifier_config),
                NOISE_AWARE_SECTION_NAME: asdict(noise_aware_config),
            },
            settings=wandb.Settings(init_timeout=300),
        )

    full_objective_config = replace(objective_config, significant_return_absolute_threshold=0.0)
    train_full_loader = _build_data_loader(experiment_config, full_objective_config, factor_classifier_config, "train", True)
    train_significant_loader = _build_data_loader(experiment_config, objective_config, factor_classifier_config, "train", False)
    validation_significant_loader = _build_data_loader(
        experiment_config,
        objective_config,
        factor_classifier_config,
        "validation",
        False,
    )
    validation_full_loader = _build_data_loader(
        experiment_config,
        full_objective_config,
        factor_classifier_config,
        "validation",
        False,
    )
    test_significant_loader = _build_data_loader(experiment_config, objective_config, factor_classifier_config, "test", False)
    test_full_loader = _build_data_loader(experiment_config, full_objective_config, factor_classifier_config, "test", False)
    model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
    fine_tune_checkpoint_file_path: Path | None = None
    base_checkpoint_file_path: Path | None = None
    checkpoint_initialization_type: str
    if noise_aware_config.fine_tune_checkpoint_file_path is not None:
        fine_tune_checkpoint_file_path = Path(noise_aware_config.fine_tune_checkpoint_file_path)
        if not fine_tune_checkpoint_file_path.is_absolute():
            fine_tune_checkpoint_file_path = (PROJECT_ROOT / fine_tune_checkpoint_file_path).resolve()
        _load_checkpoint(
            model=model,
            checkpoint_file_path=fine_tune_checkpoint_file_path,
            device=torch.device(experiment_config.experiment.device),
        )
        checkpoint_initialization_type = "factor_wrapper_fine_tune"
    elif noise_aware_config.base_checkpoint_file_path is not None:
        base_checkpoint_file_path = Path(noise_aware_config.base_checkpoint_file_path)
        if not base_checkpoint_file_path.is_absolute():
            base_checkpoint_file_path = (PROJECT_ROOT / base_checkpoint_file_path).resolve()
        _load_frozen_base_checkpoint(
            model=model,
            checkpoint_file_path=base_checkpoint_file_path,
        )
        checkpoint_initialization_type = "frozen_base_initialize_factor_wrapper"
    else:
        raise ConfigurationError(
            "factor_noise_aware_training must provide a checkpoint initialization path."
        )
    _apply_noise_aware_training_epoch_state(model, epoch_index=1)
    write_json_file(
        run_directory / "trainable_parameter_summary.json",
        {
            "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "base_trainable_parameters": int(
                sum(parameter.numel() for parameter in model.base_model.parameters() if parameter.requires_grad)
            ),
            "checkpoint_initialization_type": checkpoint_initialization_type,
        },
    )
    (run_directory / "model_architecture.txt").write_text(str(model), encoding="utf-8")

    training_summary = _fit_model(
        model=model,
        train_full_loader=train_full_loader,
        train_significant_loader=train_significant_loader,
        validation_significant_loader=validation_significant_loader,
        validation_full_loader=validation_full_loader,
        experiment_config=experiment_config,
        objective_config=objective_config,
        full_objective_config=full_objective_config,
        noise_aware_config=noise_aware_config,
        run_directory=run_directory,
        wandb_run=wandb_run,
    )
    best_checkpoint_file_path = Path(str(training_summary["best_checkpoint_file_path"]))

    def _build_loaded_model() -> FactorConcatLogitClassifierV1:
        loaded_model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
        return loaded_model

    validation_significant_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        _build_loaded_model(),
        validation_significant_loader,
        train_significant_loader,
        experiment_config,
        objective_config,
        noise_aware_config,
        "validation_significant",
        run_directory,
        wandb_run,
    )
    validation_full_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        _build_loaded_model(),
        validation_full_loader,
        train_significant_loader,
        experiment_config,
        full_objective_config,
        noise_aware_config,
        "validation_full",
        run_directory,
        wandb_run,
    )
    test_significant_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        _build_loaded_model(),
        test_significant_loader,
        train_significant_loader,
        experiment_config,
        objective_config,
        noise_aware_config,
        "test_significant",
        run_directory,
        wandb_run,
    )
    test_full_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        _build_loaded_model(),
        test_full_loader,
        train_significant_loader,
        experiment_config,
        full_objective_config,
        noise_aware_config,
        "test_full",
        run_directory,
        wandb_run,
    )
    pipeline_summary = {
        "config_file_path": str(parsed_arguments.config_file),
        "run_directory": str(run_directory),
        "fine_tune_checkpoint_file_path": (
            str(fine_tune_checkpoint_file_path)
            if fine_tune_checkpoint_file_path is not None
            else None
        ),
        "base_checkpoint_file_path": (
            str(base_checkpoint_file_path)
            if base_checkpoint_file_path is not None
            else None
        ),
        "checkpoint_initialization_type": checkpoint_initialization_type,
        "checkpoint_file_path": str(best_checkpoint_file_path),
        "attention_side_adapter_config": asdict(attention_side_adapter_config),
        "single_logit_objective_config": asdict(objective_config),
        "factor_concat_classifier_config": asdict(factor_classifier_config),
        "noise_aware_training_config": asdict(noise_aware_config),
        "training_summary": training_summary,
        "validation_significant_metrics": validation_significant_metrics,
        "validation_full_metrics": validation_full_metrics,
        "test_significant_metrics": test_significant_metrics,
        "test_full_metrics": test_full_metrics,
    }
    write_json_file(run_directory / "pipeline_summary.json", pipeline_summary)
    LOGGER.info("Noise-aware factor concat-logit pipeline finished: %s", pipeline_summary)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
