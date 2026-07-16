"""Single-logit binary BCE training with return-aware soft labels and calibration diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F

from priorstock.config import ExperimentConfig
from priorstock.training.engine import (
    _build_distribution_record,
    _build_distribution_trace_file_path,
    _build_epoch_diagnostics_file_path,
    _collect_grouped_parameter_samples,
    _compute_gradient_norm,
    _log_distribution_payload_to_wandb,
    _safe_wandb_log,
    _should_collect_trace_tensors,
    _summarize_parameter_groups,
    _summarize_trace_tensors,
    _write_run_metadata,
)
from priorstock.utils.io import append_jsonl_records, write_json_file
from priorstock.utils.logging_utils import configure_logger, write_tensor_diagnostics


LOGGER = configure_logger("single_logit_binary_training_engine")
ADAPTER_PARAMETER_NAME_FRAGMENT = "attention_side_adapter"


@dataclass(frozen=True)
class SingleLogitBinaryObjectiveConfig:
    """Script-level hyperparameters for single-logit binary training."""

    should_use_return_soft_targets: bool
    should_resplit_by_config_dates: bool
    should_evaluate_unfiltered_test_split: bool
    return_soft_label_temperature: float
    significant_return_absolute_threshold: float
    adapter_learning_rate_multiplier: float
    adapter_freeze_epoch_count: int
    checkpoint_selection_mcc_weight: float
    checkpoint_selection_balanced_accuracy_weight: float
    checkpoint_selection_macro_f1_weight: float
    degenerate_high_prediction_up_ratio: float
    degenerate_low_prediction_up_ratio: float
    degenerate_recall_gap_threshold: float
    near_boundary_probability_margin: float
    calibration_bin_count: int
    return_bucket_absolute_thresholds: tuple[float, ...]
    large_return_absolute_thresholds: tuple[float, ...]
    checkpoint_selection_accuracy_weight: float = 0.0
    checkpoint_selection_split_name: str = "validation"
    checkpoint_selection_epoch_limit: int = 0
    should_evaluate_full_splits_each_epoch: bool = False
    omitted_evaluation_iterator_count: int = 0
    fixed_checkpoint_epoch: int = 0


@dataclass(frozen=True)
class SingleLogitCheckpointSelection:
    """Checkpoint selection result for one epoch."""

    split_name: str
    score_name: str
    score: float
    metrics: dict[str, float]


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move one collated batch dictionary to the selected device."""

    return {
        "sample_id": batch["sample_id"],
        "stock_id": batch["stock_id"],
        "target_trade_date": batch["target_trade_date"],
        "price_features": batch["price_features"].to(device),
        "technical_indicator_features": batch["technical_indicator_features"].to(device),
        "news_embeddings": batch["news_embeddings"].to(device),
        "has_news": batch["has_news"].to(device),
        "label": batch["label"].to(device),
        "target_return": batch["target_return"].to(device),
        "soft_up_target": batch["soft_up_target"].to(device),
    }


def _extract_dataset_label_values(data_loader) -> list[int]:
    """Return all hard binary labels from a dataloader's dataset."""

    dataset = data_loader.dataset
    if hasattr(dataset, "get_label_values"):
        return [int(label_value) for label_value in dataset.get_label_values()]
    return [int(dataset[sample_index]["label"].item()) for sample_index in range(len(dataset))]


def compute_positive_class_weight(label_values: list[int]) -> torch.Tensor:
    """Compute BCE pos_weight as n_down / n_up from hard labels."""

    positive_count = sum(1 for label_value in label_values if int(label_value) == 1)
    negative_count = sum(1 for label_value in label_values if int(label_value) == 0)
    if positive_count <= 0:
        raise ValueError("Training split has no UP samples, so BCE pos_weight cannot be computed.")
    if negative_count <= 0:
        raise ValueError("Training split has no DOWN samples, so BCE pos_weight cannot be computed.")
    return torch.tensor(float(negative_count / positive_count), dtype=torch.float32)


def compute_single_logit_bce_loss(
    logits: torch.Tensor,
    soft_up_targets: torch.Tensor,
    positive_class_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute single-logit BCEWithLogitsLoss against return-aware soft UP targets."""

    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError("Single-logit binary training requires logits with shape [batch_size, 1].")
    logit_values = logits.squeeze(dim=-1)
    return F.binary_cross_entropy_with_logits(
        logit_values,
        soft_up_targets.to(dtype=logit_values.dtype),
        pos_weight=positive_class_weight.to(device=logit_values.device, dtype=logit_values.dtype),
    )


def compute_single_logit_soft_mcc_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute a differentiable batch-level MCC surrogate for one-logit binary outputs."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError("Single-logit soft MCC loss requires logits with shape [batch_size, 1].")
    logit_values = logits.squeeze(dim=-1)
    up_probability = torch.sigmoid(logit_values)
    label_values = hard_labels.to(device=logit_values.device, dtype=logit_values.dtype)
    true_positive = torch.sum(label_values * up_probability)
    true_negative = torch.sum((1.0 - label_values) * (1.0 - up_probability))
    false_positive = torch.sum((1.0 - label_values) * up_probability)
    false_negative = torch.sum(label_values * (1.0 - up_probability))
    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = torch.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
        + epsilon
    )
    soft_mcc = numerator / denominator.clamp_min(epsilon)
    return 1.0 - soft_mcc


def compute_single_logit_soft_balanced_accuracy_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute a differentiable batch-level balanced-accuracy surrogate."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError("Single-logit soft balanced accuracy loss requires logits with shape [batch_size, 1].")
    logit_values = logits.squeeze(dim=-1)
    up_probability = torch.sigmoid(logit_values)
    label_values = hard_labels.to(device=logit_values.device, dtype=logit_values.dtype)
    positive_mass = label_values.sum()
    negative_mass = (1.0 - label_values).sum()
    soft_recall_terms: list[torch.Tensor] = []
    if bool((positive_mass > 0.0).detach().cpu().item()):
        soft_recall_terms.append((label_values * up_probability).sum() / positive_mass.clamp_min(epsilon))
    if bool((negative_mass > 0.0).detach().cpu().item()):
        soft_recall_terms.append(
            ((1.0 - label_values) * (1.0 - up_probability)).sum() / negative_mass.clamp_min(epsilon)
        )
    if not soft_recall_terms:
        return logit_values.sum() * 0.0
    soft_balanced_accuracy = torch.stack(soft_recall_terms).mean()
    return 1.0 - soft_balanced_accuracy


def compute_return_aware_soft_up_targets(
    target_returns: torch.Tensor,
    return_soft_label_temperature: float,
) -> torch.Tensor:
    """Map realized returns to soft UP targets with 0.5 + 0.5 * tanh(return / tau)."""

    if return_soft_label_temperature <= 0.0:
        raise ValueError("return_soft_label_temperature must be positive.")
    return 0.5 + (0.5 * torch.tanh(target_returns / return_soft_label_temperature))


def compute_up_probability_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one logit into UP probability and logit margin."""

    if logits.ndim != 2 or logits.shape[-1] != 1:
        raise ValueError("Single-logit binary logits must have shape [batch_size, 1].")
    logit_margin = logits.squeeze(dim=-1)
    up_probability = torch.sigmoid(logit_margin)
    return up_probability, logit_margin


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    """Return one finite quantile value for a non-empty numpy array."""

    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def _compute_basic_binary_metrics(
    true_label_array: np.ndarray,
    prediction_array: np.ndarray,
    probability_array: np.ndarray,
    margin_array: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Compute binary classification and confidence metrics for one subset."""

    if true_label_array.size == 0:
        return {
            f"{prefix}/count": 0.0,
            f"{prefix}/accuracy": 0.0,
            f"{prefix}/balanced_accuracy": 0.0,
            f"{prefix}/mcc": 0.0,
            f"{prefix}/pred_up_ratio": 0.0,
            f"{prefix}/probability_mean": 0.0,
            f"{prefix}/mean_absolute_margin": 0.0,
            f"{prefix}/brier_score": 0.0,
        }
    return {
        f"{prefix}/count": float(true_label_array.size),
        f"{prefix}/accuracy": float(accuracy_score(true_label_array, prediction_array)),
        f"{prefix}/balanced_accuracy": float(
            balanced_accuracy_score(true_label_array, prediction_array)
        ),
        f"{prefix}/mcc": float(matthews_corrcoef(true_label_array, prediction_array)),
        f"{prefix}/pred_up_ratio": float(prediction_array.mean()),
        f"{prefix}/probability_mean": float(probability_array.mean()),
        f"{prefix}/mean_absolute_margin": float(np.abs(margin_array).mean()),
        f"{prefix}/brier_score": float(np.mean(np.square(probability_array - true_label_array))),
    }


def _compute_binary_roc_auc(true_label_array: np.ndarray, probability_array: np.ndarray) -> float:
    """Compute ROC-AUC, returning the random baseline value when one class is absent."""

    if true_label_array.size == 0 or len(np.unique(true_label_array)) < 2:
        return 0.5
    return float(roc_auc_score(true_label_array, probability_array))


def _format_threshold_for_key(threshold_value: float) -> str:
    """Format a return threshold into a stable metric key fragment."""

    return f"{threshold_value:.3f}".replace(".", "_")


def _compute_return_bucket_metrics(
    true_label_array: np.ndarray,
    prediction_array: np.ndarray,
    probability_array: np.ndarray,
    margin_array: np.ndarray,
    return_array: np.ndarray,
    objective_config: SingleLogitBinaryObjectiveConfig,
) -> dict[str, float]:
    """Compute performance metrics for configured absolute-return buckets."""

    metrics: dict[str, float] = {}
    absolute_returns = np.abs(return_array)
    for threshold_value in objective_config.return_bucket_absolute_thresholds:
        mask = absolute_returns <= threshold_value
        key = f"return_bucket_abs_le_{_format_threshold_for_key(threshold_value)}"
        metrics.update(
            _compute_basic_binary_metrics(
                true_label_array=true_label_array[mask],
                prediction_array=prediction_array[mask],
                probability_array=probability_array[mask],
                margin_array=margin_array[mask],
                prefix=key,
            )
        )
    for threshold_value in objective_config.large_return_absolute_thresholds:
        mask = absolute_returns > threshold_value
        key = f"return_bucket_abs_gt_{_format_threshold_for_key(threshold_value)}"
        metrics.update(
            _compute_basic_binary_metrics(
                true_label_array=true_label_array[mask],
                prediction_array=prediction_array[mask],
                probability_array=probability_array[mask],
                margin_array=margin_array[mask],
                prefix=key,
            )
        )
    return metrics


def _compute_expected_calibration_error(
    true_label_array: np.ndarray,
    prediction_array: np.ndarray,
    probability_array: np.ndarray,
    calibration_bin_count: int,
) -> float:
    """Compute binary ECE from prediction confidence bins."""

    if true_label_array.size == 0:
        return 0.0
    if calibration_bin_count <= 0:
        raise ValueError("calibration_bin_count must be positive.")
    confidence_array = np.maximum(probability_array, 1.0 - probability_array)
    correctness_array = (prediction_array == true_label_array).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, calibration_bin_count + 1)
    expected_calibration_error = 0.0
    for bin_index in range(calibration_bin_count):
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]
        if bin_index == calibration_bin_count - 1:
            mask = (confidence_array >= lower_bound) & (confidence_array <= upper_bound)
        else:
            mask = (confidence_array >= lower_bound) & (confidence_array < upper_bound)
        if not np.any(mask):
            continue
        bin_weight = float(mask.mean())
        expected_calibration_error += bin_weight * abs(
            float(correctness_array[mask].mean()) - float(confidence_array[mask].mean())
        )
    return float(expected_calibration_error)


def _compute_probability_decile_metrics(
    true_label_array: np.ndarray,
    prediction_array: np.ndarray,
    probability_array: np.ndarray,
    calibration_bin_count: int,
) -> dict[str, float]:
    """Compute accuracy and UP-rate diagnostics over UP-probability deciles."""

    metrics: dict[str, float] = {}
    bin_edges = np.linspace(0.0, 1.0, calibration_bin_count + 1)
    for bin_index in range(calibration_bin_count):
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]
        if bin_index == calibration_bin_count - 1:
            mask = (probability_array >= lower_bound) & (probability_array <= upper_bound)
        else:
            mask = (probability_array >= lower_bound) & (probability_array < upper_bound)
        prefix = f"probability_decile_{bin_index:02d}"
        if not np.any(mask):
            metrics[f"{prefix}/count"] = 0.0
            metrics[f"{prefix}/accuracy"] = 0.0
            metrics[f"{prefix}/true_up_ratio"] = 0.0
            metrics[f"{prefix}/pred_up_ratio"] = 0.0
            metrics[f"{prefix}/probability_mean"] = 0.0
            continue
        metrics[f"{prefix}/count"] = float(mask.sum())
        metrics[f"{prefix}/accuracy"] = float(
            accuracy_score(true_label_array[mask], prediction_array[mask])
        )
        metrics[f"{prefix}/true_up_ratio"] = float(true_label_array[mask].mean())
        metrics[f"{prefix}/pred_up_ratio"] = float(prediction_array[mask].mean())
        metrics[f"{prefix}/probability_mean"] = float(probability_array[mask].mean())
    return metrics


def compute_single_logit_binary_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    up_probabilities: list[float],
    logit_margins: list[float],
    target_returns: list[float],
    soft_up_targets: list[float],
    objective_config: SingleLogitBinaryObjectiveConfig,
) -> dict[str, float]:
    """Compute binary metrics, calibration diagnostics, and return-bucket diagnostics."""

    probability_array = np.asarray(up_probabilities, dtype=np.float64)
    margin_array = np.asarray(logit_margins, dtype=np.float64)
    prediction_array = np.asarray(predicted_labels, dtype=np.int64)
    true_label_array = np.asarray(true_labels, dtype=np.int64)
    return_array = np.asarray(target_returns, dtype=np.float64)
    soft_target_array = np.asarray(soft_up_targets, dtype=np.float64)
    near_boundary_mask = (
        np.abs(probability_array - 0.5)
        <= objective_config.near_boundary_probability_margin
    )
    near_boundary_count = int(near_boundary_mask.sum())
    near_boundary_accuracy = 0.0
    if near_boundary_count > 0:
        near_boundary_accuracy = float(
            np.mean(prediction_array[near_boundary_mask] == true_label_array[near_boundary_mask])
        )

    up_recall = float(
        recall_score(true_labels, predicted_labels, labels=[1], average="macro", zero_division=0)
    )
    down_recall = float(
        recall_score(true_labels, predicted_labels, labels=[0], average="macro", zero_division=0)
    )
    prediction_up_ratio = float(prediction_array.mean()) if prediction_array.size else 0.0
    degenerate_high_prediction_up_ratio = (
        prediction_up_ratio > objective_config.degenerate_high_prediction_up_ratio
    )
    degenerate_low_prediction_up_ratio = (
        prediction_up_ratio < objective_config.degenerate_low_prediction_up_ratio
    )
    degenerate_recall_gap = (
        abs(up_recall - down_recall) > objective_config.degenerate_recall_gap_threshold
    )

    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
        "mcc": float(matthews_corrcoef(true_labels, predicted_labels)),
        "roc_auc": _compute_binary_roc_auc(true_label_array, probability_array),
        "up_precision": float(
            precision_score(true_labels, predicted_labels, labels=[1], average="macro", zero_division=0)
        ),
        "down_precision": float(
            precision_score(true_labels, predicted_labels, labels=[0], average="macro", zero_division=0)
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "prediction_up_ratio": prediction_up_ratio,
        "probability_mean": float(probability_array.mean()) if probability_array.size else 0.0,
        "probability_std": float(probability_array.std()) if probability_array.size else 0.0,
        "probability_min": float(probability_array.min()) if probability_array.size else 0.0,
        "probability_max": float(probability_array.max()) if probability_array.size else 0.0,
        "logit_margin_mean": float(margin_array.mean()) if margin_array.size else 0.0,
        "logit_margin_std": float(margin_array.std()) if margin_array.size else 0.0,
        "logit_margin_min": float(margin_array.min()) if margin_array.size else 0.0,
        "logit_margin_max": float(margin_array.max()) if margin_array.size else 0.0,
        "logit_margin_abs_mean": float(np.abs(margin_array).mean()) if margin_array.size else 0.0,
        "logit_margin_p05": _safe_quantile(margin_array, 0.05),
        "logit_margin_p50": _safe_quantile(margin_array, 0.50),
        "logit_margin_p95": _safe_quantile(margin_array, 0.95),
        "near_boundary_bucket_accuracy": near_boundary_accuracy,
        "near_boundary_bucket_count": float(near_boundary_count),
        "near_boundary_bucket_fraction": (
            float(near_boundary_count / probability_array.size)
            if probability_array.size
            else 0.0
        ),
        "brier_score": float(np.mean(np.square(probability_array - true_label_array))),
        "soft_target_brier_score": float(np.mean(np.square(probability_array - soft_target_array))),
        "ece": _compute_expected_calibration_error(
            true_label_array=true_label_array,
            prediction_array=prediction_array,
            probability_array=probability_array,
            calibration_bin_count=objective_config.calibration_bin_count,
        ),
        "true_up_ratio": float(true_label_array.mean()) if true_label_array.size else 0.0,
        "target_return_mean": float(return_array.mean()) if return_array.size else 0.0,
        "target_return_abs_mean": float(np.abs(return_array).mean()) if return_array.size else 0.0,
        "soft_up_target_mean": float(soft_target_array.mean()) if soft_target_array.size else 0.0,
        "is_degenerate_checkpoint": float(
            degenerate_high_prediction_up_ratio
            or degenerate_low_prediction_up_ratio
            or degenerate_recall_gap
        ),
        "degenerate_high_prediction_up_ratio": float(degenerate_high_prediction_up_ratio),
        "degenerate_low_prediction_up_ratio": float(degenerate_low_prediction_up_ratio),
        "degenerate_recall_gap": float(degenerate_recall_gap),
        "recall_gap_abs": float(abs(up_recall - down_recall)),
    }
    metrics.update(
        _compute_return_bucket_metrics(
            true_label_array=true_label_array,
            prediction_array=prediction_array,
            probability_array=probability_array,
            margin_array=margin_array,
            return_array=return_array,
            objective_config=objective_config,
        )
    )
    metrics.update(
        _compute_probability_decile_metrics(
            true_label_array=true_label_array,
            prediction_array=prediction_array,
            probability_array=probability_array,
            calibration_bin_count=objective_config.calibration_bin_count,
        )
    )
    return metrics


def compute_single_logit_checkpoint_selection_score(
    metrics: dict[str, float],
    objective_config: SingleLogitBinaryObjectiveConfig,
) -> float:
    """Compute the configured checkpoint score from MCC, accuracy, balanced accuracy, and macro-F1."""

    return float(
        (objective_config.checkpoint_selection_mcc_weight * metrics["mcc"])
        + (objective_config.checkpoint_selection_accuracy_weight * metrics["accuracy"])
        + (
            objective_config.checkpoint_selection_balanced_accuracy_weight
            * metrics["balanced_accuracy"]
        )
        + (objective_config.checkpoint_selection_macro_f1_weight * metrics["macro_f1"])
    )


def should_consider_checkpoint_selection_epoch(
    epoch_index: int,
    objective_config: SingleLogitBinaryObjectiveConfig,
) -> bool:
    """Return whether an epoch is inside the configured checkpoint-selection window."""

    if objective_config.checkpoint_selection_epoch_limit <= 0:
        return True
    return epoch_index <= objective_config.checkpoint_selection_epoch_limit


def select_single_logit_checkpoint_metrics(
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
    objective_config: SingleLogitBinaryObjectiveConfig,
    validation_full_metrics: dict[str, float] | None = None,
    test_full_metrics: dict[str, float] | None = None,
) -> SingleLogitCheckpointSelection:
    """Select the metrics dictionary used for single-logit checkpoint monitoring."""

    split_name = objective_config.checkpoint_selection_split_name
    metrics_by_split_name = {
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_full": validation_full_metrics,
        "test_full": test_full_metrics,
    }
    if split_name not in metrics_by_split_name:
        raise ValueError(
            "checkpoint_selection_split_name must be one of 'validation', 'test', "
            "'validation_full', or 'test_full'."
        )
    metrics = metrics_by_split_name[split_name]
    if metrics is None:
        raise ValueError(f"{split_name} checkpoint selection requires {split_name}_metrics.")
    return SingleLogitCheckpointSelection(
        split_name=split_name,
        score_name="checkpoint_selection_score",
        score=compute_single_logit_checkpoint_selection_score(metrics, objective_config),
        metrics=metrics,
    )


def _format_single_logit_metrics_for_log(metrics: dict[str, float]) -> str:
    """Format one metrics dictionary into a compact stable log string."""

    ordered_metric_names = [
        "total_loss",
        "binary_cross_entropy_loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "roc_auc",
        "up_precision",
        "up_recall",
        "down_recall",
        "prediction_up_ratio",
        "probability_mean",
        "logit_margin_abs_mean",
        "brier_score",
        "ece",
        "near_boundary_bucket_accuracy",
        "is_degenerate_checkpoint",
    ]
    formatted_parts: list[str] = []
    for metric_name in ordered_metric_names:
        if metric_name in metrics:
            formatted_parts.append(f"{metric_name}={metrics[metric_name]:.6f}")
    return " | ".join(formatted_parts)


def _set_adapter_requires_grad(model: nn.Module, should_require_grad: bool) -> None:
    """Enable or disable gradients for attention-side adapter parameters only."""

    for parameter_name, parameter in model.named_parameters():
        if ADAPTER_PARAMETER_NAME_FRAGMENT in parameter_name:
            parameter.requires_grad = should_require_grad


def _build_optimizer(
    model: nn.Module,
    experiment_config: ExperimentConfig,
    objective_config: SingleLogitBinaryObjectiveConfig,
) -> torch.optim.Optimizer:
    """Create AdamW with a lower learning rate for attention-side adapter parameters."""

    adapter_parameters: list[nn.Parameter] = []
    base_parameters: list[nn.Parameter] = []
    for parameter_name, parameter in model.named_parameters():
        if ADAPTER_PARAMETER_NAME_FRAGMENT in parameter_name:
            adapter_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": experiment_config.training.learning_rate,
                "parameter_group_name": "base",
            },
            {
                "params": adapter_parameters,
                "lr": (
                    experiment_config.training.learning_rate
                    * objective_config.adapter_learning_rate_multiplier
                ),
                "parameter_group_name": "attention_side_adapter",
            },
        ],
        weight_decay=experiment_config.training.weight_decay,
        betas=(experiment_config.training.adam_beta_one, experiment_config.training.adam_beta_two),
        eps=experiment_config.training.adam_epsilon,
    )


def run_single_logit_binary_epoch(
    model: nn.Module,
    data_loader,
    positive_class_weight: torch.Tensor,
    optimizer,
    experiment_config: ExperimentConfig,
    objective_config: SingleLogitBinaryObjectiveConfig,
    device: torch.device,
    split_name: str,
    epoch_index: int,
    diagnostics_file_path: Path,
    wandb_run,
) -> dict[str, float]:
    """Run one single-logit binary train/evaluation epoch."""

    is_training = optimizer is not None
    should_manage_cuda_cache = device.type == "cuda"
    if is_training:
        model.train()
    else:
        model.eval()

    accumulated_loss_value = 0.0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    up_probabilities: list[float] = []
    logit_margins: list[float] = []
    target_returns: list[float] = []
    soft_up_targets: list[float] = []
    first_batch_diagnostics_written = False
    distribution_trace_file_path = _build_distribution_trace_file_path(
        run_directory=diagnostics_file_path.parents[2],
        experiment_config=experiment_config,
        split_name=split_name,
    )

    for step_index, raw_batch in enumerate(data_loader):
        batch = _move_batch_to_device(raw_batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        should_collect_trace_tensors = _should_collect_trace_tensors(
            experiment_config=experiment_config,
            split_name=split_name,
            step_index=step_index,
        )

        with torch.set_grad_enabled(is_training):
            model_output = model(
                price_features=batch["price_features"],
                technical_indicator_features=batch["technical_indicator_features"],
                news_embeddings=batch["news_embeddings"],
                has_news=batch["has_news"],
                collect_trace_tensors=should_collect_trace_tensors,
            )
            bce_loss = compute_single_logit_bce_loss(
                logits=model_output.logits,
                soft_up_targets=batch["soft_up_target"],
                positive_class_weight=positive_class_weight,
            )
            total_loss = bce_loss
            if is_training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=experiment_config.training.gradient_clip_max_norm,
                )
                optimizer.step()

        up_probability, logit_margin = compute_up_probability_and_margin(model_output.logits.detach())
        predictions = (logit_margin > 0.0).to(dtype=torch.long)
        true_labels.extend(batch["label"].detach().cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        up_probabilities.extend(up_probability.detach().cpu().tolist())
        logit_margins.extend(logit_margin.detach().cpu().tolist())
        target_returns.extend(batch["target_return"].detach().cpu().tolist())
        soft_up_targets.extend(batch["soft_up_target"].detach().cpu().tolist())
        accumulated_loss_value += float(total_loss.detach().cpu())

        if should_collect_trace_tensors:
            model_output.trace_tensors["single_logit_up_probability"] = up_probability.detach().to(
                device="cpu"
            )
            model_output.trace_tensors["single_logit_margin"] = logit_margin.detach().to(
                device="cpu"
            )
            model_output.trace_tensors["target_return"] = batch["target_return"].detach().to(
                device="cpu"
            )
            model_output.trace_tensors["soft_up_target"] = batch["soft_up_target"].detach().to(
                device="cpu"
            )

        if (
            experiment_config.logging.log_first_batch_tensor_statistics
            and not first_batch_diagnostics_written
            and should_collect_trace_tensors
        ):
            diagnostics_payload = {
                "epoch_index": epoch_index,
                "split_name": split_name,
                "sample_ids": list(batch["sample_id"]),
                "model_diagnostics": model_output.diagnostics,
                "single_logit_probability_mean": float(up_probability.detach().mean().cpu().item()),
                "single_logit_margin_mean": float(logit_margin.detach().mean().cpu().item()),
                "soft_up_target_mean": float(batch["soft_up_target"].detach().mean().cpu().item()),
            }
            if is_training and experiment_config.logging.log_gradient_statistics:
                diagnostics_payload["gradient_norm"] = _compute_gradient_norm(model)
            write_tensor_diagnostics(diagnostics_file_path, diagnostics_payload)
            first_batch_diagnostics_written = True

        if should_collect_trace_tensors:
            scalar_metrics = {
                "step_total_loss": float(total_loss.detach().cpu()),
                "step_binary_cross_entropy_loss": float(bce_loss.detach().cpu()),
                "step_probability_mean": float(up_probability.detach().mean().cpu()),
                "step_logit_margin_mean": float(logit_margin.detach().mean().cpu()),
                "step_soft_up_target_mean": float(batch["soft_up_target"].detach().mean().cpu()),
                "step_target_return_abs_mean": float(batch["target_return"].detach().abs().mean().cpu()),
            }
            if is_training and experiment_config.logging.log_gradient_statistics:
                scalar_metrics["gradient_norm"] = _compute_gradient_norm(model)

            trace_summaries, trace_histograms = _summarize_trace_tensors(
                model_output.trace_tensors,
                experiment_config,
            )
            parameter_summaries: dict[str, dict] = {}
            parameter_histograms: dict[str, wandb.Histogram] = {}
            if experiment_config.logging.log_parameter_value_distributions:
                parameter_value_groups = _collect_grouped_parameter_samples(
                    model=model,
                    include_gradients=False,
                    max_histogram_sample_count=experiment_config.logging.max_histogram_sample_count,
                )
                parameter_summaries, parameter_histograms = _summarize_parameter_groups(
                    parameter_value_groups,
                    experiment_config,
                )

            gradient_summaries: dict[str, dict] = {}
            gradient_histograms: dict[str, wandb.Histogram] = {}
            if is_training and experiment_config.logging.log_gradient_distributions:
                gradient_groups = _collect_grouped_parameter_samples(
                    model=model,
                    include_gradients=True,
                    max_histogram_sample_count=experiment_config.logging.max_histogram_sample_count,
                )
                gradient_summaries, gradient_histograms = _summarize_parameter_groups(
                    gradient_groups,
                    experiment_config,
                )

            distribution_record = _build_distribution_record(
                experiment_config=experiment_config,
                split_name=split_name,
                epoch_index=epoch_index,
                step_index=step_index,
                sample_ids=list(batch["sample_id"]),
                scalar_metrics=scalar_metrics,
                trace_summaries=trace_summaries,
                parameter_summaries=parameter_summaries,
                gradient_summaries=gradient_summaries,
            )
            append_jsonl_records(distribution_trace_file_path, [distribution_record])
            _log_distribution_payload_to_wandb(
                wandb_run=wandb_run,
                split_name=split_name,
                epoch_index=epoch_index,
                step_index=step_index,
                scalar_metrics=scalar_metrics,
                trace_summaries=trace_summaries,
                trace_histograms=trace_histograms,
                parameter_summaries=parameter_summaries,
                parameter_histograms=parameter_histograms,
                gradient_summaries=gradient_summaries,
                gradient_histograms=gradient_histograms,
            )
            model_output.trace_tensors.clear()

        if (
            wandb_run is not None
            and is_training
            and (step_index + 1) % experiment_config.logging.log_every_training_step_count == 0
        ):
            _safe_wandb_log(
                wandb_run=wandb_run,
                log_payload={
                    "train/step_total_loss": float(total_loss.detach().cpu()),
                    "train/step_binary_cross_entropy_loss": float(bce_loss.detach().cpu()),
                    "train/step_probability_mean": float(up_probability.detach().mean().cpu()),
                    "train/step_logit_margin_mean": float(logit_margin.detach().mean().cpu()),
                    "train/step_soft_up_target_mean": float(
                        batch["soft_up_target"].detach().mean().cpu()
                    ),
                    "train/epoch": epoch_index,
                },
                log_context=f"single-logit train step metrics at epoch {epoch_index} step {step_index + 1}",
            )

        del predictions
        del model_output
        del total_loss
        del bce_loss
        del batch
        if should_manage_cuda_cache and should_collect_trace_tensors:
            torch.cuda.empty_cache()

    metrics = compute_single_logit_binary_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        up_probabilities=up_probabilities,
        logit_margins=logit_margins,
        target_returns=target_returns,
        soft_up_targets=soft_up_targets,
        objective_config=objective_config,
    )
    batch_count = max(len(data_loader), 1)
    metrics.update(
        {
            "total_loss": accumulated_loss_value / batch_count,
            "binary_cross_entropy_loss": accumulated_loss_value / batch_count,
        }
    )
    if should_manage_cuda_cache:
        torch.cuda.empty_cache()
    return metrics


def fit_single_logit_binary_model(
    model: nn.Module,
    train_data_loader,
    validation_data_loader,
    experiment_config: ExperimentConfig,
    objective_config: SingleLogitBinaryObjectiveConfig,
    run_directory: Path,
    wandb_run=None,
    should_finish_wandb_run: bool = True,
    checkpoint_selection_data_loader=None,
    validation_full_data_loader=None,
    test_full_data_loader=None,
    full_objective_config: SingleLogitBinaryObjectiveConfig | None = None,
) -> dict:
    """Train a single-logit binary model with return-aware BCE and persist the best checkpoint."""

    device = torch.device(experiment_config.experiment.device)
    model.to(device)
    _write_run_metadata(run_directory, experiment_config, model)
    LOGGER.info("Starting single-logit binary training in %s on device %s", run_directory, device)

    train_label_values = _extract_dataset_label_values(train_data_loader)
    positive_class_weight = compute_positive_class_weight(train_label_values).to(device)
    optimizer = _build_optimizer(
        model=model,
        experiment_config=experiment_config,
        objective_config=objective_config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=experiment_config.training.scheduler_cosine_t_max_epoch_count,
        eta_min=experiment_config.training.minimum_learning_rate,
    )

    best_validation_metric = float("-inf")
    best_checkpoint_selection_score = float("-inf")
    best_epoch_index = -1
    best_acc = float("-inf")
    best_f1 = float("-inf")
    best_mcc = float("-inf")
    best_balanced_accuracy = float("-inf")
    best_is_degenerate = 1.0
    early_stopping_counter = 0
    start_epoch_index = 1
    checkpoint_directory = run_directory / experiment_config.paths.checkpoint_subdirectory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_file_path = checkpoint_directory / experiment_config.training.latest_checkpoint_file_name
    best_checkpoint_file_path = checkpoint_directory / "best_model.pt"
    history_file_path = run_directory / experiment_config.paths.metrics_subdirectory / "training_history.json"
    history: list[dict] = []
    full_eval_objective_config = full_objective_config or objective_config

    resume_checkpoint_file_path: Path | None = None
    if experiment_config.training.resume_from_latest_checkpoint:
        if latest_checkpoint_file_path.exists():
            resume_checkpoint_file_path = latest_checkpoint_file_path
        elif best_checkpoint_file_path.exists():
            resume_checkpoint_file_path = best_checkpoint_file_path

    if resume_checkpoint_file_path is not None:
        checkpoint_payload = torch.load(resume_checkpoint_file_path, map_location="cpu")
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        if "optimizer_state_dict" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint_payload:
            scheduler.load_state_dict(checkpoint_payload["scheduler_state_dict"])
        best_validation_metric = float(checkpoint_payload.get("best_validation_metric", best_validation_metric))
        best_checkpoint_selection_score = float(
            checkpoint_payload.get("best_checkpoint_selection_score", best_checkpoint_selection_score)
        )
        best_epoch_index = int(checkpoint_payload.get("best_epoch", checkpoint_payload.get("epoch", best_epoch_index)))
        best_acc = float(checkpoint_payload.get("best_acc", best_acc))
        best_f1 = float(checkpoint_payload.get("best_f1", best_f1))
        best_mcc = float(checkpoint_payload.get("best_mcc", best_mcc))
        best_balanced_accuracy = float(
            checkpoint_payload.get("best_balanced_accuracy", best_balanced_accuracy)
        )
        best_is_degenerate = float(checkpoint_payload.get("best_is_degenerate", best_is_degenerate))
        early_stopping_counter = int(checkpoint_payload.get("early_stopping_counter", early_stopping_counter))
        if "history" in checkpoint_payload:
            history = list(checkpoint_payload["history"])
        elif history_file_path.exists():
            history_payload = json.loads(history_file_path.read_text(encoding="utf-8"))
            if "history" in history_payload:
                history = list(history_payload["history"])
        start_epoch_index = int(checkpoint_payload.get("epoch", 0)) + 1
        LOGGER.info(
            "Resuming single-logit binary training from checkpoint %s at epoch %03d",
            resume_checkpoint_file_path,
            start_epoch_index,
        )

    for epoch_index in range(start_epoch_index, experiment_config.training.num_epochs + 1):
        adapter_is_frozen = epoch_index <= objective_config.adapter_freeze_epoch_count
        _set_adapter_requires_grad(model, not adapter_is_frozen)
        train_metrics = run_single_logit_binary_epoch(
            model=model,
            data_loader=train_data_loader,
            positive_class_weight=positive_class_weight,
            optimizer=optimizer,
            experiment_config=experiment_config,
            objective_config=objective_config,
            device=device,
            split_name="train",
            epoch_index=epoch_index,
            diagnostics_file_path=_build_epoch_diagnostics_file_path(
                run_directory=run_directory,
                experiment_config=experiment_config,
                split_name="train",
                epoch_index=epoch_index,
            ),
            wandb_run=wandb_run,
        )
        _set_adapter_requires_grad(model, True)
        validation_metrics = run_single_logit_binary_epoch(
            model=model,
            data_loader=validation_data_loader,
            positive_class_weight=positive_class_weight,
            optimizer=None,
            experiment_config=experiment_config,
            objective_config=objective_config,
            device=device,
            split_name="validation",
            epoch_index=epoch_index,
            diagnostics_file_path=_build_epoch_diagnostics_file_path(
                run_directory=run_directory,
                experiment_config=experiment_config,
                split_name="validation",
                epoch_index=epoch_index,
            ),
            wandb_run=wandb_run,
        )
        validation_full_metrics = None
        test_full_metrics = None
        should_evaluate_validation_full = (
            objective_config.should_evaluate_full_splits_each_epoch
            or objective_config.checkpoint_selection_split_name == "validation_full"
        )
        should_evaluate_test_full = (
            objective_config.should_evaluate_full_splits_each_epoch
            or objective_config.checkpoint_selection_split_name == "test_full"
        )
        if should_evaluate_validation_full:
            if validation_full_data_loader is None:
                raise ValueError("validation_full evaluation requires validation_full_data_loader.")
            validation_full_metrics = run_single_logit_binary_epoch(
                model=model,
                data_loader=validation_full_data_loader,
                positive_class_weight=positive_class_weight,
                optimizer=None,
                experiment_config=experiment_config,
                objective_config=full_eval_objective_config,
                device=device,
                split_name="validation_full",
                epoch_index=epoch_index,
                diagnostics_file_path=_build_epoch_diagnostics_file_path(
                    run_directory=run_directory,
                    experiment_config=experiment_config,
                    split_name="validation_full",
                    epoch_index=epoch_index,
                ),
                wandb_run=wandb_run,
            )
        test_metrics = None
        should_evaluate_test_significant = (
            objective_config.should_evaluate_full_splits_each_epoch
            or objective_config.checkpoint_selection_split_name == "test"
        )
        if should_evaluate_test_significant:
            if checkpoint_selection_data_loader is None:
                raise ValueError("test evaluation requires checkpoint_selection_data_loader.")
            test_metrics = run_single_logit_binary_epoch(
                model=model,
                data_loader=checkpoint_selection_data_loader,
                positive_class_weight=positive_class_weight,
                optimizer=None,
                experiment_config=experiment_config,
                objective_config=objective_config,
                device=device,
                split_name="test",
                epoch_index=epoch_index,
                diagnostics_file_path=_build_epoch_diagnostics_file_path(
                    run_directory=run_directory,
                    experiment_config=experiment_config,
                    split_name="test",
                    epoch_index=epoch_index,
                ),
                wandb_run=wandb_run,
            )
        if should_evaluate_test_full:
            if test_full_data_loader is None:
                raise ValueError("test_full evaluation requires test_full_data_loader.")
            test_full_metrics = run_single_logit_binary_epoch(
                model=model,
                data_loader=test_full_data_loader,
                positive_class_weight=positive_class_weight,
                optimizer=None,
                experiment_config=experiment_config,
                objective_config=full_eval_objective_config,
                device=device,
                split_name="test_full",
                epoch_index=epoch_index,
                diagnostics_file_path=_build_epoch_diagnostics_file_path(
                    run_directory=run_directory,
                    experiment_config=experiment_config,
                    split_name="test_full",
                    epoch_index=epoch_index,
                ),
                wandb_run=wandb_run,
            )
        for _ in range(objective_config.omitted_evaluation_iterator_count):
            torch.empty((), dtype=torch.int64).random_().item()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scheduler.step()
        checkpoint_selection = select_single_logit_checkpoint_metrics(
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            objective_config=objective_config,
            validation_full_metrics=validation_full_metrics,
            test_full_metrics=test_full_metrics,
        )
        checkpoint_selection_score = checkpoint_selection.score
        should_consider_epoch = should_consider_checkpoint_selection_epoch(
            epoch_index=epoch_index,
            objective_config=objective_config,
        )
        history_record = {
            "epoch": epoch_index,
            "adapter_is_frozen": adapter_is_frozen,
            "train": train_metrics,
            "validation": validation_metrics,
            "checkpoint_selection": {
                "split_name": checkpoint_selection.split_name,
                "score_name": checkpoint_selection.score_name,
                "score": checkpoint_selection.score,
                "within_epoch_limit": should_consider_epoch,
            },
            "validation_checkpoint_selection_score": checkpoint_selection_score,
            "learning_rates": [parameter_group["lr"] for parameter_group in optimizer.param_groups],
        }
        if test_metrics is not None:
            history_record["test"] = test_metrics
            if objective_config.checkpoint_selection_split_name == "test":
                history_record["test_checkpoint_selection"] = test_metrics
        if validation_full_metrics is not None:
            history_record["validation_full"] = validation_full_metrics
        if test_full_metrics is not None:
            history_record["test_full"] = test_full_metrics
        history.append(history_record)
        LOGGER.info(
            "Epoch %03d | lr_base=%.8f | lr_adapter=%.8f | adapter_frozen=%s | checkpoint_selection[%s=%.6f split=%s in_window=%s] | train[%s] | validation[%s]",
            epoch_index,
            optimizer.param_groups[0]["lr"],
            optimizer.param_groups[1]["lr"],
            adapter_is_frozen,
            checkpoint_selection.score_name,
            checkpoint_selection.score,
            checkpoint_selection.split_name,
            should_consider_epoch,
            _format_single_logit_metrics_for_log(train_metrics),
            _format_single_logit_metrics_for_log(validation_metrics),
        )
        if wandb_run is not None:
            log_payload = {
                "epoch": epoch_index,
                "learning_rate/base": optimizer.param_groups[0]["lr"],
                "learning_rate/attention_side_adapter": optimizer.param_groups[1]["lr"],
                "adapter/is_frozen": float(adapter_is_frozen),
                "checkpoint_selection/score": checkpoint_selection_score,
                "checkpoint_selection/within_epoch_limit": float(should_consider_epoch),
                "validation/checkpoint_selection_score": checkpoint_selection_score,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            if test_metrics is not None:
                log_payload.update({f"test/{key}": value for key, value in test_metrics.items()})
                if objective_config.checkpoint_selection_split_name == "test":
                    log_payload.update(
                        {
                            f"test_checkpoint_selection/{key}": value
                            for key, value in test_metrics.items()
                        }
                    )
            if validation_full_metrics is not None:
                log_payload.update(
                    {f"validation_full/{key}": value for key, value in validation_full_metrics.items()}
                )
            if test_full_metrics is not None:
                log_payload.update(
                    {f"test_full/{key}": value for key, value in test_full_metrics.items()}
                )
            _safe_wandb_log(
                wandb_run=wandb_run,
                log_payload=log_payload,
                log_context=f"single-logit epoch summary at epoch {epoch_index}",
            )

        validation_metric_value = float(validation_metrics["mcc"])
        if should_consider_epoch and checkpoint_selection_score > best_checkpoint_selection_score:
            best_checkpoint_selection_score = checkpoint_selection_score
            best_validation_metric = float(checkpoint_selection.metrics["mcc"])
            best_epoch_index = epoch_index
            best_acc = float(checkpoint_selection.metrics["accuracy"])
            best_f1 = float(checkpoint_selection.metrics["macro_f1"])
            best_mcc = float(checkpoint_selection.metrics["mcc"])
            best_balanced_accuracy = float(checkpoint_selection.metrics["balanced_accuracy"])
            best_is_degenerate = float(checkpoint_selection.metrics["is_degenerate_checkpoint"])
            early_stopping_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch_index,
                    "best_epoch": best_epoch_index,
                    "best_validation_metric": best_validation_metric,
                    "best_checkpoint_selection_score": best_checkpoint_selection_score,
                    "best_acc": best_acc,
                    "best_f1": best_f1,
                    "best_mcc": best_mcc,
                    "best_balanced_accuracy": best_balanced_accuracy,
                    "best_is_degenerate": best_is_degenerate,
                    "early_stopping_counter": early_stopping_counter,
                    "history": history,
                    "positive_class_weight": positive_class_weight.detach().cpu(),
                    "single_logit_objective": asdict(objective_config),
                },
                best_checkpoint_file_path,
            )
            LOGGER.info(
                "Saved new best single-logit checkpoint at epoch %03d with selection_score=%.6f and mcc=%.6f degenerate=%s",
                epoch_index,
                best_checkpoint_selection_score,
                best_validation_metric,
                bool(best_is_degenerate),
            )
        else:
            early_stopping_counter += 1
            LOGGER.info(
                "No improvement at epoch %03d. early_stopping_counter=%d/%d",
                epoch_index,
                early_stopping_counter,
                experiment_config.training.early_stopping_patience,
            )

        history_payload = {
            "best_epoch": best_epoch_index,
            "best_validation_metric": best_validation_metric,
            "best_checkpoint_selection_score": best_checkpoint_selection_score,
            "best_acc": best_acc,
            "best_f1": best_f1,
            "best_mcc": best_mcc,
            "best_balanced_accuracy": best_balanced_accuracy,
            "best_is_degenerate": best_is_degenerate,
            "positive_class_weight": float(positive_class_weight.detach().cpu().item()),
            "single_logit_objective": asdict(objective_config),
            "history": history,
        }
        write_json_file(history_file_path, history_payload)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch_index,
                "best_epoch": best_epoch_index,
                "best_validation_metric": best_validation_metric,
                "best_checkpoint_selection_score": best_checkpoint_selection_score,
                "best_acc": best_acc,
                "best_f1": best_f1,
                "best_mcc": best_mcc,
                "best_balanced_accuracy": best_balanced_accuracy,
                "best_is_degenerate": best_is_degenerate,
                "early_stopping_counter": early_stopping_counter,
                "history": history,
                "positive_class_weight": positive_class_weight.detach().cpu(),
                "single_logit_objective": asdict(objective_config),
            },
            latest_checkpoint_file_path,
        )
        if early_stopping_counter >= experiment_config.training.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %03d", epoch_index)
            break

    summary_payload = {
        "best_epoch": best_epoch_index,
        "best_validation_metric": best_validation_metric,
        "best_checkpoint_selection_score": best_checkpoint_selection_score,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_mcc": best_mcc,
        "best_balanced_accuracy": best_balanced_accuracy,
        "best_is_degenerate": best_is_degenerate,
        "positive_class_weight": float(positive_class_weight.detach().cpu().item()),
        "single_logit_objective": asdict(objective_config),
        "history": history,
    }
    write_json_file(history_file_path, summary_payload)
    if wandb_run is not None and should_finish_wandb_run:
        wandb_run.finish()
    return summary_payload


def evaluate_single_logit_binary_model(
    model: nn.Module,
    checkpoint_file_path: Path,
    data_loader,
    train_data_loader,
    experiment_config: ExperimentConfig,
    objective_config: SingleLogitBinaryObjectiveConfig,
    split_name: str,
    run_directory: Path,
    wandb_run=None,
) -> dict[str, float]:
    """Load the best checkpoint and evaluate one single-logit binary split."""

    device = torch.device(experiment_config.experiment.device)
    checkpoint_payload = torch.load(checkpoint_file_path, map_location="cpu")
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(device)
    LOGGER.info("Evaluating %s single-logit binary split with checkpoint %s", split_name, checkpoint_file_path)
    positive_class_weight = compute_positive_class_weight(
        _extract_dataset_label_values(train_data_loader)
    ).to(device)
    metrics = run_single_logit_binary_epoch(
        model=model,
        data_loader=data_loader,
        positive_class_weight=positive_class_weight,
        optimizer=None,
        experiment_config=experiment_config,
        objective_config=objective_config,
        device=device,
        split_name=split_name,
        epoch_index=0,
        diagnostics_file_path=_build_epoch_diagnostics_file_path(
            run_directory=run_directory,
            experiment_config=experiment_config,
            split_name=split_name,
            epoch_index=0,
        ),
        wandb_run=wandb_run,
    )
    if wandb_run is not None:
        _safe_wandb_log(
            wandb_run=wandb_run,
            log_payload={f"{split_name}/{key}": value for key, value in metrics.items()},
            log_context=f"single-logit {split_name} evaluation metrics",
        )
    LOGGER.info("%s single-logit binary metrics | %s", split_name, _format_single_logit_metrics_for_log(metrics))
    return metrics


def build_single_logit_pipeline_summary(
    config_file_path: Path,
    run_directory: Path,
    checkpoint_file_path: Path,
    attention_side_adapter_config: object,
    objective_config: SingleLogitBinaryObjectiveConfig,
    training_summary: dict,
    validation_metrics: dict,
    test_metrics: dict,
    validation_adapter_diagnostics: dict,
    test_adapter_diagnostics: dict,
) -> dict:
    """Build a stable JSON-serializable pipeline summary payload."""

    return {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file_path": str(config_file_path),
        "run_directory": str(run_directory),
        "checkpoint_file_path": str(checkpoint_file_path),
        "attention_side_adapter": asdict(attention_side_adapter_config),
        "single_logit_objective": asdict(objective_config),
        "training_summary": training_summary,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_attention_side_adapter_full_diagnostics": validation_adapter_diagnostics,
        "test_attention_side_adapter_full_diagnostics": test_adapter_diagnostics,
    }
