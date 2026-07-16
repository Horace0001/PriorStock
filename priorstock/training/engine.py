"""Training loop, evaluation loop, and checkpoint orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.nn import functional as F

from priorstock.config import ExperimentConfig
from priorstock.training.metrics import compute_classification_metrics
from priorstock.utils.io import append_jsonl_records, ensure_directory, write_json_file
from priorstock.utils.logging_utils import (
    configure_logger,
    sample_tensor_values,
    summarize_tensor_distribution,
    write_tensor_diagnostics,
)


LOGGER = configure_logger("training_engine")


def compute_class_weights(label_values: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for the configured classification task."""

    label_counter = Counter(label_values)
    total_sample_count = sum(label_counter.values())
    weight_values = []
    for label_index in range(num_classes):
        label_count = label_counter[label_index]
        if label_count <= 0:
            raise ValueError(f"Training split is missing class {label_index}, so class weights cannot be computed.")
        weight_values.append(total_sample_count / (num_classes * label_count))
    return torch.tensor(weight_values, dtype=torch.float32)


def load_training_split_class_weights(experiment_config: ExperimentConfig) -> torch.Tensor:
    """Compute class weights from the persisted training split to keep all stages consistent."""

    training_index_file_path = (
        Path(experiment_config.experiment.artifact_output_root)
        / experiment_config.data.market_code.lower().replace("-", "_")
        / experiment_config.paths.sample_index_subdirectory
        / "train.csv"
    )
    train_split_frame = pd.read_csv(training_index_file_path)
    return compute_class_weights(train_split_frame["label"].tolist(), experiment_config.model.num_classes)


def compute_checkpoint_selection_score(
    metrics: dict[str, float],
    experiment_config: ExperimentConfig,
) -> float:
    """Compute the configured checkpoint-selection score from one validation metrics payload."""

    if not experiment_config.evaluation.checkpoint_selection_uses_composite_score:
        return float(metrics[experiment_config.evaluation.primary_metric_name])
    return float(
        (
            experiment_config.evaluation.checkpoint_selection_macro_f1_weight
            * metrics["macro_f1"]
        )
        + (
            experiment_config.evaluation.checkpoint_selection_mcc_weight
            * metrics["mcc"]
        )
    )


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move one collated batch dictionary to the training device."""

    return {
        "sample_id": batch["sample_id"],
        "stock_id": batch["stock_id"],
        "target_trade_date": batch["target_trade_date"],
        "price_features": batch["price_features"].to(device),
        "technical_indicator_features": batch["technical_indicator_features"].to(device),
        "news_embeddings": batch["news_embeddings"].to(device),
        "has_news": batch["has_news"].to(device),
        "label": batch["label"].to(device),
    }


def _compute_gradient_norm(model: nn.Module) -> float:
    """Measure the global L2 gradient norm over all trainable parameters."""

    squared_norm_sum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        squared_norm_sum += float(torch.sum(parameter.grad.detach() ** 2).cpu())
    return squared_norm_sum ** 0.5


def _compute_technical_indicator_auxiliary_contrastive_loss(
    technical_indicator_representation: torch.Tensor,
    labels: torch.Tensor,
    experiment_config: ExperimentConfig,
) -> torch.Tensor:
    """Pull configured positive-class technical indicator representations together and push configured negatives apart."""

    if not experiment_config.training.use_technical_indicator_auxiliary_contrastive_loss:
        return torch.zeros((), device=technical_indicator_representation.device)

    normalized_representation = F.normalize(technical_indicator_representation, p=2, dim=-1)
    positive_class_index = experiment_config.training.technical_indicator_auxiliary_positive_class_index
    negative_class_index = experiment_config.training.technical_indicator_auxiliary_negative_class_index
    positive_mask = labels == positive_class_index
    negative_mask = labels == negative_class_index

    loss_terms: list[torch.Tensor] = []
    if int(positive_mask.sum().item()) >= 2:
        positive_representation = normalized_representation[positive_mask]
        positive_similarity = positive_representation @ positive_representation.transpose(0, 1)
        upper_triangle_row_indices, upper_triangle_column_indices = torch.triu_indices(
            positive_similarity.shape[0],
            positive_similarity.shape[1],
            offset=1,
            device=positive_similarity.device,
        )
        if upper_triangle_row_indices.numel() > 0:
            positive_pair_similarity = positive_similarity[
                upper_triangle_row_indices,
                upper_triangle_column_indices,
            ]
            loss_terms.append((1.0 - positive_pair_similarity).mean())

    if int(positive_mask.sum().item()) > 0 and int(negative_mask.sum().item()) > 0:
        positive_representation = normalized_representation[positive_mask]
        negative_representation = normalized_representation[negative_mask]
        cross_class_similarity = positive_representation @ negative_representation.transpose(0, 1)
        loss_terms.append(
            torch.relu(
                cross_class_similarity
                - experiment_config.training.technical_indicator_auxiliary_negative_similarity_margin
            ).mean()
        )

    if not loss_terms:
        return torch.zeros((), device=technical_indicator_representation.device)

    return torch.stack(loss_terms).mean()


def compute_soft_macro_f1_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    experiment_config: ExperimentConfig,
) -> torch.Tensor:
    """Compute batch-level soft macro-F1 loss from hard labels and soft predictions.

    Args:
        logits: Raw class logits with shape ``[batch_size, num_classes]``.
        labels: Hard integer labels with shape ``[batch_size]``.
        experiment_config: Experiment configuration containing class count and epsilon.

    Returns:
        Scalar tensor equal to ``1 - mean_k soft_F1_k``.
    """

    if not experiment_config.training.use_soft_macro_f1_loss:
        return torch.zeros((), device=logits.device)

    class_probabilities = torch.softmax(logits, dim=-1)
    hard_label_matrix = F.one_hot(
        labels,
        num_classes=experiment_config.model.num_classes,
    ).to(dtype=class_probabilities.dtype)
    soft_confusion_matrix = hard_label_matrix.transpose(0, 1) @ class_probabilities
    true_positive_values = torch.diagonal(soft_confusion_matrix)
    predicted_class_mass = soft_confusion_matrix.sum(dim=0)
    true_class_mass = soft_confusion_matrix.sum(dim=1)
    soft_f1_values = (
        2.0
        * true_positive_values
        / (
            predicted_class_mass
            + true_class_mass
            + experiment_config.training.soft_macro_f1_epsilon
        )
    )
    return 1.0 - soft_f1_values.mean()


def _build_epoch_diagnostics_file_path(
    run_directory: Path,
    experiment_config: ExperimentConfig,
    split_name: str,
    epoch_index: int,
) -> Path:
    """Return one per-epoch diagnostics file path so shape logs are never overwritten."""

    diagnostics_directory = ensure_directory(
        run_directory / experiment_config.paths.metrics_subdirectory / "tensor_diagnostics"
    )
    if epoch_index <= 0:
        return diagnostics_directory / f"{split_name}_evaluation_diagnostics.json"
    return diagnostics_directory / f"epoch_{epoch_index:03d}_{split_name}_diagnostics.json"


def _build_distribution_trace_file_path(
    run_directory: Path,
    experiment_config: ExperimentConfig,
    split_name: str,
) -> Path:
    """Resolve the JSONL file used for step-level distribution summaries."""

    distribution_directory = ensure_directory(
        run_directory
        / experiment_config.paths.metrics_subdirectory
        / experiment_config.logging.distribution_summary_subdirectory
    )
    return distribution_directory / f"{split_name}_{experiment_config.logging.distribution_trace_file_name}"


def _should_collect_trace_tensors(
    experiment_config: ExperimentConfig,
    split_name: str,
    step_index: int,
) -> bool:
    """Decide whether the current batch should emit full trace tensors."""

    is_first_batch = step_index == 0
    if is_first_batch and experiment_config.logging.log_first_batch_tensor_statistics:
        return True
    if split_name != "train":
        return False
    return (step_index + 1) % experiment_config.logging.distribution_log_every_training_step_count == 0


def _build_attention_distribution_summary(attention_tensor: torch.Tensor, max_sample_count: int) -> dict[str, float | int | list[int] | str]:
    """Add attention-specific structure metrics on top of the generic tensor summary."""

    summary_payload = summarize_tensor_distribution(attention_tensor, max_sample_count)
    detached_attention_tensor = attention_tensor.detach().to(dtype=torch.float32, device="cpu")
    attention_floor = torch.finfo(detached_attention_tensor.dtype).tiny
    stable_attention_tensor = detached_attention_tensor.clamp_min(attention_floor)
    attention_entropy = -(stable_attention_tensor * stable_attention_tensor.log()).sum(dim=-1)
    diagonal_attention_mass = torch.diagonal(detached_attention_tensor, dim1=-2, dim2=-1)
    summary_payload.update(
        {
            "attention_entropy_mean": float(attention_entropy.mean().item()),
            "attention_entropy_std": float(attention_entropy.std(unbiased=False).item()),
            "diagonal_attention_mass_mean": float(diagonal_attention_mass.mean().item()),
            "diagonal_attention_mass_max": float(diagonal_attention_mass.max().item()),
        }
    )
    return summary_payload


def _summarize_trace_tensors(
    trace_tensors: dict[str, torch.Tensor],
    experiment_config: ExperimentConfig,
) -> tuple[dict[str, dict], dict[str, wandb.Histogram]]:
    """Summarize and optionally build W&B histograms for activation and attention traces."""

    trace_summaries: dict[str, dict] = {}
    histogram_payload: dict[str, wandb.Histogram] = {}
    for trace_name, trace_tensor in trace_tensors.items():
        is_attention_trace = "attention_weights" in trace_name
        if is_attention_trace and not experiment_config.logging.log_attention_distributions:
            continue
        if (not is_attention_trace) and not experiment_config.logging.log_activation_distributions:
            continue

        if is_attention_trace:
            trace_summaries[trace_name] = _build_attention_distribution_summary(
                trace_tensor,
                experiment_config.logging.max_distribution_summary_sample_count,
            )
        else:
            trace_summaries[trace_name] = summarize_tensor_distribution(
                trace_tensor,
                experiment_config.logging.max_distribution_summary_sample_count,
            )

        if experiment_config.logging.log_histograms_to_wandb:
            histogram_values = sample_tensor_values(
                trace_tensor,
                experiment_config.logging.max_histogram_sample_count,
            )
            histogram_object = _build_wandb_histogram(histogram_values)
            if histogram_object is not None:
                histogram_payload[trace_name] = histogram_object
    return trace_summaries, histogram_payload


def _derive_parameter_group_name(parameter_name: str) -> str:
    """Collapse one parameter name into a stable module-level distribution group."""

    if parameter_name.startswith("transformer_layers."):
        name_parts = parameter_name.split(".")
        layer_index = int(name_parts[1]) + 1
        if "layer_tech_proj" in parameter_name:
            return f"layer_{layer_index}/layer_tech_projection"
        if "layer_news_proj" in parameter_name:
            return f"layer_{layer_index}/layer_news_projection"
        if "layer_tech_layer_norm" in parameter_name:
            return f"layer_{layer_index}/layer_tech_layer_norm"
        if "layer_news_layer_norm" in parameter_name:
            return f"layer_{layer_index}/layer_news_layer_norm"
        if "self_attention" in parameter_name:
            return f"layer_{layer_index}/self_attention"
        if "feed_forward" in parameter_name:
            return f"layer_{layer_index}/feed_forward"
        if "tech_gate" in parameter_name:
            return f"layer_{layer_index}/tech_gate"
        if "news_gate" in parameter_name:
            return f"layer_{layer_index}/news_gate"
        if "attention_layer_norm" in parameter_name:
            return f"layer_{layer_index}/attention_layer_norm"
        if "feed_forward_layer_norm" in parameter_name:
            return f"layer_{layer_index}/feed_forward_layer_norm"
        if "post_injection_layer_norm" in parameter_name:
            return f"layer_{layer_index}/post_injection_layer_norm"
        return f"layer_{layer_index}/other"
    if "." in parameter_name:
        return parameter_name.split(".", maxsplit=1)[0]
    return parameter_name


def _collect_grouped_parameter_samples(
    model: nn.Module,
    include_gradients: bool,
    max_histogram_sample_count: int,
) -> dict[str, torch.Tensor]:
    """Group sampled parameter values or gradients by module family."""

    grouped_tensors: dict[str, list[torch.Tensor]] = {}
    for parameter_name, parameter in model.named_parameters():
        source_tensor = parameter.grad if include_gradients else parameter.detach()
        if source_tensor is None:
            continue
        group_name = _derive_parameter_group_name(parameter_name)
        grouped_tensors.setdefault(group_name, []).append(
            sample_tensor_values(source_tensor, max_histogram_sample_count)
        )
    return {
        group_name: torch.cat(tensor_list, dim=0)
        for group_name, tensor_list in grouped_tensors.items()
        if tensor_list
    }


def _summarize_parameter_groups(
    grouped_tensors: dict[str, torch.Tensor],
    experiment_config: ExperimentConfig,
) -> tuple[dict[str, dict], dict[str, wandb.Histogram]]:
    """Convert grouped parameter samples into scalar summaries and optional W&B histograms."""

    summary_payload: dict[str, dict] = {}
    histogram_payload: dict[str, wandb.Histogram] = {}
    for group_name, group_tensor in grouped_tensors.items():
        summary_payload[group_name] = summarize_tensor_distribution(
            group_tensor,
            experiment_config.logging.max_distribution_summary_sample_count,
        )
        if experiment_config.logging.log_histograms_to_wandb:
            histogram_object = _build_wandb_histogram(group_tensor)
            if histogram_object is not None:
                histogram_payload[group_name] = histogram_object
    return summary_payload, histogram_payload


def _build_wandb_histogram(sample_tensor: torch.Tensor) -> wandb.Histogram | None:
    """Build one W&B histogram robustly, even for degenerate near-constant tensors."""

    histogram_values = sample_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
    finite_values = histogram_values[np.isfinite(histogram_values)]
    if finite_values.size == 0:
        return None

    minimum_value = float(finite_values.min())
    maximum_value = float(finite_values.max())
    if minimum_value == maximum_value:
        epsilon_value = max(abs(minimum_value) * 1.0e-6, 1.0e-6)
        bin_edges = np.array(
            [minimum_value - epsilon_value, minimum_value + epsilon_value],
            dtype=np.float64,
        )
        bin_counts = np.array([finite_values.size], dtype=np.float64)
        return wandb.Histogram(np_histogram=(bin_counts, bin_edges))

    try:
        return wandb.Histogram(finite_values)
    except ValueError:
        safe_bin_edges = np.linspace(minimum_value, maximum_value, num=65, dtype=np.float64)
        if np.unique(safe_bin_edges).size < 2:
            epsilon_value = max(abs(minimum_value) * 1.0e-6, 1.0e-6)
            safe_bin_edges = np.array(
                [minimum_value - epsilon_value, minimum_value + epsilon_value],
                dtype=np.float64,
            )
        bin_counts, safe_bin_edges = np.histogram(finite_values, bins=safe_bin_edges)
        return wandb.Histogram(np_histogram=(bin_counts.astype(np.float64), safe_bin_edges.astype(np.float64)))


def _safe_wandb_log(wandb_run, log_payload: dict[str, object], log_context: str) -> None:
    """Log to W&B without letting transient service-transport failures abort training."""

    if wandb_run is None or not log_payload:
        return
    try:
        wandb_run.log(log_payload)
    except (ConnectionResetError, BrokenPipeError, OSError, RuntimeError) as error:
        LOGGER.warning("W&B logging failed during %s: %s", log_context, error)


def _build_distribution_record(
    experiment_config: ExperimentConfig,
    split_name: str,
    epoch_index: int,
    step_index: int,
    sample_ids: list[str],
    scalar_metrics: dict[str, float],
    trace_summaries: dict[str, dict],
    parameter_summaries: dict[str, dict],
    gradient_summaries: dict[str, dict],
) -> dict:
    """Build one persistent distribution-trace record."""

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": split_name,
        "epoch_index": epoch_index,
        "step_index_within_epoch": step_index + 1,
        "sample_ids": sample_ids,
        "scalar_metrics": scalar_metrics,
        "activation_and_attention_distributions": trace_summaries,
        "parameter_value_distributions": parameter_summaries if experiment_config.logging.log_parameter_value_distributions else {},
        "gradient_distributions": gradient_summaries if experiment_config.logging.log_gradient_distributions else {},
    }


def _log_distribution_payload_to_wandb(
    wandb_run,
    split_name: str,
    epoch_index: int,
    step_index: int,
    scalar_metrics: dict[str, float],
    trace_summaries: dict[str, dict],
    trace_histograms: dict[str, wandb.Histogram],
    parameter_summaries: dict[str, dict],
    parameter_histograms: dict[str, wandb.Histogram],
    gradient_summaries: dict[str, dict],
    gradient_histograms: dict[str, wandb.Histogram],
) -> None:
    """Send one detailed distribution payload to W&B."""

    if wandb_run is None:
        return

    scalar_log_payload: dict[str, object] = {
        f"{split_name}/distribution_epoch": epoch_index,
        f"{split_name}/distribution_step": step_index + 1,
        **{f"{split_name}/scalar/{metric_name}": metric_value for metric_name, metric_value in scalar_metrics.items()},
    }
    for trace_name, summary_payload in trace_summaries.items():
        for summary_name, summary_value in summary_payload.items():
            if isinstance(summary_value, (float, int)):
                scalar_log_payload[f"{split_name}/distribution_stats/{trace_name}/{summary_name}"] = summary_value
    for group_name, summary_payload in parameter_summaries.items():
        for summary_name, summary_value in summary_payload.items():
            if isinstance(summary_value, (float, int)):
                scalar_log_payload[f"{split_name}/parameter_stats/{group_name}/{summary_name}"] = summary_value
    for group_name, summary_payload in gradient_summaries.items():
        for summary_name, summary_value in summary_payload.items():
            if isinstance(summary_value, (float, int)):
                scalar_log_payload[f"{split_name}/gradient_stats/{group_name}/{summary_name}"] = summary_value
    _safe_wandb_log(
        wandb_run=wandb_run,
        log_payload=scalar_log_payload,
        log_context=f"{split_name} distribution scalar summaries at epoch {epoch_index} step {step_index + 1}",
    )

    histogram_payload_items = [
        *[
            (f"{split_name}/distribution_hist/{histogram_name}", histogram_value)
            for histogram_name, histogram_value in trace_histograms.items()
        ],
        *[
            (f"{split_name}/parameter_hist/{histogram_name}", histogram_value)
            for histogram_name, histogram_value in parameter_histograms.items()
        ],
        *[
            (f"{split_name}/gradient_hist/{histogram_name}", histogram_value)
            for histogram_name, histogram_value in gradient_histograms.items()
        ],
    ]
    histogram_chunk_size = 16
    for chunk_start_index in range(0, len(histogram_payload_items), histogram_chunk_size):
        histogram_chunk = dict(histogram_payload_items[chunk_start_index : chunk_start_index + histogram_chunk_size])
        _safe_wandb_log(
            wandb_run=wandb_run,
            log_payload=histogram_chunk,
            log_context=(
                f"{split_name} histogram summaries at epoch {epoch_index} step {step_index + 1} "
                f"(chunk {chunk_start_index // histogram_chunk_size + 1})"
            ),
        )


def _write_run_metadata(
    run_directory: Path,
    experiment_config: ExperimentConfig,
    model: nn.Module,
) -> None:
    """Persist the exact config snapshot and model structure for one run."""

    write_json_file(run_directory / "config_snapshot.json", asdict(experiment_config))
    ensure_directory(run_directory)
    with (run_directory / "model_architecture.txt").open("w", encoding="utf-8") as file_handle:
        file_handle.write(str(model))
        file_handle.write("\n")


def _format_metrics_for_log(metrics: dict[str, float]) -> str:
    """Format one metrics dictionary into a stable compact log string."""

    ordered_metric_names = [
        "total_loss",
        "cross_entropy_loss",
        "soft_macro_f1_loss",
        "auxiliary_contrastive_loss",
        "gate_regularization",
        "accuracy",
        "macro_f1",
        "mcc",
        "up_precision",
        "up_recall",
    ]
    formatted_parts: list[str] = []
    for metric_name in ordered_metric_names:
        if metric_name in metrics:
            formatted_parts.append(f"{metric_name}={metrics[metric_name]:.6f}")
    return " | ".join(formatted_parts)


def run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer,
    experiment_config: ExperimentConfig,
    device: torch.device,
    split_name: str,
    epoch_index: int,
    diagnostics_file_path: Path,
    wandb_run,
) -> dict:
    """Run one full training or evaluation epoch and return metrics plus losses."""

    is_training = optimizer is not None
    should_manage_cuda_cache = device.type == "cuda"
    if is_training:
        model.train()
    else:
        model.eval()

    accumulated_loss_value = 0.0
    accumulated_cross_entropy_loss_value = 0.0
    accumulated_soft_macro_f1_loss_value = 0.0
    accumulated_auxiliary_contrastive_loss_value = 0.0
    accumulated_tech_gate_regularization_value = 0.0
    accumulated_news_gate_regularization_value = 0.0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
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
            cross_entropy_loss = criterion(model_output.logits, batch["label"])
            soft_macro_f1_loss = compute_soft_macro_f1_loss(
                logits=model_output.logits,
                labels=batch["label"],
                experiment_config=experiment_config,
            )
            auxiliary_contrastive_loss = _compute_technical_indicator_auxiliary_contrastive_loss(
                model_output.technical_indicator_representation,
                batch["label"],
                experiment_config,
            )
            gate_regularization_loss = (
                experiment_config.training.tech_gate_regularization_weight
                * model_output.tech_gate_regularization_loss
            ) + (
                experiment_config.training.news_gate_regularization_weight
                * model_output.news_gate_regularization_loss
            )
            total_loss = cross_entropy_loss + (
                experiment_config.training.soft_macro_f1_loss_weight * soft_macro_f1_loss
            ) + (
                experiment_config.training.technical_indicator_auxiliary_loss_weight * auxiliary_contrastive_loss
            ) + gate_regularization_loss

            if is_training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=experiment_config.training.gradient_clip_max_norm,
                )
                optimizer.step()

        predictions = torch.argmax(model_output.logits.detach(), dim=-1)
        true_labels.extend(batch["label"].detach().cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        accumulated_loss_value += float(total_loss.detach().cpu())
        accumulated_cross_entropy_loss_value += float(cross_entropy_loss.detach().cpu())
        accumulated_soft_macro_f1_loss_value += float(soft_macro_f1_loss.detach().cpu())
        accumulated_auxiliary_contrastive_loss_value += float(auxiliary_contrastive_loss.detach().cpu())
        accumulated_tech_gate_regularization_value += float(model_output.tech_gate_regularization_loss.detach().cpu())
        accumulated_news_gate_regularization_value += float(model_output.news_gate_regularization_loss.detach().cpu())

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
            }
            if is_training and experiment_config.logging.log_gradient_statistics:
                diagnostics_payload["gradient_norm"] = _compute_gradient_norm(model)
            write_tensor_diagnostics(diagnostics_file_path, diagnostics_payload)
            first_batch_diagnostics_written = True

        should_log_distribution_trace = should_collect_trace_tensors
        if should_log_distribution_trace:
            scalar_metrics = {
                "step_total_loss": float(total_loss.detach().cpu()),
                "step_cross_entropy_loss": float(cross_entropy_loss.detach().cpu()),
                "step_soft_macro_f1_loss": float(soft_macro_f1_loss.detach().cpu()),
                "step_auxiliary_contrastive_loss": float(auxiliary_contrastive_loss.detach().cpu()),
                "step_tech_gate_regularization": float(model_output.tech_gate_regularization_loss.detach().cpu()),
                "step_news_gate_regularization": float(model_output.news_gate_regularization_loss.detach().cpu()),
                "step_weighted_gate_regularization": float(gate_regularization_loss.detach().cpu()),
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
            trace_summaries.clear()
            trace_histograms.clear()
            parameter_summaries.clear()
            parameter_histograms.clear()
            gradient_summaries.clear()
            gradient_histograms.clear()
            model_output.trace_tensors.clear()

        if wandb_run is not None and is_training and (step_index + 1) % experiment_config.logging.log_every_training_step_count == 0:
            _safe_wandb_log(
                wandb_run=wandb_run,
                log_payload={
                    "train/step_total_loss": float(total_loss.detach().cpu()),
                    "train/step_cross_entropy_loss": float(cross_entropy_loss.detach().cpu()),
                    "train/step_soft_macro_f1_loss": float(soft_macro_f1_loss.detach().cpu()),
                    "train/step_auxiliary_contrastive_loss": float(auxiliary_contrastive_loss.detach().cpu()),
                    "train/step_tech_gate_regularization": float(model_output.tech_gate_regularization_loss.detach().cpu()),
                    "train/step_news_gate_regularization": float(model_output.news_gate_regularization_loss.detach().cpu()),
                    "train/step_weighted_gate_regularization": float(gate_regularization_loss.detach().cpu()),
                    "train/epoch": epoch_index,
                },
                log_context=f"train step metrics at epoch {epoch_index} step {step_index + 1}",
            )

        del predictions
        del model_output
        del total_loss
        del cross_entropy_loss
        del soft_macro_f1_loss
        del auxiliary_contrastive_loss
        del gate_regularization_loss
        del batch
        if should_manage_cuda_cache and should_collect_trace_tensors:
            torch.cuda.empty_cache()

    metrics = compute_classification_metrics(true_labels, predicted_labels, experiment_config.evaluation)
    batch_count = max(len(data_loader), 1)
    metrics.update(
        {
            "total_loss": accumulated_loss_value / batch_count,
            "cross_entropy_loss": accumulated_cross_entropy_loss_value / batch_count,
            "soft_macro_f1_loss": accumulated_soft_macro_f1_loss_value / batch_count,
            "auxiliary_contrastive_loss": accumulated_auxiliary_contrastive_loss_value / batch_count,
            "tech_gate_regularization": accumulated_tech_gate_regularization_value / batch_count,
            "news_gate_regularization": accumulated_news_gate_regularization_value / batch_count,
            "weighted_gate_regularization": (
                (
                    experiment_config.training.tech_gate_regularization_weight
                    * accumulated_tech_gate_regularization_value
                )
                + (
                    experiment_config.training.news_gate_regularization_weight
                    * accumulated_news_gate_regularization_value
                )
            )
            / batch_count,
        }
    )
    if should_manage_cuda_cache:
        torch.cuda.empty_cache()
    return metrics


def fit_model(
    model: nn.Module,
    train_data_loader,
    validation_data_loader,
    experiment_config: ExperimentConfig,
    run_directory: Path,
    wandb_run=None,
    should_finish_wandb_run: bool = True,
) -> dict:
    """Train the model with early stopping and persist the best checkpoint."""

    device = torch.device(experiment_config.experiment.device)
    model.to(device)
    _write_run_metadata(run_directory, experiment_config, model)
    LOGGER.info("Starting training in %s on device %s", run_directory, device)

    class_weights = load_training_split_class_weights(experiment_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=experiment_config.training.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment_config.training.learning_rate,
        weight_decay=experiment_config.training.weight_decay,
        betas=(experiment_config.training.adam_beta_one, experiment_config.training.adam_beta_two),
        eps=experiment_config.training.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=experiment_config.training.scheduler_cosine_t_max_epoch_count,
        eta_min=experiment_config.training.minimum_learning_rate,
    )

    if wandb_run is None and experiment_config.logging.use_wandb:
        wandb_run = wandb.init(
            project=experiment_config.logging.wandb_project_name,
            entity=experiment_config.logging.wandb_entity_name,
            mode=experiment_config.logging.wandb_mode,
            dir=str(run_directory),
            config=asdict(experiment_config),
            name=run_directory.name,
            settings=wandb.Settings(init_timeout=300),
        )

    best_validation_metric = float("-inf")
    best_checkpoint_selection_score = float("-inf")
    best_epoch_index = -1
    best_acc = float("-inf")
    best_f1 = float("-inf")
    best_mcc = float("-inf")
    early_stopping_counter = 0
    start_epoch_index = 1
    checkpoint_directory = run_directory / experiment_config.paths.checkpoint_subdirectory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_file_path = checkpoint_directory / experiment_config.training.latest_checkpoint_file_name
    best_checkpoint_file_path = checkpoint_directory / "best_model.pt"
    history_file_path = run_directory / experiment_config.paths.metrics_subdirectory / "training_history.json"
    history: list[dict] = []

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
            checkpoint_payload.get(
                "best_checkpoint_selection_score",
                checkpoint_payload.get("best_validation_metric", best_checkpoint_selection_score),
            )
        )
        best_epoch_index = int(checkpoint_payload.get("best_epoch", checkpoint_payload.get("epoch", best_epoch_index)))
        best_acc = float(checkpoint_payload.get("best_acc", best_acc))
        best_f1 = float(checkpoint_payload.get("best_f1", best_f1))
        best_mcc = float(checkpoint_payload.get("best_mcc", best_mcc))
        early_stopping_counter = int(checkpoint_payload.get("early_stopping_counter", early_stopping_counter))
        if "history" in checkpoint_payload:
            history = list(checkpoint_payload["history"])
        elif history_file_path.exists():
            history_payload = json.loads(history_file_path.read_text(encoding="utf-8"))
            if "history" in history_payload:
                history = list(history_payload["history"])
        start_epoch_index = int(checkpoint_payload.get("epoch", 0)) + 1
        LOGGER.info(
            "Resuming training from checkpoint %s at epoch %03d",
            resume_checkpoint_file_path,
            start_epoch_index,
        )

    if history:
        write_json_file(
            history_file_path,
            {
                "best_epoch": best_epoch_index,
                "best_validation_metric": best_validation_metric,
                "best_checkpoint_selection_score": best_checkpoint_selection_score,
                "best_acc": best_acc,
                "best_f1": best_f1,
                "best_mcc": best_mcc,
                "history": history,
            },
        )

    for epoch_index in range(start_epoch_index, experiment_config.training.num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            experiment_config=experiment_config,
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
        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_data_loader,
            criterion=criterion,
            optimizer=None,
            experiment_config=experiment_config,
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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scheduler.step()

        history_record = {
            "epoch": epoch_index,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_checkpoint_selection_score": compute_checkpoint_selection_score(
                validation_metrics,
                experiment_config,
            ),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(history_record)
        LOGGER.info(
            "Epoch %03d | lr=%.8f | train[%s] | validation[%s]",
            epoch_index,
            scheduler.get_last_lr()[0],
            _format_metrics_for_log(train_metrics),
            _format_metrics_for_log(validation_metrics),
        )

        if wandb_run is not None:
            _safe_wandb_log(
                wandb_run=wandb_run,
                log_payload={
                    "epoch": epoch_index,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "validation/checkpoint_selection_score": history_record["validation_checkpoint_selection_score"],
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"validation/{key}": value for key, value in validation_metrics.items()},
                },
                log_context=f"epoch summary at epoch {epoch_index}",
            )

        validation_metric_value = float(validation_metrics[experiment_config.evaluation.primary_metric_name])
        checkpoint_selection_score = history_record["validation_checkpoint_selection_score"]
        if checkpoint_selection_score > best_checkpoint_selection_score:
            best_checkpoint_selection_score = checkpoint_selection_score
            best_validation_metric = validation_metric_value
            best_epoch_index = epoch_index
            best_acc = float(validation_metrics["accuracy"])
            best_f1 = float(validation_metrics["macro_f1"])
            best_mcc = float(validation_metrics["mcc"])
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
                    "early_stopping_counter": early_stopping_counter,
                    "history": history,
                },
                best_checkpoint_file_path,
            )
            LOGGER.info(
                "Saved new best checkpoint at epoch %03d with checkpoint_selection_score=%.6f and %s=%.6f",
                epoch_index,
                best_checkpoint_selection_score,
                experiment_config.evaluation.primary_metric_name,
                best_validation_metric,
            )
        else:
            early_stopping_counter += 1
            LOGGER.info(
                "No improvement at epoch %03d. early_stopping_counter=%d/%d",
                epoch_index,
                early_stopping_counter,
                experiment_config.training.early_stopping_patience,
            )
        write_json_file(
            history_file_path,
            {
                "best_epoch": best_epoch_index,
                "best_validation_metric": best_validation_metric,
                "best_checkpoint_selection_score": best_checkpoint_selection_score,
                "best_acc": best_acc,
                "best_f1": best_f1,
                "best_mcc": best_mcc,
                "history": history,
            },
        )
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
                "early_stopping_counter": early_stopping_counter,
                "history": history,
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
        "history": history,
    }
    write_json_file(history_file_path, summary_payload)

    if wandb_run is not None and should_finish_wandb_run:
        wandb_run.finish()

    return summary_payload


def fit_model_two_stage_learning_rate_reset(
    model: nn.Module,
    train_data_loader,
    validation_data_loader,
    experiment_config: ExperimentConfig,
    run_directory: Path,
    wandb_run,
    first_stage_epoch_count: int,
    first_stage_learning_rate: float,
    second_stage_epoch_count: int,
    second_stage_learning_rate: float,
    second_stage_minimum_learning_rate: float,
    should_finish_wandb_run: bool,
) -> dict:
    """Train with a greedy first stage and a reset-optimizer second stage.

    Args:
        model: Model instance to train.
        train_data_loader: DataLoader for the training split.
        validation_data_loader: DataLoader for the validation split.
        experiment_config: Fully validated experiment configuration.
        run_directory: Directory where checkpoints and diagnostics are written.
        wandb_run: Optional active W&B run.
        first_stage_epoch_count: Number of epochs in the initial high-learning-rate phase.
        first_stage_learning_rate: Learning rate used for the first phase.
        second_stage_epoch_count: Number of epochs in the low-learning-rate refinement phase.
        second_stage_learning_rate: Learning rate used at the start of the second phase.
        second_stage_minimum_learning_rate: Cosine schedule floor for the second phase.
        should_finish_wandb_run: Whether this function should finish the active W&B run.

    Returns:
        Training summary containing per-phase history and the final best checkpoint metrics.
    """

    if first_stage_epoch_count <= 0:
        raise ValueError("first_stage_epoch_count must be positive.")
    if second_stage_epoch_count <= 0:
        raise ValueError("second_stage_epoch_count must be positive.")
    if first_stage_learning_rate <= 0.0:
        raise ValueError("first_stage_learning_rate must be positive.")
    if second_stage_learning_rate <= 0.0:
        raise ValueError("second_stage_learning_rate must be positive.")
    if second_stage_minimum_learning_rate < 0.0:
        raise ValueError("second_stage_minimum_learning_rate must be non-negative.")
    if second_stage_minimum_learning_rate > second_stage_learning_rate:
        raise ValueError("second_stage_minimum_learning_rate cannot exceed second_stage_learning_rate.")

    device = torch.device(experiment_config.experiment.device)
    model.to(device)
    _write_run_metadata(run_directory, experiment_config, model)
    LOGGER.info(
        "Starting two-stage training in %s on device %s | stage1_epochs=%d stage1_lr=%.8f "
        "| stage2_epochs=%d stage2_lr=%.8f stage2_eta_min=%.8f",
        run_directory,
        device,
        first_stage_epoch_count,
        first_stage_learning_rate,
        second_stage_epoch_count,
        second_stage_learning_rate,
        second_stage_minimum_learning_rate,
    )

    class_weights = load_training_split_class_weights(experiment_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=experiment_config.training.label_smoothing,
    )

    if wandb_run is None and experiment_config.logging.use_wandb:
        wandb_run = wandb.init(
            project=experiment_config.logging.wandb_project_name,
            entity=experiment_config.logging.wandb_entity_name,
            mode=experiment_config.logging.wandb_mode,
            dir=str(run_directory),
            config=asdict(experiment_config),
            name=run_directory.name,
            settings=wandb.Settings(init_timeout=300),
        )

    checkpoint_directory = run_directory / experiment_config.paths.checkpoint_subdirectory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    best_checkpoint_file_path = checkpoint_directory / "best_model.pt"
    first_stage_best_checkpoint_file_path = checkpoint_directory / "stage1_best_model_weights.pt"
    second_stage_best_checkpoint_file_path = checkpoint_directory / "stage2_best_model_weights.pt"
    history_file_path = run_directory / experiment_config.paths.metrics_subdirectory / "training_history.json"
    history: list[dict] = []

    best_validation_metric = float("-inf")
    best_checkpoint_selection_score = float("-inf")
    best_epoch_index = -1
    best_phase_name = ""
    best_acc = float("-inf")
    best_f1 = float("-inf")
    best_mcc = float("-inf")
    stage_summaries: dict[str, dict[str, float | int | str]] = {}

    def build_optimizer(learning_rate: float) -> torch.optim.Optimizer:
        """Create a fresh optimizer for one training phase."""

        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=experiment_config.training.weight_decay,
            betas=(experiment_config.training.adam_beta_one, experiment_config.training.adam_beta_two),
            eps=experiment_config.training.adam_epsilon,
        )

    def run_training_phase(
        phase_name: str,
        start_epoch_index: int,
        epoch_count: int,
        learning_rate: float,
        eta_minimum: float,
        stage_best_checkpoint_file_path: Path,
    ) -> dict[str, float | int | str]:
        """Run one fixed-length phase and save model-weight-only stage checkpoints."""

        nonlocal best_validation_metric
        nonlocal best_checkpoint_selection_score
        nonlocal best_epoch_index
        nonlocal best_phase_name
        nonlocal best_acc
        nonlocal best_f1
        nonlocal best_mcc

        optimizer = build_optimizer(learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_count,
            eta_min=eta_minimum,
        )
        stage_best_score = float("-inf")
        stage_best_epoch_index = -1
        stage_best_metric = float("-inf")
        stage_best_acc = float("-inf")
        stage_best_f1 = float("-inf")
        stage_best_mcc = float("-inf")

        for phase_epoch_index in range(1, epoch_count + 1):
            epoch_index = start_epoch_index + phase_epoch_index - 1
            current_learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = run_epoch(
                model=model,
                data_loader=train_data_loader,
                criterion=criterion,
                optimizer=optimizer,
                experiment_config=experiment_config,
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
            validation_metrics = run_epoch(
                model=model,
                data_loader=validation_data_loader,
                criterion=criterion,
                optimizer=None,
                experiment_config=experiment_config,
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
            scheduler.step()

            validation_metric_value = float(
                validation_metrics[experiment_config.evaluation.primary_metric_name]
            )
            checkpoint_selection_score = compute_checkpoint_selection_score(
                validation_metrics,
                experiment_config,
            )
            history_record = {
                "epoch": epoch_index,
                "phase": phase_name,
                "phase_epoch": phase_epoch_index,
                "train": train_metrics,
                "validation": validation_metrics,
                "validation_checkpoint_selection_score": checkpoint_selection_score,
                "learning_rate": current_learning_rate,
            }
            history.append(history_record)

            LOGGER.info(
                "Phase %s | Epoch %03d | phase_epoch=%03d | lr=%.8f | train[%s] | validation[%s]",
                phase_name,
                epoch_index,
                phase_epoch_index,
                current_learning_rate,
                _format_metrics_for_log(train_metrics),
                _format_metrics_for_log(validation_metrics),
            )

            if wandb_run is not None:
                _safe_wandb_log(
                    wandb_run=wandb_run,
                    log_payload={
                        "epoch": epoch_index,
                        "phase_epoch": phase_epoch_index,
                        "learning_rate": current_learning_rate,
                        "training_phase/name": phase_name,
                        "validation/checkpoint_selection_score": checkpoint_selection_score,
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **{f"validation/{key}": value for key, value in validation_metrics.items()},
                    },
                    log_context=f"{phase_name} epoch summary at epoch {epoch_index}",
                )

            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch_index,
                "phase": phase_name,
                "phase_epoch": phase_epoch_index,
                "validation_metric": validation_metric_value,
                "checkpoint_selection_score": checkpoint_selection_score,
                "accuracy": float(validation_metrics["accuracy"]),
                "macro_f1": float(validation_metrics["macro_f1"]),
                "mcc": float(validation_metrics["mcc"]),
            }

            if checkpoint_selection_score > stage_best_score:
                stage_best_score = checkpoint_selection_score
                stage_best_epoch_index = epoch_index
                stage_best_metric = validation_metric_value
                stage_best_acc = float(validation_metrics["accuracy"])
                stage_best_f1 = float(validation_metrics["macro_f1"])
                stage_best_mcc = float(validation_metrics["mcc"])
                torch.save(checkpoint_payload, stage_best_checkpoint_file_path)
                LOGGER.info(
                    "Saved %s model-weight checkpoint at epoch %03d with checkpoint_selection_score=%.6f",
                    phase_name,
                    epoch_index,
                    checkpoint_selection_score,
                )

            if checkpoint_selection_score > best_checkpoint_selection_score:
                best_checkpoint_selection_score = checkpoint_selection_score
                best_validation_metric = validation_metric_value
                best_epoch_index = epoch_index
                best_phase_name = phase_name
                best_acc = float(validation_metrics["accuracy"])
                best_f1 = float(validation_metrics["macro_f1"])
                best_mcc = float(validation_metrics["mcc"])
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch_index,
                        "phase": phase_name,
                        "phase_epoch": phase_epoch_index,
                        "best_epoch": best_epoch_index,
                        "best_phase": best_phase_name,
                        "best_validation_metric": best_validation_metric,
                        "best_checkpoint_selection_score": best_checkpoint_selection_score,
                        "best_acc": best_acc,
                        "best_f1": best_f1,
                        "best_mcc": best_mcc,
                        "history": history,
                    },
                    best_checkpoint_file_path,
                )
                LOGGER.info(
                    "Saved final-best model-weight checkpoint at epoch %03d with checkpoint_selection_score=%.6f",
                    epoch_index,
                    best_checkpoint_selection_score,
                )

            write_json_file(
                history_file_path,
                {
                    "best_epoch": best_epoch_index,
                    "best_phase": best_phase_name,
                    "best_validation_metric": best_validation_metric,
                    "best_checkpoint_selection_score": best_checkpoint_selection_score,
                    "best_acc": best_acc,
                    "best_f1": best_f1,
                    "best_mcc": best_mcc,
                    "stage_summaries": stage_summaries,
                    "history": history,
                },
            )

        return {
            "phase": phase_name,
            "best_epoch": stage_best_epoch_index,
            "best_validation_metric": stage_best_metric,
            "best_checkpoint_selection_score": stage_best_score,
            "best_acc": stage_best_acc,
            "best_f1": stage_best_f1,
            "best_mcc": stage_best_mcc,
        }

    stage_summaries["stage1"] = run_training_phase(
        phase_name="stage1_high_lr",
        start_epoch_index=1,
        epoch_count=first_stage_epoch_count,
        learning_rate=first_stage_learning_rate,
        eta_minimum=experiment_config.training.minimum_learning_rate,
        stage_best_checkpoint_file_path=first_stage_best_checkpoint_file_path,
    )

    first_stage_checkpoint_payload = torch.load(
        first_stage_best_checkpoint_file_path,
        map_location="cpu",
    )
    model.load_state_dict(first_stage_checkpoint_payload["model_state_dict"])
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    LOGGER.info(
        "Reset optimizer/scheduler and rolled model back to first-stage best epoch %03d before stage2.",
        int(first_stage_checkpoint_payload["epoch"]),
    )

    stage_summaries["stage2"] = run_training_phase(
        phase_name="stage2_low_lr_reset",
        start_epoch_index=first_stage_epoch_count + 1,
        epoch_count=second_stage_epoch_count,
        learning_rate=second_stage_learning_rate,
        eta_minimum=second_stage_minimum_learning_rate,
        stage_best_checkpoint_file_path=second_stage_best_checkpoint_file_path,
    )

    summary_payload = {
        "best_epoch": best_epoch_index,
        "best_phase": best_phase_name,
        "best_validation_metric": best_validation_metric,
        "best_checkpoint_selection_score": best_checkpoint_selection_score,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_mcc": best_mcc,
        "stage_summaries": stage_summaries,
        "history": history,
    }
    write_json_file(history_file_path, summary_payload)

    if wandb_run is not None and should_finish_wandb_run:
        wandb_run.finish()

    return summary_payload


def evaluate_model(
    model: nn.Module,
    checkpoint_file_path: Path,
    data_loader,
    experiment_config: ExperimentConfig,
    split_name: str,
    run_directory: Path,
    wandb_run=None,
) -> dict:
    """Load one checkpoint and evaluate it on one split."""

    checkpoint_payload = torch.load(checkpoint_file_path, map_location="cpu")
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(torch.device(experiment_config.experiment.device))
    LOGGER.info("Evaluating %s split with checkpoint %s", split_name, checkpoint_file_path)

    class_weights = load_training_split_class_weights(experiment_config).to(
        torch.device(experiment_config.experiment.device)
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=experiment_config.training.label_smoothing,
    )
    metrics = run_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=None,
        experiment_config=experiment_config,
        device=torch.device(experiment_config.experiment.device),
        split_name=split_name,
        epoch_index=0,
        diagnostics_file_path=_build_epoch_diagnostics_file_path(
            run_directory=run_directory,
            experiment_config=experiment_config,
            split_name=split_name,
            epoch_index=0,
        ),
        wandb_run=None,
    )
    if experiment_config.experiment.device == "cuda":
        torch.cuda.empty_cache()
    write_json_file(run_directory / experiment_config.paths.metrics_subdirectory / f"{split_name}_metrics.json", metrics)
    LOGGER.info("%s metrics | %s", split_name, _format_metrics_for_log(metrics))
    if wandb_run is not None:
        _safe_wandb_log(
            wandb_run=wandb_run,
            log_payload={f"{split_name}/{key}": value for key, value in metrics.items()},
            log_context=f"{split_name} final metrics",
        )
    return metrics
