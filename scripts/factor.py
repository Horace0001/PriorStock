"""Train a factor-attention concat classifier on top of a frozen wide adapter checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import torch
import wandb
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from priorstock.config import load_experiment_config
from priorstock.exceptions import ConfigurationError
from priorstock.training.engine import _safe_wandb_log
from priorstock.training.single_logit_binary_engine import (
    SingleLogitBinaryObjectiveConfig,
    compute_positive_class_weight,
    compute_single_logit_bce_loss,
    compute_single_logit_binary_metrics,
    compute_single_logit_checkpoint_selection_score,
    compute_single_logit_soft_mcc_loss,
    compute_up_probability_and_margin,
    select_single_logit_checkpoint_metrics,
    should_consider_checkpoint_selection_epoch,
)
from priorstock.utils.environment import load_project_local_environment_files
from priorstock.utils.io import write_json_file
from priorstock.utils.logging_utils import configure_logger
from priorstock.utils.seed import set_global_seed
from priorstock.versioned.ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_v1 import (
    BASE_ATTENTION_SIDE_ADAPTER_VARIANT_NAME,
    FACTOR_PROJECTION_MODE_MLP_THEN_ADD_RANK,
    FACTOR_PROJECTION_MODE_RANK_CONCAT_LINEAR,
    FactorConcatLogitClassifierConfig,
    FactorConcatLogitClassifierV1,
    PriorStockOHLCV124GroupedFactorEmbeddingSingleLogitDataset,
)
from scripts.base import (
    ATTENTION_SIDE_ADAPTER_SECTION_NAME,
    BINARY_CLASSIFICATION_SECTION_NAME,
    SINGLE_LOGIT_SECTION_NAME,
    _coerce_float,
    _coerce_int,
    _load_merged_raw_config,
    _parse_attention_side_adapter_config,
    _parse_single_logit_objective_config,
    _write_single_logit_labeling_contract,
)


EXPECTED_FRAMEWORK_VARIANT_NAME = (
    "ohlcv124_group_token_mixer_attention_side_adapter_factor_concat_logit_classifier_v1"
)
FACTOR_CONCAT_CLASSIFIER_SECTION_NAME = "factor_concat_logit_classifier"
LOGGER = configure_logger("priorstock.factor")


def _resolve_project_path(path_text: str) -> Path:
    """Resolve one path relative to the project root."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move one collated batch dictionary to the selected device."""

    return {
        "sample_id": batch["sample_id"],
        "stock_id": batch["stock_id"],
        "target_trade_date": batch["target_trade_date"],
        "price_features": batch["price_features"].to(device),
        "technical_indicator_features": batch["technical_indicator_features"].to(device),
        "news_embeddings": batch["news_embeddings"].to(device),
        "has_news": batch["has_news"].to(device),
        "factor_embeddings": batch["factor_embeddings"].to(device),
        "has_factors": batch["has_factors"].to(device),
        "label": batch["label"].to(device),
        "target_return": batch["target_return"].to(device),
        "soft_up_target": batch["soft_up_target"].to(device),
    }


def _build_data_loader(
    experiment_config,
    objective_config: SingleLogitBinaryObjectiveConfig,
    factor_classifier_config: FactorConcatLogitClassifierConfig,
    split_name: str,
    should_shuffle: bool,
) -> DataLoader:
    """Build a factor concat-logit dataloader."""

    dataset = PriorStockOHLCV124GroupedFactorEmbeddingSingleLogitDataset(
        experiment_config=experiment_config,
        split_name=split_name,
        objective_config=objective_config,
        factor_embedding_cache_directory=_resolve_project_path(
            factor_classifier_config.factor_embedding_cache_directory
        ),
        expected_factor_count=factor_classifier_config.factor_count,
        expected_factor_embedding_dim=factor_classifier_config.factor_input_dim,
        should_load_factor_embeddings=True,
    )
    return DataLoader(
        dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=should_shuffle,
        num_workers=0,
        pin_memory=experiment_config.training.pin_memory,
    )


def _load_frozen_base_checkpoint(
    model: FactorConcatLogitClassifierV1,
    checkpoint_file_path: Path,
) -> None:
    """Load the base checkpoint into the wrapper and freeze it."""

    checkpoint_payload = torch.load(checkpoint_file_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict) or "model_state_dict" not in checkpoint_payload:
        raise ValueError(f"Checkpoint must contain model_state_dict: {checkpoint_file_path}")
    model.base_model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.freeze_base_model()


def _extract_dataset_label_values(data_loader: DataLoader) -> list[int]:
    """Return all hard binary labels from a dataloader's dataset."""

    dataset = data_loader.dataset
    if hasattr(dataset, "get_label_values"):
        return [int(label_value) for label_value in dataset.get_label_values()]
    return [int(dataset[sample_index]["label"].item()) for sample_index in range(len(dataset))]


def _flatten_factor_diagnostics(raw_diagnostics: dict[str, Any]) -> dict[str, float]:
    """Flatten scalar and per-factor diagnostics into numeric metric keys."""

    flattened: dict[str, float] = {}
    for key, value in raw_diagnostics.items():
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                flattened[f"{key}_{index}"] = float(item)
        else:
            flattened[key] = float(value)
    return flattened


def _parse_factor_concat_classifier_config(raw_section: object) -> FactorConcatLogitClassifierConfig:
    """Parse the factor concat-logit classifier YAML section."""

    if not isinstance(raw_section, dict):
        raise ConfigurationError("factor_concat_logit_classifier must be a YAML mapping.")
    shared_field_names = {
        "base_checkpoint_file_path",
        "factor_embedding_cache_directory",
        "factor_count",
        "factor_input_dim",
        "factor_projector_hidden_dim",
        "attention_head_count",
        "dropout_probability",
    }
    optional_field_names = {
        "should_use_factor_transformer_ffn",
        "factor_transformer_ffn_hidden_dim",
        "factor_transformer_ffn_dropout_probability",
        "should_train_base_model",
        "base_frozen_epoch_count",
        "factor_learning_rate",
        "base_tail_learning_rate",
        "base_classifier_learning_rate",
        "base_trainable_scope",
        "should_freeze_fusion_gate_during_joint_training",
        "allow_partial_base_checkpoint_loading",
        "soft_mcc_loss_weight",
        "soft_metric_loss_epsilon",
        "factor_projection_mode",
    }
    single_hidden_field_names = {"classifier_hidden_dim"}
    two_hidden_field_names = {"classifier_hidden_dim_1", "classifier_hidden_dim_2"}
    provided_field_names = set(raw_section.keys())
    uses_single_hidden_schema = single_hidden_field_names.issubset(provided_field_names)
    uses_two_hidden_schema = two_hidden_field_names.issubset(provided_field_names)
    if uses_single_hidden_schema == uses_two_hidden_schema:
        raise ConfigurationError(
            "factor_concat_logit_classifier must provide either classifier_hidden_dim "
            "or classifier_hidden_dim_1 and classifier_hidden_dim_2, but not both."
        )
    expected_field_names = (
        shared_field_names | single_hidden_field_names
        if uses_single_hidden_schema
        else shared_field_names | two_hidden_field_names
    )
    expected_field_names = expected_field_names | (provided_field_names & optional_field_names)
    missing_field_names = sorted(expected_field_names - provided_field_names)
    unexpected_field_names = sorted(provided_field_names - expected_field_names)
    if missing_field_names:
        raise ConfigurationError(
            "factor_concat_logit_classifier missing fields: " + ", ".join(missing_field_names)
        )
    if unexpected_field_names:
        raise ConfigurationError(
            "factor_concat_logit_classifier unexpected fields: " + ", ".join(unexpected_field_names)
        )
    raw_should_use_factor_transformer_ffn = raw_section.get(
        "should_use_factor_transformer_ffn",
        False,
    )
    if not isinstance(raw_should_use_factor_transformer_ffn, bool):
        raise ConfigurationError("should_use_factor_transformer_ffn must be a boolean.")
    if raw_should_use_factor_transformer_ffn:
        required_factor_transformer_field_names = {
            "factor_transformer_ffn_hidden_dim",
            "factor_transformer_ffn_dropout_probability",
        }
        missing_factor_transformer_field_names = sorted(
            required_factor_transformer_field_names - provided_field_names
        )
        if missing_factor_transformer_field_names:
            raise ConfigurationError(
                "factor_concat_logit_classifier missing enabled transformer FFN fields: "
                + ", ".join(missing_factor_transformer_field_names)
            )
    raw_should_train_base_model = raw_section.get("should_train_base_model", False)
    if not isinstance(raw_should_train_base_model, bool):
        raise ConfigurationError("should_train_base_model must be a boolean.")
    raw_should_freeze_fusion_gate = raw_section.get(
        "should_freeze_fusion_gate_during_joint_training",
        True,
    )
    if not isinstance(raw_should_freeze_fusion_gate, bool):
        raise ConfigurationError("should_freeze_fusion_gate_during_joint_training must be a boolean.")
    raw_allow_partial_base_checkpoint_loading = raw_section.get(
        "allow_partial_base_checkpoint_loading",
        False,
    )
    if not isinstance(raw_allow_partial_base_checkpoint_loading, bool):
        raise ConfigurationError("allow_partial_base_checkpoint_loading must be a boolean.")
    raw_base_trainable_scope = str(raw_section.get("base_trainable_scope", "tail"))
    if raw_should_train_base_model:
        required_joint_training_field_names = {
            "base_frozen_epoch_count",
            "factor_learning_rate",
            "base_tail_learning_rate",
            "base_classifier_learning_rate",
        }
        missing_joint_training_field_names = sorted(
            required_joint_training_field_names - provided_field_names
        )
        if missing_joint_training_field_names:
            raise ConfigurationError(
                "factor_concat_logit_classifier missing enabled joint-training fields: "
                + ", ".join(missing_joint_training_field_names)
            )
    classifier_hidden_dimensions = (
        (_coerce_int(raw_section["classifier_hidden_dim"], "classifier_hidden_dim"),)
        if uses_single_hidden_schema
        else (
            _coerce_int(raw_section["classifier_hidden_dim_1"], "classifier_hidden_dim_1"),
            _coerce_int(raw_section["classifier_hidden_dim_2"], "classifier_hidden_dim_2"),
        )
    )
    config = FactorConcatLogitClassifierConfig(
        base_checkpoint_file_path=str(raw_section["base_checkpoint_file_path"]),
        factor_embedding_cache_directory=str(raw_section["factor_embedding_cache_directory"]),
        factor_count=_coerce_int(raw_section["factor_count"], "factor_count"),
        factor_input_dim=_coerce_int(raw_section["factor_input_dim"], "factor_input_dim"),
        factor_projector_hidden_dim=_coerce_int(
            raw_section["factor_projector_hidden_dim"],
            "factor_projector_hidden_dim",
        ),
        classifier_hidden_dimensions=classifier_hidden_dimensions,
        attention_head_count=_coerce_int(raw_section["attention_head_count"], "attention_head_count"),
        dropout_probability=_coerce_float(raw_section["dropout_probability"], "dropout_probability"),
        base_framework_variant_name=BASE_ATTENTION_SIDE_ADAPTER_VARIANT_NAME,
        should_use_factor_transformer_ffn=raw_should_use_factor_transformer_ffn,
        factor_transformer_ffn_hidden_dim=_coerce_int(
            raw_section.get("factor_transformer_ffn_hidden_dim", 0),
            "factor_transformer_ffn_hidden_dim",
        ),
        factor_transformer_ffn_dropout_probability=_coerce_float(
            raw_section.get("factor_transformer_ffn_dropout_probability", 0.0),
            "factor_transformer_ffn_dropout_probability",
        ),
        should_train_base_model=raw_should_train_base_model,
        base_frozen_epoch_count=_coerce_int(
            raw_section.get("base_frozen_epoch_count", 0),
            "base_frozen_epoch_count",
        ),
        factor_learning_rate=_coerce_float(
            raw_section.get("factor_learning_rate", 0.0),
            "factor_learning_rate",
        ),
        base_tail_learning_rate=_coerce_float(
            raw_section.get("base_tail_learning_rate", 0.0),
            "base_tail_learning_rate",
        ),
        base_classifier_learning_rate=_coerce_float(
            raw_section.get("base_classifier_learning_rate", 0.0),
            "base_classifier_learning_rate",
        ),
        base_trainable_scope=raw_base_trainable_scope,
        should_freeze_fusion_gate_during_joint_training=raw_should_freeze_fusion_gate,
        allow_partial_base_checkpoint_loading=raw_allow_partial_base_checkpoint_loading,
        soft_mcc_loss_weight=_coerce_float(
            raw_section.get("soft_mcc_loss_weight", 0.0),
            "soft_mcc_loss_weight",
        ),
        soft_metric_loss_epsilon=_coerce_float(
            raw_section.get("soft_metric_loss_epsilon", 1.0e-8),
            "soft_metric_loss_epsilon",
        ),
        factor_projection_mode=str(
            raw_section.get(
                "factor_projection_mode",
                FACTOR_PROJECTION_MODE_MLP_THEN_ADD_RANK,
            )
        ),
    )
    if config.factor_count <= 0 or config.factor_input_dim <= 0:
        raise ConfigurationError("factor count and input dimension must be positive.")
    if config.factor_projector_hidden_dim <= 0 or any(
        hidden_dimension <= 0 for hidden_dimension in config.classifier_hidden_dimensions
    ):
        raise ConfigurationError("factor concat classifier hidden dimensions must be positive.")
    if config.attention_head_count <= 0:
        raise ConfigurationError("attention_head_count must be positive.")
    if not 0.0 <= config.dropout_probability < 1.0:
        raise ConfigurationError("dropout_probability must be in [0, 1).")
    if config.should_use_factor_transformer_ffn and config.factor_transformer_ffn_hidden_dim <= 0:
        raise ConfigurationError("factor_transformer_ffn_hidden_dim must be positive when enabled.")
    if not 0.0 <= config.factor_transformer_ffn_dropout_probability < 1.0:
        raise ConfigurationError("factor_transformer_ffn_dropout_probability must be in [0, 1).")
    if config.base_frozen_epoch_count < 0:
        raise ConfigurationError("base_frozen_epoch_count must be non-negative.")
    if config.should_train_base_model and (
        config.factor_learning_rate <= 0.0
        or config.base_tail_learning_rate <= 0.0
        or config.base_classifier_learning_rate <= 0.0
    ):
        raise ConfigurationError("joint-training learning rates must be positive when enabled.")
    if config.base_trainable_scope not in {"tail", "full"}:
        raise ConfigurationError("base_trainable_scope must be either 'tail' or 'full'.")
    if config.soft_mcc_loss_weight < 0.0:
        raise ConfigurationError("soft_mcc_loss_weight must be non-negative.")
    if config.soft_metric_loss_epsilon <= 0.0:
        raise ConfigurationError("soft_metric_loss_epsilon must be positive.")
    supported_factor_projection_modes = {
        FACTOR_PROJECTION_MODE_MLP_THEN_ADD_RANK,
        FACTOR_PROJECTION_MODE_RANK_CONCAT_LINEAR,
    }
    if config.factor_projection_mode not in supported_factor_projection_modes:
        raise ConfigurationError(
            "factor_projection_mode must be one of "
            + ", ".join(sorted(supported_factor_projection_modes))
            + "."
        )
    return config


def _load_config_with_factor_concat_classifier_extensions(config_file_path: Path, run_directory: Path):
    """Load ExperimentConfig while accepting concat classifier script-local sections."""

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
    sanitized_config.pop(BINARY_CLASSIFICATION_SECTION_NAME, None)
    run_directory.mkdir(parents=True, exist_ok=True)
    sanitized_config_file_path = run_directory / "experiment_config_without_script_extensions.yaml"
    with sanitized_config_file_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(sanitized_config, file_handle, sort_keys=False, allow_unicode=False)
    experiment_config = load_experiment_config(sanitized_config_file_path)
    return experiment_config, attention_side_adapter_config, objective_config, factor_classifier_config


def _build_model(
    experiment_config,
    attention_side_adapter_config,
    factor_classifier_config: FactorConcatLogitClassifierConfig,
) -> FactorConcatLogitClassifierV1:
    """Instantiate the factor concat-logit classifier wrapper model."""

    return FactorConcatLogitClassifierV1(
        experiment_config=experiment_config,
        attention_side_adapter_config=attention_side_adapter_config,
        factor_classifier_config=factor_classifier_config,
    )


def _iter_factor_concat_trainable_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return trainable wrapper-side parameters outside the wide-adapter base model."""

    return [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if not parameter_name.startswith("base_model.")
    ]


def _iter_base_tail_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return parameters from the final temporal transformer layer of the base model."""

    transformer_layers = getattr(model.base_model, "transformer_layers", None)
    if transformer_layers is None or len(transformer_layers) == 0:
        raise ConfigurationError("Joint training requires base_model.transformer_layers to be non-empty.")
    return list(transformer_layers[-1].parameters())


def _iter_base_full_parameters_without_classifier(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return base-model parameters outside the classifier head."""

    return [
        parameter
        for parameter_name, parameter in model.base_model.named_parameters()
        if not parameter_name.startswith("classifier.")
    ]


def _iter_base_joint_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return base-model parameters selected for joint training."""

    base_trainable_scope = getattr(model.config, "base_trainable_scope", "tail")
    if base_trainable_scope == "full":
        return _iter_base_full_parameters_without_classifier(model)
    return _iter_base_tail_parameters(model)


def _iter_base_classifier_parameters(model: FactorConcatLogitClassifierV1) -> list[nn.Parameter]:
    """Return parameters from the base model classifier head."""

    classifier = getattr(model.base_model, "classifier", None)
    if classifier is None:
        raise ConfigurationError("Joint training requires base_model.classifier.")
    return list(classifier.parameters())


def _set_requires_grad(parameters: list[nn.Parameter], should_require_grad: bool) -> None:
    """Set requires_grad for each parameter in a list."""

    for parameter in parameters:
        parameter.requires_grad = should_require_grad


def _apply_factor_concat_joint_training_epoch_state(
    model: FactorConcatLogitClassifierV1,
    epoch_index: int,
) -> None:
    """Apply the factor concat joint-training freeze schedule for one epoch."""

    factor_classifier_config = model.config
    _set_requires_grad(_iter_factor_concat_trainable_parameters(model), True)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    if not factor_classifier_config.should_train_base_model:
        return
    if epoch_index > factor_classifier_config.base_frozen_epoch_count:
        _set_requires_grad(_iter_base_joint_parameters(model), True)
        _set_requires_grad(_iter_base_classifier_parameters(model), True)
    if factor_classifier_config.should_freeze_fusion_gate_during_joint_training and hasattr(
        model.base_model,
        "fusion_gate",
    ):
        for parameter in model.base_model.fusion_gate.parameters():
            parameter.requires_grad = False


def _build_factor_concat_optimizer(
    model: FactorConcatLogitClassifierV1,
    experiment_config,
) -> torch.optim.Optimizer:
    """Build AdamW with optional joint-training parameter groups."""

    factor_classifier_config = model.config
    optimizer_common_kwargs = {
        "betas": (experiment_config.training.adam_beta_one, experiment_config.training.adam_beta_two),
        "eps": experiment_config.training.adam_epsilon,
        "weight_decay": experiment_config.training.weight_decay,
    }
    if not factor_classifier_config.should_train_base_model:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=experiment_config.training.learning_rate,
            **optimizer_common_kwargs,
        )
    return torch.optim.AdamW(
        [
            {
                "params": _iter_factor_concat_trainable_parameters(model),
                "lr": factor_classifier_config.factor_learning_rate,
                "name": "factor",
            },
            {
                "params": _iter_base_joint_parameters(model),
                "lr": factor_classifier_config.base_tail_learning_rate,
                "name": f"base_{factor_classifier_config.base_trainable_scope}",
            },
            {
                "params": _iter_base_classifier_parameters(model),
                "lr": factor_classifier_config.base_classifier_learning_rate,
                "name": "base_classifier",
            },
        ],
        **optimizer_common_kwargs,
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


def _run_factor_concat_epoch(
    model: FactorConcatLogitClassifierV1,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    experiment_config,
    objective_config,
    positive_class_weight: torch.Tensor,
    split_name: str,
) -> dict[str, float]:
    """Run one factor concat train/eval epoch with optional joint-training diagnostics."""

    is_training = optimizer is not None
    device = torch.device(experiment_config.experiment.device)
    model.train(mode=is_training)
    if not is_training or not model.config.should_train_base_model:
        model.base_model.eval()
    total_loss = 0.0
    bce_loss_sum = 0.0
    soft_mcc_loss_sum = 0.0
    total_samples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    up_probabilities: list[float] = []
    logit_margins: list[float] = []
    target_returns: list[float] = []
    soft_up_targets: list[float] = []
    factor_diagnostic_weighted_sums: dict[str, float] = {}
    base_gradient_norm_sum = 0.0
    factor_gradient_norm_sum = 0.0
    gradient_batch_count = 0
    factor_parameters = _iter_factor_concat_trainable_parameters(model)
    base_parameters = list(model.base_model.parameters())
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
            bce_loss = compute_single_logit_bce_loss(
                logits=model_output.logits,
                soft_up_targets=batch["soft_up_target"],
                positive_class_weight=positive_class_weight,
            )
            soft_mcc_loss = compute_single_logit_soft_mcc_loss(
                logits=model_output.logits,
                hard_labels=batch["label"].to(dtype=torch.float32),
                epsilon=model.config.soft_metric_loss_epsilon,
            )
            loss = bce_loss + (model.config.soft_mcc_loss_weight * soft_mcc_loss)
            if is_training:
                loss.backward()
                base_gradient_norm_sum += _compute_parameter_gradient_norm(base_parameters)
                factor_gradient_norm_sum += _compute_parameter_gradient_norm(factor_parameters)
                gradient_batch_count += 1
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    experiment_config.training.gradient_clip_max_norm,
                )
                optimizer.step()
            batch_size = int(batch["label"].shape[0])
            total_samples += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            bce_loss_sum += float(bce_loss.detach().cpu()) * batch_size
            soft_mcc_loss_sum += float(soft_mcc_loss.detach().cpu()) * batch_size
            up_probability, logit_margin = compute_up_probability_and_margin(model_output.logits)
            predicted_label = (up_probability >= 0.5).to(dtype=torch.long)
            true_labels.extend([int(value) for value in batch["label"].detach().cpu().tolist()])
            predicted_labels.extend([int(value) for value in predicted_label.detach().cpu().tolist()])
            up_probabilities.extend([float(value) for value in up_probability.detach().cpu().tolist()])
            logit_margins.extend([float(value) for value in logit_margin.detach().cpu().tolist()])
            target_returns.extend([float(value) for value in batch["target_return"].detach().cpu().tolist()])
            soft_up_targets.extend([float(value) for value in batch["soft_up_target"].detach().cpu().tolist()])
            factor_diagnostics = _flatten_factor_diagnostics(
                model_output.diagnostics["factor_enhanced_classifier"]
            )
            for metric_name, metric_value in factor_diagnostics.items():
                factor_diagnostic_weighted_sums[metric_name] = factor_diagnostic_weighted_sums.get(
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
    metrics["binary_cross_entropy_loss"] = bce_loss_sum / float(max(total_samples, 1))
    metrics["soft_mcc_loss"] = soft_mcc_loss_sum / float(max(total_samples, 1))
    metrics.update(
        {
            f"factor_enhanced/{metric_name}": metric_sum / float(max(total_samples, 1))
            for metric_name, metric_sum in factor_diagnostic_weighted_sums.items()
        }
    )
    metrics["training/base_grad_norm"] = base_gradient_norm_sum / float(max(gradient_batch_count, 1))
    metrics["training/factor_grad_norm"] = factor_gradient_norm_sum / float(max(gradient_batch_count, 1))
    metrics["selection_score"] = compute_single_logit_checkpoint_selection_score(
        metrics,
        objective_config,
    )
    LOGGER.info(
        "%s loss=%.6f bce=%.6f soft_mcc=%.6f score=%.6f mcc=%.6f auc=%.6f acc=%.6f pred_up=%.6f "
        "up_recall=%.6f down_recall=%.6f recall_gap=%.6f base_grad=%.6f "
        "factor_grad=%.6f delta_abs=%.8f",
        split_name,
        metrics["total_loss"],
        metrics["binary_cross_entropy_loss"],
        metrics["soft_mcc_loss"],
        metrics["selection_score"],
        metrics["mcc"],
        metrics["roc_auc"],
        metrics["accuracy"],
        metrics["prediction_up_ratio"],
        metrics["up_recall"],
        metrics["down_recall"],
        metrics["recall_gap_abs"],
        metrics["training/base_grad_norm"],
        metrics["training/factor_grad_norm"],
        metrics.get("factor_enhanced/delta_logit_abs_mean", 0.0),
    )
    return metrics


def _save_checkpoint(
    checkpoint_file_path: Path,
    model: FactorConcatLogitClassifierV1,
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    validation_metrics: dict[str, float],
    checkpoint_selection_metrics: dict[str, float] | None = None,
    checkpoint_selection_score: float | None = None,
    checkpoint_selection_metric_name: str = "selection_score",
    checkpoint_selection_split_name: str = "validation",
) -> None:
    """Save a factor concat classifier checkpoint."""

    resolved_selection_metrics = (
        validation_metrics if checkpoint_selection_metrics is None else checkpoint_selection_metrics
    )
    resolved_selection_score = (
        float(resolved_selection_metrics["selection_score"])
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
            "checkpoint_selection_metrics": resolved_selection_metrics,
            "selection_metric_name": checkpoint_selection_metric_name,
            "selection_split_name": checkpoint_selection_split_name,
            "selection_score": resolved_selection_score,
        },
        checkpoint_file_path,
    )


def _load_checkpoint(
    model: FactorConcatLogitClassifierV1,
    checkpoint_file_path: Path,
    device: torch.device,
) -> None:
    """Load a saved factor concat classifier checkpoint."""

    checkpoint_payload = torch.load(checkpoint_file_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.freeze_base_model()


def _load_base_checkpoint_for_factor_concat(
    model: FactorConcatLogitClassifierV1,
    checkpoint_file_path: Path,
) -> None:
    """Load base checkpoint, optionally allowing added layers to stay initialized."""

    checkpoint_payload = torch.load(checkpoint_file_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict) or "model_state_dict" not in checkpoint_payload:
        raise ValueError(f"Checkpoint must contain model_state_dict: {checkpoint_file_path}")
    checkpoint_state_dict = checkpoint_payload["model_state_dict"]
    if not model.config.allow_partial_base_checkpoint_loading:
        model.base_model.load_state_dict(checkpoint_state_dict, strict=True)
        model.freeze_base_model()
        return

    current_state_dict = model.base_model.state_dict()
    compatible_state_dict = {}
    skipped_missing_names = []
    skipped_shape_names = []
    for parameter_name, checkpoint_value in checkpoint_state_dict.items():
        current_value = current_state_dict.get(parameter_name)
        if current_value is None:
            skipped_missing_names.append(parameter_name)
            continue
        if tuple(current_value.shape) != tuple(checkpoint_value.shape):
            skipped_shape_names.append(parameter_name)
            continue
        compatible_state_dict[parameter_name] = checkpoint_value
    if not compatible_state_dict:
        raise ValueError(
            "No compatible base checkpoint tensors were found for partial loading: "
            f"{checkpoint_file_path}"
        )
    current_state_dict.update(compatible_state_dict)
    model.base_model.load_state_dict(current_state_dict, strict=True)
    initialized_from_scratch_names = sorted(set(current_state_dict) - set(compatible_state_dict))
    LOGGER.info(
        "Partially loaded base checkpoint %s: loaded=%d initialized_from_scratch=%d "
        "checkpoint_missing_in_model=%d checkpoint_shape_mismatch=%d scratch_examples=%s",
        checkpoint_file_path,
        len(compatible_state_dict),
        len(initialized_from_scratch_names),
        len(skipped_missing_names),
        len(skipped_shape_names),
        initialized_from_scratch_names[:20],
    )
    model.freeze_base_model()


def _fit_model(
    model: FactorConcatLogitClassifierV1,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    validation_full_data_loader: DataLoader | None,
    experiment_config,
    objective_config: SingleLogitBinaryObjectiveConfig,
    full_objective_config: SingleLogitBinaryObjectiveConfig,
    run_directory: Path,
    wandb_run,
) -> dict[str, Any]:
    """Train the factor concat classifier with optional base-tail joint training."""

    device = torch.device(experiment_config.experiment.device)
    model.to(device)
    model.freeze_base_model()
    _apply_factor_concat_joint_training_epoch_state(model, epoch_index=1)
    positive_class_weight = compute_positive_class_weight(_extract_dataset_label_values(train_data_loader))
    optimizer = _build_factor_concat_optimizer(model, experiment_config)
    checkpoint_directory = run_directory / experiment_config.paths.checkpoint_subdirectory
    best_checkpoint_file_path = checkpoint_directory / "best_model.pt"
    latest_checkpoint_file_path = checkpoint_directory / experiment_config.training.latest_checkpoint_file_name
    history: list[dict[str, Any]] = []
    best_validation_score = -float("inf")
    best_checkpoint_selection_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch_index in range(1, experiment_config.training.num_epochs + 1):
        _apply_factor_concat_joint_training_epoch_state(model, epoch_index=epoch_index)
        train_metrics = _run_factor_concat_epoch(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            experiment_config=experiment_config,
            objective_config=objective_config,
            positive_class_weight=positive_class_weight,
            split_name=f"train epoch={epoch_index:03d}",
        )
        validation_metrics = _run_factor_concat_epoch(
            model=model,
            data_loader=validation_data_loader,
            optimizer=None,
            experiment_config=experiment_config,
            objective_config=objective_config,
            positive_class_weight=positive_class_weight,
            split_name=f"validation epoch={epoch_index:03d}",
        )
        validation_full_metrics = None
        should_evaluate_validation_full = (
            objective_config.should_evaluate_full_splits_each_epoch
            or objective_config.checkpoint_selection_split_name == "validation_full"
        )
        if should_evaluate_validation_full:
            if validation_full_data_loader is None:
                raise ValueError("validation_full evaluation requires validation_full_data_loader.")
            validation_full_metrics = _run_factor_concat_epoch(
                model=model,
                data_loader=validation_full_data_loader,
                optimizer=None,
                experiment_config=experiment_config,
                objective_config=full_objective_config,
                positive_class_weight=positive_class_weight,
                split_name=f"validation_full epoch={epoch_index:03d}",
            )
        omitted_iterator_count = objective_config.omitted_evaluation_iterator_count
        for _ in range(omitted_iterator_count):
            torch.empty((), dtype=torch.int64).random_().item()
        checkpoint_selection = select_single_logit_checkpoint_metrics(
            validation_metrics=validation_metrics,
            objective_config=objective_config,
            validation_full_metrics=validation_full_metrics,
        )
        should_consider_epoch = should_consider_checkpoint_selection_epoch(
            epoch_index=epoch_index,
            objective_config=objective_config,
        )
        epoch_record = {
            "epoch": epoch_index,
            "train": train_metrics,
            "validation": validation_metrics,
            "checkpoint_selection": {
                "split_name": checkpoint_selection.split_name,
                "score_name": checkpoint_selection.score_name,
                "score": checkpoint_selection.score,
                "within_epoch_limit": should_consider_epoch,
            },
            "learning_rates": {
                str(parameter_group.get("name", f"group_{group_index}")): float(parameter_group["lr"])
                for group_index, parameter_group in enumerate(optimizer.param_groups)
            },
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
            checkpoint_selection_metrics=checkpoint_selection.metrics,
            checkpoint_selection_score=checkpoint_selection.score,
            checkpoint_selection_metric_name=checkpoint_selection.score_name,
            checkpoint_selection_split_name=checkpoint_selection.split_name,
        )
        should_save_checkpoint = (
            should_consider_epoch
            and checkpoint_selection.score > best_checkpoint_selection_score
        )
        if should_save_checkpoint:
            best_checkpoint_selection_score = float(checkpoint_selection.score)
            best_validation_score = float(validation_metrics["selection_score"])
            best_epoch = epoch_index
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_file_path,
                model,
                optimizer,
                epoch_index,
                validation_metrics,
                checkpoint_selection_metrics=checkpoint_selection.metrics,
                checkpoint_selection_score=checkpoint_selection.score,
                checkpoint_selection_metric_name=checkpoint_selection.score_name,
                checkpoint_selection_split_name=checkpoint_selection.split_name,
            )
        else:
            epochs_without_improvement += 1
        log_payload = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in validation_metrics.items()},
            "epoch": epoch_index,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{
                f"learning_rate/{parameter_group.get('name', f'group_{group_index}')}": float(
                    parameter_group["lr"]
                )
                for group_index, parameter_group in enumerate(optimizer.param_groups)
            },
            "checkpoint_selection/score": checkpoint_selection.score,
            "checkpoint_selection/within_epoch_limit": float(should_consider_epoch),
            "best_validation_score": best_validation_score,
            "best_checkpoint_selection_score": best_checkpoint_selection_score,
        }
        if validation_full_metrics is not None:
            log_payload.update(
                {f"validation_full/{key}": value for key, value in validation_full_metrics.items()}
            )
        _safe_wandb_log(
            wandb_run,
            log_payload,
            log_context=f"factor concat epoch {epoch_index}",
        )
        LOGGER.info(
            "Epoch %03d best_epoch=%03d best_checkpoint_selection_score=%.6f "
            "current_checkpoint_selection_score=%.6f split=%s in_window=%s no_improve=%d",
            epoch_index,
            best_epoch,
            best_checkpoint_selection_score,
            checkpoint_selection.score,
            checkpoint_selection.split_name,
            should_consider_epoch,
            epochs_without_improvement,
        )
        if epochs_without_improvement >= experiment_config.training.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %03d.", epoch_index)
            break
    if best_epoch == 0:
        raise RuntimeError("No eligible checkpoint was selected.")
    write_json_file(run_directory / "training_history.json", history)
    return {
        "best_epoch": best_epoch,
        "best_validation_score": best_validation_score,
        "best_checkpoint_selection_score": best_checkpoint_selection_score,
        "checkpoint_selection_split_name": objective_config.checkpoint_selection_split_name,
        "checkpoint_selection_metric_name": "checkpoint_selection_score",
        "history": history,
        "best_checkpoint_file_path": str(best_checkpoint_file_path),
        "positive_class_weight": float(positive_class_weight.item()),
    }


def _evaluate_checkpoint(
    checkpoint_file_path: Path,
    model: FactorConcatLogitClassifierV1,
    data_loader: DataLoader,
    train_data_loader: DataLoader,
    experiment_config,
    objective_config,
    split_name: str,
    run_directory: Path,
    wandb_run,
) -> dict[str, float]:
    """Evaluate one saved factor concat checkpoint."""

    device = torch.device(experiment_config.experiment.device)
    _load_checkpoint(model=model, checkpoint_file_path=checkpoint_file_path, device=device)
    model.to(device)
    positive_class_weight = compute_positive_class_weight(_extract_dataset_label_values(train_data_loader))
    metrics = _run_factor_concat_epoch(
        model=model,
        data_loader=data_loader,
        optimizer=None,
        experiment_config=experiment_config,
        objective_config=objective_config,
        positive_class_weight=positive_class_weight,
        split_name=split_name,
    )
    write_json_file(run_directory / f"{split_name}_metrics.json", metrics)
    _safe_wandb_log(
        wandb_run,
        {f"{split_name}/{key}": value for key, value in metrics.items()},
        log_context=f"{split_name} final evaluation",
    )
    return metrics


def main() -> None:
    """Run factor concat-logit classifier single-logit training and evaluation."""

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
    ) = _load_config_with_factor_concat_classifier_extensions(parsed_arguments.config_file, run_directory)
    if experiment_config.model.framework_variant_name != EXPECTED_FRAMEWORK_VARIANT_NAME:
        raise ValueError(
            "Factor concat-logit classifier runner expects model.framework_variant_name to equal "
            f"{EXPECTED_FRAMEWORK_VARIANT_NAME}."
        )
    if experiment_config.model.num_classes != 1:
        raise ValueError("Factor concat-logit classifier runner requires model.num_classes to equal 1.")
    set_global_seed(experiment_config.experiment.random_seed)
    write_json_file(run_directory / "attention_side_adapter_config_snapshot.json", asdict(attention_side_adapter_config))
    write_json_file(run_directory / "single_logit_objective_config_snapshot.json", asdict(objective_config))
    write_json_file(run_directory / "factor_concat_classifier_config_snapshot.json", asdict(factor_classifier_config))
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
            },
            settings=wandb.Settings(init_timeout=300),
        )

    train_data_loader = _build_data_loader(experiment_config, objective_config, factor_classifier_config, "train", True)
    validation_data_loader = _build_data_loader(
        experiment_config,
        objective_config,
        factor_classifier_config,
        "validation",
        False,
    )
    test_data_loader = _build_data_loader(experiment_config, objective_config, factor_classifier_config, "test", False)
    full_objective_config = replace(objective_config, significant_return_absolute_threshold=0.0)
    should_build_full_epoch_loaders = (
        objective_config.should_evaluate_full_splits_each_epoch
        or objective_config.checkpoint_selection_split_name == "validation_full"
    )
    validation_full_data_loader = None
    if should_build_full_epoch_loaders:
        validation_full_data_loader = _build_data_loader(
            experiment_config,
            full_objective_config,
            factor_classifier_config,
            "validation",
            False,
        )
    base_checkpoint_file_path = _resolve_project_path(factor_classifier_config.base_checkpoint_file_path)
    model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
    _load_base_checkpoint_for_factor_concat(model, base_checkpoint_file_path)
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
        },
    )
    (run_directory / "model_architecture.txt").write_text(str(model), encoding="utf-8")

    training_summary = _fit_model(
        model=model,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        validation_full_data_loader=validation_full_data_loader,
        experiment_config=experiment_config,
        objective_config=objective_config,
        full_objective_config=full_objective_config,
        run_directory=run_directory,
        wandb_run=wandb_run,
    )
    best_checkpoint_file_path = Path(str(training_summary["best_checkpoint_file_path"]))
    validation_model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
    _load_base_checkpoint_for_factor_concat(validation_model, base_checkpoint_file_path)
    validation_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        validation_model,
        validation_data_loader,
        train_data_loader,
        experiment_config,
        objective_config,
        "validation",
        run_directory,
        wandb_run,
    )
    test_model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
    _load_base_checkpoint_for_factor_concat(test_model, base_checkpoint_file_path)
    test_metrics = _evaluate_checkpoint(
        best_checkpoint_file_path,
        test_model,
        test_data_loader,
        train_data_loader,
        experiment_config,
        objective_config,
        "test",
        run_directory,
        wandb_run,
    )
    unfiltered_test_metrics = None
    if objective_config.should_evaluate_unfiltered_test_split:
        unfiltered_objective_config = replace(objective_config, significant_return_absolute_threshold=0.0)
        unfiltered_test_data_loader = _build_data_loader(
            experiment_config,
            unfiltered_objective_config,
            factor_classifier_config,
            "test",
            False,
        )
        unfiltered_model = _build_model(experiment_config, attention_side_adapter_config, factor_classifier_config)
        _load_base_checkpoint_for_factor_concat(unfiltered_model, base_checkpoint_file_path)
        unfiltered_test_metrics = _evaluate_checkpoint(
            best_checkpoint_file_path,
            unfiltered_model,
            unfiltered_test_data_loader,
            train_data_loader,
            experiment_config,
            unfiltered_objective_config,
            "test_unfiltered",
            run_directory,
            wandb_run,
        )
    pipeline_summary = {
        "config_file_path": str(parsed_arguments.config_file),
        "run_directory": str(run_directory),
        "base_checkpoint_file_path": str(base_checkpoint_file_path),
        "checkpoint_file_path": str(best_checkpoint_file_path),
        "attention_side_adapter_config": asdict(attention_side_adapter_config),
        "single_logit_objective_config": asdict(objective_config),
        "factor_concat_classifier_config": asdict(factor_classifier_config),
        "training_summary": training_summary,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    if unfiltered_test_metrics is not None:
        pipeline_summary["test_unfiltered_metrics"] = unfiltered_test_metrics
    write_json_file(run_directory / "pipeline_summary.json", pipeline_summary)
    LOGGER.info("Factor concat-logit classifier single-logit pipeline finished: %s", pipeline_summary)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
