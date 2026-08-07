"""Train and evaluate single-logit return-aware attention-side adapter experiments."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from priorstock.config import get_market_artifact_root, load_experiment_config
from priorstock.exceptions import ConfigurationError
from priorstock.training.single_logit_binary_engine import (
    SingleLogitBinaryObjectiveConfig,
    build_single_logit_pipeline_summary,
    evaluate_single_logit_binary_model,
    fit_single_logit_binary_model,
)
from priorstock.utils.environment import load_project_local_environment_files
from priorstock.utils.io import path_from_serialized_value, write_json_file
from priorstock.utils.logging_utils import configure_logger
from priorstock.utils.seed import set_global_seed
from priorstock.versioned.ohlcv124_group_fusion_v1.dataset import (
    PriorStockOHLCV124GroupedDataset,
)
from priorstock.versioned.ohlcv124_group_token_mixer_attention_side_adapter_v1.model import (
    AttentionSideAdapterConfig,
    PriorStockV3OHLCV124GroupTokenMixerAttentionSideAdapter,
)
EXPECTED_FRAMEWORK_VARIANT_NAME = "ohlcv124_group_token_mixer_attention_side_adapter_v1"
ATTENTION_SIDE_ADAPTER_SECTION_NAME = "attention_side_adapter"
SINGLE_LOGIT_SECTION_NAME = "single_logit_binary_classification"
BINARY_CLASSIFICATION_SECTION_NAME = "binary_classification"
BASE_CONFIG_FILE_FIELD_NAME = "base_config_file"
OVERRIDES_FIELD_NAME = "overrides"
LOGGER = configure_logger("priorstock.base")


class PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset(PriorStockOHLCV124GroupedDataset):
    """OHLCV-124 dataset wrapper that emits hard labels, returns, and soft UP targets."""

    def __init__(
        self,
        experiment_config,
        split_name: str,
        objective_config: SingleLogitBinaryObjectiveConfig,
    ) -> None:
        """Create the return-aware binary dataset wrapper."""

        super().__init__(experiment_config, split_name)
        self._objective_config = objective_config
        self._apply_config_date_resplit(split_name)
        self._binary_label_cache: list[int] | None = None
        self._apply_significant_return_filter(split_name)

    def _apply_config_date_resplit(self, split_name: str) -> None:
        """Rebuild the sample index for one split from all split CSVs using configured dates."""

        if not self._objective_config.should_resplit_by_config_dates:
            return
        market_artifact_root = get_market_artifact_root(self._experiment_config)
        sample_index_directory = (
            market_artifact_root / self._experiment_config.paths.sample_index_subdirectory
        )
        split_frame_parts = [
            pd.read_csv(sample_index_directory / f"{source_split_name}.csv")
            for source_split_name in ("train", "validation", "test")
        ]
        combined_sample_index_frame = pd.concat(split_frame_parts, ignore_index=True)
        date_ranges_by_split_name = {
            "train": (
                self._experiment_config.data.train_start_date,
                self._experiment_config.data.train_end_date,
            ),
            "validation": (
                self._experiment_config.data.validation_start_date,
                self._experiment_config.data.validation_end_date,
            ),
            "test": (
                self._experiment_config.data.test_start_date,
                self._experiment_config.data.test_end_date,
            ),
        }
        if split_name not in date_ranges_by_split_name:
            raise ConfigurationError(f"Unsupported split name for date resplit: {split_name}.")
        start_date, end_date = date_ranges_by_split_name[split_name]
        target_trade_dates = pd.to_datetime(combined_sample_index_frame["target_trade_date"])
        filtered_sample_index_frame = combined_sample_index_frame.loc[
            (target_trade_dates >= pd.Timestamp(start_date))
            & (target_trade_dates <= pd.Timestamp(end_date))
        ].copy()
        if filtered_sample_index_frame.empty:
            raise ConfigurationError(
                f"Date-resplit split '{split_name}' has no samples between {start_date} and {end_date}."
            )
        filtered_sample_index_frame["split_name"] = split_name
        filtered_sample_index_frame = filtered_sample_index_frame.reset_index(drop=True)
        LOGGER.info(
            "Config-date resplit split=%s start=%s end=%s retained=%d source_total=%d",
            split_name,
            start_date,
            end_date,
            int(filtered_sample_index_frame.shape[0]),
            int(combined_sample_index_frame.shape[0]),
        )
        self._sample_index_frame = filtered_sample_index_frame

    def _compute_target_return_for_row(self, sample_row: pd.Series) -> float:
        """Compute one target return from previous and target effective close."""

        if "computed_target_return" in sample_row.index and not pd.isna(
            sample_row["computed_target_return"]
        ):
            return float(sample_row["computed_target_return"])
        stock_id = str(sample_row["stock_id"])
        price_frame = self._load_price_frame(stock_id)
        target_row_index = int(sample_row["target_row_index"])
        previous_effective_close = float(price_frame.iloc[target_row_index - 1]["effective_close"])
        target_effective_close = float(price_frame.iloc[target_row_index]["effective_close"])
        return (target_effective_close / previous_effective_close) - 1.0

    def _apply_significant_return_filter(self, split_name: str) -> None:
        """Keep only samples whose absolute one-step return exceeds the configured threshold."""

        threshold = self._objective_config.significant_return_absolute_threshold
        if threshold <= 0.0:
            return
        target_returns = [
            self._compute_target_return_for_row(self._sample_index_frame.iloc[row_index])
            for row_index in range(len(self._sample_index_frame))
        ]
        filtered_sample_index_frame = self._sample_index_frame.assign(
            computed_target_return=target_returns
        )
        filtered_sample_index_frame = filtered_sample_index_frame.loc[
            filtered_sample_index_frame["computed_target_return"].abs() > threshold
        ].reset_index(drop=True)
        if filtered_sample_index_frame.empty:
            raise ConfigurationError(
                f"Split '{split_name}' has no samples with |return| > {threshold:.6f}."
            )
        LOGGER.info(
            "Significant-only filter split=%s threshold=%.6f retained=%d original=%d retention_ratio=%.6f",
            split_name,
            threshold,
            int(filtered_sample_index_frame.shape[0]),
            int(len(target_returns)),
            float(filtered_sample_index_frame.shape[0] / max(len(target_returns), 1)),
        )
        self._sample_index_frame = filtered_sample_index_frame

    def _compute_bce_up_target(self, target_return: float, hard_label: int) -> float:
        """Compute either the return-aware soft target or the hard binary sign target."""

        if not self._objective_config.should_use_return_soft_targets:
            return float(hard_label)
        return 0.5 + (
            0.5
            * math.tanh(
                target_return / self._objective_config.return_soft_label_temperature
            )
        )

    def get_label_values(self) -> list[int]:
        """Return all recomputed hard binary labels for this split."""

        if self._binary_label_cache is None:
            self._binary_label_cache = [
                1
                if self._compute_target_return_for_row(self._sample_index_frame.iloc[row_index])
                > 0.0
                else 0
                for row_index in range(len(self._sample_index_frame))
            ]
        return list(self._binary_label_cache)

    def __getitem__(self, sample_index: int) -> dict[str, torch.Tensor | str]:
        """Load one sample and replace stale labels with return-aware binary fields."""

        sample = super().__getitem__(sample_index)
        target_return = self._compute_target_return_for_row(
            self._sample_index_frame.iloc[sample_index]
        )
        hard_label = 1 if target_return > 0.0 else 0
        soft_up_target = self._compute_bce_up_target(target_return, hard_label)
        sample["label"] = torch.tensor(hard_label, dtype=torch.long)
        sample["target_return"] = torch.tensor(target_return, dtype=torch.float32)
        sample["soft_up_target"] = torch.tensor(soft_up_target, dtype=torch.float32)
        return sample


def _coerce_float(raw_value: object, field_name: str) -> float:
    """Validate one floating-point script-level configuration field."""

    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a float.")
    return float(raw_value)


def _coerce_int(raw_value: object, field_name: str) -> int:
    """Validate one integer script-level configuration field."""

    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ConfigurationError(f"{field_name} must be an integer.")
    return int(raw_value)


def _coerce_bool(raw_value: object, field_name: str) -> bool:
    """Validate one boolean script-level configuration field."""

    if not isinstance(raw_value, bool):
        raise ConfigurationError(f"{field_name} must be a boolean.")
    return raw_value


def _coerce_float_tuple(raw_value: object, field_name: str) -> tuple[float, ...]:
    """Validate one sequence of floating-point values."""

    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigurationError(f"{field_name} must be a non-empty list.")
    return tuple(_coerce_float(item, f"{field_name}[]") for item in raw_value)


def _parse_attention_side_adapter_config(raw_section: object) -> AttentionSideAdapterConfig:
    """Parse and validate the attention-side adapter configuration."""

    if not isinstance(raw_section, dict):
        raise ConfigurationError("attention_side_adapter must be a YAML mapping.")
    required_field_names = {
        "adapter_rank",
        "adapter_dropout_probability",
        "alpha_initial_value",
        "condition_projection_initialization_std",
        "up_projection_initialization_std",
        "scale_control_target_ratio",
        "scale_control_max_scale",
        "scale_control_epsilon",
    }
    optional_field_names = {"should_replace_attention_update_with_adapter_delta"}
    raw_field_names = set(raw_section.keys())
    missing_field_names = sorted(required_field_names - raw_field_names)
    unexpected_field_names = sorted(
        raw_field_names - required_field_names - optional_field_names
    )
    if missing_field_names:
        raise ConfigurationError(
            "attention_side_adapter is missing required fields: "
            + ", ".join(missing_field_names)
            + "."
        )
    if unexpected_field_names:
        raise ConfigurationError(
            "attention_side_adapter contains unexpected fields: "
            + ", ".join(unexpected_field_names)
            + "."
        )
    adapter_config = AttentionSideAdapterConfig(
        adapter_rank=_coerce_int(
            raw_section["adapter_rank"],
            "attention_side_adapter.adapter_rank",
        ),
        adapter_dropout_probability=_coerce_float(
            raw_section["adapter_dropout_probability"],
            "attention_side_adapter.adapter_dropout_probability",
        ),
        alpha_initial_value=_coerce_float(
            raw_section["alpha_initial_value"],
            "attention_side_adapter.alpha_initial_value",
        ),
        should_replace_attention_update_with_adapter_delta=_coerce_bool(
            raw_section.get("should_replace_attention_update_with_adapter_delta", False),
            "attention_side_adapter.should_replace_attention_update_with_adapter_delta",
        ),
        condition_projection_initialization_std=_coerce_float(
            raw_section["condition_projection_initialization_std"],
            "attention_side_adapter.condition_projection_initialization_std",
        ),
        up_projection_initialization_std=_coerce_float(
            raw_section["up_projection_initialization_std"],
            "attention_side_adapter.up_projection_initialization_std",
        ),
        scale_control_target_ratio=_coerce_float(
            raw_section["scale_control_target_ratio"],
            "attention_side_adapter.scale_control_target_ratio",
        ),
        scale_control_max_scale=_coerce_float(
            raw_section["scale_control_max_scale"],
            "attention_side_adapter.scale_control_max_scale",
        ),
        scale_control_epsilon=_coerce_float(
            raw_section["scale_control_epsilon"],
            "attention_side_adapter.scale_control_epsilon",
        ),
    )
    if adapter_config.adapter_rank <= 0:
        raise ConfigurationError("attention_side_adapter.adapter_rank must be positive.")
    if not 0.0 <= adapter_config.adapter_dropout_probability < 1.0:
        raise ConfigurationError(
            "attention_side_adapter.adapter_dropout_probability must be in [0, 1)."
        )
    if adapter_config.condition_projection_initialization_std <= 0.0:
        raise ConfigurationError(
            "attention_side_adapter.condition_projection_initialization_std must be positive."
        )
    if adapter_config.up_projection_initialization_std <= 0.0:
        raise ConfigurationError(
            "attention_side_adapter.up_projection_initialization_std must be positive."
        )
    if (
        adapter_config.should_replace_attention_update_with_adapter_delta
        and abs(adapter_config.alpha_initial_value - 1.0)
        > adapter_config.scale_control_epsilon
    ):
        raise ConfigurationError(
            "attention_side_adapter.alpha_initial_value must be 1.0 when "
            "should_replace_attention_update_with_adapter_delta is true."
        )
    if adapter_config.scale_control_target_ratio <= 0.0:
        raise ConfigurationError(
            "attention_side_adapter.scale_control_target_ratio must be positive."
        )
    if adapter_config.scale_control_max_scale <= 0.0:
        raise ConfigurationError(
            "attention_side_adapter.scale_control_max_scale must be positive."
        )
    if adapter_config.scale_control_epsilon <= 0.0:
        raise ConfigurationError("attention_side_adapter.scale_control_epsilon must be positive.")
    return adapter_config


def _parse_single_logit_objective_config(raw_section: object) -> SingleLogitBinaryObjectiveConfig:
    """Parse and validate the single_logit_binary_classification YAML section."""

    if not isinstance(raw_section, dict):
        raise ConfigurationError("single_logit_binary_classification must be a YAML mapping.")
    expected_field_names = {
        "should_use_return_soft_targets",
        "should_resplit_by_config_dates",
        "should_evaluate_unfiltered_test_split",
        "return_soft_label_temperature",
        "significant_return_absolute_threshold",
        "adapter_learning_rate_multiplier",
        "adapter_freeze_epoch_count",
        "checkpoint_selection_mcc_weight",
        "checkpoint_selection_balanced_accuracy_weight",
        "checkpoint_selection_macro_f1_weight",
        "degenerate_high_prediction_up_ratio",
        "degenerate_low_prediction_up_ratio",
        "degenerate_recall_gap_threshold",
        "near_boundary_probability_margin",
        "calibration_bin_count",
        "return_bucket_absolute_thresholds",
        "large_return_absolute_thresholds",
    }
    optional_field_names = {
        "checkpoint_selection_accuracy_weight",
        "checkpoint_selection_split_name",
        "checkpoint_selection_epoch_limit",
        "should_evaluate_full_splits_each_epoch",
        "omitted_evaluation_iterator_count",
    }
    raw_field_names = set(raw_section.keys())
    missing_field_names = sorted(expected_field_names - raw_field_names)
    unexpected_field_names = sorted(raw_field_names - expected_field_names - optional_field_names)
    if missing_field_names:
        raise ConfigurationError(
            "single_logit_binary_classification is missing required fields: "
            + ", ".join(missing_field_names)
            + "."
        )
    if unexpected_field_names:
        raise ConfigurationError(
            "single_logit_binary_classification contains unexpected fields: "
            + ", ".join(unexpected_field_names)
            + "."
        )
    objective_config = SingleLogitBinaryObjectiveConfig(
        should_use_return_soft_targets=_coerce_bool(
            raw_section["should_use_return_soft_targets"],
            "single_logit_binary_classification.should_use_return_soft_targets",
        ),
        should_resplit_by_config_dates=_coerce_bool(
            raw_section["should_resplit_by_config_dates"],
            "single_logit_binary_classification.should_resplit_by_config_dates",
        ),
        should_evaluate_unfiltered_test_split=_coerce_bool(
            raw_section["should_evaluate_unfiltered_test_split"],
            "single_logit_binary_classification.should_evaluate_unfiltered_test_split",
        ),
        return_soft_label_temperature=_coerce_float(
            raw_section["return_soft_label_temperature"],
            "single_logit_binary_classification.return_soft_label_temperature",
        ),
        significant_return_absolute_threshold=_coerce_float(
            raw_section["significant_return_absolute_threshold"],
            "single_logit_binary_classification.significant_return_absolute_threshold",
        ),
        adapter_learning_rate_multiplier=_coerce_float(
            raw_section["adapter_learning_rate_multiplier"],
            "single_logit_binary_classification.adapter_learning_rate_multiplier",
        ),
        adapter_freeze_epoch_count=_coerce_int(
            raw_section["adapter_freeze_epoch_count"],
            "single_logit_binary_classification.adapter_freeze_epoch_count",
        ),
        checkpoint_selection_mcc_weight=_coerce_float(
            raw_section["checkpoint_selection_mcc_weight"],
            "single_logit_binary_classification.checkpoint_selection_mcc_weight",
        ),
        checkpoint_selection_balanced_accuracy_weight=_coerce_float(
            raw_section["checkpoint_selection_balanced_accuracy_weight"],
            "single_logit_binary_classification.checkpoint_selection_balanced_accuracy_weight",
        ),
        checkpoint_selection_macro_f1_weight=_coerce_float(
            raw_section["checkpoint_selection_macro_f1_weight"],
            "single_logit_binary_classification.checkpoint_selection_macro_f1_weight",
        ),
        degenerate_high_prediction_up_ratio=_coerce_float(
            raw_section["degenerate_high_prediction_up_ratio"],
            "single_logit_binary_classification.degenerate_high_prediction_up_ratio",
        ),
        degenerate_low_prediction_up_ratio=_coerce_float(
            raw_section["degenerate_low_prediction_up_ratio"],
            "single_logit_binary_classification.degenerate_low_prediction_up_ratio",
        ),
        degenerate_recall_gap_threshold=_coerce_float(
            raw_section["degenerate_recall_gap_threshold"],
            "single_logit_binary_classification.degenerate_recall_gap_threshold",
        ),
        near_boundary_probability_margin=_coerce_float(
            raw_section["near_boundary_probability_margin"],
            "single_logit_binary_classification.near_boundary_probability_margin",
        ),
        calibration_bin_count=_coerce_int(
            raw_section["calibration_bin_count"],
            "single_logit_binary_classification.calibration_bin_count",
        ),
        return_bucket_absolute_thresholds=_coerce_float_tuple(
            raw_section["return_bucket_absolute_thresholds"],
            "single_logit_binary_classification.return_bucket_absolute_thresholds",
        ),
        large_return_absolute_thresholds=_coerce_float_tuple(
            raw_section["large_return_absolute_thresholds"],
            "single_logit_binary_classification.large_return_absolute_thresholds",
        ),
        checkpoint_selection_accuracy_weight=_coerce_float(
            raw_section.get("checkpoint_selection_accuracy_weight", 0.0),
            "single_logit_binary_classification.checkpoint_selection_accuracy_weight",
        ),
        checkpoint_selection_split_name=str(
            raw_section.get("checkpoint_selection_split_name", "validation")
        ),
        checkpoint_selection_epoch_limit=_coerce_int(
            raw_section.get("checkpoint_selection_epoch_limit", 0),
            "single_logit_binary_classification.checkpoint_selection_epoch_limit",
        ),
        should_evaluate_full_splits_each_epoch=_coerce_bool(
            raw_section.get("should_evaluate_full_splits_each_epoch", False),
            "single_logit_binary_classification.should_evaluate_full_splits_each_epoch",
        ),
        omitted_evaluation_iterator_count=int(
            raw_section.get("omitted_evaluation_iterator_count", 0)
        ),
    )
    if objective_config.return_soft_label_temperature <= 0.0:
        raise ConfigurationError(
            "single_logit_binary_classification.return_soft_label_temperature must be positive."
        )
    if objective_config.significant_return_absolute_threshold < 0.0:
        raise ConfigurationError(
            "single_logit_binary_classification.significant_return_absolute_threshold must be non-negative."
        )
    if objective_config.adapter_learning_rate_multiplier <= 0.0:
        raise ConfigurationError(
            "single_logit_binary_classification.adapter_learning_rate_multiplier must be positive."
        )
    if objective_config.adapter_freeze_epoch_count < 0:
        raise ConfigurationError(
            "single_logit_binary_classification.adapter_freeze_epoch_count must be non-negative."
        )
    if objective_config.calibration_bin_count <= 0:
        raise ConfigurationError(
            "single_logit_binary_classification.calibration_bin_count must be positive."
        )
    if objective_config.checkpoint_selection_accuracy_weight < 0.0:
        raise ConfigurationError(
            "single_logit_binary_classification.checkpoint_selection_accuracy_weight must be non-negative."
        )
    supported_selection_split_names = {"validation", "validation_full"}
    if objective_config.checkpoint_selection_split_name not in supported_selection_split_names:
        raise ConfigurationError(
            "single_logit_binary_classification.checkpoint_selection_split_name must be one of "
            + ", ".join(sorted(supported_selection_split_names))
            + "."
        )
    if objective_config.checkpoint_selection_epoch_limit < 0:
        raise ConfigurationError(
            "single_logit_binary_classification.checkpoint_selection_epoch_limit must be non-negative."
        )
    if not 0.0 < objective_config.near_boundary_probability_margin < 0.5:
        raise ConfigurationError(
            "single_logit_binary_classification.near_boundary_probability_margin must be in (0, 0.5)."
        )
    if not 0.0 < objective_config.degenerate_low_prediction_up_ratio < 1.0:
        raise ConfigurationError(
            "single_logit_binary_classification.degenerate_low_prediction_up_ratio must be in (0, 1)."
        )
    if not 0.0 < objective_config.degenerate_high_prediction_up_ratio < 1.0:
        raise ConfigurationError(
            "single_logit_binary_classification.degenerate_high_prediction_up_ratio must be in (0, 1)."
        )
    if (
        objective_config.degenerate_low_prediction_up_ratio
        >= objective_config.degenerate_high_prediction_up_ratio
    ):
        raise ConfigurationError(
            "single_logit_binary_classification.degenerate_low_prediction_up_ratio must be below the high threshold."
        )
    if objective_config.degenerate_recall_gap_threshold <= 0.0:
        raise ConfigurationError(
            "single_logit_binary_classification.degenerate_recall_gap_threshold must be positive."
        )
    return objective_config


def _deep_merge_mapping(base_mapping: dict[str, Any], override_mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge where override values replace base values."""

    merged_mapping = dict(base_mapping)
    for key, override_value in override_mapping.items():
        base_value = merged_mapping.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged_mapping[key] = _deep_merge_mapping(base_value, override_value)
        else:
            merged_mapping[key] = override_value
    return merged_mapping


def _read_yaml_mapping(config_file_path: Path) -> dict[str, Any]:
    """Read one YAML file and require its root to be a mapping."""

    try:
        with config_file_path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' was not found.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' is not valid YAML.") from error
    if not isinstance(raw_config, dict):
        raise ConfigurationError(f"Configuration file '{config_file_path}' must contain a YAML mapping.")
    return raw_config


def _load_merged_raw_config(config_file_path: Path) -> dict[str, Any]:
    """Load a config file, optionally merging it over a base_config_file."""

    raw_config = _read_yaml_mapping(config_file_path)
    if BASE_CONFIG_FILE_FIELD_NAME not in raw_config:
        return raw_config
    base_config_path = Path(str(raw_config[BASE_CONFIG_FILE_FIELD_NAME]))
    if not base_config_path.is_absolute():
        base_config_path = (PROJECT_ROOT / base_config_path).resolve()
    base_config = _load_merged_raw_config(base_config_path)
    overrides = raw_config.get(OVERRIDES_FIELD_NAME)
    if not isinstance(overrides, dict):
        raise ConfigurationError("A config with base_config_file must define mapping overrides.")
    merged_config = _deep_merge_mapping(base_config, overrides)
    for key, value in raw_config.items():
        if key not in {BASE_CONFIG_FILE_FIELD_NAME, OVERRIDES_FIELD_NAME}:
            merged_config[key] = value
    return merged_config


def _load_config_with_single_logit_extensions(
    config_file_path: Path,
    run_directory: Path,
) -> tuple[
    object,
    AttentionSideAdapterConfig,
    SingleLogitBinaryObjectiveConfig,
]:
    """Load ExperimentConfig while allowing adapter and single-logit script-local sections."""

    raw_config = _load_merged_raw_config(config_file_path)
    sanitized_config = dict(raw_config)
    if ATTENTION_SIDE_ADAPTER_SECTION_NAME not in sanitized_config:
        raise ConfigurationError(
            f"Configuration file '{config_file_path}' must define "
            f"{ATTENTION_SIDE_ADAPTER_SECTION_NAME}."
        )
    if SINGLE_LOGIT_SECTION_NAME not in sanitized_config:
        raise ConfigurationError(
            f"Configuration file '{config_file_path}' must define {SINGLE_LOGIT_SECTION_NAME}."
        )
    attention_side_adapter_config = _parse_attention_side_adapter_config(
        sanitized_config.pop(ATTENTION_SIDE_ADAPTER_SECTION_NAME)
    )
    single_logit_objective_config = _parse_single_logit_objective_config(
        sanitized_config.pop(SINGLE_LOGIT_SECTION_NAME)
    )
    sanitized_config.pop(BINARY_CLASSIFICATION_SECTION_NAME, None)
    if "two_stage_training" in sanitized_config:
        raise ConfigurationError("Single-logit attention-side adapter runner does not support two_stage_training.")

    run_directory.mkdir(parents=True, exist_ok=True)
    sanitized_config_file_path = run_directory / "experiment_config_without_script_extensions.yaml"
    with sanitized_config_file_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(sanitized_config, file_handle, sort_keys=False, allow_unicode=False)
    experiment_config = load_experiment_config(sanitized_config_file_path)
    return (
        experiment_config,
        attention_side_adapter_config,
        single_logit_objective_config,
    )


def _build_data_loader(
    experiment_config,
    objective_config: SingleLogitBinaryObjectiveConfig,
    split_name: str,
    should_shuffle: bool,
) -> DataLoader:
    """Build one single-logit return-aware OHLCV-124 grouped dataset loader."""

    effective_num_workers = experiment_config.training.num_data_loader_workers
    if os.name == "nt" and effective_num_workers > 0:
        effective_num_workers = 0
    dataset = PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset(
        experiment_config,
        split_name,
        objective_config,
    )
    return DataLoader(
        dataset,
        batch_size=experiment_config.training.batch_size,
        shuffle=should_shuffle,
        num_workers=effective_num_workers,
        pin_memory=experiment_config.training.pin_memory,
    )


def _build_model(
    experiment_config,
    attention_side_adapter_config: AttentionSideAdapterConfig,
):
    """Build the attention-side adapter model for single-logit binary classification."""

    return PriorStockV3OHLCV124GroupTokenMixerAttentionSideAdapter(
        experiment_config=experiment_config,
        attention_side_adapter_config=attention_side_adapter_config,
    )


def _write_single_logit_labeling_contract(
    experiment_config,
    objective_config: SingleLogitBinaryObjectiveConfig,
    run_directory: Path,
) -> None:
    """Persist an explicit record of adjusted-close labels and return-aware targets."""

    manifest_file_path = get_market_artifact_root(experiment_config) / "stock_manifest.csv"
    manifest_frame = pd.read_csv(manifest_file_path)
    first_processed_price_file_path = path_from_serialized_value(
        manifest_frame.iloc[0]["processed_price_file_path"]
    )
    first_processed_price_frame = pd.read_csv(first_processed_price_file_path, nrows=3)
    write_json_file(
        run_directory / "single_logit_labeling_contract.json",
        {
            "hard_label_rule": "UP=1 iff effective_close[target_row_index] > effective_close[target_row_index - 1], else DOWN=0.",
            "soft_target_rule": "p_up_target = 0.5 + 0.5 * tanh(return / tau).",
            "bce_target_rule": (
                "Use tanh return-aware soft targets when should_use_return_soft_targets=true; "
                "otherwise use hard binary sign targets."
            ),
            "return_definition": "effective_close[target_row_index] / effective_close[target_row_index - 1] - 1.",
            "should_use_return_soft_targets": objective_config.should_use_return_soft_targets,
            "should_resplit_by_config_dates": objective_config.should_resplit_by_config_dates,
            "should_evaluate_unfiltered_test_split": objective_config.should_evaluate_unfiltered_test_split,
            "should_evaluate_full_splits_each_epoch": objective_config.should_evaluate_full_splits_each_epoch,
            "omitted_evaluation_iterator_count": objective_config.omitted_evaluation_iterator_count,
            "return_soft_label_temperature": objective_config.return_soft_label_temperature,
            "significant_return_absolute_threshold": objective_config.significant_return_absolute_threshold,
            "raw_effective_close_column_name": experiment_config.data.effective_close_column_name,
            "processed_effective_close_column_name": "effective_close",
            "should_adjust_ohlc_with_effective_close": experiment_config.data.should_adjust_ohlc_with_effective_close,
            "sample_processed_price_file_path": str(first_processed_price_file_path),
            "sample_processed_price_columns": list(first_processed_price_frame.columns),
            "sample_effective_close_values": first_processed_price_frame[
                "effective_close"
            ].astype(float).tolist(),
        },
    )


def main() -> None:
    """Run single-logit training, evaluation, and full adapter diagnostics."""

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
    ) = _load_config_with_single_logit_extensions(parsed_arguments.config_file, run_directory)
    if experiment_config.model.framework_variant_name != EXPECTED_FRAMEWORK_VARIANT_NAME:
        raise ValueError(
            "Single-logit attention-side adapter runner expects model.framework_variant_name "
            f"to equal '{EXPECTED_FRAMEWORK_VARIANT_NAME}'."
        )
    if experiment_config.model.num_classes != 1:
        raise ValueError("Single-logit attention-side adapter runner requires model.num_classes to equal 1.")
    set_global_seed(experiment_config.experiment.random_seed)
    write_json_file(
        run_directory / "attention_side_adapter_config_snapshot.json",
        asdict(attention_side_adapter_config),
    )
    write_json_file(
        run_directory / "single_logit_objective_config_snapshot.json",
        asdict(objective_config),
    )
    _write_single_logit_labeling_contract(
        experiment_config,
        objective_config,
        run_directory,
    )

    wandb_run = None
    if experiment_config.logging.use_wandb:
        wandb_run_id_file_path = run_directory / "wandb_run_id.txt"
        wandb_config = asdict(experiment_config)
        wandb_config[ATTENTION_SIDE_ADAPTER_SECTION_NAME] = asdict(attention_side_adapter_config)
        wandb_config[SINGLE_LOGIT_SECTION_NAME] = asdict(objective_config)
        wandb_initialization_arguments = {
            "project": experiment_config.logging.wandb_project_name,
            "entity": experiment_config.logging.wandb_entity_name,
            "mode": experiment_config.logging.wandb_mode,
            "dir": str(run_directory),
            "config": wandb_config,
            "name": run_directory.name,
        }
        if wandb_run_id_file_path.exists():
            existing_wandb_run_id = wandb_run_id_file_path.read_text(encoding="utf-8").strip()
            if existing_wandb_run_id:
                wandb_initialization_arguments["id"] = existing_wandb_run_id
                wandb_initialization_arguments["resume"] = "allow"
        wandb_run = wandb.init(
            **wandb_initialization_arguments,
            settings=wandb.Settings(init_timeout=300),
        )
        if wandb_run is not None:
            wandb_run_id_file_path.write_text(wandb_run.id, encoding="utf-8")

    train_data_loader = _build_data_loader(
        experiment_config,
        objective_config,
        "train",
        should_shuffle=True,
    )
    validation_data_loader = _build_data_loader(
        experiment_config,
        objective_config,
        "validation",
        should_shuffle=False,
    )
    test_data_loader = _build_data_loader(
        experiment_config,
        objective_config,
        "test",
        should_shuffle=False,
    )
    full_objective_config = replace(
        objective_config,
        significant_return_absolute_threshold=0.0,
    )
    should_build_full_epoch_loaders = (
        objective_config.should_evaluate_full_splits_each_epoch
        or objective_config.checkpoint_selection_split_name == "validation_full"
    )
    validation_full_data_loader = None
    if should_build_full_epoch_loaders:
        validation_full_data_loader = _build_data_loader(
            experiment_config,
            full_objective_config,
            "validation",
            should_shuffle=False,
        )
    training_summary = fit_single_logit_binary_model(
        model=_build_model(
            experiment_config,
            attention_side_adapter_config,
        ),
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        experiment_config=experiment_config,
        objective_config=objective_config,
        run_directory=run_directory,
        wandb_run=wandb_run,
        should_finish_wandb_run=False,
        validation_full_data_loader=validation_full_data_loader,
        full_objective_config=full_objective_config,
    )
    checkpoint_file_path = run_directory / experiment_config.paths.checkpoint_subdirectory / "best_model.pt"
    validation_metrics = evaluate_single_logit_binary_model(
        model=_build_model(
            experiment_config,
            attention_side_adapter_config,
        ),
        checkpoint_file_path=checkpoint_file_path,
        data_loader=validation_data_loader,
        train_data_loader=train_data_loader,
        experiment_config=experiment_config,
        objective_config=objective_config,
        split_name="validation",
        run_directory=run_directory,
        wandb_run=wandb_run,
    )
    test_metrics = evaluate_single_logit_binary_model(
        model=_build_model(
            experiment_config,
            attention_side_adapter_config,
        ),
        checkpoint_file_path=checkpoint_file_path,
        data_loader=test_data_loader,
        train_data_loader=train_data_loader,
        experiment_config=experiment_config,
        objective_config=objective_config,
        split_name="test",
        run_directory=run_directory,
        wandb_run=wandb_run,
    )
    unfiltered_test_metrics = None
    if objective_config.should_evaluate_unfiltered_test_split:
        unfiltered_test_data_loader = _build_data_loader(
            experiment_config,
            full_objective_config,
            "test",
            should_shuffle=False,
        )
        unfiltered_test_metrics = evaluate_single_logit_binary_model(
            model=_build_model(
                experiment_config,
                attention_side_adapter_config,
            ),
            checkpoint_file_path=checkpoint_file_path,
            data_loader=unfiltered_test_data_loader,
            train_data_loader=train_data_loader,
            experiment_config=experiment_config,
            objective_config=full_objective_config,
            split_name="test_unfiltered",
            run_directory=run_directory,
            wandb_run=wandb_run,
        )
    validation_adapter_diagnostics = {}
    test_adapter_diagnostics = {}
    pipeline_summary = build_single_logit_pipeline_summary(
        config_file_path=parsed_arguments.config_file,
        run_directory=run_directory,
        checkpoint_file_path=checkpoint_file_path,
        attention_side_adapter_config=attention_side_adapter_config,
        objective_config=objective_config,
        training_summary=training_summary,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_adapter_diagnostics=validation_adapter_diagnostics,
        test_adapter_diagnostics=test_adapter_diagnostics,
    )
    if unfiltered_test_metrics is not None:
        pipeline_summary["test_unfiltered_metrics"] = unfiltered_test_metrics
    write_json_file(run_directory / "pipeline_summary.json", pipeline_summary)
    LOGGER.info("Full single-logit train/evaluate pipeline finished: %s", pipeline_summary)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
