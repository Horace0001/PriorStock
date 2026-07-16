"""Configuration loading for the lightweight technical-text quality probe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from priorstock.config import (
    EvaluationConfig,
    ExperimentMetadataConfig,
    LoggingConfig,
    _instantiate_dataclass,
)
from priorstock.exceptions import ConfigurationError
import yaml


@dataclass(frozen=True)
class ProbePathConfig:
    """Artifact-relative directories required by the text-quality probe."""

    sample_index_subdirectory: str
    tech_embedding_subdirectory: str
    checkpoint_subdirectory: str
    metrics_subdirectory: str


@dataclass(frozen=True)
class ProbeDataConfig:
    """Dataset lookup settings for the technical-text probe."""

    market_code: str
    train_split_file_name: str
    validation_split_file_name: str
    test_split_file_name: str
    stock_manifest_file_name: str
    stock_id_column_name: str
    sample_id_column_name: str
    label_column_name: str
    target_trade_date_column_name: str
    embedding_row_index_column_name: str
    included_label_values: list[int]
    binary_positive_label_values: list[int] | None


@dataclass(frozen=True)
class ProbeModelConfig:
    """MLP hyperparameters for the text-quality probe."""

    input_embedding_dim: int
    projection_dim: int
    hidden_dropout_probability: float
    num_classes: int


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Optimization settings for the text-quality probe."""

    batch_size: int
    num_data_loader_workers: int
    pin_memory: bool
    num_epochs: int
    learning_rate: float
    minimum_learning_rate: float
    adam_beta_one: float
    adam_beta_two: float
    adam_epsilon: float
    weight_decay: float
    label_smoothing: float
    use_class_weights: bool
    use_auxiliary_contrastive_loss: bool
    auxiliary_contrastive_loss_weight: float
    auxiliary_positive_class_index: int
    auxiliary_negative_class_index: int
    auxiliary_negative_similarity_margin: float
    gradient_clip_max_norm: float
    early_stopping_patience: int
    resume_from_latest_checkpoint: bool
    latest_checkpoint_file_name: str


@dataclass(frozen=True)
class ProbeExperimentConfig:
    """Top-level configuration object for the text-quality probe."""

    experiment: ExperimentMetadataConfig
    paths: ProbePathConfig
    data: ProbeDataConfig
    model: ProbeModelConfig
    training: ProbeTrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def _validate_probe_constraints(experiment_config: ProbeExperimentConfig) -> None:
    """Validate cross-field invariants for the probe configuration."""

    if experiment_config.model.input_embedding_dim <= 0:
        raise ConfigurationError("model.input_embedding_dim must be positive.")
    if experiment_config.model.projection_dim <= 0:
        raise ConfigurationError("model.projection_dim must be positive.")
    if experiment_config.model.num_classes <= 1:
        raise ConfigurationError("model.num_classes must be greater than 1.")
    if not experiment_config.data.included_label_values:
        raise ConfigurationError("data.included_label_values must not be empty.")
    if len(set(experiment_config.data.included_label_values)) != len(experiment_config.data.included_label_values):
        raise ConfigurationError("data.included_label_values must not contain duplicates.")
    if any(label_value < 0 for label_value in experiment_config.data.included_label_values):
        raise ConfigurationError("data.included_label_values must contain only non-negative class indices.")
    if experiment_config.data.binary_positive_label_values is not None:
        if experiment_config.model.num_classes != 2:
            raise ConfigurationError(
                "model.num_classes must be 2 when data.binary_positive_label_values is configured."
            )
        if not experiment_config.data.binary_positive_label_values:
            raise ConfigurationError("data.binary_positive_label_values must not be empty when provided.")
        if len(set(experiment_config.data.binary_positive_label_values)) != len(
            experiment_config.data.binary_positive_label_values
        ):
            raise ConfigurationError("data.binary_positive_label_values must not contain duplicates.")
        positive_label_value_set = set(experiment_config.data.binary_positive_label_values)
        included_label_value_set = set(experiment_config.data.included_label_values)
        if not positive_label_value_set.issubset(included_label_value_set):
            raise ConfigurationError(
                "data.binary_positive_label_values must be a subset of data.included_label_values."
            )
        if positive_label_value_set == included_label_value_set:
            raise ConfigurationError(
                "data.binary_positive_label_values cannot cover every included label; at least one negative label is required."
            )
    if not 0.0 <= experiment_config.model.hidden_dropout_probability < 1.0:
        raise ConfigurationError("model.hidden_dropout_probability must be in [0.0, 1.0).")
    if experiment_config.training.batch_size <= 0:
        raise ConfigurationError("training.batch_size must be positive.")
    if experiment_config.training.num_epochs <= 0:
        raise ConfigurationError("training.num_epochs must be positive.")
    if experiment_config.training.learning_rate <= 0.0:
        raise ConfigurationError("training.learning_rate must be positive.")
    if experiment_config.training.minimum_learning_rate <= 0.0:
        raise ConfigurationError("training.minimum_learning_rate must be positive.")
    if experiment_config.training.minimum_learning_rate > experiment_config.training.learning_rate:
        raise ConfigurationError("training.minimum_learning_rate cannot exceed training.learning_rate.")
    if not 0.0 <= experiment_config.training.label_smoothing < 1.0:
        raise ConfigurationError("training.label_smoothing must be in [0.0, 1.0).")
    if experiment_config.training.gradient_clip_max_norm <= 0.0:
        raise ConfigurationError("training.gradient_clip_max_norm must be positive.")
    if experiment_config.training.early_stopping_patience <= 0:
        raise ConfigurationError("training.early_stopping_patience must be positive.")
    if experiment_config.training.auxiliary_contrastive_loss_weight < 0.0:
        raise ConfigurationError("training.auxiliary_contrastive_loss_weight must be non-negative.")
    if experiment_config.training.auxiliary_positive_class_index < 0:
        raise ConfigurationError("training.auxiliary_positive_class_index must be non-negative.")
    if experiment_config.training.auxiliary_negative_class_index < 0:
        raise ConfigurationError("training.auxiliary_negative_class_index must be non-negative.")
    if experiment_config.training.auxiliary_positive_class_index >= experiment_config.model.num_classes:
        raise ConfigurationError("training.auxiliary_positive_class_index must be smaller than model.num_classes.")
    if experiment_config.training.auxiliary_negative_class_index >= experiment_config.model.num_classes:
        raise ConfigurationError("training.auxiliary_negative_class_index must be smaller than model.num_classes.")
    if experiment_config.training.auxiliary_positive_class_index == experiment_config.training.auxiliary_negative_class_index:
        raise ConfigurationError(
            "training.auxiliary_positive_class_index and training.auxiliary_negative_class_index must differ."
        )
    if experiment_config.evaluation.up_class_label_index >= experiment_config.model.num_classes:
        raise ConfigurationError("evaluation.up_class_label_index must be smaller than model.num_classes.")
    if experiment_config.logging.log_every_training_step_count <= 0:
        raise ConfigurationError("logging.log_every_training_step_count must be positive.")
    if experiment_config.logging.distribution_log_every_training_step_count <= 0:
        raise ConfigurationError("logging.distribution_log_every_training_step_count must be positive.")
    if experiment_config.logging.max_distribution_summary_sample_count <= 0:
        raise ConfigurationError("logging.max_distribution_summary_sample_count must be positive.")
    if experiment_config.logging.max_histogram_sample_count <= 0:
        raise ConfigurationError("logging.max_histogram_sample_count must be positive.")


def load_probe_experiment_config(config_file_path: Path) -> ProbeExperimentConfig:
    """Load and validate one YAML probe configuration file."""

    try:
        with config_file_path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' was not found.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' is not valid YAML.") from error

    experiment_config = _instantiate_dataclass(ProbeExperimentConfig, raw_config, "root")
    _validate_probe_constraints(experiment_config)
    return experiment_config


def build_probe_run_name(experiment_config: ProbeExperimentConfig, timestamp_value: datetime) -> str:
    """Build one deterministic run name for the probe experiment."""

    timestamp_fragment = timestamp_value.strftime("%Y%m%d_%H%M%S")
    return (
        f"{experiment_config.experiment.market_name}_"
        f"{experiment_config.experiment.architecture_tag}_"
        f"{timestamp_fragment}"
    )


def get_probe_market_artifact_root(experiment_config: ProbeExperimentConfig) -> Path:
    """Return the artifact root that stores the selected market's prepared inputs."""

    safe_market_name = experiment_config.data.market_code.lower().replace("-", "_")
    return Path(experiment_config.experiment.artifact_output_root) / safe_market_name


def get_probe_run_root(experiment_config: ProbeExperimentConfig) -> Path:
    """Return the base directory for probe runtime outputs."""

    return Path(experiment_config.experiment.run_output_root)
