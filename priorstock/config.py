"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from priorstock.exceptions import ConfigurationError


@dataclass(frozen=True)
class ExperimentMetadataConfig:
    """High-level experiment metadata and runtime placement."""

    experiment_name: str
    architecture_tag: str
    market_name: str
    run_output_root: str
    artifact_output_root: str
    random_seed: int
    device: str


@dataclass(frozen=True)
class PathConfig:
    """Relative paths used by the project pipeline."""

    dataset_root: str
    price_raw_subdirectory: str
    price_preprocessed_subdirectory: str
    news_raw_subdirectory: str
    news_preprocessed_subdirectory: str
    processed_price_subdirectory: str
    aligned_news_subdirectory: str
    sample_index_subdirectory: str
    tech_text_output_subdirectory: str
    tech_text_validation_subdirectory: str
    tech_embedding_subdirectory: str
    news_embedding_subdirectory: str
    checkpoint_subdirectory: str
    metrics_subdirectory: str


@dataclass(frozen=True)
class DataConfig:
    """Dataset parsing and split behavior."""

    market_code: str
    effective_close_column_name: str
    should_adjust_ohlc_with_effective_close: bool
    date_column_name: str
    open_column_name: str
    high_column_name: str
    low_column_name: str
    close_column_name: str
    volume_column_name: str
    us_news_separator: str
    cn_news_excel_sheet_name: str
    cn_name_mapping_strategy: str
    lookback_window_size: int
    label_threshold: float
    train_start_date: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: str
    split_on_target_date_only: bool
    trading_calendar_mode: str
    minimum_indicator_history_mode: str
    use_binary_up_vs_non_up_labeling: bool
    use_dynamic_volatility_aware_three_class_labeling: bool
    dynamic_label_minimum_absolute_return: float
    dynamic_label_volatility_scale: float
    dynamic_label_volatility_window: int


@dataclass(frozen=True)
class IndicatorConfig:
    """Technical-indicator hyperparameters."""

    moving_average_windows: list[int]
    macd_fast_span: int
    macd_slow_span: int
    macd_signal_span: int
    kdj_lookback_period: int
    kdj_initial_k: float
    kdj_initial_d: float
    kdj_smoothing_alpha: float
    rsi_windows: list[int]
    bollinger_window: int
    bollinger_standard_deviation_multiplier: float
    volume_ratio_window: int
    epsilon: float


@dataclass(frozen=True)
class TextGenerationConfig:
    """Third-party API generation and validation configuration."""

    system_prompt_path: str
    user_prompt_path: str
    api_base_url: str
    api_model_name: str
    api_key_environment_variable: str
    requests_per_second: int
    max_worker_count: int
    request_timeout_seconds: int
    max_retry_attempt_count: int
    initial_retry_delay_seconds: float
    retry_backoff_multiplier: float
    retryable_status_codes: list[int]
    max_consecutive_failure_count: int
    generation_checkpoint_file_name: str
    generation_failure_log_file_name: str
    sample_validation_size: int
    save_every_response_count: int
    minimum_output_word_count: int
    maximum_output_word_count: int
    banned_phrases: list[str]
    validation_random_seed: int
    bandwidth_validation_tolerance: float
    percent_b_validation_tolerance: float
    require_zero_digit_output: bool
    forbid_number_word_output: bool
    require_single_paragraph_output: bool
    require_local_indicator_consistency_check: bool
    enable_word_count_check: bool
    require_all_samples_to_pass: bool
    enable_full_generation_after_validation: bool


@dataclass(frozen=True)
class TextEncoderConfig:
    """Frozen text encoder or embedding-API configuration."""

    backend_type: str
    pretrained_model_name: Optional[str]
    api_base_url: Optional[str]
    api_model_name: Optional[str]
    api_key_environment_variable: Optional[str]
    tokenizer_max_length: int
    encoder_batch_size: int
    cache_device: str
    local_model_cache_directory: str
    output_embedding_dim: int
    pooling_strategy: str
    normalize_embeddings: bool
    model_dtype: str
    instruction_prefix: str
    requests_per_second: int
    request_timeout_seconds: int
    max_retry_attempt_count: int
    initial_retry_delay_seconds: float
    retry_backoff_multiplier: float
    retryable_status_codes: list[int]
    max_consecutive_failure_count: int
    save_every_record_count: int
    checkpoint_file_name: str
    failure_log_file_name: str
    quota_exhaustion_error_codes: list[str]


@dataclass(frozen=True)
class ModelConfig:
    """PriorStock model hyperparameters."""

    framework_variant_name: str
    input_price_feature_count: int
    technical_indicator_feature_count: int
    technical_indicator_representation_dim: int
    technical_indicator_mlp_hidden_dim: int
    use_group_specific_technical_indicator_mlp_hidden_dims: bool
    technical_indicator_group_mlp_hidden_dims: list[int]
    technical_indicator_group_mlp_dropout_probability: float
    use_controlled_technical_indicator_global_mixer: bool
    technical_indicator_global_mixer_hidden_dim: int
    technical_indicator_global_mixer_dropout_probability: float
    technical_indicator_global_mixer_residual_alpha_initial_value: float
    use_vector_technical_indicator_global_mixer_residual_alpha: bool
    technical_indicator_group_token_dim: int
    technical_indicator_group_token_mixer_layer_count: int
    technical_indicator_group_token_mixer_head_count: int
    technical_indicator_group_token_mixer_feed_forward_dim: int
    technical_indicator_group_token_mixer_dropout_probability: float
    use_technical_indicator_group_identity_embedding: bool
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    num_classes: int
    dropout_probability: float
    gate_hidden_dim: int
    use_raw_technical_indicator_branch: bool
    use_technical_indicator_sequence_as_main_modality: bool
    use_news_only_auxiliary_branch: bool
    use_technical_indicator_pre_layer_norm: bool
    tech_gate_bias_initial_value: float
    news_gate_bias_initial_value: float
    positional_embedding_initialization_std: float
    no_news_embedding_initialization_std: float
    use_post_injection_layer_norm: bool
    use_layer_specific_text_projections: bool
    trace_layer_specific_projected_text_features: bool
    disable_outer_no_news_placeholder_embedding: bool
    use_block_level_news_hard_mask: bool
    use_shared_text_bottlenecks: bool
    shared_text_bottleneck_dim: int
    use_pre_projection_input_dropout: bool
    input_dropout_probability: float


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, scheduler, and loader hyperparameters."""

    batch_size: int
    num_data_loader_workers: int
    pin_memory: bool
    num_epochs: int
    learning_rate: float
    minimum_learning_rate: float
    scheduler_cosine_t_max_epoch_count: int
    adam_beta_one: float
    adam_beta_two: float
    adam_epsilon: float
    weight_decay: float
    label_smoothing: float
    use_soft_macro_f1_loss: bool
    soft_macro_f1_loss_weight: float
    soft_macro_f1_epsilon: float
    tech_gate_regularization_weight: float
    news_gate_regularization_weight: float
    use_technical_indicator_auxiliary_contrastive_loss: bool
    technical_indicator_auxiliary_loss_weight: float
    technical_indicator_auxiliary_positive_class_index: int
    technical_indicator_auxiliary_negative_class_index: int
    technical_indicator_auxiliary_negative_similarity_margin: float
    resume_from_latest_checkpoint: bool
    latest_checkpoint_file_name: str
    gradient_clip_max_norm: float
    early_stopping_patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation metric settings."""

    primary_metric_name: str
    checkpoint_selection_uses_composite_score: bool
    checkpoint_selection_macro_f1_weight: float
    checkpoint_selection_mcc_weight: float
    up_class_label_index: int


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging and W&B controls."""

    use_wandb: bool
    wandb_project_name: str
    wandb_entity_name: Optional[str]
    wandb_mode: str
    log_every_training_step_count: int
    log_gradient_statistics: bool
    log_tensor_shapes: bool
    log_first_batch_tensor_statistics: bool
    log_activation_distributions: bool
    log_attention_distributions: bool
    log_gradient_distributions: bool
    log_parameter_value_distributions: bool
    log_histograms_to_wandb: bool
    distribution_log_every_training_step_count: int
    distribution_summary_subdirectory: str
    distribution_trace_file_name: str
    max_distribution_summary_sample_count: int
    max_histogram_sample_count: int
    tensor_statistics_output_file_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration object."""

    experiment: ExperimentMetadataConfig
    paths: PathConfig
    data: DataConfig
    indicator: IndicatorConfig
    text_generation: TextGenerationConfig
    technical_text_encoder: TextEncoderConfig
    news_encoder: TextEncoderConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def _coerce_scalar_value(expected_type: type[Any], raw_value: Any, dotted_name: str) -> Any:
    """Convert a scalar YAML value into the declared dataclass field type."""

    if expected_type is str:
        if isinstance(raw_value, str):
            return raw_value
        if isinstance(raw_value, date):
            return raw_value.isoformat()
        raise ConfigurationError(f"Field '{dotted_name}' must be a string.")

    if expected_type is int:
        if isinstance(raw_value, bool):
            raise ConfigurationError(f"Field '{dotted_name}' must be an integer, not a boolean.")
        if isinstance(raw_value, int):
            return raw_value
        raise ConfigurationError(f"Field '{dotted_name}' must be an integer.")

    if expected_type is float:
        if isinstance(raw_value, bool):
            raise ConfigurationError(f"Field '{dotted_name}' must be a float, not a boolean.")
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        raise ConfigurationError(f"Field '{dotted_name}' must be a float.")

    if expected_type is bool:
        if isinstance(raw_value, bool):
            return raw_value
        raise ConfigurationError(f"Field '{dotted_name}' must be a boolean.")

    if expected_type is Path:
        if isinstance(raw_value, str):
            return Path(raw_value)
        raise ConfigurationError(f"Field '{dotted_name}' must be a filesystem path string.")

    raise ConfigurationError(f"Unsupported scalar type '{expected_type}' for field '{dotted_name}'.")


def _coerce_value(expected_type: Any, raw_value: Any, dotted_name: str) -> Any:
    """Recursively validate and convert YAML content into the expected Python type."""

    origin = get_origin(expected_type)

    if origin in {Union, UnionType}:
        union_args = [item for item in get_args(expected_type) if item is not type(None)]
        if raw_value is None:
            return None
        if len(union_args) != 1:
            raise ConfigurationError(f"Field '{dotted_name}' uses an unsupported Union type.")
        return _coerce_value(union_args[0], raw_value, dotted_name)

    if origin is list:
        if not isinstance(raw_value, list):
            raise ConfigurationError(f"Field '{dotted_name}' must be a YAML list.")
        inner_type = get_args(expected_type)[0]
        return [_coerce_value(inner_type, item, f"{dotted_name}[{index}]") for index, item in enumerate(raw_value)]

    if is_dataclass(expected_type):
        return _instantiate_dataclass(expected_type, raw_value, dotted_name)

    return _coerce_scalar_value(expected_type, raw_value, dotted_name)


def _instantiate_dataclass(dataclass_type: type[Any], raw_section: Any, dotted_name: str) -> Any:
    """Instantiate one dataclass section from a validated YAML mapping."""

    if not isinstance(raw_section, dict):
        raise ConfigurationError(f"Section '{dotted_name}' must be a YAML mapping.")

    field_names = {field_definition.name for field_definition in fields(dataclass_type)}
    raw_names = set(raw_section.keys())

    missing_field_names = sorted(field_names - raw_names)
    unexpected_field_names = sorted(raw_names - field_names)

    if missing_field_names:
        raise ConfigurationError(
            f"Section '{dotted_name}' is missing required fields: {', '.join(missing_field_names)}."
        )
    if unexpected_field_names:
        raise ConfigurationError(
            f"Section '{dotted_name}' contains unexpected fields: {', '.join(unexpected_field_names)}."
        )

    type_hints = get_type_hints(dataclass_type)
    coerced_values: dict[str, Any] = {}
    for field_definition in fields(dataclass_type):
        coerced_values[field_definition.name] = _coerce_value(
            type_hints[field_definition.name],
            raw_section[field_definition.name],
            f"{dotted_name}.{field_definition.name}",
        )
    return dataclass_type(**coerced_values)


def _validate_cross_section_constraints(experiment_config: ExperimentConfig) -> None:
    """Validate cross-field invariants that cannot be expressed by simple type checks."""

    valid_encoder_backend_types = {"huggingface_transformers", "openai_compatible_embedding_api"}
    valid_pooling_strategies = {"cls", "mean", "last_token"}
    valid_model_dtypes = {"float32", "float16", "bfloat16"}

    if experiment_config.model.d_model % experiment_config.model.num_heads != 0:
        raise ConfigurationError("model.d_model must be divisible by model.num_heads.")

    if experiment_config.model.technical_indicator_feature_count <= 0:
        raise ConfigurationError("model.technical_indicator_feature_count must be positive.")

    if experiment_config.model.technical_indicator_representation_dim <= 0:
        raise ConfigurationError("model.technical_indicator_representation_dim must be positive.")

    if experiment_config.model.technical_indicator_mlp_hidden_dim <= 0:
        raise ConfigurationError("model.technical_indicator_mlp_hidden_dim must be positive.")

    if not experiment_config.model.technical_indicator_group_mlp_hidden_dims:
        raise ConfigurationError("model.technical_indicator_group_mlp_hidden_dims must not be empty.")

    if any(
        hidden_dim <= 0
        for hidden_dim in experiment_config.model.technical_indicator_group_mlp_hidden_dims
    ):
        raise ConfigurationError(
            "model.technical_indicator_group_mlp_hidden_dims must contain only positive integers."
        )

    if not 0.0 <= experiment_config.model.technical_indicator_group_mlp_dropout_probability < 1.0:
        raise ConfigurationError(
            "model.technical_indicator_group_mlp_dropout_probability must be in the range [0.0, 1.0)."
        )

    if experiment_config.model.technical_indicator_global_mixer_hidden_dim <= 0:
        raise ConfigurationError("model.technical_indicator_global_mixer_hidden_dim must be positive.")

    if not 0.0 <= experiment_config.model.technical_indicator_global_mixer_dropout_probability < 1.0:
        raise ConfigurationError(
            "model.technical_indicator_global_mixer_dropout_probability must be in the range [0.0, 1.0)."
        )

    if experiment_config.model.technical_indicator_global_mixer_residual_alpha_initial_value < 0.0:
        raise ConfigurationError(
            "model.technical_indicator_global_mixer_residual_alpha_initial_value must be non-negative."
        )

    if experiment_config.model.technical_indicator_group_token_dim <= 0:
        raise ConfigurationError("model.technical_indicator_group_token_dim must be positive.")

    if experiment_config.model.technical_indicator_group_token_mixer_layer_count <= 0:
        raise ConfigurationError(
            "model.technical_indicator_group_token_mixer_layer_count must be positive."
        )

    if experiment_config.model.technical_indicator_group_token_mixer_head_count <= 0:
        raise ConfigurationError(
            "model.technical_indicator_group_token_mixer_head_count must be positive."
        )

    if (
        experiment_config.model.technical_indicator_group_token_dim
        % experiment_config.model.technical_indicator_group_token_mixer_head_count
        != 0
    ):
        raise ConfigurationError(
            "model.technical_indicator_group_token_dim must be divisible by "
            "model.technical_indicator_group_token_mixer_head_count."
        )

    if experiment_config.model.technical_indicator_group_token_mixer_feed_forward_dim <= 0:
        raise ConfigurationError(
            "model.technical_indicator_group_token_mixer_feed_forward_dim must be positive."
        )

    if not 0.0 <= experiment_config.model.technical_indicator_group_token_mixer_dropout_probability < 1.0:
        raise ConfigurationError(
            "model.technical_indicator_group_token_mixer_dropout_probability must be in the range [0.0, 1.0)."
        )

    if experiment_config.model.gate_hidden_dim <= 0:
        raise ConfigurationError("model.gate_hidden_dim must be positive.")

    if experiment_config.model.shared_text_bottleneck_dim <= 0:
        raise ConfigurationError("model.shared_text_bottleneck_dim must be positive.")

    if not 0.0 <= experiment_config.model.input_dropout_probability < 1.0:
        raise ConfigurationError("model.input_dropout_probability must be in the range [0.0, 1.0).")

    if not experiment_config.model.framework_variant_name.strip():
        raise ConfigurationError("model.framework_variant_name must not be empty.")

    if experiment_config.text_generation.requests_per_second <= 0:
        raise ConfigurationError("text_generation.requests_per_second must be positive.")

    if experiment_config.text_generation.max_worker_count <= 0:
        raise ConfigurationError("text_generation.max_worker_count must be positive.")

    if experiment_config.text_generation.request_timeout_seconds <= 0:
        raise ConfigurationError("text_generation.request_timeout_seconds must be positive.")

    if experiment_config.text_generation.max_retry_attempt_count <= 0:
        raise ConfigurationError("text_generation.max_retry_attempt_count must be positive.")

    if experiment_config.text_generation.initial_retry_delay_seconds <= 0:
        raise ConfigurationError("text_generation.initial_retry_delay_seconds must be positive.")

    if experiment_config.text_generation.retry_backoff_multiplier < 1.0:
        raise ConfigurationError("text_generation.retry_backoff_multiplier must be at least 1.0.")

    if not experiment_config.text_generation.retryable_status_codes:
        raise ConfigurationError("text_generation.retryable_status_codes must not be empty.")

    if any(status_code < 400 or status_code > 599 for status_code in experiment_config.text_generation.retryable_status_codes):
        raise ConfigurationError("text_generation.retryable_status_codes must contain only HTTP error status codes.")

    if experiment_config.text_generation.max_consecutive_failure_count <= 0:
        raise ConfigurationError("text_generation.max_consecutive_failure_count must be positive.")

    if experiment_config.text_generation.sample_validation_size <= 0:
        raise ConfigurationError("text_generation.sample_validation_size must be positive.")

    if experiment_config.text_generation.save_every_response_count <= 0:
        raise ConfigurationError("text_generation.save_every_response_count must be positive.")

    if experiment_config.training.batch_size <= 0:
        raise ConfigurationError("training.batch_size must be positive.")

    if experiment_config.training.num_epochs <= 0:
        raise ConfigurationError("training.num_epochs must be positive.")

    if experiment_config.training.scheduler_cosine_t_max_epoch_count <= 0:
        raise ConfigurationError("training.scheduler_cosine_t_max_epoch_count must be positive.")

    if experiment_config.training.gradient_clip_max_norm <= 0:
        raise ConfigurationError("training.gradient_clip_max_norm must be positive.")

    if experiment_config.training.minimum_learning_rate > experiment_config.training.learning_rate:
        raise ConfigurationError("training.minimum_learning_rate cannot exceed training.learning_rate.")

    if experiment_config.training.tech_gate_regularization_weight < 0.0:
        raise ConfigurationError("training.tech_gate_regularization_weight must be non-negative.")

    if experiment_config.training.news_gate_regularization_weight < 0.0:
        raise ConfigurationError("training.news_gate_regularization_weight must be non-negative.")

    if experiment_config.training.soft_macro_f1_loss_weight < 0.0:
        raise ConfigurationError("training.soft_macro_f1_loss_weight must be non-negative.")

    if experiment_config.training.soft_macro_f1_epsilon <= 0.0:
        raise ConfigurationError("training.soft_macro_f1_epsilon must be positive.")

    if (
        not experiment_config.training.use_soft_macro_f1_loss
        and experiment_config.training.soft_macro_f1_loss_weight != 0.0
    ):
        raise ConfigurationError(
            "training.soft_macro_f1_loss_weight must be 0.0 when "
            "training.use_soft_macro_f1_loss is false."
        )

    if experiment_config.training.technical_indicator_auxiliary_loss_weight < 0.0:
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_loss_weight must be non-negative."
        )

    if experiment_config.evaluation.checkpoint_selection_macro_f1_weight < 0.0:
        raise ConfigurationError(
            "evaluation.checkpoint_selection_macro_f1_weight must be non-negative."
        )

    if experiment_config.evaluation.checkpoint_selection_mcc_weight < 0.0:
        raise ConfigurationError(
            "evaluation.checkpoint_selection_mcc_weight must be non-negative."
        )

    if (
        experiment_config.evaluation.checkpoint_selection_macro_f1_weight
        + experiment_config.evaluation.checkpoint_selection_mcc_weight
        <= 0.0
    ):
        raise ConfigurationError(
            "At least one checkpoint selection weight must be positive."
        )

    if experiment_config.training.technical_indicator_auxiliary_positive_class_index < 0:
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_positive_class_index must be non-negative."
        )

    if experiment_config.training.technical_indicator_auxiliary_negative_class_index < 0:
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_negative_class_index must be non-negative."
        )

    if (
        experiment_config.training.use_technical_indicator_auxiliary_contrastive_loss
        and experiment_config.training.technical_indicator_auxiliary_positive_class_index
        == experiment_config.training.technical_indicator_auxiliary_negative_class_index
    ):
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_positive_class_index and "
            "training.technical_indicator_auxiliary_negative_class_index must be different."
        )

    if (
        experiment_config.training.technical_indicator_auxiliary_positive_class_index
        >= experiment_config.model.num_classes
    ):
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_positive_class_index must be smaller than model.num_classes."
        )

    if (
        experiment_config.training.technical_indicator_auxiliary_negative_class_index
        >= experiment_config.model.num_classes
    ):
        raise ConfigurationError(
            "training.technical_indicator_auxiliary_negative_class_index must be smaller than model.num_classes."
        )

    if experiment_config.data.lookback_window_size <= 0:
        raise ConfigurationError("data.lookback_window_size must be positive.")

    if experiment_config.data.dynamic_label_minimum_absolute_return < 0.0:
        raise ConfigurationError("data.dynamic_label_minimum_absolute_return must be non-negative.")

    if experiment_config.data.dynamic_label_volatility_scale < 0.0:
        raise ConfigurationError("data.dynamic_label_volatility_scale must be non-negative.")

    if experiment_config.data.dynamic_label_volatility_window <= 0:
        raise ConfigurationError("data.dynamic_label_volatility_window must be positive.")

    if experiment_config.data.use_binary_up_vs_non_up_labeling:
        if experiment_config.model.num_classes not in {1, 2}:
            raise ConfigurationError(
                "model.num_classes must be 1 or 2 when binary labeling is enabled."
            )
        if experiment_config.evaluation.up_class_label_index != 1:
            raise ConfigurationError("evaluation.up_class_label_index must be 1 when binary labeling is enabled.")
    else:
        if experiment_config.model.num_classes != 3:
            raise ConfigurationError("model.num_classes must be 3 when binary labeling is disabled.")
        if experiment_config.evaluation.up_class_label_index != 2:
            raise ConfigurationError("evaluation.up_class_label_index must be 2 when binary labeling is disabled.")

    for encoder_name, encoder_config in [
        ("technical_text_encoder", experiment_config.technical_text_encoder),
        ("news_encoder", experiment_config.news_encoder),
    ]:
        if encoder_config.backend_type not in valid_encoder_backend_types:
            raise ConfigurationError(
                f"{encoder_name}.backend_type must be one of: {', '.join(sorted(valid_encoder_backend_types))}."
            )

        if encoder_config.pooling_strategy not in valid_pooling_strategies:
            raise ConfigurationError(
                f"{encoder_name}.pooling_strategy must be one of: {', '.join(sorted(valid_pooling_strategies))}."
            )

        if encoder_config.model_dtype not in valid_model_dtypes:
            raise ConfigurationError(
                f"{encoder_name}.model_dtype must be one of: {', '.join(sorted(valid_model_dtypes))}."
            )

        if encoder_config.output_embedding_dim <= 0:
            raise ConfigurationError(f"{encoder_name}.output_embedding_dim must be positive.")

        if encoder_config.encoder_batch_size <= 0:
            raise ConfigurationError(f"{encoder_name}.encoder_batch_size must be positive.")

        if encoder_config.tokenizer_max_length <= 0:
            raise ConfigurationError(f"{encoder_name}.tokenizer_max_length must be positive.")

        if encoder_config.requests_per_second <= 0:
            raise ConfigurationError(f"{encoder_name}.requests_per_second must be positive.")

        if encoder_config.request_timeout_seconds <= 0:
            raise ConfigurationError(f"{encoder_name}.request_timeout_seconds must be positive.")

        if encoder_config.max_retry_attempt_count <= 0:
            raise ConfigurationError(f"{encoder_name}.max_retry_attempt_count must be positive.")

        if encoder_config.initial_retry_delay_seconds <= 0:
            raise ConfigurationError(f"{encoder_name}.initial_retry_delay_seconds must be positive.")

        if encoder_config.retry_backoff_multiplier < 1.0:
            raise ConfigurationError(f"{encoder_name}.retry_backoff_multiplier must be at least 1.0.")

        if not encoder_config.retryable_status_codes:
            raise ConfigurationError(f"{encoder_name}.retryable_status_codes must not be empty.")

        if any(status_code < 400 or status_code > 599 for status_code in encoder_config.retryable_status_codes):
            raise ConfigurationError(f"{encoder_name}.retryable_status_codes must contain only HTTP error status codes.")

        if encoder_config.max_consecutive_failure_count <= 0:
            raise ConfigurationError(f"{encoder_name}.max_consecutive_failure_count must be positive.")

        if encoder_config.save_every_record_count <= 0:
            raise ConfigurationError(f"{encoder_name}.save_every_record_count must be positive.")

        if not encoder_config.quota_exhaustion_error_codes:
            raise ConfigurationError(f"{encoder_name}.quota_exhaustion_error_codes must not be empty.")

        if encoder_config.backend_type == "huggingface_transformers" and not encoder_config.pretrained_model_name:
            raise ConfigurationError(f"{encoder_name}.pretrained_model_name is required for Hugging Face encoders.")

        if encoder_config.backend_type == "openai_compatible_embedding_api":
            for required_field_name in ["api_base_url", "api_model_name", "api_key_environment_variable"]:
                required_field_value = getattr(encoder_config, required_field_name)
                if not required_field_value:
                    raise ConfigurationError(
                        f"{encoder_name}.{required_field_name} is required for embedding-API encoders."
                    )


def load_experiment_config(config_file_path: Path) -> ExperimentConfig:
    """Load, validate, and materialize a YAML experiment configuration."""

    try:
        with config_file_path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' was not found.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Configuration file '{config_file_path}' is not valid YAML.") from error

    experiment_config = _instantiate_dataclass(ExperimentConfig, raw_config, "root")
    _validate_cross_section_constraints(experiment_config)
    return experiment_config


def build_run_name(experiment_config: ExperimentConfig, timestamp_value: datetime) -> str:
    """Build a deterministic run name from config metadata and a timestamp."""

    timestamp_fragment = timestamp_value.strftime("%Y%m%d_%H%M%S")
    return (
        f"{experiment_config.experiment.market_name}_"
        f"{experiment_config.experiment.architecture_tag}_"
        f"{timestamp_fragment}"
    )


def get_market_dataset_root(experiment_config: ExperimentConfig) -> Path:
    """Return the selected market root directory inside the raw dataset tree."""

    return Path(experiment_config.paths.dataset_root) / experiment_config.data.market_code


def get_market_artifact_root(experiment_config: ExperimentConfig) -> Path:
    """Return the root directory that stores all derived artifacts for the selected market."""

    safe_market_name = experiment_config.data.market_code.lower().replace("-", "_")
    return Path(experiment_config.experiment.artifact_output_root) / safe_market_name


def get_run_root(experiment_config: ExperimentConfig) -> Path:
    """Return the base output directory for runtime logs and checkpoints."""

    return Path(experiment_config.experiment.run_output_root)
