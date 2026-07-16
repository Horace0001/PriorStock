"""Configuration parsing for LLM-extracted news-factor experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from priorstock.exceptions import ConfigurationError


@dataclass(frozen=True)
class ApiEndpointConfig:
    """OpenAI-compatible API endpoint settings."""

    base_url: str
    api_key_environment_variable: str
    request_timeout_seconds: float
    retry_count: int
    retry_wait_seconds: float
    requests_per_second: float
    max_workers: int


@dataclass(frozen=True)
class ChatGenerationConfig:
    """Chat-completion factor extraction settings."""

    models: tuple[str, ...]
    disabled_models: tuple[str, ...]
    temperature: float
    top_p: float
    max_output_tokens: int
    max_output_tokens_by_model: dict[str, int]
    retry_max_output_tokens_by_model: dict[str, int]
    reasoning_effort_by_model: dict[str, str]
    prompt_version: str
    prompt_template: str


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding settings for extracted factors."""

    model_name: str
    embedding_dimension: int
    batch_size: int
    max_workers: int
    requests_per_second: float
    reduced_dimension: int


@dataclass(frozen=True)
class SamplingConfig:
    """Round-specific sample selection and filtering settings."""

    round_name: str
    split_sizes: dict[str, int]
    train_start_date: str
    train_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: str
    significant_return_absolute_threshold: float
    recent_news_lookback_trading_days: int
    exclude_target_date_news: bool
    max_samples_per_stock_per_split: int
    base_confidence_bucket_count: int
    news_volume_bucket_count: int
    random_seed: int


@dataclass(frozen=True)
class NewsTextConfig:
    """Controls for turning raw news rows into a bounded LLM prompt section."""

    raw_news_directory: Path
    max_news_items_per_sample: int
    max_news_items_per_day: int
    max_news_characters_per_item: int
    max_prompt_news_characters: int
    news_text_source_field: str


@dataclass(frozen=True)
class BaseModelConfig:
    """Frozen wide-adapter baseline used for stratification diagnostics."""

    config_file: Path
    checkpoint_file: Path
    batch_size: int


@dataclass(frozen=True)
class ProbeConfig:
    """MLP probe training settings."""

    seeds: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    dropout_probability: float
    hidden_dimension: int
    attention_hidden_dimension: int
    return_soft_label_temperature: float
    score_mcc_weight: float
    score_balanced_accuracy_weight: float


@dataclass(frozen=True)
class OutputConfig:
    """Output directories and flush cadence."""

    artifact_root: Path
    run_root: Path
    save_every_n_records: int


@dataclass(frozen=True)
class NewsFactorExperimentConfig:
    """Complete configuration for one LLM-factor experiment round."""

    api: ApiEndpointConfig
    chat: ChatGenerationConfig
    embedding: EmbeddingConfig
    sampling: SamplingConfig
    news_text: NewsTextConfig
    base_model: BaseModelConfig
    probe: ProbeConfig
    output: OutputConfig


def _require_mapping(raw_value: object, field_name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a configuration error."""

    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"{field_name} must be a YAML mapping.")
    return raw_value


def _require_string(raw_mapping: dict[str, Any], field_name: str) -> str:
    """Return a required string field from a mapping."""

    raw_value = raw_mapping.get(field_name)
    if not isinstance(raw_value, str) or not raw_value:
        raise ConfigurationError(f"{field_name} must be a non-empty string.")
    return raw_value


def _require_float(raw_mapping: dict[str, Any], field_name: str) -> float:
    """Return a required float field from a mapping."""

    raw_value = raw_mapping.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a numeric value.")
    return float(raw_value)


def _require_int(raw_mapping: dict[str, Any], field_name: str) -> int:
    """Return a required integer field from a mapping."""

    raw_value = raw_mapping.get(field_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ConfigurationError(f"{field_name} must be an integer.")
    return raw_value


def _require_bool(raw_mapping: dict[str, Any], field_name: str) -> bool:
    """Return a required boolean field from a mapping."""

    raw_value = raw_mapping.get(field_name)
    if not isinstance(raw_value, bool):
        raise ConfigurationError(f"{field_name} must be a boolean.")
    return raw_value


def _require_string_tuple(raw_mapping: dict[str, Any], field_name: str) -> tuple[str, ...]:
    """Return a required non-empty string list as a tuple."""

    raw_value = raw_mapping.get(field_name)
    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigurationError(f"{field_name} must be a non-empty list.")
    parsed_values = tuple(str(item) for item in raw_value if isinstance(item, str) and item)
    if len(parsed_values) != len(raw_value):
        raise ConfigurationError(f"{field_name} must contain only non-empty strings.")
    return parsed_values


def _optional_string_tuple(raw_mapping: dict[str, Any], field_name: str) -> tuple[str, ...]:
    """Return an optional string list as a tuple."""

    raw_value = raw_mapping.get(field_name, [])
    if not isinstance(raw_value, list):
        raise ConfigurationError(f"{field_name} must be a list when provided.")
    parsed_values = tuple(str(item) for item in raw_value if isinstance(item, str) and item)
    if len(parsed_values) != len(raw_value):
        raise ConfigurationError(f"{field_name} must contain only non-empty strings.")
    return parsed_values


def _require_int_dict(raw_mapping: dict[str, Any], field_name: str) -> dict[str, int]:
    """Return a required string-to-int mapping."""

    raw_value = raw_mapping.get(field_name)
    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"{field_name} must be a mapping.")
    parsed_values: dict[str, int] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{field_name} must map strings to integers.")
        parsed_values[key] = value
    return parsed_values


def _optional_string_dict(raw_mapping: dict[str, Any], field_name: str) -> dict[str, str]:
    """Return an optional string-to-string mapping."""

    raw_value = raw_mapping.get(field_name, {})
    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"{field_name} must be a mapping when provided.")
    parsed_values: dict[str, str] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise ConfigurationError(f"{field_name} must map strings to non-empty strings.")
        parsed_values[key] = value
    return parsed_values


def _optional_int_dict(raw_mapping: dict[str, Any], field_name: str) -> dict[str, int]:
    """Return an optional string-to-int mapping."""

    raw_value = raw_mapping.get(field_name, {})
    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"{field_name} must be a mapping when provided.")
    parsed_values: dict[str, int] = {}
    for key, value in raw_value.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{field_name} must map strings to integers.")
        parsed_values[key] = value
    return parsed_values


def _require_int_tuple(raw_mapping: dict[str, Any], field_name: str) -> tuple[int, ...]:
    """Return a required integer list as a tuple."""

    raw_value = raw_mapping.get(field_name)
    if not isinstance(raw_value, list) or not raw_value:
        raise ConfigurationError(f"{field_name} must be a non-empty list.")
    parsed_values: list[int] = []
    for item in raw_value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigurationError(f"{field_name} must contain only integers.")
        parsed_values.append(item)
    return tuple(parsed_values)


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    """Resolve one YAML path relative to the project root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_news_factor_experiment_config(
    config_file_path: Path,
    project_root: Path,
) -> NewsFactorExperimentConfig:
    """Load and validate a news-factor experiment YAML file."""

    with config_file_path.open("r", encoding="utf-8") as file_handle:
        raw_config = yaml.safe_load(file_handle)
    root_mapping = _require_mapping(raw_config, "root")
    api_mapping = _require_mapping(root_mapping.get("api"), "api")
    chat_mapping = _require_mapping(root_mapping.get("chat"), "chat")
    embedding_mapping = _require_mapping(root_mapping.get("embedding"), "embedding")
    sampling_mapping = _require_mapping(root_mapping.get("sampling"), "sampling")
    news_text_mapping = _require_mapping(root_mapping.get("news_text"), "news_text")
    base_model_mapping = _require_mapping(root_mapping.get("base_model"), "base_model")
    probe_mapping = _require_mapping(root_mapping.get("probe"), "probe")
    output_mapping = _require_mapping(root_mapping.get("output"), "output")

    return NewsFactorExperimentConfig(
        api=ApiEndpointConfig(
            base_url=_require_string(api_mapping, "base_url"),
            api_key_environment_variable=_require_string(
                api_mapping,
                "api_key_environment_variable",
            ),
            request_timeout_seconds=_require_float(api_mapping, "request_timeout_seconds"),
            retry_count=_require_int(api_mapping, "retry_count"),
            retry_wait_seconds=_require_float(api_mapping, "retry_wait_seconds"),
            requests_per_second=_require_float(api_mapping, "requests_per_second"),
            max_workers=_require_int(api_mapping, "max_workers"),
        ),
        chat=ChatGenerationConfig(
            models=_require_string_tuple(chat_mapping, "models"),
            disabled_models=_optional_string_tuple(chat_mapping, "disabled_models"),
            temperature=_require_float(chat_mapping, "temperature"),
            top_p=_require_float(chat_mapping, "top_p"),
            max_output_tokens=_require_int(chat_mapping, "max_output_tokens"),
            max_output_tokens_by_model=_optional_int_dict(
                chat_mapping,
                "max_output_tokens_by_model",
            ),
            retry_max_output_tokens_by_model=_optional_int_dict(
                chat_mapping,
                "retry_max_output_tokens_by_model",
            ),
            reasoning_effort_by_model=_optional_string_dict(
                chat_mapping,
                "reasoning_effort_by_model",
            ),
            prompt_version=_require_string(chat_mapping, "prompt_version"),
            prompt_template=_require_string(chat_mapping, "prompt_template"),
        ),
        embedding=EmbeddingConfig(
            model_name=_require_string(embedding_mapping, "model_name"),
            embedding_dimension=_require_int(embedding_mapping, "embedding_dimension"),
            batch_size=_require_int(embedding_mapping, "batch_size"),
            max_workers=_require_int(embedding_mapping, "max_workers"),
            requests_per_second=_require_float(embedding_mapping, "requests_per_second"),
            reduced_dimension=_require_int(embedding_mapping, "reduced_dimension"),
        ),
        sampling=SamplingConfig(
            round_name=_require_string(sampling_mapping, "round_name"),
            split_sizes=_require_int_dict(sampling_mapping, "split_sizes"),
            train_start_date=_require_string(sampling_mapping, "train_start_date"),
            train_end_date=_require_string(sampling_mapping, "train_end_date"),
            validation_start_date=_require_string(sampling_mapping, "validation_start_date"),
            validation_end_date=_require_string(sampling_mapping, "validation_end_date"),
            test_start_date=_require_string(sampling_mapping, "test_start_date"),
            test_end_date=_require_string(sampling_mapping, "test_end_date"),
            significant_return_absolute_threshold=_require_float(
                sampling_mapping,
                "significant_return_absolute_threshold",
            ),
            recent_news_lookback_trading_days=_require_int(
                sampling_mapping,
                "recent_news_lookback_trading_days",
            ),
            exclude_target_date_news=_require_bool(sampling_mapping, "exclude_target_date_news"),
            max_samples_per_stock_per_split=_require_int(
                sampling_mapping,
                "max_samples_per_stock_per_split",
            ),
            base_confidence_bucket_count=_require_int(
                sampling_mapping,
                "base_confidence_bucket_count",
            ),
            news_volume_bucket_count=_require_int(
                sampling_mapping,
                "news_volume_bucket_count",
            ),
            random_seed=_require_int(sampling_mapping, "random_seed"),
        ),
        news_text=NewsTextConfig(
            raw_news_directory=_resolve_path(
                _require_string(news_text_mapping, "raw_news_directory"),
                project_root,
            ),
            max_news_items_per_sample=_require_int(
                news_text_mapping,
                "max_news_items_per_sample",
            ),
            max_news_items_per_day=_require_int(news_text_mapping, "max_news_items_per_day"),
            max_news_characters_per_item=_require_int(
                news_text_mapping,
                "max_news_characters_per_item",
            ),
            max_prompt_news_characters=_require_int(
                news_text_mapping,
                "max_prompt_news_characters",
            ),
            news_text_source_field=_require_string(news_text_mapping, "news_text_source_field"),
        ),
        base_model=BaseModelConfig(
            config_file=_resolve_path(_require_string(base_model_mapping, "config_file"), project_root),
            checkpoint_file=_resolve_path(
                _require_string(base_model_mapping, "checkpoint_file"),
                project_root,
            ),
            batch_size=_require_int(base_model_mapping, "batch_size"),
        ),
        probe=ProbeConfig(
            seeds=_require_int_tuple(probe_mapping, "seeds"),
            learning_rate=_require_float(probe_mapping, "learning_rate"),
            weight_decay=_require_float(probe_mapping, "weight_decay"),
            batch_size=_require_int(probe_mapping, "batch_size"),
            max_epochs=_require_int(probe_mapping, "max_epochs"),
            early_stopping_patience=_require_int(probe_mapping, "early_stopping_patience"),
            dropout_probability=_require_float(probe_mapping, "dropout_probability"),
            hidden_dimension=_require_int(probe_mapping, "hidden_dimension"),
            attention_hidden_dimension=_require_int(probe_mapping, "attention_hidden_dimension"),
            return_soft_label_temperature=_require_float(
                probe_mapping,
                "return_soft_label_temperature",
            ),
            score_mcc_weight=_require_float(probe_mapping, "score_mcc_weight"),
            score_balanced_accuracy_weight=_require_float(
                probe_mapping,
                "score_balanced_accuracy_weight",
            ),
        ),
        output=OutputConfig(
            artifact_root=_resolve_path(_require_string(output_mapping, "artifact_root"), project_root),
            run_root=_resolve_path(_require_string(output_mapping, "run_root"), project_root),
            save_every_n_records=_require_int(output_mapping, "save_every_n_records"),
        ),
    )
