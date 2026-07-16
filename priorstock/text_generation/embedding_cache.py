"""Reusable embedding-cache builders for generated technical text and aligned daily news."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout
from transformers import AutoModel, AutoTokenizer

from priorstock.config import ExperimentConfig, TextEncoderConfig, get_market_artifact_root
from priorstock.exceptions import ArtifactConsistencyError, ExternalServiceError
from priorstock.utils.environment import load_project_local_environment_files
from priorstock.utils.io import (
    append_jsonl_records,
    ensure_directory,
    path_from_serialized_value,
    read_json_file,
    read_jsonl_records,
    write_json_file,
    write_jsonl_records,
)
from priorstock.utils.rate_limit import FixedIntervalRateLimiter


LOGGER = logging.getLogger(__name__)
ERROR_RESPONSE_TEXT_MAX_CHARACTERS = 1000
PARTIAL_EMBEDDING_ROOT_DIRECTORY_NAME = "tech_embedding_partials"


MODEL_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _sanitize_model_name_for_path(model_name: str) -> str:
    """Convert one Hugging Face model identifier into a filesystem-safe directory name."""

    return model_name.replace("/", "__").replace("\\", "__")


def _resolve_model_input_device(model: Any) -> torch.device:
    """Infer the device that should receive model inputs for single-device or auto-dispatched models."""

    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for mapped_device in model.hf_device_map.values():
            if mapped_device == "disk":
                continue
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str):
                return torch.device(mapped_device)

    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ArtifactConsistencyError("Loaded encoder model has no parameters.") from error


@dataclass(frozen=True)
class LoadedHuggingFaceEncoder:
    """One locally loaded Hugging Face tokenizer-model pair."""

    tokenizer: Any
    model: Any
    device: torch.device


@dataclass(frozen=True)
class EmbeddingProgressState:
    """One stock's resumable embedding progress state."""

    completed_count: int
    embedding_chunks: list[torch.Tensor]
    mask_chunks: list[torch.Tensor]
    is_complete: bool


class EmbeddingRequestError(ExternalServiceError):
    """Raised when one embedding request fails after all configured retries."""

    def __init__(self, message: str, retryable_at_batch_level: bool) -> None:
        """Store whether the failed batch should be retried by the outer resume loop."""

        super().__init__(message)
        self.retryable_at_batch_level = retryable_at_batch_level


class EmbeddingQuotaExceededError(EmbeddingRequestError):
    """Raised when the provider reports that the embedding account quota is exhausted."""

    def __init__(self, message: str) -> None:
        """Quota exhaustion should never be retried at the outer batch level."""

        super().__init__(message=message, retryable_at_batch_level=False)


def _load_huggingface_encoder(text_encoder_config: TextEncoderConfig) -> LoadedHuggingFaceEncoder:
    """Instantiate one frozen local Hugging Face text encoder from config."""

    model_dtype = MODEL_DTYPE_BY_NAME[text_encoder_config.model_dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_config.pretrained_model_name,
        cache_dir=text_encoder_config.local_model_cache_directory,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_load_kwargs: dict[str, Any] = {
        "cache_dir": text_encoder_config.local_model_cache_directory,
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if text_encoder_config.cache_device == "auto":
        offload_directory = ensure_directory(
            Path(text_encoder_config.local_model_cache_directory)
            / "offload"
            / _sanitize_model_name_for_path(text_encoder_config.pretrained_model_name or "encoder")
        )
        model_load_kwargs["device_map"] = "auto"
        model_load_kwargs["offload_folder"] = str(offload_directory)

    model = AutoModel.from_pretrained(
        text_encoder_config.pretrained_model_name,
        **model_load_kwargs,
    )
    model.eval()
    if text_encoder_config.cache_device == "auto":
        device = _resolve_model_input_device(model)
    else:
        device = torch.device(text_encoder_config.cache_device)
        model.to(device)
    return LoadedHuggingFaceEncoder(tokenizer=tokenizer, model=model, device=device)


def _apply_instruction_prefix(text_batch: list[str], instruction_prefix: str) -> list[str]:
    """Prepend one configured instruction string to every text in a batch."""

    if not instruction_prefix:
        return text_batch
    return [f"{instruction_prefix}{text_item}" for text_item in text_batch]


def _mean_pool_last_hidden_state(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with the tokenizer attention mask."""

    expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
    masked_hidden_state = last_hidden_state * expanded_attention_mask
    token_count_tensor = expanded_attention_mask.sum(dim=1).clamp_min(1)
    return masked_hidden_state.sum(dim=1) / token_count_tensor


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-padding token embedding, matching the official E5-Mistral recipe."""

    is_left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if is_left_padded:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_index = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_index, sequence_lengths]


def _run_huggingface_encoder_forward(loaded_encoder: LoadedHuggingFaceEncoder, tokenized_batch: dict[str, torch.Tensor]):
    """Execute the correct forward path for encoder-only and encoder-decoder models."""

    if hasattr(loaded_encoder.model, "encoder"):
        return loaded_encoder.model.encoder(
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
        )
    return loaded_encoder.model(**tokenized_batch)


def _encode_text_batch(
    text_batch: list[str],
    text_encoder_config: TextEncoderConfig,
    loaded_encoder: LoadedHuggingFaceEncoder,
) -> torch.Tensor:
    """Encode one text batch with a local Hugging Face model and configured pooling."""

    prepared_text_batch = _apply_instruction_prefix(text_batch, text_encoder_config.instruction_prefix)
    tokenized_batch = loaded_encoder.tokenizer(
        prepared_text_batch,
        padding=True,
        truncation=True,
        max_length=text_encoder_config.tokenizer_max_length,
        return_tensors="pt",
    )
    tokenized_batch = {key: value.to(loaded_encoder.device) for key, value in tokenized_batch.items()}
    with torch.no_grad():
        model_output = _run_huggingface_encoder_forward(loaded_encoder, tokenized_batch)
        last_hidden_state = model_output.last_hidden_state
        if text_encoder_config.pooling_strategy == "cls":
            encoded_batch = last_hidden_state[:, 0, :]
        elif text_encoder_config.pooling_strategy == "mean":
            encoded_batch = _mean_pool_last_hidden_state(last_hidden_state, tokenized_batch["attention_mask"])
        elif text_encoder_config.pooling_strategy == "last_token":
            encoded_batch = _last_token_pool(last_hidden_state, tokenized_batch["attention_mask"])
        else:
            raise ValueError(
                "text encoder pooling_strategy must be one of 'cls', 'mean', or 'last_token'."
            )
        if text_encoder_config.normalize_embeddings:
            encoded_batch = F.normalize(encoded_batch, p=2, dim=1)
        encoded_batch = encoded_batch.detach().cpu()

    if encoded_batch.shape[-1] != text_encoder_config.output_embedding_dim:
        raise ArtifactConsistencyError(
            "Local encoder output dimension does not match configuration. "
            f"Expected {text_encoder_config.output_embedding_dim}, got {encoded_batch.shape[-1]}."
        )
    return encoded_batch


class OpenAICompatibleEmbeddingClient:
    """Thin requests-based client for OpenAI-compatible embedding endpoints."""

    def __init__(self, text_encoder_config: TextEncoderConfig) -> None:
        """Create the client and validate the required API key environment variable."""

        load_project_local_environment_files()
        api_key = os.getenv(text_encoder_config.api_key_environment_variable or "")
        if not api_key:
            raise ExternalServiceError(
                "The embedding API key is missing. "
                f"Set environment variable '{text_encoder_config.api_key_environment_variable}'."
            )

        self._text_encoder_config = text_encoder_config
        self._rate_limiter = FixedIntervalRateLimiter(text_encoder_config.requests_per_second)
        self._session = requests.Session()
        http_adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        self._session.mount("http://", http_adapter)
        self._session.mount("https://", http_adapter)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_http_error_message(response: Response) -> str:
        """Format one HTTP error message with a truncated provider response body."""

        response_text_excerpt = response.text[:ERROR_RESPONSE_TEXT_MAX_CHARACTERS]
        return f"Embedding request failed with HTTP {response.status_code}. Response body excerpt: {response_text_excerpt}"

    @staticmethod
    def _extract_provider_error_code(response: Response) -> str | None:
        """Return the provider-specific error code when the response body is JSON."""

        try:
            payload = response.json()
        except ValueError:
            return None

        if not isinstance(payload, dict):
            return None
        error_object = payload.get("error")
        if not isinstance(error_object, dict):
            return None
        error_code = error_object.get("code")
        if isinstance(error_code, str) and error_code:
            return error_code
        return None

    def encode_text_batch(self, text_batch: list[str]) -> torch.Tensor:
        """Send one embedding request for a text batch and return a float tensor."""

        prepared_text_batch = _apply_instruction_prefix(text_batch, self._text_encoder_config.instruction_prefix)
        retry_delay_seconds = self._text_encoder_config.initial_retry_delay_seconds
        last_error_message: str | None = None

        for attempt_index in range(1, self._text_encoder_config.max_retry_attempt_count + 1):
            self._rate_limiter.acquire()
            try:
                response = self._session.post(
                    self._text_encoder_config.api_base_url,
                    headers=self._headers,
                    json={
                        "model": self._text_encoder_config.api_model_name,
                        "input": prepared_text_batch,
                        "encoding_format": "float",
                    },
                    timeout=self._text_encoder_config.request_timeout_seconds,
                )
                if response.status_code >= 400:
                    raise HTTPError(self._build_http_error_message(response), response=response)
                payload = response.json()
                return self._parse_embedding_payload(payload, len(text_batch))
            except Timeout as error:
                last_error_message = (
                    "Embedding API request timed out on attempt "
                    f"{attempt_index}/{self._text_encoder_config.max_retry_attempt_count}: {error}"
                )
            except RequestsConnectionError as error:
                last_error_message = (
                    "Embedding API request hit a network connection error on attempt "
                    f"{attempt_index}/{self._text_encoder_config.max_retry_attempt_count}: {error}"
                )
            except HTTPError as error:
                response = error.response
                status_code = response.status_code if response is not None else None
                provider_error_code = self._extract_provider_error_code(response) if response is not None else None
                if provider_error_code in self._text_encoder_config.quota_exhaustion_error_codes:
                    raise EmbeddingQuotaExceededError(
                        "Embedding API request failed because the provider reported exhausted account quota "
                        f"({provider_error_code}): {error}"
                    ) from error
                if status_code not in self._text_encoder_config.retryable_status_codes:
                    raise EmbeddingRequestError(
                        f"Embedding API request failed with a non-retryable HTTP status code {status_code}: {error}",
                        retryable_at_batch_level=False,
                    ) from error
                last_error_message = (
                    f"Embedding API request received a retryable HTTP status code {status_code} on attempt "
                    f"{attempt_index}/{self._text_encoder_config.max_retry_attempt_count}: {error}"
                )
            except ValueError as error:
                last_error_message = (
                    "Embedding API returned non-JSON content on attempt "
                    f"{attempt_index}/{self._text_encoder_config.max_retry_attempt_count}: {error}"
                )
            except ExternalServiceError as error:
                raise EmbeddingRequestError(str(error), retryable_at_batch_level=False) from error

            if attempt_index == self._text_encoder_config.max_retry_attempt_count:
                break
            time.sleep(retry_delay_seconds)
            retry_delay_seconds *= self._text_encoder_config.retry_backoff_multiplier

        raise EmbeddingRequestError(
            last_error_message or "Embedding API request failed.",
            retryable_at_batch_level=True,
        )

    def _parse_embedding_payload(self, payload: dict[str, Any], expected_batch_size: int) -> torch.Tensor:
        """Validate and convert one provider embedding payload into a float tensor."""

        try:
            raw_data_items = payload["data"]
        except KeyError as error:
            raise ExternalServiceError("Embedding API response is missing the 'data' field.") from error

        if not isinstance(raw_data_items, list):
            raise ExternalServiceError("Embedding API response field 'data' must be a list.")

        if any(isinstance(item, dict) and "index" in item for item in raw_data_items):
            try:
                raw_data_items = sorted(raw_data_items, key=lambda item: int(item["index"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ExternalServiceError("Embedding API returned invalid embedding indices.") from error

        embedding_rows: list[list[float]] = []
        for raw_item in raw_data_items:
            if not isinstance(raw_item, dict):
                raise ExternalServiceError("Embedding API response items must be JSON objects.")
            raw_embedding = raw_item.get("embedding")
            if not isinstance(raw_embedding, list):
                raise ExternalServiceError("Embedding API response item is missing a valid embedding list.")
            try:
                embedding_rows.append([float(value) for value in raw_embedding])
            except (TypeError, ValueError) as error:
                raise ExternalServiceError("Embedding API returned non-numeric embedding values.") from error

        embedding_tensor = torch.tensor(embedding_rows, dtype=torch.float32)
        if embedding_tensor.ndim != 2 or embedding_tensor.shape[0] != expected_batch_size:
            raise ExternalServiceError("Embedding API returned an unexpected batch shape.")
        if embedding_tensor.shape[-1] != self._text_encoder_config.output_embedding_dim:
            raise ExternalServiceError(
                "Embedding API returned a vector dimension that does not match configuration. "
                f"Expected {self._text_encoder_config.output_embedding_dim}, got {embedding_tensor.shape[-1]}."
            )
        return embedding_tensor


def _build_encoder_runtime(
    text_encoder_config: TextEncoderConfig,
) -> tuple[LoadedHuggingFaceEncoder | None, OpenAICompatibleEmbeddingClient | None]:
    """Instantiate exactly one backend runtime according to the encoder configuration."""

    if text_encoder_config.backend_type == "huggingface_transformers":
        return _load_huggingface_encoder(text_encoder_config), None
    if text_encoder_config.backend_type == "openai_compatible_embedding_api":
        return None, OpenAICompatibleEmbeddingClient(text_encoder_config)
    raise ValueError("Unsupported text encoder backend type.")


def _encode_text_sequence(
    text_sequence: list[str],
    text_encoder_config: TextEncoderConfig,
    loaded_encoder: LoadedHuggingFaceEncoder | None,
    embedding_client: OpenAICompatibleEmbeddingClient | None,
) -> torch.Tensor:
    """Encode one arbitrarily long text sequence with the configured backend in mini-batches."""

    batch_embeddings: list[torch.Tensor] = []
    for batch_start_index in range(0, len(text_sequence), text_encoder_config.encoder_batch_size):
        batch_texts = text_sequence[batch_start_index : batch_start_index + text_encoder_config.encoder_batch_size]
        if loaded_encoder is not None:
            batch_embeddings.append(_encode_text_batch(batch_texts, text_encoder_config, loaded_encoder))
        elif embedding_client is not None:
            batch_embeddings.append(embedding_client.encode_text_batch(batch_texts))
        else:
            raise ValueError("At least one encoder runtime must be available.")
    return torch.cat(batch_embeddings, dim=0)


def _save_embedding_artifact(
    artifact_file_path: Path,
    trade_dates: list[str],
    embeddings: torch.Tensor,
    has_embedding_mask: torch.Tensor,
    encoder_fingerprint: str | None = None,
) -> None:
    """Persist one aligned daily embedding artifact to disk."""

    ensure_directory(artifact_file_path.parent)
    payload = {
        "trade_dates": trade_dates,
        "embeddings": embeddings.cpu(),
        "has_embedding_mask": has_embedding_mask.cpu(),
    }
    if encoder_fingerprint is not None:
        payload["encoder_fingerprint"] = encoder_fingerprint
    torch.save(payload, artifact_file_path)


def _compute_encoder_fingerprint(text_encoder_config: TextEncoderConfig) -> str:
    """Build one stable fingerprint for the active text-encoder configuration."""

    fingerprint_payload = {
        "backend_type": text_encoder_config.backend_type,
        "pretrained_model_name": text_encoder_config.pretrained_model_name,
        "api_base_url": text_encoder_config.api_base_url,
        "api_model_name": text_encoder_config.api_model_name,
        "tokenizer_max_length": text_encoder_config.tokenizer_max_length,
        "output_embedding_dim": text_encoder_config.output_embedding_dim,
        "pooling_strategy": text_encoder_config.pooling_strategy,
        "normalize_embeddings": text_encoder_config.normalize_embeddings,
        "model_dtype": text_encoder_config.model_dtype,
        "instruction_prefix": text_encoder_config.instruction_prefix,
    }
    serialized_payload = json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _build_embedding_checkpoint_paths(
    market_artifact_root: Path,
    checkpoint_subdirectory: str,
    checkpoint_file_name: str,
    failure_log_file_name: str,
) -> tuple[Path, Path]:
    """Resolve filesystem paths for one resumable embedding checkpoint and failure log."""

    checkpoint_directory = ensure_directory(market_artifact_root / checkpoint_subdirectory)
    return checkpoint_directory / checkpoint_file_name, checkpoint_directory / failure_log_file_name


def _build_partial_embedding_directory(artifact_file_path: Path) -> Path:
    """Return the per-stock partial embedding directory used for resumable API encoding."""

    market_artifact_root = artifact_file_path.parent.parent
    partial_root_directory = ensure_directory(market_artifact_root / PARTIAL_EMBEDDING_ROOT_DIRECTORY_NAME)
    return partial_root_directory / artifact_file_path.stem


def _load_checkpoint_failure_count(checkpoint_file_path: Path, encoder_fingerprint: str) -> int:
    """Recover the cumulative failed-request counter from one compatible checkpoint file."""

    if not checkpoint_file_path.exists():
        return 0

    checkpoint_payload = read_json_file(checkpoint_file_path)
    existing_encoder_fingerprint = checkpoint_payload.get("encoder_fingerprint")
    if existing_encoder_fingerprint != encoder_fingerprint:
        raise ArtifactConsistencyError(
            f"Embedding checkpoint '{checkpoint_file_path}' was created with a different encoder fingerprint."
        )

    failed_request_count = checkpoint_payload.get("failed_request_count")
    if not isinstance(failed_request_count, int) or failed_request_count < 0:
        raise ArtifactConsistencyError(
            f"Embedding checkpoint '{checkpoint_file_path}' contains an invalid failed_request_count value."
        )
    return failed_request_count


def _write_embedding_checkpoint(
    checkpoint_file_path: Path,
    experiment_config: ExperimentConfig,
    text_encoder_config: TextEncoderConfig,
    encoder_fingerprint: str,
    total_record_count: int,
    completed_record_count: int,
    failed_request_count: int,
    current_stock_id: str,
    last_saved_trade_date: str,
) -> None:
    """Persist resumable embedding progress metadata."""

    checkpoint_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_code": experiment_config.data.market_code,
        "encoder_backend_type": text_encoder_config.backend_type,
        "encoder_model_name": text_encoder_config.api_model_name or text_encoder_config.pretrained_model_name,
        "encoder_fingerprint": encoder_fingerprint,
        "requests_per_second": text_encoder_config.requests_per_second,
        "encoder_batch_size": text_encoder_config.encoder_batch_size,
        "save_every_record_count": text_encoder_config.save_every_record_count,
        "total_record_count": total_record_count,
        "completed_record_count": completed_record_count,
        "remaining_record_count": max(total_record_count - completed_record_count, 0),
        "failed_request_count": failed_request_count,
        "current_stock_id": current_stock_id,
        "last_saved_trade_date": last_saved_trade_date,
    }
    write_json_file(checkpoint_file_path, checkpoint_payload)


def _load_saved_embedding_progress(
    artifact_file_path: Path,
    ordered_trade_dates: list[str],
    output_embedding_dim: int,
    encoder_fingerprint: str,
) -> EmbeddingProgressState:
    """Load one final or partial embedding artifact and validate that it matches the current encoder."""

    if not artifact_file_path.exists():
        return EmbeddingProgressState(completed_count=0, embedding_chunks=[], mask_chunks=[], is_complete=False)

    raw_payload = torch.load(artifact_file_path, map_location="cpu")
    required_keys = {"trade_dates", "embeddings", "has_embedding_mask"}
    missing_keys = required_keys - set(raw_payload.keys())
    if missing_keys:
        raise ArtifactConsistencyError(
            f"Embedding artifact '{artifact_file_path}' is missing keys: {', '.join(sorted(missing_keys))}."
        )

    saved_trade_dates = list(raw_payload["trade_dates"])
    embeddings = raw_payload["embeddings"].float()
    has_embedding_mask = raw_payload["has_embedding_mask"].float()
    saved_encoder_fingerprint = raw_payload.get("encoder_fingerprint")
    if saved_encoder_fingerprint is not None and saved_encoder_fingerprint != encoder_fingerprint:
        raise ArtifactConsistencyError(
            f"Embedding artifact '{artifact_file_path}' was created with a different encoder fingerprint."
        )

    if embeddings.ndim != 2 or embeddings.shape[-1] != output_embedding_dim:
        raise ArtifactConsistencyError(
            f"Embedding artifact '{artifact_file_path}' has an unexpected tensor shape {tuple(embeddings.shape)}."
        )
    if len(saved_trade_dates) != embeddings.shape[0] or len(saved_trade_dates) != has_embedding_mask.shape[0]:
        raise ArtifactConsistencyError(
            f"Embedding artifact '{artifact_file_path}' has mismatched date and tensor lengths."
        )
    if ordered_trade_dates[: len(saved_trade_dates)] != saved_trade_dates:
        raise ArtifactConsistencyError(
            f"Embedding artifact '{artifact_file_path}' does not match the expected trade-date prefix."
        )

    return EmbeddingProgressState(
        completed_count=len(saved_trade_dates),
        embedding_chunks=[embeddings] if embeddings.shape[0] > 0 else [],
        mask_chunks=[has_embedding_mask] if has_embedding_mask.shape[0] > 0 else [],
        is_complete=len(saved_trade_dates) == len(ordered_trade_dates),
    )


def _load_partial_chunk_progress(
    partial_stock_directory: Path,
    ordered_trade_dates: list[str],
    output_embedding_dim: int,
    encoder_fingerprint: str,
) -> EmbeddingProgressState:
    """Load all per-stock partial chunk files and reconstruct the completed prefix."""

    if not partial_stock_directory.exists():
        return EmbeddingProgressState(completed_count=0, embedding_chunks=[], mask_chunks=[], is_complete=False)

    chunk_file_paths = sorted(partial_stock_directory.glob("chunk_*.pt"))
    if not chunk_file_paths:
        return EmbeddingProgressState(completed_count=0, embedding_chunks=[], mask_chunks=[], is_complete=False)

    accumulated_trade_dates: list[str] = []
    embedding_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []

    for chunk_file_path in chunk_file_paths:
        raw_payload = torch.load(chunk_file_path, map_location="cpu")
        required_keys = {"trade_dates", "embeddings", "has_embedding_mask", "encoder_fingerprint"}
        missing_keys = required_keys - set(raw_payload.keys())
        if missing_keys:
            raise ArtifactConsistencyError(
                f"Partial embedding chunk '{chunk_file_path}' is missing keys: {', '.join(sorted(missing_keys))}."
            )
        if raw_payload["encoder_fingerprint"] != encoder_fingerprint:
            raise ArtifactConsistencyError(
                f"Partial embedding chunk '{chunk_file_path}' was created with a different encoder fingerprint."
            )

        chunk_trade_dates = list(raw_payload["trade_dates"])
        chunk_embeddings = raw_payload["embeddings"].float()
        chunk_mask = raw_payload["has_embedding_mask"].float()
        if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[-1] != output_embedding_dim:
            raise ArtifactConsistencyError(
                f"Partial embedding chunk '{chunk_file_path}' has an unexpected tensor shape {tuple(chunk_embeddings.shape)}."
            )
        if len(chunk_trade_dates) != chunk_embeddings.shape[0] or len(chunk_trade_dates) != chunk_mask.shape[0]:
            raise ArtifactConsistencyError(
                f"Partial embedding chunk '{chunk_file_path}' has mismatched date and tensor lengths."
            )

        accumulated_trade_dates.extend(chunk_trade_dates)
        embedding_chunks.append(chunk_embeddings)
        mask_chunks.append(chunk_mask)

    if ordered_trade_dates[: len(accumulated_trade_dates)] != accumulated_trade_dates:
        raise ArtifactConsistencyError(
            f"Partial embedding chunks under '{partial_stock_directory}' do not match the expected trade-date prefix."
        )

    return EmbeddingProgressState(
        completed_count=len(accumulated_trade_dates),
        embedding_chunks=embedding_chunks,
        mask_chunks=mask_chunks,
        is_complete=False,
    )


def _flush_partial_embedding_chunk(
    partial_stock_directory: Path,
    chunk_index: int,
    trade_dates: list[str],
    embedding_rows: list[torch.Tensor],
    encoder_fingerprint: str,
) -> None:
    """Persist one partial embedding chunk for safe resume."""

    ensure_directory(partial_stock_directory)
    embeddings = torch.stack(embedding_rows, dim=0)
    has_embedding_mask = torch.ones(len(trade_dates), dtype=torch.float32)
    torch.save(
        {
            "trade_dates": trade_dates,
            "embeddings": embeddings.cpu(),
            "has_embedding_mask": has_embedding_mask.cpu(),
            "encoder_fingerprint": encoder_fingerprint,
        },
        partial_stock_directory / f"chunk_{chunk_index:06d}.pt",
    )


def _finalize_partial_chunks_into_artifact(
    final_artifact_file_path: Path,
    ordered_trade_dates: list[str],
    embedding_chunks: list[torch.Tensor],
    mask_chunks: list[torch.Tensor],
    encoder_fingerprint: str,
    partial_stock_directory: Path,
) -> None:
    """Merge all chunk tensors into the final aligned artifact and delete the partial directory."""

    if not embedding_chunks or not mask_chunks:
        raise ArtifactConsistencyError("Cannot finalize an embedding artifact without any encoded rows.")

    embeddings = torch.cat(embedding_chunks, dim=0)
    has_embedding_mask = torch.cat(mask_chunks, dim=0)
    if embeddings.shape[0] != len(ordered_trade_dates):
        raise ArtifactConsistencyError(
            f"Cannot finalize '{final_artifact_file_path}' because the embedding row count does not match trade dates."
        )

    _save_embedding_artifact(
        artifact_file_path=final_artifact_file_path,
        trade_dates=ordered_trade_dates,
        embeddings=embeddings,
        has_embedding_mask=has_embedding_mask,
        encoder_fingerprint=encoder_fingerprint,
    )
    if partial_stock_directory.exists():
        shutil.rmtree(partial_stock_directory)


def _build_embedding_failure_payload(
    stock_id: str,
    stock_name: str,
    trade_dates: list[str],
    error: ExternalServiceError,
    failure_kind: str,
    consecutive_failure_count: int,
) -> dict:
    """Build one durable failure record for later resume diagnostics."""

    trade_date_start = trade_dates[0] if trade_dates else ""
    trade_date_end = trade_dates[-1] if trade_dates else ""
    return {
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stock_id": stock_id,
        "stock_name": stock_name,
        "trade_date_start": trade_date_start,
        "trade_date_end": trade_date_end,
        "batch_size": len(trade_dates),
        "failure_kind": failure_kind,
        "consecutive_failure_count": consecutive_failure_count,
        "error_message": str(error),
    }


def _load_and_normalize_generated_technical_text_records(
    generated_text_file_path: Path,
    stock_id: str,
    ordered_trade_dates: list[str],
) -> dict[str, str]:
    """Load one generated technical-text file, keep the latest record per date, and rewrite duplicates away."""

    generated_records = read_jsonl_records(generated_text_file_path)
    generated_record_by_trade_date: dict[str, dict] = {}
    duplicate_record_count = 0

    for generated_record in generated_records:
        record_stock_id = str(generated_record.get("stock_id"))
        trade_date = str(generated_record.get("trade_date"))
        if record_stock_id != stock_id:
            raise ArtifactConsistencyError(
                f"Generated technical-text file '{generated_text_file_path}' contains stock_id '{record_stock_id}', "
                f"expected '{stock_id}'."
            )
        if trade_date in generated_record_by_trade_date:
            duplicate_record_count += 1
        generated_record_by_trade_date[trade_date] = generated_record

    missing_trade_dates = [trade_date for trade_date in ordered_trade_dates if trade_date not in generated_record_by_trade_date]
    if missing_trade_dates:
        raise ArtifactConsistencyError(
            f"Technical-text generation is incomplete for stock '{stock_id}'. "
            f"Missing {len(missing_trade_dates)} trade dates."
        )

    ordered_records = [generated_record_by_trade_date[trade_date] for trade_date in ordered_trade_dates]
    if duplicate_record_count > 0 or len(generated_records) != len(ordered_records):
        write_jsonl_records(generated_text_file_path, ordered_records)

    return {
        trade_date: str(generated_record_by_trade_date[trade_date]["generated_text"])
        for trade_date in ordered_trade_dates
    }


def encode_technical_text_artifacts(experiment_config: ExperimentConfig) -> None:
    """Encode generated technical-indicator texts into daily embedding caches with save-and-resume support."""

    text_encoder_config = experiment_config.technical_text_encoder
    loaded_encoder, embedding_client = _build_encoder_runtime(text_encoder_config)
    market_artifact_root = get_market_artifact_root(experiment_config)
    manifest_frame = pd.read_csv(market_artifact_root / "stock_manifest.csv")
    encoder_fingerprint = _compute_encoder_fingerprint(text_encoder_config)
    checkpoint_file_path, failure_log_file_path = _build_embedding_checkpoint_paths(
        market_artifact_root=market_artifact_root,
        checkpoint_subdirectory=experiment_config.paths.checkpoint_subdirectory,
        checkpoint_file_name=text_encoder_config.checkpoint_file_name,
        failure_log_file_name=text_encoder_config.failure_log_file_name,
    )
    failed_request_count = _load_checkpoint_failure_count(checkpoint_file_path, encoder_fingerprint)

    stock_runtime_plans: list[dict[str, Any]] = []
    total_record_count = 0
    completed_record_count = 0

    for _, manifest_row in manifest_frame.iterrows():
        processed_price_frame = pd.read_csv(path_from_serialized_value(manifest_row["processed_price_file_path"]))
        trade_dates = processed_price_frame["trade_date"].tolist()
        final_artifact_file_path = path_from_serialized_value(manifest_row["tech_embedding_file_path"])
        partial_stock_directory = _build_partial_embedding_directory(final_artifact_file_path)
        final_progress_state = _load_saved_embedding_progress(
            artifact_file_path=final_artifact_file_path,
            ordered_trade_dates=trade_dates,
            output_embedding_dim=text_encoder_config.output_embedding_dim,
            encoder_fingerprint=encoder_fingerprint,
        )

        if final_progress_state.is_complete:
            if partial_stock_directory.exists():
                shutil.rmtree(partial_stock_directory)
            initial_progress_state = final_progress_state
        else:
            initial_progress_state = _load_partial_chunk_progress(
                partial_stock_directory=partial_stock_directory,
                ordered_trade_dates=trade_dates,
                output_embedding_dim=text_encoder_config.output_embedding_dim,
                encoder_fingerprint=encoder_fingerprint,
            )

        total_record_count += len(trade_dates)
        completed_record_count += initial_progress_state.completed_count
        stock_runtime_plans.append(
            {
                "stock_id": str(manifest_row["stock_id"]),
                "stock_name": str(manifest_row["stock_name"]),
                "generated_text_file_path": path_from_serialized_value(manifest_row["tech_text_output_file_path"]),
                "final_artifact_file_path": final_artifact_file_path,
                "partial_stock_directory": partial_stock_directory,
                "trade_dates": trade_dates,
                "initial_progress_state": initial_progress_state,
            }
        )

    LOGGER.info(
        "Technical embedding resume scan complete. total=%s completed=%s pending=%s encoder=%s rps_limit=%s",
        total_record_count,
        completed_record_count,
        total_record_count - completed_record_count,
        text_encoder_config.api_model_name or text_encoder_config.pretrained_model_name,
        text_encoder_config.requests_per_second,
    )
    _write_embedding_checkpoint(
        checkpoint_file_path=checkpoint_file_path,
        experiment_config=experiment_config,
        text_encoder_config=text_encoder_config,
        encoder_fingerprint=encoder_fingerprint,
        total_record_count=total_record_count,
        completed_record_count=completed_record_count,
        failed_request_count=failed_request_count,
        current_stock_id="",
        last_saved_trade_date="",
    )

    for stock_runtime_plan in stock_runtime_plans:
        initial_progress_state = stock_runtime_plan["initial_progress_state"]
        trade_dates = stock_runtime_plan["trade_dates"]
        if initial_progress_state.is_complete:
            continue

        generated_text_by_trade_date = _load_and_normalize_generated_technical_text_records(
            generated_text_file_path=stock_runtime_plan["generated_text_file_path"],
            stock_id=stock_runtime_plan["stock_id"],
            ordered_trade_dates=trade_dates,
        )
        ordered_texts = [generated_text_by_trade_date[trade_date] for trade_date in trade_dates]
        embedding_chunks = list(initial_progress_state.embedding_chunks)
        mask_chunks = list(initial_progress_state.mask_chunks)
        pending_trade_dates: list[str] = []
        pending_embedding_rows: list[torch.Tensor] = []
        completed_for_stock = initial_progress_state.completed_count
        next_chunk_index = len(embedding_chunks)
        consecutive_batch_failure_count = 0

        for batch_start_index in range(
            initial_progress_state.completed_count,
            len(ordered_texts),
            text_encoder_config.encoder_batch_size,
        ):
            batch_trade_dates = trade_dates[
                batch_start_index : batch_start_index + text_encoder_config.encoder_batch_size
            ]
            batch_texts = ordered_texts[
                batch_start_index : batch_start_index + text_encoder_config.encoder_batch_size
            ]

            while True:
                try:
                    if loaded_encoder is not None:
                        batch_embeddings = _encode_text_batch(batch_texts, text_encoder_config, loaded_encoder)
                    elif embedding_client is not None:
                        batch_embeddings = embedding_client.encode_text_batch(batch_texts)
                    else:
                        raise ValueError("At least one encoder runtime must be available.")
                    consecutive_batch_failure_count = 0
                    break
                except EmbeddingQuotaExceededError as error:
                    failed_request_count += 1
                    append_jsonl_records(
                        failure_log_file_path,
                        [
                            _build_embedding_failure_payload(
                                stock_id=stock_runtime_plan["stock_id"],
                                stock_name=stock_runtime_plan["stock_name"],
                                trade_dates=batch_trade_dates,
                                error=error,
                                failure_kind="quota_exhausted",
                                consecutive_failure_count=1,
                            )
                        ],
                    )
                    if pending_trade_dates:
                        _flush_partial_embedding_chunk(
                            partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
                            chunk_index=next_chunk_index,
                            trade_dates=pending_trade_dates,
                            embedding_rows=pending_embedding_rows,
                            encoder_fingerprint=encoder_fingerprint,
                        )
                        embedding_chunks.append(torch.stack(pending_embedding_rows, dim=0))
                        mask_chunks.append(torch.ones(len(pending_trade_dates), dtype=torch.float32))
                        completed_for_stock += len(pending_trade_dates)
                        completed_record_count += len(pending_trade_dates)
                    _write_embedding_checkpoint(
                        checkpoint_file_path=checkpoint_file_path,
                        experiment_config=experiment_config,
                        text_encoder_config=text_encoder_config,
                        encoder_fingerprint=encoder_fingerprint,
                        total_record_count=total_record_count,
                        completed_record_count=completed_record_count,
                        failed_request_count=failed_request_count,
                        current_stock_id=stock_runtime_plan["stock_id"],
                        last_saved_trade_date=trade_dates[completed_for_stock - 1] if completed_for_stock > 0 else "",
                    )
                    raise RuntimeError(
                        "Stopping technical embedding generation because the provider reported exhausted account quota "
                        "(insufficient_user_quota). Restore quota and rerun the same command to resume."
                    ) from error
                except EmbeddingRequestError as error:
                    failed_request_count += 1
                    if error.retryable_at_batch_level:
                        consecutive_batch_failure_count += 1
                    else:
                        consecutive_batch_failure_count = 1
                    append_jsonl_records(
                        failure_log_file_path,
                        [
                            _build_embedding_failure_payload(
                                stock_id=stock_runtime_plan["stock_id"],
                                stock_name=stock_runtime_plan["stock_name"],
                                trade_dates=batch_trade_dates,
                                error=error,
                                failure_kind=(
                                    "retryable_batch_failure"
                                    if error.retryable_at_batch_level
                                    else "non_retryable_batch_failure"
                                ),
                                consecutive_failure_count=consecutive_batch_failure_count,
                            )
                        ],
                    )
                    _write_embedding_checkpoint(
                        checkpoint_file_path=checkpoint_file_path,
                        experiment_config=experiment_config,
                        text_encoder_config=text_encoder_config,
                        encoder_fingerprint=encoder_fingerprint,
                        total_record_count=total_record_count,
                        completed_record_count=completed_record_count,
                        failed_request_count=failed_request_count,
                        current_stock_id=stock_runtime_plan["stock_id"],
                        last_saved_trade_date=trade_dates[completed_for_stock - 1] if completed_for_stock > 0 else "",
                    )
                    if not error.retryable_at_batch_level:
                        if pending_trade_dates:
                            _flush_partial_embedding_chunk(
                                partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
                                chunk_index=next_chunk_index,
                                trade_dates=pending_trade_dates,
                                embedding_rows=pending_embedding_rows,
                                encoder_fingerprint=encoder_fingerprint,
                            )
                            embedding_chunks.append(torch.stack(pending_embedding_rows, dim=0))
                            mask_chunks.append(torch.ones(len(pending_trade_dates), dtype=torch.float32))
                            completed_for_stock += len(pending_trade_dates)
                            completed_record_count += len(pending_trade_dates)
                        _write_embedding_checkpoint(
                            checkpoint_file_path=checkpoint_file_path,
                            experiment_config=experiment_config,
                            text_encoder_config=text_encoder_config,
                            encoder_fingerprint=encoder_fingerprint,
                            total_record_count=total_record_count,
                            completed_record_count=completed_record_count,
                            failed_request_count=failed_request_count,
                            current_stock_id=stock_runtime_plan["stock_id"],
                            last_saved_trade_date=trade_dates[completed_for_stock - 1] if completed_for_stock > 0 else "",
                        )
                        raise RuntimeError(
                            "Stopping technical embedding generation because the provider returned a "
                            f"non-retryable batch error: {error}"
                        ) from error
                    if consecutive_batch_failure_count >= text_encoder_config.max_consecutive_failure_count:
                        if pending_trade_dates:
                            _flush_partial_embedding_chunk(
                                partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
                                chunk_index=next_chunk_index,
                                trade_dates=pending_trade_dates,
                                embedding_rows=pending_embedding_rows,
                                encoder_fingerprint=encoder_fingerprint,
                            )
                            embedding_chunks.append(torch.stack(pending_embedding_rows, dim=0))
                            mask_chunks.append(torch.ones(len(pending_trade_dates), dtype=torch.float32))
                            completed_for_stock += len(pending_trade_dates)
                            completed_record_count += len(pending_trade_dates)
                        _write_embedding_checkpoint(
                            checkpoint_file_path=checkpoint_file_path,
                            experiment_config=experiment_config,
                            text_encoder_config=text_encoder_config,
                            encoder_fingerprint=encoder_fingerprint,
                            total_record_count=total_record_count,
                            completed_record_count=completed_record_count,
                            failed_request_count=failed_request_count,
                            current_stock_id=stock_runtime_plan["stock_id"],
                            last_saved_trade_date=trade_dates[completed_for_stock - 1] if completed_for_stock > 0 else "",
                        )
                        raise RuntimeError(
                            "Stopping technical embedding generation because one batch kept failing after "
                            f"{consecutive_batch_failure_count} consecutive batch-level retry cycles: {error}"
                        ) from error

                    retry_delay_seconds = text_encoder_config.initial_retry_delay_seconds * (
                        text_encoder_config.retry_backoff_multiplier ** (consecutive_batch_failure_count - 1)
                    )
                    LOGGER.warning(
                        "Technical embedding batch failed for %s|%s on recovery attempt %s/%s. "
                        "Sleeping %.2fs before retrying the same batch. Error: %s",
                        stock_runtime_plan["stock_id"],
                        batch_trade_dates[0] if batch_trade_dates else "",
                        consecutive_batch_failure_count,
                        text_encoder_config.max_consecutive_failure_count,
                        retry_delay_seconds,
                        error,
                    )
                    time.sleep(retry_delay_seconds)

            for row_index, trade_date in enumerate(batch_trade_dates):
                pending_trade_dates.append(trade_date)
                pending_embedding_rows.append(batch_embeddings[row_index].detach().cpu())
                if len(pending_trade_dates) >= text_encoder_config.save_every_record_count:
                    _flush_partial_embedding_chunk(
                        partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
                        chunk_index=next_chunk_index,
                        trade_dates=pending_trade_dates,
                        embedding_rows=pending_embedding_rows,
                        encoder_fingerprint=encoder_fingerprint,
                    )
                    embedding_chunks.append(torch.stack(pending_embedding_rows, dim=0))
                    mask_chunks.append(torch.ones(len(pending_trade_dates), dtype=torch.float32))
                    completed_for_stock += len(pending_trade_dates)
                    completed_record_count += len(pending_trade_dates)
                    _write_embedding_checkpoint(
                        checkpoint_file_path=checkpoint_file_path,
                        experiment_config=experiment_config,
                        text_encoder_config=text_encoder_config,
                        encoder_fingerprint=encoder_fingerprint,
                        total_record_count=total_record_count,
                        completed_record_count=completed_record_count,
                        failed_request_count=failed_request_count,
                        current_stock_id=stock_runtime_plan["stock_id"],
                        last_saved_trade_date=pending_trade_dates[-1],
                    )
                    LOGGER.info(
                        "Saved %s technical embeddings. completed=%s remaining=%s last_record=%s|%s",
                        len(pending_trade_dates),
                        completed_record_count,
                        max(total_record_count - completed_record_count, 0),
                        stock_runtime_plan["stock_id"],
                        pending_trade_dates[-1],
                    )
                    next_chunk_index += 1
                    pending_trade_dates = []
                    pending_embedding_rows = []

        if pending_trade_dates:
            _flush_partial_embedding_chunk(
                partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
                chunk_index=next_chunk_index,
                trade_dates=pending_trade_dates,
                embedding_rows=pending_embedding_rows,
                encoder_fingerprint=encoder_fingerprint,
            )
            embedding_chunks.append(torch.stack(pending_embedding_rows, dim=0))
            mask_chunks.append(torch.ones(len(pending_trade_dates), dtype=torch.float32))
            completed_for_stock += len(pending_trade_dates)
            completed_record_count += len(pending_trade_dates)
            _write_embedding_checkpoint(
                checkpoint_file_path=checkpoint_file_path,
                experiment_config=experiment_config,
                text_encoder_config=text_encoder_config,
                encoder_fingerprint=encoder_fingerprint,
                total_record_count=total_record_count,
                completed_record_count=completed_record_count,
                failed_request_count=failed_request_count,
                current_stock_id=stock_runtime_plan["stock_id"],
                last_saved_trade_date=pending_trade_dates[-1],
            )
            LOGGER.info(
                "Saved %s technical embeddings. completed=%s remaining=%s last_record=%s|%s",
                len(pending_trade_dates),
                completed_record_count,
                max(total_record_count - completed_record_count, 0),
                stock_runtime_plan["stock_id"],
                pending_trade_dates[-1],
            )

        _finalize_partial_chunks_into_artifact(
            final_artifact_file_path=stock_runtime_plan["final_artifact_file_path"],
            ordered_trade_dates=trade_dates,
            embedding_chunks=embedding_chunks,
            mask_chunks=mask_chunks,
            encoder_fingerprint=encoder_fingerprint,
            partial_stock_directory=stock_runtime_plan["partial_stock_directory"],
        )
        _write_embedding_checkpoint(
            checkpoint_file_path=checkpoint_file_path,
            experiment_config=experiment_config,
            text_encoder_config=text_encoder_config,
            encoder_fingerprint=encoder_fingerprint,
            total_record_count=total_record_count,
            completed_record_count=completed_record_count,
            failed_request_count=failed_request_count,
            current_stock_id=stock_runtime_plan["stock_id"],
            last_saved_trade_date=trade_dates[-1],
        )
        LOGGER.info(
            "Finalized technical embedding artifact for %s with %s records.",
            stock_runtime_plan["stock_id"],
            len(trade_dates),
        )

    LOGGER.info(
        "Technical embedding generation finished. total=%s completed=%s failed=%s remaining=%s",
        total_record_count,
        completed_record_count,
        failed_request_count,
        max(total_record_count - completed_record_count, 0),
    )
    _write_embedding_checkpoint(
        checkpoint_file_path=checkpoint_file_path,
        experiment_config=experiment_config,
        text_encoder_config=text_encoder_config,
        encoder_fingerprint=encoder_fingerprint,
        total_record_count=total_record_count,
        completed_record_count=completed_record_count,
        failed_request_count=failed_request_count,
        current_stock_id="",
        last_saved_trade_date="",
    )


def encode_news_artifacts(experiment_config: ExperimentConfig) -> None:
    """Encode aligned daily news into one average daily embedding vector per trade date."""

    text_encoder_config = experiment_config.news_encoder
    loaded_encoder, embedding_client = _build_encoder_runtime(text_encoder_config)
    market_artifact_root = get_market_artifact_root(experiment_config)
    manifest_frame = pd.read_csv(market_artifact_root / "stock_manifest.csv")
    encoder_fingerprint = _compute_encoder_fingerprint(text_encoder_config)

    for _, manifest_row in manifest_frame.iterrows():
        processed_price_frame = pd.read_csv(path_from_serialized_value(manifest_row["processed_price_file_path"]))
        aligned_news_map = read_json_file(path_from_serialized_value(manifest_row["aligned_news_file_path"]))
        trade_dates = processed_price_frame["trade_date"].tolist()
        embedding_rows: list[torch.Tensor] = []
        has_embedding_values: list[float] = []

        for trade_date in trade_dates:
            headlines = list(aligned_news_map[str(trade_date)])
            if not headlines:
                embedding_rows.append(torch.zeros(text_encoder_config.output_embedding_dim, dtype=torch.float32))
                has_embedding_values.append(0.0)
                continue

            headline_embeddings = _encode_text_sequence(
                headlines,
                text_encoder_config,
                loaded_encoder,
                embedding_client,
            )
            averaged_embedding = headline_embeddings.mean(dim=0)
            embedding_rows.append(averaged_embedding)
            has_embedding_values.append(1.0)

        embedding_tensor = torch.stack(embedding_rows, dim=0)
        has_embedding_mask = torch.tensor(has_embedding_values, dtype=torch.float32)
        _save_embedding_artifact(
            artifact_file_path=path_from_serialized_value(manifest_row["news_embedding_file_path"]),
            trade_dates=trade_dates,
            embeddings=embedding_tensor,
            has_embedding_mask=has_embedding_mask,
            encoder_fingerprint=encoder_fingerprint,
        )
