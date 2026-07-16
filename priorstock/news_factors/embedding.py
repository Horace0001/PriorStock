"""Embedding cache builder for parsed LLM factors."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from priorstock.news_factors.api_client import OpenAICompatibleClient
from priorstock.news_factors.config import NewsFactorExperimentConfig
from priorstock.news_factors.factors import RequestRateLimiter


def embed_factor_records(
    config: NewsFactorExperimentConfig,
    factor_records_file_path: Path,
    output_file_path: Path,
) -> None:
    """Embed parsed factor records and persist a compressed NPZ cache."""

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    if output_file_path.exists():
        return
    batch_directory_path = output_file_path.with_suffix("")
    batch_directory_path = batch_directory_path.parent / f"{batch_directory_path.name}_batches"
    batch_directory_path.mkdir(parents=True, exist_ok=True)
    partial_file_path = output_file_path.with_suffix(".partial.pt")
    valid_record_by_key: dict[tuple[str, str], dict[str, object]] = {}
    texts: list[str] = []
    with factor_records_file_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record["llm_model"]) in set(config.chat.disabled_models):
                continue
            if not bool(record["is_valid"]):
                continue
            parsed_factors = record["parsed_factors"]
            if not isinstance(parsed_factors, list) or len(parsed_factors) != 5:
                continue
            key = (str(record["sample_id"]), str(record["llm_model"]))
            valid_record_by_key[key] = record
    records = list(valid_record_by_key.values())
    records.sort(key=lambda item: (str(item["llm_model"]), str(item["split_name"]), str(item["sample_id"])))
    for record in records:
        parsed_factors = record["parsed_factors"]
        texts.extend(str(factor_text) for factor_text in parsed_factors)
    _migrate_partial_embeddings_if_needed(
        partial_file_path=partial_file_path,
        batch_directory_path=batch_directory_path,
        batch_size=config.embedding.batch_size,
    )
    _write_embedding_batches(
        config=config,
        texts=texts,
        batch_directory_path=batch_directory_path,
    )
    embeddings = _load_embedding_batches(
        batch_directory_path=batch_directory_path,
        total_text_count=len(texts),
        batch_size=config.embedding.batch_size,
    )
    if embeddings.shape[0] != len(records) * 5:
        raise RuntimeError("Embedded factor count does not match record count.")
    embeddings = embeddings.reshape(len(records), 5, embeddings.shape[-1])
    sample_ids = np.asarray([str(record["sample_id"]) for record in records])
    split_names = np.asarray([str(record["split_name"]) for record in records])
    llm_models = np.asarray([str(record["llm_model"]) for record in records])
    np.savez_compressed(
        output_file_path,
        embeddings=embeddings.astype(np.float32),
        sample_ids=sample_ids,
        split_names=split_names,
        llm_models=llm_models,
    )
    partial_file_path.unlink(missing_ok=True)


def _migrate_partial_embeddings_if_needed(
    partial_file_path: Path,
    batch_directory_path: Path,
    batch_size: int,
) -> None:
    """Convert the legacy sequential partial file into shard files once."""

    if not partial_file_path.exists():
        return
    partial_payload = torch.load(partial_file_path, map_location="cpu")
    embeddings = np.asarray(partial_payload["embeddings"], dtype=np.float32)
    for start_index in range(0, embeddings.shape[0], batch_size):
        shard_file_path = _batch_file_path(batch_directory_path, start_index)
        if shard_file_path.exists():
            continue
        batch_embeddings = embeddings[start_index : start_index + batch_size]
        np.save(shard_file_path, batch_embeddings)


def _write_embedding_batches(
    config: NewsFactorExperimentConfig,
    texts: list[str],
    batch_directory_path: Path,
) -> None:
    """Write missing embedding shards with bounded concurrency."""

    client = OpenAICompatibleClient(config.api)
    rate_limiter = RequestRateLimiter(config.embedding.requests_per_second)
    missing_batch_starts = [
        start_index
        for start_index in range(0, len(texts), config.embedding.batch_size)
        if not _batch_file_path(batch_directory_path, start_index).exists()
    ]
    if not missing_batch_starts:
        return

    def _embed_one_batch(start_index: int) -> tuple[int, np.ndarray]:
        """Embed one batch and return its start index with an array."""

        batch_texts = texts[start_index : start_index + config.embedding.batch_size]
        rate_limiter.wait_for_slot()
        embeddings = client.create_embeddings(
            model_name=config.embedding.model_name,
            input_texts=batch_texts,
        )
        return start_index, np.asarray(embeddings, dtype=np.float32)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=config.embedding.max_workers) as executor:
        future_to_start = {
            executor.submit(_embed_one_batch, start_index): start_index
            for start_index in missing_batch_starts
        }
        for future in as_completed(future_to_start):
            start_index, batch_embeddings = future.result()
            np.save(_batch_file_path(batch_directory_path, start_index), batch_embeddings)
            completed_count += int(batch_embeddings.shape[0])
            print(
                f"embedding_progress completed_missing_texts={completed_count} "
                f"missing_texts={len(missing_batch_starts) * config.embedding.batch_size} "
                f"batch_start={start_index}",
                flush=True,
            )


def _load_embedding_batches(
    batch_directory_path: Path,
    total_text_count: int,
    batch_size: int,
) -> np.ndarray:
    """Load embedding shard files in text order."""

    batch_arrays: list[np.ndarray] = []
    for start_index in range(0, total_text_count, batch_size):
        shard_file_path = _batch_file_path(batch_directory_path, start_index)
        if not shard_file_path.exists():
            raise RuntimeError(f"Missing embedding shard: {shard_file_path}")
        batch_arrays.append(np.load(shard_file_path))
    return np.concatenate(batch_arrays, axis=0)


def _batch_file_path(batch_directory_path: Path, start_index: int) -> Path:
    """Return the deterministic shard path for one batch start index."""

    return batch_directory_path / f"batch_{start_index:08d}.npy"
