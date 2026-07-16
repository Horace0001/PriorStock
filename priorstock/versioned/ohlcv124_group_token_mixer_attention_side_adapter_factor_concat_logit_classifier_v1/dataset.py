"""Single-logit OHLCV-124 dataset with LLM-factor embedding lookup."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run_train_and_evaluate_ohlcv124_group_token_mixer_attention_side_adapter_single_logit import (
    PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset,
    SingleLogitBinaryObjectiveConfig,
)


@dataclass(frozen=True)
class FactorEmbeddingCacheMetadata:
    """Metadata for a memory-mapped factor embedding cache."""

    embedding_file_paths: tuple[Path, ...]
    shard_row_offsets: tuple[int, ...]
    index_file_path: Path
    record_count: int
    factor_count: int
    embedding_dimension: int
    embedding_model_name: str


def load_factor_embedding_cache_metadata(cache_directory: Path) -> FactorEmbeddingCacheMetadata:
    """Load and validate the factor embedding cache metadata file."""

    metadata_file_path = cache_directory / "factor_embedding_metadata.json"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Factor embedding metadata file does not exist: {metadata_file_path}")
    with metadata_file_path.open("r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)
    raw_embedding_file_paths = metadata.get("embedding_file_paths")
    if raw_embedding_file_paths is None:
        raw_embedding_file_paths = [metadata["embedding_file_path"]]
    embedding_file_paths = tuple(Path(str(value)) for value in raw_embedding_file_paths)
    index_file_path = Path(str(metadata["index_file_path"]))
    embedding_file_paths = tuple(
        path if path.is_absolute() else (metadata_file_path.parent / path).resolve()
        for path in embedding_file_paths
    )
    if not index_file_path.is_absolute():
        index_file_path = (metadata_file_path.parent / index_file_path).resolve()
    return FactorEmbeddingCacheMetadata(
        embedding_file_paths=embedding_file_paths,
        shard_row_offsets=tuple(int(value) for value in metadata.get("shard_row_offsets", [0])),
        index_file_path=index_file_path,
        record_count=int(metadata["record_count"]),
        factor_count=int(metadata["factor_count"]),
        embedding_dimension=int(metadata["embedding_dimension"]),
        embedding_model_name=str(metadata["embedding_model_name"]),
    )


class PriorStockOHLCV124GroupedFactorEmbeddingSingleLogitDataset(
    PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset
):
    """Dataset wrapper that adds five LLM-factor embeddings per stock-date sample."""

    def __init__(
        self,
        experiment_config,
        objective_config: SingleLogitBinaryObjectiveConfig,
        split_name: str,
        factor_embedding_cache_directory: Path,
        expected_factor_count: int,
        expected_factor_embedding_dim: int,
        should_load_factor_embeddings: bool = True,
    ) -> None:
        """Initialize the base dataset and memory-map the factor embedding cache."""

        super().__init__(
            experiment_config=experiment_config,
            split_name=split_name,
            objective_config=objective_config,
        )
        self._expected_factor_count = expected_factor_count
        self._expected_factor_embedding_dim = expected_factor_embedding_dim
        self._should_load_factor_embeddings = should_load_factor_embeddings
        if not should_load_factor_embeddings:
            self._cache_metadata = None
            self._factor_embeddings = None
            self._sample_id_to_row_index = {}
            return
        self._cache_metadata = load_factor_embedding_cache_metadata(
            factor_embedding_cache_directory
        )
        self._validate_cache_contract()
        self._factor_embedding_shards = tuple(
            np.load(path, mmap_mode="r")
            for path in self._cache_metadata.embedding_file_paths
        )
        self._sample_id_to_row_index = self._load_sample_id_to_row_index(
            self._cache_metadata.index_file_path
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample with factor embeddings and factor masks added."""

        sample = super().__getitem__(index)
        if not self._should_load_factor_embeddings:
            sample["factor_embeddings"] = torch.zeros(
                (self._expected_factor_count, 1),
                dtype=torch.float32,
            )
            sample["has_factors"] = torch.zeros(
                (self._expected_factor_count,),
                dtype=torch.bool,
            )
            return sample
        sample_id = str(self._sample_index_frame.iloc[index]["sample_id"])
        row_index = self._sample_id_to_row_index.get(sample_id)
        if row_index is None:
            factor_embeddings = torch.zeros(
                (self._expected_factor_count, self._expected_factor_embedding_dim),
                dtype=torch.float32,
            )
            has_factors = torch.zeros((self._expected_factor_count,), dtype=torch.bool)
        else:
            shard_index = bisect_right(self._cache_metadata.shard_row_offsets, row_index) - 1
            shard_row_index = row_index - self._cache_metadata.shard_row_offsets[shard_index]
            factor_embeddings = torch.tensor(
                np.array(
                    self._factor_embedding_shards[shard_index][shard_row_index],
                    dtype=np.float32,
                    copy=True,
                ),
                dtype=torch.float32,
            )
            has_factors = torch.ones((self._expected_factor_count,), dtype=torch.bool)
        sample["factor_embeddings"] = factor_embeddings
        sample["has_factors"] = has_factors
        return sample

    def _validate_cache_contract(self) -> None:
        """Validate that the cache dimensions match this experiment."""

        if self._cache_metadata.factor_count != self._expected_factor_count:
            raise ValueError(
                f"Expected {self._expected_factor_count} factors, got {self._cache_metadata.factor_count}."
            )
        if self._cache_metadata.embedding_dimension != self._expected_factor_embedding_dim:
            raise ValueError(
                "Expected factor embedding dimension "
                f"{self._expected_factor_embedding_dim}, got {self._cache_metadata.embedding_dimension}."
            )
        if len(self._cache_metadata.embedding_file_paths) != len(
            self._cache_metadata.shard_row_offsets
        ):
            raise ValueError("Factor embedding shard paths and offsets do not match.")
        for embedding_file_path in self._cache_metadata.embedding_file_paths:
            if not embedding_file_path.exists():
                raise FileNotFoundError(
                    f"Factor embedding file does not exist: {embedding_file_path}"
                )
        if not self._cache_metadata.index_file_path.exists():
            raise FileNotFoundError(
                f"Factor embedding index file does not exist: {self._cache_metadata.index_file_path}"
            )

    @staticmethod
    def _load_sample_id_to_row_index(index_file_path: Path) -> dict[str, int]:
        """Load a sample ID to factor-cache row-index mapping."""

        sample_id_to_row_index: dict[str, int] = {}
        with index_file_path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                if sample_id in sample_id_to_row_index:
                    raise ValueError(f"Duplicate factor cache sample_id: {sample_id}")
                sample_id_to_row_index[sample_id] = int(record["row_index"])
        return sample_id_to_row_index
