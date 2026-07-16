"""Dataset for the OHLCV-124 grouped-fusion experiment variant."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from priorstock.config import ExperimentConfig, get_market_artifact_root
from priorstock.exceptions import ArtifactConsistencyError
from priorstock.utils.io import path_from_serialized_value
from priorstock.versioned.ohlcv124_group_fusion_v1.feature_schema import ALL_FEATURE_NAMES
from priorstock.versioned.ohlcv124_group_fusion_v1.features import build_ohlcv124_feature_frame


_FEATURE_CACHE_FORMAT_VERSION = "ohlcv124_feature_frames_v1_20260714"
_FEATURE_SET_NAME = "expert_ohlcv124"
_PROCESS_PRICE_FRAME_CACHE: dict[str, pd.DataFrame] = {}
_PROCESS_FEATURE_FRAME_CACHE: dict[tuple[str, str, float], pd.DataFrame] = {}


def _read_csv_with_out_of_memory_fallback(csv_file_path: Path) -> pd.DataFrame:
    """Read one CSV, retrying with the Python parser if pandas hits a parser memory error."""

    try:
        return pd.read_csv(csv_file_path)
    except pd.errors.ParserError as error:
        if "out of memory" not in str(error).lower():
            raise
        return pd.read_csv(csv_file_path, engine="python")


def _normalize_price_window(price_window_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply the project-standard per-window OHLCV normalization for trace comparability."""

    reference_close_value = price_window_tensor[0, 3]
    normalized_price_tensor = torch.empty_like(price_window_tensor)
    normalized_price_tensor[:, 0:4] = (price_window_tensor[:, 0:4] / (reference_close_value + epsilon)) - 1.0
    volume_tensor = price_window_tensor[:, 4]
    normalized_price_tensor[:, 4] = (volume_tensor - volume_tensor.mean()) / (
        volume_tensor.std(unbiased=False) + epsilon
    )
    return normalized_price_tensor


class PriorStockOHLCV124GroupedDataset(Dataset):
    """Load supervised samples with on-the-fly OHLCV-124 feature windows."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        split_name: str,
    ) -> None:
        """Create the dataset from prepared price artifacts and split index files."""

        self._experiment_config = experiment_config
        self._feature_set_name = _FEATURE_SET_NAME
        market_artifact_root = get_market_artifact_root(experiment_config)
        self._sample_index_frame = pd.read_csv(
            market_artifact_root / experiment_config.paths.sample_index_subdirectory / f"{split_name}.csv"
        )
        self._manifest_frame = pd.read_csv(market_artifact_root / "stock_manifest.csv")
        self._manifest_by_stock_id = {
            row["stock_id"]: row for _, row in self._manifest_frame.iterrows()
        }
        self._price_frame_cache: dict[str, pd.DataFrame] = {}
        self._feature_frame_cache: dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""

        return int(self._sample_index_frame.shape[0])

    def _load_price_frame(self, stock_id: str) -> pd.DataFrame:
        """Lazily load one processed price table."""

        if stock_id not in self._price_frame_cache:
            manifest_row = self._manifest_by_stock_id[stock_id]
            price_file_path = path_from_serialized_value(
                manifest_row["processed_price_file_path"]
            ).resolve()
            process_cache_key = str(price_file_path)
            if process_cache_key not in _PROCESS_PRICE_FRAME_CACHE:
                _PROCESS_PRICE_FRAME_CACHE[process_cache_key] = (
                    _read_csv_with_out_of_memory_fallback(price_file_path)
                )
            self._price_frame_cache[stock_id] = _PROCESS_PRICE_FRAME_CACHE[
                process_cache_key
            ]
        return self._price_frame_cache[stock_id]

    def _resolve_feature_cache_paths(self, stock_id: str) -> tuple[Path, Path]:
        """Return data and metadata paths for one deterministic feature-frame cache."""

        market_artifact_root = get_market_artifact_root(self._experiment_config)
        cache_directory = (
            market_artifact_root
            / _FEATURE_CACHE_FORMAT_VERSION
            / self._feature_set_name
        )
        cache_directory.mkdir(parents=True, exist_ok=True)
        return (
            cache_directory / f"{stock_id}.pkl",
            cache_directory / f"{stock_id}.metadata.json",
        )

    def _load_valid_disk_feature_cache(
        self,
        stock_id: str,
        price_file_path: Path,
    ) -> pd.DataFrame | None:
        """Load a local cache only when its source-file and schema contract still match."""

        cache_file_path, metadata_file_path = self._resolve_feature_cache_paths(stock_id)
        if not cache_file_path.exists() or not metadata_file_path.exists():
            return None
        try:
            metadata = json.loads(metadata_file_path.read_text(encoding="utf-8"))
            source_stat = price_file_path.stat()
            if metadata.get("cache_format_version") != _FEATURE_CACHE_FORMAT_VERSION:
                return None
            if metadata.get("feature_set_name") != self._feature_set_name:
                return None
            if int(metadata.get("source_file_size", -1)) != int(source_stat.st_size):
                return None
            if int(metadata.get("source_file_mtime_ns", -1)) != int(source_stat.st_mtime_ns):
                return None
            feature_frame = pd.read_pickle(cache_file_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if feature_frame.shape[1] != self._experiment_config.model.technical_indicator_feature_count:
            return None
        return feature_frame

    def _write_disk_feature_cache(
        self,
        stock_id: str,
        price_file_path: Path,
        feature_frame: pd.DataFrame,
    ) -> None:
        """Atomically persist a feature frame so later stages avoid identical recomputation."""

        cache_file_path, metadata_file_path = self._resolve_feature_cache_paths(stock_id)
        temporary_cache_file_path = cache_file_path.with_suffix(".tmp.pkl")
        temporary_metadata_file_path = metadata_file_path.with_suffix(".tmp.json")
        feature_frame.to_pickle(temporary_cache_file_path)
        source_stat = price_file_path.stat()
        temporary_metadata_file_path.write_text(
            json.dumps(
                {
                    "cache_format_version": _FEATURE_CACHE_FORMAT_VERSION,
                    "feature_set_name": self._feature_set_name,
                    "source_file_path": str(price_file_path),
                    "source_file_size": int(source_stat.st_size),
                    "source_file_mtime_ns": int(source_stat.st_mtime_ns),
                    "row_count": int(feature_frame.shape[0]),
                    "feature_count": int(feature_frame.shape[1]),
                    "feature_names": list(feature_frame.columns),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_cache_file_path, cache_file_path)
        os.replace(temporary_metadata_file_path, metadata_file_path)

    def _load_feature_frame(self, stock_id: str) -> pd.DataFrame:
        """Lazily compute and cache one stock's OHLCV-124 feature table."""

        if stock_id not in self._feature_frame_cache:
            manifest_row = self._manifest_by_stock_id[stock_id]
            price_file_path = path_from_serialized_value(
                manifest_row["processed_price_file_path"]
            ).resolve()
            process_cache_key = (
                self._feature_set_name,
                str(price_file_path),
                float(self._experiment_config.indicator.epsilon),
            )
            feature_frame = _PROCESS_FEATURE_FRAME_CACHE.get(process_cache_key)
            if feature_frame is None:
                feature_frame = self._load_valid_disk_feature_cache(
                    stock_id,
                    price_file_path,
                )
            if feature_frame is None:
                price_frame = self._load_price_frame(stock_id)
                feature_frame = build_ohlcv124_feature_frame(
                    price_frame,
                    self._experiment_config.indicator.epsilon,
                )
                self._write_disk_feature_cache(
                    stock_id,
                    price_file_path,
                    feature_frame,
                )
            if feature_frame.shape[1] != self._experiment_config.model.technical_indicator_feature_count:
                raise ArtifactConsistencyError(
                    f"OHLCV-124 feature width for stock '{stock_id}' does not match "
                    "model.technical_indicator_feature_count."
                )
            missing_feature_names = set(ALL_FEATURE_NAMES) - set(feature_frame.columns)
            if missing_feature_names:
                raise ArtifactConsistencyError(
                    f"OHLCV-124 feature table for stock '{stock_id}' is missing features: "
                    f"{', '.join(sorted(missing_feature_names))}."
                )
            _PROCESS_FEATURE_FRAME_CACHE[process_cache_key] = feature_frame
            self._feature_frame_cache[stock_id] = feature_frame
        return self._feature_frame_cache[stock_id]

    def __getitem__(self, sample_index: int) -> dict[str, torch.Tensor | str]:
        """Load one normalized price window, OHLCV-124 window, and label."""

        sample_row = self._sample_index_frame.iloc[sample_index]
        stock_id = str(sample_row["stock_id"])
        price_frame = self._load_price_frame(stock_id)
        feature_frame = self._load_feature_frame(stock_id)

        window_start_index = int(sample_row["window_start_index"])
        window_end_index = int(sample_row["window_end_index"])
        price_window_frame = price_frame.iloc[window_start_index : window_end_index + 1]
        raw_price_window_tensor = torch.tensor(
            price_window_frame[
                ["effective_open", "effective_high", "effective_low", "effective_close", "volume"]
            ].to_numpy(dtype="float32")
        )
        normalized_price_window_tensor = _normalize_price_window(
            raw_price_window_tensor,
            self._experiment_config.indicator.epsilon,
        )

        feature_window_tensor = torch.tensor(
            feature_frame.iloc[window_start_index : window_end_index + 1].to_numpy(dtype="float32")
        )
        if feature_window_tensor.shape[-1] != self._experiment_config.model.technical_indicator_feature_count:
            raise ArtifactConsistencyError(
                f"OHLCV-124 feature window for stock '{stock_id}' has unexpected final dimension."
            )

        sequence_length = feature_window_tensor.shape[0]
        return {
            "sample_id": str(sample_row["sample_id"]),
            "stock_id": stock_id,
            "target_trade_date": str(sample_row["target_trade_date"]),
            "price_features": normalized_price_window_tensor,
            "technical_indicator_features": feature_window_tensor,
            "news_embeddings": torch.zeros((sequence_length, 1), dtype=torch.float32),
            "has_news": torch.zeros((sequence_length, 1), dtype=torch.float32),
            "label": torch.tensor(int(sample_row["label"]), dtype=torch.long),
        }
