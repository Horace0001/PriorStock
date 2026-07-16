"""Sample-pool construction for LLM news-factor experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from priorstock.config import get_market_artifact_root
from priorstock.news_factors.config import NewsFactorExperimentConfig
from priorstock.news_factors.news_text import RawNewsRepository
from scripts.run_train_and_evaluate_ohlcv124_group_token_mixer_attention_side_adapter_single_logit import (
    PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset,
    _build_model,
    _load_config_with_single_logit_extensions,
)


def build_llm_factor_sample_pool(
    config: NewsFactorExperimentConfig,
    output_file_path: Path,
    cache_directory: Path,
) -> None:
    """Build a stratified sample pool with base-model logits and recent raw news."""

    if output_file_path.exists():
        return
    cache_directory.mkdir(parents=True, exist_ok=True)
    (
        experiment_config,
        attention_side_adapter_config,
        objective_config,
        abj_logit_decomposition_config,
    ) = _load_config_with_single_logit_extensions(
        config.base_model.config_file,
        cache_directory / "wide_base_config_loader",
    )
    model = _build_model(
        experiment_config,
        attention_side_adapter_config,
        abj_logit_decomposition_config,
    )
    checkpoint = torch.load(config.base_model.checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    news_repository = RawNewsRepository(config.news_text)
    all_candidate_rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        dataset = PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset(
            experiment_config,
            split_name,
            objective_config,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=config.base_model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        split_logits = _collect_base_logits(model, data_loader)
        split_rows = _build_split_candidate_rows(
            dataset=dataset,
            split_name=split_name,
            logits=split_logits,
            news_repository=news_repository,
            config=config,
        )
        all_candidate_rows.extend(split_rows)
    candidate_frame = pd.DataFrame(all_candidate_rows)
    if candidate_frame.empty:
        raise RuntimeError("No candidate sample has significant return and recent news.")
    candidate_pool_file_path = output_file_path.with_name("candidate_pool.jsonl")
    with candidate_pool_file_path.open("w", encoding="utf-8") as file_handle:
        for record in candidate_frame.to_dict(orient="records"):
            file_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    selected_frames = []
    for split_name, split_size in config.sampling.split_sizes.items():
        split_frame = candidate_frame.loc[candidate_frame["split_name"] == split_name].copy()
        selected_frames.append(
            _select_stratified_split_samples(
                split_frame=split_frame,
                target_size=split_size,
                max_samples_per_stock=config.sampling.max_samples_per_stock_per_split,
                random_seed=config.sampling.random_seed,
            )
        )
    selected_frame = pd.concat(selected_frames, ignore_index=True)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as file_handle:
        for record in selected_frame.to_dict(orient="records"):
            file_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _write_sample_pool_summary(
        selected_frame=selected_frame,
        candidate_frame=candidate_frame,
        output_file_path=output_file_path.with_suffix(".summary.json"),
        config=config,
    )


def _collect_base_logits(model: torch.nn.Module, data_loader: DataLoader) -> list[float]:
    """Run the frozen baseline model and collect one logit per sample."""

    logits: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            model_output = model(
                price_features=batch["price_features"],
                technical_indicator_features=batch["technical_indicator_features"],
                news_embeddings=batch["news_embeddings"],
                has_news=batch["has_news"],
                collect_trace_tensors=False,
            )
            logits.extend(model_output.logits.squeeze(dim=-1).cpu().tolist())
    return logits


def _build_split_candidate_rows(
    dataset: PriorStockOHLCV124GroupedReturnAwareSingleLogitDataset,
    split_name: str,
    logits: list[float],
    news_repository: RawNewsRepository,
    config: NewsFactorExperimentConfig,
) -> list[dict[str, Any]]:
    """Combine dataset rows, base logits, and recent-news windows for one split."""

    if len(dataset) != len(logits):
        raise RuntimeError("Dataset length and collected logit count differ.")
    candidate_rows: list[dict[str, Any]] = []
    for sample_index in range(len(dataset)):
        sample_row = dataset._sample_index_frame.iloc[sample_index]
        stock_id = str(sample_row["stock_id"])
        price_frame = dataset._load_price_frame(stock_id)
        target_row_index = int(sample_row["target_row_index"])
        candidate_dates = _build_candidate_news_dates(
            price_frame=price_frame,
            target_row_index=target_row_index,
            config=config,
        )
        news_window = news_repository.build_recent_news_window(stock_id, candidate_dates)
        if news_window.news_item_count <= 0:
            continue
        target_return = float(dataset._compute_target_return_for_row(sample_row))
        if abs(target_return) <= config.sampling.significant_return_absolute_threshold:
            continue
        hard_label = 1 if target_return > 0.0 else 0
        base_logit = float(logits[sample_index])
        base_probability = float(1.0 / (1.0 + np.exp(-base_logit)))
        base_prediction = 1 if base_logit > 0.0 else 0
        candidate_rows.append(
            {
                "sample_id": str(sample_row["sample_id"]),
                "split_name": split_name,
                "stock_id": stock_id,
                "stock_name": str(sample_row["stock_name"]),
                "industry": "Unknown",
                "target_trade_date": str(sample_row["target_trade_date"]),
                "target_return": target_return,
                "hard_label": hard_label,
                "base_logit": base_logit,
                "base_probability": base_probability,
                "base_correct": int(base_prediction == hard_label),
                "base_abs_logit": abs(base_logit),
                "news_item_count": int(news_window.news_item_count),
                "news_source_dates": list(news_window.source_dates),
                "formatted_recent_news": news_window.formatted_news,
                "news_text_source": config.news_text.news_text_source_field,
            }
        )
    return _assign_selection_buckets(pd.DataFrame(candidate_rows), config).to_dict(
        orient="records"
    )


def _build_candidate_news_dates(
    price_frame: pd.DataFrame,
    target_row_index: int,
    config: NewsFactorExperimentConfig,
) -> list[str]:
    """Return recent trading dates allowed for news input, nearest first."""

    end_index = target_row_index if config.sampling.exclude_target_date_news else target_row_index + 1
    start_index = max(0, end_index - config.sampling.recent_news_lookback_trading_days)
    candidate_dates = [
        str(trade_date)
        for trade_date in price_frame.iloc[start_index:end_index]["trade_date"].tolist()
    ]
    candidate_dates.reverse()
    return candidate_dates


def _assign_selection_buckets(
    candidate_frame: pd.DataFrame,
    config: NewsFactorExperimentConfig,
) -> pd.DataFrame:
    """Add base-confidence and news-volume buckets used by stratified sampling."""

    if candidate_frame.empty:
        return candidate_frame
    candidate_frame = candidate_frame.copy()
    candidate_frame["base_confidence_bucket"] = _safe_quantile_bucket(
        candidate_frame["base_abs_logit"],
        config.sampling.base_confidence_bucket_count,
    )
    candidate_frame["news_volume_bucket"] = _safe_quantile_bucket(
        candidate_frame["news_item_count"],
        config.sampling.news_volume_bucket_count,
    )
    return candidate_frame


def _safe_quantile_bucket(values: pd.Series, bucket_count: int) -> list[str]:
    """Assign quantile buckets while tolerating repeated values."""

    if bucket_count <= 1 or values.nunique(dropna=True) <= 1:
        return ["all" for _ in range(len(values))]
    ranked_values = values.rank(method="first")
    bucket_codes = pd.qcut(ranked_values, q=bucket_count, labels=False, duplicates="drop")
    return [f"q{int(code)}" for code in bucket_codes]


def _select_stratified_split_samples(
    split_frame: pd.DataFrame,
    target_size: int,
    max_samples_per_stock: int,
    random_seed: int,
) -> pd.DataFrame:
    """Select one split while preserving its original significant-sample label ratio."""

    if split_frame.shape[0] < target_size:
        raise RuntimeError(
            f"Split {split_frame['split_name'].iloc[0]} only has {split_frame.shape[0]} "
            f"eligible samples, below requested {target_size}."
        )
    rng = np.random.default_rng(random_seed)
    split_frame = split_frame.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    label_targets = _compute_label_targets(split_frame, target_size)
    selected_indices: list[int] = []
    stock_counts: dict[str, int] = {}
    for hard_label, label_target in label_targets.items():
        label_frame = split_frame.loc[split_frame["hard_label"] == hard_label]
        label_indices = _select_within_label_strata(
            split_frame=split_frame,
            label_frame=label_frame,
            target_size=label_target,
            max_samples_per_stock=max_samples_per_stock,
            stock_counts=stock_counts,
            rng=rng,
        )
        selected_indices.extend(label_indices)
    if len(selected_indices) < target_size:
        remaining_frame = split_frame.drop(index=selected_indices)
        for candidate_index, row in remaining_frame.iterrows():
            stock_id = str(row["stock_id"])
            if stock_counts.get(stock_id, 0) >= max_samples_per_stock:
                continue
            selected_indices.append(candidate_index)
            stock_counts[stock_id] = stock_counts.get(stock_id, 0) + 1
            if len(selected_indices) >= target_size:
                break
    if len(selected_indices) < target_size:
        raise RuntimeError("Could not satisfy split sample target under stock cap.")
    return split_frame.loc[selected_indices].sample(frac=1.0, random_state=random_seed).reset_index(
        drop=True
    )


def _compute_label_targets(split_frame: pd.DataFrame, target_size: int) -> dict[int, int]:
    """Compute integer label quotas from the eligible split's original label distribution."""

    label_counts = split_frame["hard_label"].value_counts().sort_index()
    raw_targets = {
        int(label): float(count) / float(split_frame.shape[0]) * float(target_size)
        for label, count in label_counts.items()
    }
    label_targets = {label: int(np.floor(raw_target)) for label, raw_target in raw_targets.items()}
    remaining_count = target_size - sum(label_targets.values())
    remainders = sorted(
        raw_targets.items(),
        key=lambda item: item[1] - np.floor(item[1]),
        reverse=True,
    )
    for label, _ in remainders[:remaining_count]:
        label_targets[label] += 1
    return label_targets


def _select_within_label_strata(
    split_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    target_size: int,
    max_samples_per_stock: int,
    stock_counts: dict[str, int],
    rng: np.random.Generator,
) -> list[int]:
    """Select samples within a fixed label across diagnostic strata."""

    strata_columns = [
        "base_correct",
        "base_confidence_bucket",
        "news_volume_bucket",
    ]
    grouped_indices = {
        stratum_key: list(group_frame.index)
        for stratum_key, group_frame in label_frame.groupby(strata_columns, sort=False)
    }
    for indices in grouped_indices.values():
        rng.shuffle(indices)
    selected_indices: list[int] = []
    while len(selected_indices) < target_size and grouped_indices:
        for stratum_key in list(grouped_indices.keys()):
            indices = grouped_indices[stratum_key]
            while indices:
                candidate_index = indices.pop()
                stock_id = str(split_frame.loc[candidate_index, "stock_id"])
                if stock_counts.get(stock_id, 0) >= max_samples_per_stock:
                    continue
                selected_indices.append(candidate_index)
                stock_counts[stock_id] = stock_counts.get(stock_id, 0) + 1
                break
            if not indices:
                grouped_indices.pop(stratum_key, None)
            if len(selected_indices) >= target_size:
                break
    if len(selected_indices) < target_size:
        remaining_frame = label_frame.drop(index=selected_indices)
        for candidate_index, row in remaining_frame.iterrows():
            stock_id = str(row["stock_id"])
            if stock_counts.get(stock_id, 0) >= max_samples_per_stock:
                continue
            selected_indices.append(candidate_index)
            stock_counts[stock_id] = stock_counts.get(stock_id, 0) + 1
            if len(selected_indices) >= target_size:
                break
    return selected_indices


def _write_sample_pool_summary(
    selected_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    output_file_path: Path,
    config: NewsFactorExperimentConfig,
) -> None:
    """Persist candidate and selected sample distribution diagnostics."""

    summary = {
        "config": {
            "sampling": asdict(config.sampling),
            "news_text": {
                **asdict(config.news_text),
                "raw_news_directory": str(config.news_text.raw_news_directory),
            },
        },
        "candidate_count": int(candidate_frame.shape[0]),
        "selected_count": int(selected_frame.shape[0]),
        "selected_by_split": selected_frame.groupby("split_name").size().to_dict(),
        "selected_label_mean_by_split": selected_frame.groupby("split_name")[
            "hard_label"
        ].mean().to_dict(),
        "selected_news_count_mean_by_split": selected_frame.groupby("split_name")[
            "news_item_count"
        ].mean().to_dict(),
        "selected_base_correct_mean_by_split": selected_frame.groupby("split_name")[
            "base_correct"
        ].mean().to_dict(),
    }
    with output_file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
