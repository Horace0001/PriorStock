"""Structured runtime logging and diagnostics helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from priorstock.utils.io import ensure_directory


def configure_logger(logger_name: str) -> logging.Logger:
    """Create one console logger with a stable message format."""

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def write_tensor_diagnostics(diagnostics_file_path: Path, diagnostics_payload: dict[str, Any]) -> None:
    """Write one JSON diagnostics payload for tensor shapes and scalar statistics."""

    ensure_directory(diagnostics_file_path.parent)
    with diagnostics_file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(diagnostics_payload, file_handle, ensure_ascii=False, indent=2)


def _downsample_flat_tensor(flat_tensor: torch.Tensor, max_sample_count: int) -> torch.Tensor:
    """Take one deterministic strided sample from a flattened tensor."""

    if flat_tensor.numel() <= max_sample_count:
        return flat_tensor
    step_size = max(1, flat_tensor.numel() // max_sample_count)
    return flat_tensor[::step_size][:max_sample_count]


def sample_tensor_values(tensor: torch.Tensor, max_sample_count: int) -> torch.Tensor:
    """Detach one tensor, flatten it, deterministically downsample it, and move it to CPU."""

    flattened_tensor = tensor.detach().reshape(-1)
    sampled_tensor = _downsample_flat_tensor(flattened_tensor, max_sample_count)
    return sampled_tensor.to(dtype=torch.float32, device="cpu")


def summarize_tensor_distribution(tensor: torch.Tensor, max_sample_count: int) -> dict[str, Any]:
    """Build one robust scalar summary for a tensor distribution."""

    flattened_tensor = tensor.detach().reshape(-1)
    element_count = int(flattened_tensor.numel())
    if element_count == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "element_count": 0,
        }

    flattened_tensor_cpu = flattened_tensor.to(dtype=torch.float32, device="cpu")
    nan_count = int(torch.isnan(flattened_tensor_cpu).sum().item())
    inf_count = int(torch.isinf(flattened_tensor_cpu).sum().item())
    finite_mask = torch.isfinite(flattened_tensor_cpu)
    finite_tensor = flattened_tensor_cpu[finite_mask]
    finite_element_count = int(finite_tensor.numel())
    if finite_element_count == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "element_count": element_count,
            "finite_element_count": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

    sampled_finite_tensor = _downsample_flat_tensor(finite_tensor, max_sample_count)
    quantile_levels = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float32)
    quantile_values = torch.quantile(sampled_finite_tensor, quantile_levels)
    zero_fraction = float((finite_tensor == 0).float().mean().item())
    positive_fraction = float((finite_tensor > 0).float().mean().item())
    negative_fraction = float((finite_tensor < 0).float().mean().item())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "element_count": element_count,
        "finite_element_count": finite_element_count,
        "sampled_element_count": int(sampled_finite_tensor.numel()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mean": float(sampled_finite_tensor.mean().item()),
        "std": float(sampled_finite_tensor.std(unbiased=False).item()),
        "min": float(sampled_finite_tensor.min().item()),
        "max": float(sampled_finite_tensor.max().item()),
        "abs_mean": float(sampled_finite_tensor.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(sampled_finite_tensor).item()),
        "zero_fraction": zero_fraction,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "p01": float(quantile_values[0].item()),
        "p05": float(quantile_values[1].item()),
        "p25": float(quantile_values[2].item()),
        "p50": float(quantile_values[3].item()),
        "p75": float(quantile_values[4].item()),
        "p95": float(quantile_values[5].item()),
        "p99": float(quantile_values[6].item()),
    }
