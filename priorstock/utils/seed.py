"""Deterministic seed helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(random_seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

