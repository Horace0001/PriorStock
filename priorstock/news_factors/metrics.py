"""Metrics for binary news-factor probes."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_binary_probe_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mcc_score_weight: float,
    balanced_accuracy_score_weight: float,
) -> dict[str, float]:
    """Compute hard-label and probability metrics for a binary probe."""

    predictions = (probabilities >= 0.5).astype(np.int64)
    if len(np.unique(labels)) < 2:
        auc_value = 0.5
    else:
        auc_value = float(roc_auc_score(labels, probabilities))
    mcc_value = float(matthews_corrcoef(labels, predictions))
    balanced_accuracy_value = float(balanced_accuracy_score(labels, predictions))
    normalized_mcc = (mcc_value + 1.0) / 2.0
    selection_score = (
        mcc_score_weight * normalized_mcc
        + balanced_accuracy_score_weight * balanced_accuracy_value
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": balanced_accuracy_value,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": mcc_value,
        "roc_auc": auc_value,
        "prediction_up_ratio": float(predictions.mean()) if predictions.size else 0.0,
        "selection_score": float(selection_score),
    }
