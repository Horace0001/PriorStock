"""Evaluation metrics for PriorStock classification experiments."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

from priorstock.config import EvaluationConfig


def compute_classification_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    evaluation_config: EvaluationConfig,
) -> dict[str, float]:
    """Compute the exact evaluation metrics required by the technical specification."""

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
        "mcc": float(matthews_corrcoef(true_labels, predicted_labels)),
        "up_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                labels=[evaluation_config.up_class_label_index],
                average="macro",
                zero_division=0,
            )
        ),
        "up_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                labels=[evaluation_config.up_class_label_index],
                average="macro",
                zero_division=0,
            )
        ),
    }
