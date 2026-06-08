"""Evaluation metrics tailored to highly-imbalanced fraud detection.

Accuracy is meaningless at a ~3.5 % base rate, so we report:

- **ROC-AUC** — ranking quality across all thresholds
- **AUC-PR** (average precision) — the primary metric under heavy imbalance
- **best F1** and its threshold — a balanced operating point
- **recall @ precision >= 0.90** — "if we only act on alerts we're 90 % sure of,
  what share of fraud do we still catch?" — the kind of number a fraud-ops team
  actually negotiates around
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float = 0.90) -> dict[str, float]:
    """Compute imbalance-aware classification metrics from predicted probabilities."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    roc_auc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # F1 across the curve; precision/recall have one more element than thresholds.
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    best_f1 = float(f1[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # Highest recall achievable while keeping precision >= min_precision.
    feasible = precision[:-1] >= min_precision
    recall_at_min_precision = float(recall[:-1][feasible].max()) if feasible.any() else 0.0

    return {
        "roc_auc": float(roc_auc),
        "auc_pr": float(auc_pr),
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "recall_at_p90": recall_at_min_precision,
    }
