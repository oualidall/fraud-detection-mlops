"""Unit tests for feature engineering and evaluation, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TARGET, engineer
from src.training.evaluate import evaluate


def _toy_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": rng.integers(0, 3600 * 24 * 30, size=n),
            "TransactionAmt": rng.gamma(2.0, 50.0, size=n),
            "ProductCD": rng.choice(["W", "C", "R"], size=n),
            "card4": rng.choice(["visa", "mastercard", None], size=n),
            TARGET: rng.integers(0, 2, size=n),
        }
    )


def test_engineer_adds_expected_features() -> None:
    out, _ = engineer(_toy_frame())
    for col in ["TransactionAmt_log", "dt_hour", "dt_dow", "n_missing"]:
        assert col in out.columns

    # hour-of-day and day-of-week stay within range
    assert out["dt_hour"].between(0, 23).all()
    assert out["dt_dow"].between(0, 6).all()
    # log1p is monotonic and finite
    assert np.isfinite(out["TransactionAmt_log"]).all()


def test_engineer_label_encodes_categoricals() -> None:
    out, maps = engineer(_toy_frame())
    # object columns should now be integer codes (NaN -> -1)
    assert out["ProductCD"].dtype.kind in "iu"
    assert out["card4"].dtype.kind in "iu"
    assert (out["card4"] == -1).any()  # the None category becomes -1
    # category_maps must contain the encoded columns
    assert "ProductCD" in maps
    assert "card4" in maps


def test_engineer_is_sorted_by_time() -> None:
    out, _ = engineer(_toy_frame())
    assert out["TransactionDT"].is_monotonic_increasing


def test_evaluate_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate(y_true, y_proba)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)
    assert metrics["best_f1"] == pytest.approx(1.0)


def test_evaluate_returns_all_keys() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=100)
    y_proba = rng.random(size=100)
    metrics = evaluate(y_true, y_proba)
    assert set(metrics) == {"roc_auc", "auc_pr", "best_f1", "best_threshold", "recall_at_p90"}
