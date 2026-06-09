"""Model loading and single-transaction inference for the API.

The trained artefact (``models/xgboost.joblib``) contains:
  - ``model``          : fitted XGBClassifier
  - ``feature_names``  : ordered list of columns the model expects
  - ``category_maps``  : {col: {category_str -> int_code}} for label encoding
  - ``best_threshold`` : F1-optimal decision threshold from training

Inference pipeline for a single transaction
-------------------------------------------
1. Accept a dict of raw transaction fields.
2. Build a single-row DataFrame aligned to ``feature_names`` (NaN for
   any missing column — XGBoost handles this natively).
3. Apply the same deterministic transformations used during training
   (log1p, hour-of-day, day-of-week, n_missing, categorical encoding).
4. Call ``predict_proba`` and threshold the probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "xgboost"


class ModelBundle:
    """Wraps a trained model with its feature contract."""

    def __init__(self, bundle: dict[str, Any], model_name: str) -> None:
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.category_maps: dict[str, dict[str, int]] = bundle.get("category_maps", {})
        self.best_threshold: float = bundle.get("best_threshold", 0.5)
        self.model_name = model_name

    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Return fraud probability and binary label for one transaction."""
        row = self._build_row(transaction)
        prob = float(self.model.predict_proba(row)[:, 1][0])
        return {
            "fraud_probability": prob,
            "is_fraud": prob >= self.best_threshold,
            "threshold": self.best_threshold,
            "model_name": self.model_name,
        }

    def _build_row(self, transaction: dict[str, Any]) -> pd.DataFrame:
        """Transform a raw transaction dict into a model-ready single-row DataFrame."""
        # Start from a frame with all expected columns initialised to NaN.
        row = pd.DataFrame([{col: np.nan for col in self.feature_names}])

        # Fill in whatever the caller provided.
        for col, val in transaction.items():
            if col in row.columns:
                row.at[0, col] = val

        # ---- Engineered features (same logic as build_features.py) ----
        amt = float(transaction.get("TransactionAmt", np.nan))
        row["TransactionAmt_log"] = np.log1p(amt) if not np.isnan(amt) else np.nan

        dt = transaction.get("TransactionDT")
        if dt is not None:
            row["dt_hour"] = (int(dt) // 3600) % 24
            row["dt_dow"] = (int(dt) // (3600 * 24)) % 7
        else:
            row["dt_hour"] = np.nan
            row["dt_dow"] = np.nan

        row["n_missing"] = row.isna().sum(axis=1).astype("int32")

        # ---- Categorical encoding ----
        for col, mapping in self.category_maps.items():
            if col in row.columns:
                raw = row.at[0, col]
                row.at[0, col] = mapping.get(raw, -1) if pd.notna(raw) else -1

        # ---- Align columns to training order ----
        row = row.reindex(columns=self.feature_names, fill_value=np.nan)
        return row


# ── Module-level singleton (loaded once at startup) ───────────────────────────

_bundle: ModelBundle | None = None


def load_model(model_name: str = _DEFAULT_MODEL_NAME) -> ModelBundle:
    """Load the model bundle from disk (cached after first call)."""
    global _bundle
    if _bundle is not None and _bundle.model_name == model_name:
        return _bundle

    model_path: Path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run `python -m src.training.train --model xgboost` first."
        )

    logger.info("loading model bundle from %s", model_path)
    raw = joblib.load(model_path)
    _bundle = ModelBundle(raw, model_name=model_name)
    logger.info(
        "model loaded: %s features, threshold=%.4f",
        len(_bundle.feature_names),
        _bundle.best_threshold,
    )
    return _bundle


def get_bundle() -> ModelBundle:
    """FastAPI dependency that returns the cached model bundle."""
    return load_model()
