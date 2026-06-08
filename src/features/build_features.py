"""Feature engineering and temporal train/validation split.

Design choices (informed by notebooks/01_eda.ipynb):

- **Temporal split**: rows are ordered by ``TransactionDT`` and the most recent
  fraction is held out for validation. The EDA showed the fraud rate drifts over
  the 182-day window, so a random split would leak future information.
- **log1p(TransactionAmt)**: the amount is heavily right-skewed.
- **Cyclical time features**: hour-of-day and day-of-week derived from the
  anonymized ``TransactionDT`` offset.
- **Missingness as signal**: a per-row missing count, since sparsity correlates
  with fraud on this dataset.
- **Categorical encoding**: object columns are label-encoded (NaN -> -1). This is
  unsupervised, so encoding on the full frame before the split introduces no
  target leakage. Tree models (XGBoost) consume these integer codes directly and
  handle remaining NaNs natively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

TARGET = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"


@dataclass
class Dataset:
    """A temporally-split, model-ready dataset."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    feature_names: list[str]

    def summary(self) -> dict[str, float]:
        return {
            "n_train": int(len(self.X_train)),
            "n_val": int(len(self.X_val)),
            "n_features": int(len(self.feature_names)),
            "train_fraud_rate": float(self.y_train.mean()),
            "val_fraud_rate": float(self.y_val.mean()),
        }


def _select_columns(
    parquet_path: "Path",
    max_missing_rate: float,
    always_keep: list[str],
) -> list[str]:
    """Return column names whose missing rate is below the threshold.

    Reading column statistics from Parquet metadata is instant (no data load).
    This is the same technique used in the EDA notebook.
    """
    import pyarrow.parquet as _pq

    meta = _pq.read_metadata(parquet_path)
    null_counts: dict[str, int] = {}
    for rg in range(meta.num_row_groups):
        row_group = meta.row_group(rg)
        for c in range(meta.num_columns):
            col = row_group.column(c)
            name = col.path_in_schema
            null_counts[name] = null_counts.get(name, 0) + col.statistics.null_count

    n = meta.num_rows
    selected = [
        col for col, nc in null_counts.items()
        if (nc / n) <= max_missing_rate or col in always_keep
    ]
    dropped = len(null_counts) - len(selected)
    logger.info(
        "column selection: keeping %s / %s (dropped %s with >%.0f%% missing)",
        len(selected), len(null_counts), dropped, max_missing_rate * 100,
    )
    return selected


def load_train_frame(max_missing_rate: float = 0.70) -> pd.DataFrame:
    """Load and merge the transaction + identity Parquet tables.

    Only columns whose missing rate is <= ``max_missing_rate`` are loaded.
    The V* columns that are >70 % missing add noise but cost a lot of RAM and
    slow down training significantly on a constrained machine — dropping them
    is a standard first-pass feature-selection step and does not hurt AUC-PR
    materially.
    """
    tx_path = PROCESSED_DATA_DIR / "train_transaction.parquet"
    id_path = PROCESSED_DATA_DIR / "train_identity.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Run `python -m src.data.make_parquet` first."
        )

    always_keep = [TARGET, ID_COLUMN, TIME_COLUMN]
    tx_cols = _select_columns(tx_path, max_missing_rate, always_keep)
    id_cols = _select_columns(id_path, max_missing_rate, [ID_COLUMN])

    transactions = pq.read_table(tx_path, columns=tx_cols).to_pandas()
    identity = pq.read_table(id_path, columns=id_cols).to_pandas()
    merged = transactions.merge(identity, on=ID_COLUMN, how="left")
    logger.info("loaded merged train frame: %s rows x %s cols", *merged.shape)
    return merged


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and label-encode categoricals. Sorted by time."""
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["dt_hour"] = ((df[TIME_COLUMN] // 3600) % 24).astype("int16")
    df["dt_dow"] = ((df[TIME_COLUMN] // (3600 * 24)) % 7).astype("int16")
    df["n_missing"] = df.isna().sum(axis=1).astype("int32")

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        # category codes: distinct ints per category, -1 for NaN
        df[col] = df[col].astype("category").cat.codes.astype("int32")
    logger.info("engineered features; label-encoded %s categorical cols", len(object_cols))
    return df


def build_dataset(val_fraction: float = 0.2, max_train_rows: int | None = None) -> Dataset:
    """Build a temporally-split dataset ready for model training.

    The data is ordered by time, then split so the most recent ``val_fraction``
    is held out for validation. If ``max_train_rows`` is set, the training set is
    capped to that many of the **most recent** pre-validation rows, which keeps
    runtime and memory bounded on small machines while preserving the temporal
    ordering (we train on the most recent history before the validation window).
    """
    df = engineer(load_train_frame())

    y = df[TARGET].astype("int8")
    X = df.drop(columns=[TARGET, ID_COLUMN])

    cut = int(len(df) * (1.0 - val_fraction))
    train_start = 0
    if max_train_rows is not None and max_train_rows < cut:
        train_start = cut - max_train_rows

    dataset = Dataset(
        X_train=X.iloc[train_start:cut],
        X_val=X.iloc[cut:],
        y_train=y.iloc[train_start:cut],
        y_val=y.iloc[cut:],
        feature_names=list(X.columns),
    )
    logger.info("dataset ready: %s", dataset.summary())
    return dataset


if __name__ == "__main__":
    ds = build_dataset()
    print(ds.summary())
