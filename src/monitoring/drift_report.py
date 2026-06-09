"""Data drift report using Evidently AI.

Compares a reference sample (training data) against a current sample
(the most recent validation window) and produces an HTML report.

Usage::

    python -m src.monitoring.drift_report

The report is written to ``docs/drift_report.html`` and can be opened
directly in any browser — no server required.

Why drift detection matters
---------------------------
The EDA showed the daily fraud rate and transaction volumes are not
stationary over the 182-day window. In production, if the feature
distribution shifts (e.g. seasonal spending patterns, new fraud
techniques), the model's AUC-PR degrades silently. Evidently lets us
quantify *how much* the current data has drifted from the training
reference, so we can decide when to retrain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "drift_report.html"

# Features to include in the drift report — a representative subset
# that covers the main signal types (amount, time, card type, target).
DRIFT_FEATURES = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "TransactionDT",
    "isFraud",
]

# Sample sizes: large enough to be statistically meaningful, small
# enough to keep the report generation fast.
REFERENCE_SAMPLE = 20_000
CURRENT_SAMPLE = 10_000


def _load_sample(n_rows: int, skip_rows: int = 0) -> pd.DataFrame:
    """Load a time-ordered sample from the transaction parquet."""
    tx_path = PROCESSED_DATA_DIR / "train_transaction.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Run `python -m src.data.make_parquet` first."
        )
    cols = [c for c in DRIFT_FEATURES
            if c in pq.read_schema(tx_path).names]
    full = pq.read_table(tx_path, columns=cols).to_pandas()
    full = full.sort_values("TransactionDT").reset_index(drop=True)
    end = skip_rows + n_rows
    return full.iloc[skip_rows:end]


def generate_report(output_path: Path = REPORT_PATH) -> Path:
    """Generate the drift report and return the output path."""
    try:
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise ImportError(
            "evidently is not installed. Run `pip install evidently`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading reference sample (%s rows)", REFERENCE_SAMPLE)
    reference = _load_sample(REFERENCE_SAMPLE, skip_rows=0)

    logger.info("loading current sample (%s rows)", CURRENT_SAMPLE)
    # The "current" window is the most recent transactions — the ones
    # after the reference window.
    current = _load_sample(CURRENT_SAMPLE, skip_rows=REFERENCE_SAMPLE)

    # Drop the time column — it's not a model feature, just used for ordering.
    for df in [reference, current]:
        df.drop(columns=["TransactionDT"], inplace=True, errors="ignore")

    logger.info("running Evidently drift report")
    report = Report(
        metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=None,
    )
    report.save_html(str(output_path))
    logger.info("drift report saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_report()
    print(f"Report written to: {path}")
    print("Open it in your browser to explore drift metrics.")
