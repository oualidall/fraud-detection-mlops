"""Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Usage:
    python -m src.data.download

Prerequisites:
    1. Kaggle account + API token (~/.kaggle/kaggle.json) OR
       KAGGLE_USERNAME and KAGGLE_KEY set in the environment.
    2. Accept the competition rules once at:
       https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

from src.config import RAW_DATA_DIR, settings

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


def _ensure_kaggle_credentials() -> None:
    """Make sure the kaggle library can find credentials before it imports."""
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        logger.info("kaggle credentials loaded from environment")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        logger.info("kaggle credentials found at %s", kaggle_json)
        return

    raise RuntimeError(
        "Kaggle credentials not found. Either set KAGGLE_USERNAME and KAGGLE_KEY "
        "in .env, or place kaggle.json in ~/.kaggle/."
    )


def download_dataset(competition: str | None = None, output_dir: Path | None = None) -> Path:
    """Download and unzip the IEEE-CIS competition data."""
    _ensure_kaggle_credentials()

    # Import after credentials are set; kaggle reads them at import time
    from kaggle.api.kaggle_api_extended import KaggleApi

    competition = competition or settings.kaggle_competition
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    logger.info("downloading kaggle competition %s into %s", competition, output_dir)
    api.competition_download_files(competition, path=str(output_dir), quiet=False)

    archive = output_dir / f"{competition}.zip"
    if archive.exists():
        logger.info("extracting %s", archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(output_dir)
        archive.unlink()
        logger.info("removed archive %s", archive)

    logger.info("dataset ready in %s", output_dir)
    return output_dir


if __name__ == "__main__":
    download_dataset()
