"""Project configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables (.env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow fields prefixed "model_" (model_name, model_stage) without colliding
        # with Pydantic 2's reserved namespace.
        protected_namespaces=(),
    )

    # Kaggle
    kaggle_username: str | None = None
    kaggle_key: str | None = None

    # MLflow. Defaults to a local file store so training works without a
    # running server; Phase 4's docker-compose overrides this with the
    # containerized tracking server URI.
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "fraud-detection"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Model registry
    model_name: str = "fraud_xgboost"
    model_stage: str = "Production"

    # Dataset (IEEE-CIS Fraud Detection on Kaggle)
    kaggle_competition: str = "ieee-fraud-detection"


settings = Settings()
