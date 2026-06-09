"""Train a fraud-detection model with MLflow tracking.

Usage::

    python -m src.training.train --model xgboost
    python -m src.training.train --model baseline

Everything (params, metrics, the serialized model) is logged to MLflow. The
fitted model is also saved to ``models/<name>.joblib`` for the API to load.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import mlflow

from src.config import MODELS_DIR, settings
from src.features.build_features import build_dataset
from src.models.factory import build_baseline, build_xgboost
from src.training.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


def train(
    model_name: str,
    val_fraction: float = 0.2,
    max_train_rows: int | None = None,
) -> dict[str, float]:
    dataset = build_dataset(val_fraction=val_fraction, max_train_rows=max_train_rows)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name, "val_fraction": val_fraction})
        mlflow.log_params(dataset.summary())

        if model_name == "xgboost":
            neg = int((dataset.y_train == 0).sum())
            pos = int((dataset.y_train == 1).sum())
            scale_pos_weight = neg / max(pos, 1)
            model = build_xgboost(scale_pos_weight=scale_pos_weight)
            mlflow.log_params(model.get_params())
            model.fit(
                dataset.X_train,
                dataset.y_train,
                eval_set=[(dataset.X_val, dataset.y_val)],
                verbose=25,  # print the eval metric every 25 boosting rounds
            )
            mlflow.log_metric("best_iteration", float(model.best_iteration))
        elif model_name == "baseline":
            model = build_baseline()
            model.fit(dataset.X_train, dataset.y_train)
        else:
            raise ValueError(f"unknown model: {model_name!r} (use 'xgboost' or 'baseline')")

        y_proba = model.predict_proba(dataset.X_val)[:, 1]
        metrics = evaluate(dataset.y_val, y_proba)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_names": dataset.feature_names,
                "category_maps": dataset.category_maps,
                "best_threshold": metrics["best_threshold"],
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path))

    logger.info("training complete: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud-detection model.")
    parser.add_argument(
        "--model",
        choices=["xgboost", "baseline"],
        default="xgboost",
        help="which model to train",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=80_000,
        help="cap training rows to the most recent N (use 0 for all rows)",
    )
    args = parser.parse_args()

    max_train_rows = args.max_train_rows if args.max_train_rows > 0 else None
    metrics = train(
        args.model,
        val_fraction=args.val_fraction,
        max_train_rows=max_train_rows,
    )
    print("\n=== Validation metrics ===")
    for key, value in metrics.items():
        print(f"{key:18s}: {value:.4f}")


if __name__ == "__main__":
    main()
