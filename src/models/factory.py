"""Model factory functions.

Keeping model construction in one place makes the training script thin and the
hyper-parameters easy to track in MLflow.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_xgboost(scale_pos_weight: float, random_state: int = 42) -> XGBClassifier:
    """Gradient-boosted trees, the workhorse for tabular fraud detection.

    - ``scale_pos_weight`` rebalances the loss for the rare positive class.
    - ``tree_method="hist"`` is fast and memory-friendly on CPU.
    - ``eval_metric="aucpr"`` + early stopping optimize for the imbalanced metric.
    - NaNs are handled natively, so no imputation is required.
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        max_bin=128,  # fewer histogram bins -> less memory and faster on CPU
        eval_metric="aucpr",
        early_stopping_rounds=40,
        n_jobs=4,  # cap threads so we don't oversubscribe a small machine
        random_state=random_state,
    )


def build_baseline(random_state: int = 42) -> Pipeline:
    """A logistic-regression baseline for a sanity reference.

    Linear models need explicit NaN handling and scaling, hence the pipeline.
    ``class_weight="balanced"`` accounts for the imbalance.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
