"""FastAPI application for fraud-detection inference.

Endpoints
---------
GET  /healthz      liveness / readiness probe
POST /predict      fraud-score a single transaction
GET  /metrics      Prometheus exposition format
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.api.model_loader import ModelBundle, get_bundle, load_model
from src.api.schemas import (
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "End-to-end latency of /predict requests",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_COUNTER = Counter(
    "fraud_api_predictions_total",
    "Number of predictions served",
    labelnames=["result"],  # "fraud" | "legit"
)
ERROR_COUNTER = Counter(
    "fraud_api_errors_total",
    "Number of prediction errors",
)

# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the model on startup so the first request is not slow."""
    logger.info("loading model on startup")
    load_model(settings.model_name)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud-scoring API backed by an XGBoost model trained on "
        "the IEEE-CIS Fraud Detection dataset."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness / readiness probe. Returns 200 when the model is loaded."""
    try:
        bundle = get_bundle()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=bundle.model_name,
        )
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False, model_name=None)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    request: TransactionRequest,
    bundle: ModelBundle = Depends(get_bundle),
) -> PredictionResponse:
    """Score a transaction for fraud.

    Returns the fraud probability and a binary decision at the F1-optimal
    threshold learned during training.
    """
    t0 = time.perf_counter()
    try:
        # Merge declared fields + any extra fields the caller sent
        transaction: dict[str, Any] = {
            **request.model_dump(exclude_none=True),
            **request.model_extra,
        }
        result = bundle.predict(transaction)
        label = "fraud" if result["is_fraud"] else "legit"
        PREDICTION_COUNTER.labels(result=label).inc()
        return PredictionResponse(**result)
    except Exception as exc:
        ERROR_COUNTER.inc()
        logger.exception("prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - t0)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics(request: Request) -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
