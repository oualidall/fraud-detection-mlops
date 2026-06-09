"""API integration tests using a tiny synthetic model bundle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bundle():
    """Return a minimal ModelBundle backed by a mock sklearn estimator."""
    from src.api.model_loader import ModelBundle

    mock_model = MagicMock()
    # predict_proba returns [[p_legit, p_fraud]] — use 0.8 so is_fraud=True
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    return ModelBundle(
        bundle={
            "model": mock_model,
            "feature_names": ["TransactionAmt", "TransactionAmt_log",
                               "dt_hour", "dt_dow", "n_missing"],
            "category_maps": {},
            "best_threshold": 0.5,
        },
        model_name="xgboost_test",
    )


@pytest.fixture()
def client():
    """TestClient with the model dependency overridden.

    We also patch ``load_model`` at the module level so the lifespan startup
    hook does not try to open the real ``.joblib`` file (which may not exist
    in CI or during development before training).
    """
    from src.api.main import app
    from src.api.model_loader import get_bundle

    bundle = _make_bundle()
    app.dependency_overrides[get_bundle] = lambda: bundle

    with patch("src.api.main.load_model", return_value=bundle), TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_healthz_returns_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_fraud_transaction(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        json={"TransactionAmt": 500.0, "ProductCD": "C", "card6": "credit"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert body["fraud_probability"] == pytest.approx(0.8)
    assert body["is_fraud"] is True


def test_predict_requires_amount(client: TestClient) -> None:
    resp = client.post("/predict", json={"ProductCD": "W"})
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_amount_must_be_positive(client: TestClient) -> None:
    resp = client.post("/predict", json={"TransactionAmt": -10.0})
    assert resp.status_code == 422


def test_metrics_endpoint_returns_prometheus_format(client: TestClient) -> None:
    # Trigger a prediction first so counters are non-zero
    client.post("/predict", json={"TransactionAmt": 100.0})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "fraud_api_predictions_total" in resp.text
