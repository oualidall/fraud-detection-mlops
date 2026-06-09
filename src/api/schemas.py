"""Pydantic request / response schemas for the fraud-detection API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Raw transaction fields sent by the caller.

    Only ``TransactionAmt`` is required. All other fields are optional — missing
    values are treated as NaN, which XGBoost handles natively. Extra fields
    (e.g. the full set of V* columns from a real transaction feed) are accepted
    and forwarded to the model if they match a known feature name.
    """

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: str | None = Field(None, description="Product category code")
    card4: str | None = Field(None, description="Card brand (visa, mastercard, ...)")
    card6: str | None = Field(None, description="Card type (debit, credit, ...)")
    TransactionDT: int | None = Field(
        None, description="Seconds offset from a fixed reference date"
    )


class PredictionResponse(BaseModel):
    """Fraud-score response returned by POST /predict."""

    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None


class ErrorResponse(BaseModel):
    detail: str
    extra: Any | None = None
