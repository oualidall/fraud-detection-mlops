# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System deps (needed by some wheels, e.g. psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer — only busted when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/
COPY pyproject.toml .

# The trained model is mounted at runtime via a Docker volume.
# Running the container without a mounted model will still start but
# /healthz will return {"status":"degraded","model_loaded":false}.
ENV MODEL_NAME=xgboost
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
