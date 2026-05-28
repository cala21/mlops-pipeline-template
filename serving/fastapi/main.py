import hashlib
import json
import logging
import time
import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

app = FastAPI(title="MLOps Model Serving API", version="1.0.0")

PREDICTIONS = Counter("model_predictions_total", "Total predictions", ["status"])
LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency")

# Structured audit log — EU AI Act Art. 12 (record-keeping for high-risk AI systems)
# Each prediction is logged with input hash, model version, timestamp, and output.
# Do NOT log raw feature values if they contain personal data (GDPR Art. 5 data minimisation).
audit_logger = logging.getLogger("audit")
_handler = logging.FileHandler(os.getenv("AUDIT_LOG_PATH", "audit.log"))
_handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(_handler)
audit_logger.setLevel(logging.INFO)

model = None


@app.on_event("startup")
async def load_model():
    global model
    model_uri = os.getenv("MODEL_URI", "models:/production-model/Production")
    model = mlflow.pyfunc.load_model(model_uri)


class PredictionRequest(BaseModel):
    features: list[float]
    request_id: str = ""  # caller-supplied trace ID for audit correlation


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float
    request_id: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    input_hash = hashlib.sha256(
        json.dumps(request.features, sort_keys=True).encode()
    ).hexdigest()[:16]
    model_version = os.getenv("MODEL_VERSION", "unknown")

    try:
        df = pd.DataFrame([request.features])
        prediction = float(model.predict(df)[0])
        latency = (time.time() - start) * 1000

        PREDICTIONS.labels(status="success").inc()
        LATENCY.observe(latency / 1000)

        # Art. 12 audit record: timestamp, model version, input hash (not raw data), output
        audit_logger.info(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request.request_id or input_hash,
            "model_version": model_version,
            "input_hash": input_hash,
            "prediction": prediction,
            "latency_ms": round(latency, 2),
            "status": "success",
        }))

        return PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            latency_ms=round(latency, 2),
            request_id=request.request_id or input_hash,
        )

    except Exception as e:
        PREDICTIONS.labels(status="error").inc()
        audit_logger.info(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request.request_id or input_hash,
            "model_version": model_version,
            "input_hash": input_hash,
            "status": "error",
            "error": str(e),
        }))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/override")
async def human_override(request_id: str, reason: str, override_value: float):
    """
    EU AI Act Art. 14 — human oversight endpoint.
    Allows authorised personnel to record a manual override of a model prediction.
    The override and reason are written to the audit log.
    """
    audit_logger.info(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "human_override",
        "request_id": request_id,
        "override_value": override_value,
        "reason": reason,
    }))
    return {"status": "override recorded", "request_id": request_id}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
