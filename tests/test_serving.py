"""Tests for the model serving API — mocks MLflow so no tracking server needed."""
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Stub mlflow before loading main.py so tests run without an mlflow install
_mlflow_stub = ModuleType("mlflow")
_mlflow_stub.pyfunc = MagicMock()
sys.modules.setdefault("mlflow", _mlflow_stub)
sys.modules.setdefault("mlflow.pyfunc", _mlflow_stub.pyfunc)

# Load serving/fastapi/main.py by file path to avoid collision with the `fastapi` package name
_spec = importlib.util.spec_from_file_location(
    "model_api",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "serving", "fastapi", "main.py"),
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["model_api"] = _module
_spec.loader.exec_module(_module)

app = _module.app


@pytest.fixture()
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.87]
    _module.model = mock_model
    with TestClient(app) as c:
        yield c
    _module.model = None


def test_health_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_prediction(client):
    payload = {"features": [1.0, 2.0, 3.0, 4.0], "request_id": "test-001"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "latency_ms" in body
    assert body["request_id"] == "test-001"



def test_metrics_endpoint(client):
    # Trigger a prediction first so counters are non-zero
    client.post("/predict", json={"features": [1.0, 2.0]})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"model_predictions_total" in resp.content


def test_override_endpoint(client):
    resp = client.post(
        "/override",
        params={
            "request_id": "abc123",
            "reason": "Manual QA review — edge case confirmed by engineer",
            "override_value": 0.0,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "abc123"
