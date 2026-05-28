# MLOps Pipeline Template

> Production MLOps pipeline for EU-regulated environments: MLflow experiment tracking, Kubeflow orchestration, GitHub Actions CI/CD, and FastAPI model serving with **EU AI Act Art. 12 audit logging and Art. 14 human oversight** built in. Targets AWS/GCP eu-west regions by default.

[![CI/CD](https://github.com/cala21/mlops-pipeline-template/actions/workflows/ci.yml/badge.svg)](https://github.com/cala21/mlops-pipeline-template/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange.svg)](https://mlflow.org)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Art.%2012%20%2B%2014-blue)](https://github.com/cala21/eu-ai-act-compliance-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
Data → MLflow Tracking → Model Registry → GitHub Actions → Container Registry → Kubernetes
         (experiments)    (staging/prod)    (train+test)      (build+push)       (serve)
```

**Stack:**
- **Experiment tracking:** MLflow with S3 artifact storage
- **Orchestration:** Kubeflow Pipelines
- **CI/CD:** GitHub Actions (test → train → build → deploy)
- **Serving:** FastAPI + Prometheus metrics
- **Monitoring:** Grafana dashboards + Prometheus alerting

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/cala21/mlops-pipeline-template
cd mlops-pipeline-template
pip install -r requirements.txt

# 2. Start MLflow tracking server (local)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# 3. Train a model
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/train.py

# 4. Serve the model
export MODEL_URI=models:/production-model/Production
uvicorn serving.fastapi.main:app --reload

# 5. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Repository Structure

```
├── .github/workflows/      # CI/CD pipelines (test → train → build → deploy)
├── mlflow/
│   └── experiments/        # Experiment configs and tracking settings
├── kubeflow/
│   └── pipelines/          # Pipeline definitions
├── serving/
│   └── fastapi/            # Model serving API with Prometheus metrics
├── monitoring/
│   ├── dashboards/         # Grafana dashboard JSONs
│   └── alerts/             # Prometheus alerting rules
├── scripts/
│   └── train.py            # Training entry point
├── data/
│   ├── raw/                # Raw input data (gitignored)
│   └── schemas/            # Data validation schemas
└── tests/                  # Unit and integration tests
```

## CI/CD Pipeline

The GitHub Actions workflow automates the full ML lifecycle:

| Stage | Trigger | Action |
|-------|---------|--------|
| **Test** | Every push | Run pytest, validate data schemas |
| **Train** | Merge to main | Train model, log to MLflow, register if above threshold |
| **Build** | After training | Build Docker image, push to GHCR |
| **Deploy** | After build | Rolling update to Kubernetes |

## Model Serving API

```
POST /predict     → Run inference
GET  /metrics     → Prometheus metrics (latency, error rate, prediction count)
GET  /health      → Liveness/readiness check
```

## Monitoring & Alerting

Pre-configured Prometheus alerts:
- `ModelHighLatency` — p95 latency > 500ms for 5 minutes
- `ModelHighErrorRate` — error rate > 5% for 2 minutes
- `ModelNotLoaded` — serving instance down

## EU AI Act Compliance

This template implements the technical requirements for **high-risk AI systems** under [EU AI Act (Regulation 2024/1689)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689):

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| Art. 12 | Record-keeping | Structured JSON audit log: input SHA-256 hash, model version, timestamp, prediction |
| Art. 14 | Human oversight | `POST /override` endpoint — log manual review and correction with operator reason |
| Art. 15 | Accuracy/robustness | Prometheus alert rules on error rate and latency SLOs |
| Art. 10 | Data governance | MLflow tracks dataset version and schema for every training run |

Input features are **never stored raw** — only their SHA-256 hash — satisfying GDPR Art. 5 data minimisation while maintaining the audit trail.

See [`eu-ai-act-compliance-toolkit`](https://github.com/cala21/eu-ai-act-compliance-toolkit) for risk assessment templates, Italian sector guides, and the CLI validator.

## Adapting This Template

1. Replace the training data loading in `scripts/train.py`
2. Update the feature schema in `data/schemas/`
3. Configure your MLflow tracking URI in GitHub Secrets
4. Set your Kubernetes cluster credentials in GitHub Secrets
5. Adjust alert thresholds in `monitoring/alerts/model_alerts.yml`

## License

MIT — use freely, contributions welcome.
