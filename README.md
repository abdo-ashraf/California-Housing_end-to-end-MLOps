# California Housing End-to-End MLOps

Manifest-driven, component-versioned MLOps workflow for California housing price prediction.

## Overview

This repository is organized into three layers:

1. Training and promotion layer: runs manifest-driven training, evaluation, registration, and promotion in MLflow.
2. Inference API layer: serves model predictions via FastAPI.
3. Frontend layer: provides a Streamlit UI for single and batch inference.

## Project Layout

- `training_and_promotion/`: training, evaluation, model registration, and promotion
- `app/ml_service/`: FastAPI inference service
- `app/server/`: Streamlit frontend
- `app/docker-compose.yml`: local multi-service orchestration
- `app/app.env.example`: app-layer environment template

## Documentation Map

Use the layer-specific READMEs for implementation details:

- Training and promotion guide: `training_and_promotion/README.md`
- Inference API guide: `app/ml_service/README.md`
- Streamlit frontend guide: `app/server/README.md`
- App-level Docker Compose guide: `app/README.md`

## Environment Files

- `training_and_promotion/.env`: training-layer settings (recommended for training runs).
- `app/app.env`: app-layer settings used by Docker Compose.

Start from templates:

- `training_and_promotion/.env.example`
- `app/app.env.example`

## Quick Start

### 0. Setup Python Environment And Install Dependencies

From repository root (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r training_and_promotion/requirements.txt
```

### 1. Train and Promote a Model

From repository root:

```sh
python training_and_promotion/main.py
```

This command performs one full training-and-promotion run using the configured manifest and MLflow backend.

### 2. Run the App Stack with Docker Compose

From repository root:

```sh
docker compose -f app/docker-compose.yml up --build -d
```

Optional local MLflow profile:

```sh
docker compose -f app/docker-compose.yml --profile local-mlflow up --build -d
```

## Notes

- The training model is component-driven and can change over time based on the selected trainer component in the run manifest.
- The inference service expects an existing production alias in MLflow (for example `HousingModel@champion`).

## Troubleshooting

For operational and endpoint-specific troubleshooting, use:

- `training_and_promotion/README.md`
- `app/ml_service/README.md`
- `app/server/README.md`
- `app/README.md`
