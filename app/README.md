# App Layer Guide

This folder contains the runtime application layer for the project:

- `ml_service/`: FastAPI inference API that loads `HousingModel@champion` from MLflow.
- `server/`: Streamlit frontend that calls the FastAPI API.
- `docker-compose.yml`: Local orchestration for API + frontend, with optional local MLflow.

## Services Overview

### 1) FastAPI Inference Service (`ml_service`)

Responsibilities:

- Load champion model from MLflow Model Registry at startup
- Expose prediction and operational endpoints
- Support model reload without process restart

Main endpoints:

- `GET /health`
- `GET /model_info`
- `POST /predict`
- `POST /predict_batch`
- `POST /reload_model`

Container port mapping via compose:

- Host `8000` -> Container `7860`

### 2) Streamlit Frontend (`server`)

Responsibilities:

- Provide interactive UI for single and batch inference
- Show backend and model status
- Trigger model reload from UI

Container port mapping via compose:

- Host `8501` -> Container `8501`

The frontend calls backend internally through Docker network using:

- `BACKEND_URL=http://ml_service:7860`

### 3) Optional Local MLflow (`mlflow` profile)

The compose file includes an optional local MLflow server under profile `local-mlflow`.

- Image: `ghcr.io/mlflow/mlflow:v3.5.1`
- Host port: `5000`
- Backend DB and artifacts persisted under `./assets/mlflow`

If you already use a remote MLflow backend, you do not need to run this service.

## Environment Variables

Compose reads `app.env` for both `ml_service` and `server`.

Setup:

1. Copy `app.env.example` to `app.env`.
2. Fill in your MLflow values.

Required variables:

- `MODEL_NAME`
- `MODEL_PRODUCTION_ALIAS`
- `MLFLOW_TRACKING_URI`

Optional variables:

- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

Frontend variable:

- `BACKEND_URL` (default in `app.env.example` is `http://ml_service:7860`)

## Docker Compose Usage

Run all commands from this `app/` folder.

Prerequisite:

- Your MLflow Tracking Server must already have a registered model that matches `MODEL_NAME` and `MODEL_PRODUCTION_ALIAS` (default: `HousingModel@champion`).
- If this model alias does not exist yet, run the training pipeline first to register and promote a model.

### Use existing remote MLflow backend

1. Set `MLFLOW_TRACKING_URI` in `app.env` to your remote server.
2. Start app services:

```sh
docker compose up --build -d
```

3. Open:

- API: http://localhost:8000
- UI: http://localhost:8501

### Run with local MLflow too

1. Set in `app.env`:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
```

2. Start app services + local MLflow profile:

```sh
docker compose --profile local-mlflow up --build -d
```

3. Open:

- MLflow: http://localhost:5000
- API: http://localhost:8000
- UI: http://localhost:8501

## Stop and Cleanup

Stop services:

```sh
docker compose down
```

Stop and remove volumes created by compose:

```sh
docker compose down -v
```

## Troubleshooting

- `server` cannot reach backend:
  - Ensure `ml_service` is healthy (`docker compose ps`).
  - Confirm API responds at `http://localhost:8000/health`.
- API starts but model is not loaded:
  - Check `MLFLOW_TRACKING_URI` and credentials in `app.env`.
  - Check MLflow connectivity from container logs.
- Local MLflow not available:
  - Start with `--profile local-mlflow`.

## Related Docs

- `ml_service/README.md`: Detailed API behavior and payload examples
- `server/README.md`: Detailed UI behavior and backend integration notes
