---
title: HousingModel Server
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
license: mit
---

## HousingModel Streamlit Server

This service is the frontend UI for the California Housing MLOps project.
It provides an interactive interface for single and batch inference by calling the backend API in `app/ml_service`.

## What This Component Does

- Displays backend/API health and model status
- Allows reloading the backend model alias from the UI
- Supports single-record prediction
- Supports batch prediction using JSON input
- Validates payload shape before sending requests
- Shows summary statistics and tabular batch results

## Architecture Role

This server is a presentation layer only:

- It does not train models
- It does not host the model artifact
- It communicates with FastAPI backend endpoints over HTTP

Flow:

1. User submits input in Streamlit
2. Streamlit calls ml_service endpoint
3. FastAPI runs model inference
4. Streamlit renders response

## Backend API Contract

Expected backend endpoints:

- `GET /health`
- `GET /model_info`
- `POST /reload_model`
- `POST /predict`
- `POST /predict_batch`

If any endpoint is unavailable, the UI will show a safe, user-friendly error.

## Environment Configuration

For Docker Compose usage, this service reads variables from `app/app.env`.
For standalone local Python runs, export variables in your shell before starting Streamlit.

Primary variables:

- `BACKEND_URL`: Base URL of FastAPI inference service
- `API_BASE_URL`: Optional fallback for backend base URL

Resolution order:

1. `BACKEND_URL`
2. `API_BASE_URL`
3. Default `http://localhost:8000`

Example:

```env
BACKEND_URL=https://your-ml-service.example.com
```

## Running Locally

From `app/server`:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open:

- `http://localhost:8501`

## Running with Docker

Build:

```sh
docker build -t housing-streamlit-ui .
```

Run:

```sh
docker run --rm -p 8501:8501 \
	-e BACKEND_URL=http://host.docker.internal:8000 \
	housing-streamlit-ui
```

## UI Features

### Sidebar

- API health check button
- Reload model button
- API status and model loaded metrics on one row
- Model metadata (name/version)

### Single Prediction Tab

- Structured numeric and categorical inputs
- Sends payload to `POST /predict`
- Displays formatted single prediction result

### Batch Prediction Tab

- Accepts JSON array input
- Validates required fields and types
- Sends payload to `POST /predict_batch`
- Displays min/max/average and detailed result table

## Validation and Error Handling

Implemented safeguards in `app.py`:

- Sanitized API error messages (no infrastructure leakage)
- Request timeout handling
- Connection and backend availability checks
- Input shape/type validation for batch payloads
- Clear warning when backend is unreachable on startup

## Troubleshooting

### "Cannot connect to backend"

- Ensure `BACKEND_URL` points to running `ml_service`
- Check CORS/networking/reverse proxy rules
- Verify backend health endpoint manually

### "Method Not Allowed" on reload

- Usually indicates redirect/method mismatch at gateway
- Ensure backend exposes `POST /reload_model` and proxy preserves method on redirects

### "Backend service is not responding"

- Model reload can take longer due to MLflow artifact loading
- Keep backend reachable and increase request timeout if needed

## Security Notes

- Do not expose internal backend URLs in UI
- Do not display raw backend response bodies with sensitive content
- Keep secrets in `app.env` or `.env`, not source files

## Related Components

- Backend inference API: `app/ml_service`
- Training/promotion pipeline: `CD`
