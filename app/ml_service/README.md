---
title: HousingModel-ml Service
emoji: 🏆
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# HousingModel ML Service

This component is the production inference API for the California Housing MLOps project.
It loads the champion model from MLflow Model Registry and serves prediction endpoints
for single and batch requests.

## What This Service Does

- Loads model artifact from MLflow on startup
- Exposes health and model metadata endpoints
- Supports single-record prediction
- Supports batch prediction
- Supports manual model reload without restarting the process
- Provides interactive API docs via FastAPI

## Tech Stack

- FastAPI
- MLflow PyFunc
- Pandas
- Pydantic
- Uvicorn
- python-dotenv

## API Endpoints

- GET /
	- Browser: returns a visual landing page with status and endpoint list
	- API clients: returns JSON metadata
- GET /health
	- Service and model readiness status
- GET /model_info
	- Active model name, alias, loaded flag, and model version/run id
- POST /predict
	- Single record prediction
- POST /predict_batch
	- Batch prediction
- POST /reload_model
	- Reloads champion model from MLflow

## Request Payloads

### POST /predict

```json
{
	"longitude": -122.25,
	"latitude": 37.85,
	"housing_median_age": 15.0,
	"total_rooms": 2000.0,
	"total_bedrooms": 300.0,
	"population": 800.0,
	"households": 250.0,
	"median_income": 3.5,
	"ocean_proximity": "NEAR OCEAN"
}
```

Allowed ocean_proximity values:

- NEAR BAY
- INLAND
- NEAR OCEAN
- ISLAND
- <1H OCEAN

### POST /predict_batch

```json
{
	"data": [
		{
			"longitude": -122.25,
			"latitude": 37.85,
			"housing_median_age": 15.0,
			"total_rooms": 2000.0,
			"total_bedrooms": 300.0,
			"population": 800.0,
			"households": 250.0,
			"median_income": 3.5,
			"ocean_proximity": "NEAR OCEAN"
		}
	]
}
```

## Configuration

The service loads environment variables from the repository-level .env file automatically.

Required:

- MODEL_NAME
- MODEL_PRODUCTION_ALIAS
- MLFLOW_TRACKING_URI

Optional:

- MLFLOW_TRACKING_USERNAME
- MLFLOW_TRACKING_PASSWORD

Example values:

```env
MODEL_NAME=HousingModel
MODEL_PRODUCTION_ALIAS=champion
MLFLOW_TRACKING_URI=https://your-mlflow-server
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-token
```

## Local Run

From this folder:

```sh
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000/
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Docker Run

Build:

```sh
docker build -t housing-ml-service .
```

Run:

```sh
docker run --rm -p 8000:8000 \
	-e MODEL_NAME=HousingModel \
	-e MODEL_PRODUCTION_ALIAS=champion \
	-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
	housing-ml-service
```

## Error Handling Notes

- If model is not loaded, prediction endpoints return HTTP 503
- If inference fails unexpectedly, service returns HTTP 500
- If MLflow auth is invalid or expired, startup/load errors are logged and reload may fail

## Operational Recommendations

- Put this service behind a reverse proxy in production
- Use /health for readiness checks in your platform
- Rotate MLflow credentials periodically
- Keep model alias strategy consistent (for example: champion for production)

## Related Components

- CD pipeline trains, evaluates, and promotes models to MLflow aliases
- Streamlit server consumes this API for interactive inference

