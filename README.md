# California Housing End-to-End MLOps

End-to-end MLOps workflow for California housing price prediction, including:

- Data preparation and stratified train/test split
- Feature engineering pipeline (custom sklearn transformers)
- Model training and evaluation with MLflow tracking
- Model Registry aliasing (`staging` and `champion`) and automatic promotion
- FastAPI model-serving service (single and batch inference)
- Streamlit frontend for interactive predictions

## Project Structure

```text
.
|-- CD/                          # Training and promotion pipeline
|   |-- main.py                  # Main training/evaluation/promotion entry point
|   |-- data/
|   |-- models/
|   `-- pipeline/
|-- app/
|   |-- ml_service/              # FastAPI inference service
|   |   |-- app.py
|   |   |-- Dockerfile
|   |   `-- requirements.txt
|   `-- server/                  # Streamlit frontend
|       |-- app.py
|       |-- Dockerfile
|       `-- requirements.txt
|-- assets/data/housing/housing.csv
`-- dev/requirements.txt         # Full development environment
```

## Architecture Flow

1. `CD/main.py` trains/evaluates a model and logs run metrics to MLflow.
2. The model is registered in MLflow as `HousingModel` and assigned `staging` alias.
3. `promote_if_better` compares `test_rmse` against current `champion` and promotes if better.
4. `app/ml_service` loads `models:/HousingModel@champion` and serves predictions.
5. `app/server` calls FastAPI endpoints and provides a UI for health checks and inference.

## Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized run)
- MLflow tracking backend (local MLflow server or remote URI)

## Local Setup

### 1) Create and activate environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r dev/requirements.txt
```

### 2) Configure environment variables

Set these before running training/serving.

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME="california_housing_experiment"

$env:MODEL_NAME="HousingModel"
$env:MODEL_PRODUCTION_ALIAS="champion"

# Optional (for authenticated remote MLflow)
$env:MLFLOW_TRACKING_USERNAME="<username>"
$env:MLFLOW_TRACKING_PASSWORD="<token_or_password>"

# Optional CORS setting for FastAPI
$env:FRONTEND_URL="http://localhost:8501"
```

## Optional: Run MLflow Server Locally (Docker)

```bash
docker run -d -p 5000:5000 \
	--name mlflow-server \
	-v "${PWD}/assets:/assets" \
	ghcr.io/mlflow/mlflow:v3.9.0 \
	mlflow server --host 0.0.0.0 --port 5000 \
	--backend-store-uri sqlite:////assets/mlflow/mlflow.db \
	--default-artifact-root /assets/mlflow/artifacts
```

Then use:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
```

## Run Training + Promotion Pipeline

From repo root:

```powershell
python CD/main.py
```

What it does:

- Loads `assets/data/housing/housing.csv`
- Performs stratified split
- Builds preprocessing pipeline
- Trains RandomForest model
- Logs metrics + confidence interval to MLflow
- Registers model and sets `staging`
- Promotes to `champion` if `test_rmse` is better

## Run FastAPI Inference Service

```powershell
cd app/ml_service
pip install -r requirements.txt
pip install uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

Note: `uvicorn` is required for local run.

## Run Streamlit Frontend

In a new terminal:

```powershell
cd app/server
pip install -r requirements.txt
$env:BACKEND_URL="http://localhost:8000"
streamlit run app.py
```

Open: `http://localhost:8501`

## API Endpoints

Base URL: `http://localhost:8000`

- `GET /health` - API/model health
- `GET /model_info` - current loaded model metadata
- `POST /predict` - single record inference
- `POST /predict_batch` - batch inference
- `POST /reload_model` - force reload `champion` model

### Single Prediction Payload

```json
{
	"longitude": -122.23,
	"latitude": 37.88,
	"housing_median_age": 41.0,
	"total_rooms": 880.0,
	"total_bedrooms": 129.0,
	"population": 322.0,
	"households": 126.0,
	"median_income": 8.3252,
	"ocean_proximity": "NEAR BAY"
}
```

### Batch Prediction Payload

```json
{
	"data": [
		{
			"longitude": -122.23,
			"latitude": 37.88,
			"housing_median_age": 41.0,
			"total_rooms": 880.0,
			"total_bedrooms": 129.0,
			"population": 322.0,
			"households": 126.0,
			"median_income": 8.3252,
			"ocean_proximity": "NEAR BAY"
		}
	]
}
```

## Docker

### FastAPI Service

```bash
docker build -t housing-ml-service -f app/ml_service/Dockerfile app/ml_service
docker run --rm -p 8000:8000 \
	-e MODEL_NAME=HousingModel \
	-e MODEL_PRODUCTION_ALIAS=champion \
	-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
	-e FRONTEND_URL=http://localhost:8501 \
	housing-ml-service
```

### Streamlit Frontend

```bash
docker build -t housing-streamlit -f app/server/Dockerfile app/server
docker run --rm -p 8501:8501 \
	-e BACKEND_URL=http://host.docker.internal:8000 \
	housing-streamlit
```

## Notes

- Do not commit real credentials/tokens in source files or docs.
- If the API returns `model_loaded=false`, verify:
	- `MODEL_NAME` and `MODEL_PRODUCTION_ALIAS`
	- `MLFLOW_TRACKING_URI`
	- registry alias exists (for example `champion`)
- Training script currently uses stage `prod` internally in `CD/main.py`.