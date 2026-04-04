# California Housing End-to-End MLOps

**Manifest-driven, component-versioned MLOps workflow** for California housing price prediction. Train, serve, and monitor production housing price models with full lineage tracking.

## Features

- **Manifest-Driven Orchestration**: Version and swap ML pipeline components without code changes
- **Data Preparation**: Stratified income-based train/test split for robust evaluation
- **Feature Engineering**: Custom sklearn transformers (categorical encoding, RBF clustering, log transforms, ratio features)
- **Model Training & Evaluation**: RandomForest with hyperparameter tuning, confidence intervals via bootstrap resampling
- **MLflow Integration**: Full experiment tracking, model registry, automatic alias management
- **Automatic Promotion**: Champion/staging model promotion based on RMSE comparison
- **FastAPI Inference**: Single and batch prediction endpoints with model health checks
- **Streamlit UI**: Interactive prediction frontend with API health/model status monitoring
- **Containerization**: Docker support for both inference service and frontend

## Architecture

### Three-Layer Pipeline

1. **CD (Continuous Delivery) Layer** - Training & Promotion
   - Loads run manifest defining component versions
   - Resolves versioned dataset, preprocessor, trainer, evaluator, promotion components
   - Logs full lineage (manifest, component versions, metrics) to MLflow
   - Registers trained model in MLflow Registry with `staging` alias
   - Conditionally promotes to `champion` based on configurable promotion policy

2. **Inference Layer** (FastAPI) - Model Serving
   - Loads `HousingModel@champion` from MLflow Registry on startup
   - Serves single and batch predictions
   - Provides health and model info endpoints
   - Supports hot reload of champion model

3. **Frontend Layer** (Streamlit) - User Interaction
   - Calls FastAPI endpoints for predictions
   - Displays model health, version, alias metadata
   - Provides single-record and batch prediction UI
   - Allows on-demand model reload and API health checks

### Project Structure

```
California-Housing_end-to-end-MLOps/
│
├── CD/                                  # Training pipeline (manifest-driven)
│   ├── main.py                          # Orchestrator: loads manifest, resolves & runs components
│   ├── components/
│   │   ├── __init__.py
│   │   └── component_registry.py        # Registry of all versioned component implementations
│   ├── config/
│   │   ├── __init__.py
│   │   ├── run_manifest.py              # Manifest loader & schema validation
│   │   └── manifests/
│   │       └── default_v1.json          # Default component versions & configuration
│   ├── data/
│   │   ├── housing_data_ingestion.py    # download_housing_dataset, load_housing_dataset
│   │   └── data_splitting.py            # stratified_income_train_test_split
│   ├── pipeline/
│   │   └── preprocessing_pipeline.py    # build_preprocessing_pipeline (sklearn ColumnTransformer)
│   ├── models/
│   │   ├── model_evaluation.py          # evaluate_and_register_model (RMSE, CI, MLflow logging)
│   │   ├── model_registry_promotion.py  # promote_model_if_better (champion policy)
│   │   ├── model_benchmarking.py        # train_and_benchmark_models (unused in default flow)
│   │   ├── random_forest_tuning.py      # tune_random_forest_with_grid_search (unused in default flow)
│   │   └── __init__.py
│   ├── tracking/
│   │   └── experiment_setup.py          # setup_mlflow_experiment
│   ├── __init__.py
│	├── assets/
│   └── data/
│       └── housing/
│           └── housing.csv              # California housing dataset
│
├── app/
│   ├── ml_service/                      # FastAPI inference server
│   │   ├── app.py                       # FastAPI app with /health, /model_info, /predict, /predict_batch, /reload_model
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── server/                          # Streamlit UI
│       ├── app.py                       # Streamlit multi-page app
│       ├── Dockerfile
│       └── requirements.txt
│
├── dev/
│   └── requirements.txt                 # Full development environment
│
└── README.md
```

## Manifest-Driven Component System

### Why Versioned Components?

Traditional ML pipelines couple code to data/config. This system **decouples** them:
- Upgrade dataset ingestion, preprocessing, or promotion logic independently
- **No code changes** required to switch component versions
- Full lineage: every MLflow run records which component versions were used
- Easy rollback: use an old manifest to reproduce prior run behavior

### Available Versioned Components

Each component is registered in `CD/components/component_registry.py` and resolves at runtime from the manifest.

#### Dataset Components
- `local_csv:v1` - Load CSV from project-relative path (default: `assets/data/housing/housing.csv`)

#### Tracker Components
- `mlflow_experiment:v1` - Setup MLflow experiment with dev/prod stage suffixing

#### Splitter Components
- `income_stratified_split:v1` - Stratified split by income percentile bins (0%, 1.5%, 3%, 4.5%, 6%, ∞)

#### Preprocessing Components
- `housing_preprocessing:v1` - Full sklearn ColumnTransformer pipeline:
  - Categorical: impute → merge ISLAND→NEAR_OCEAN → one-hot encode
  - Numeric (default): impute (median) → standardize
  - Ratios: bedrooms/rooms, rooms/households, people/households (impute → divide → standardize)
  - Log: log(1 + x) on: bedrooms, rooms, population, households, income
  - Geo: RBF kernel similarity to K-means clusters (latitude/longitude)

#### Trainer Components
- `random_forest_pipeline:v1` - sklearn Pipeline: preprocessing → RandomForestRegressor (configurable max_features, random_state)

#### Evaluation Components
- `mlflow_model_eval_register:v1` - Run predictions → compute RMSE → bootstrap confidence interval → log to MLflow → register model with `staging` alias

#### Promotion Components
- `champion_rmse_policy:v1` - Compare new model RMSE vs current champion RMSE. If better (lower), promote to `champion` alias. Otherwise, keep existing champion.

## Environment Files

Copy `.env.example` to `.env` in the repository root and fill in the values once. The training pipeline, FastAPI service, and Streamlit app load `.env` automatically at startup, so you do not need to export variables manually in each terminal session.

Typical values in `.env`:

- `MODEL_NAME`
- `MODEL_PRODUCTION_ALIAS`
- `MODEL_STAGING_ALIAS`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_EXPERIMENT_NAME`
- `BACKEND_URL`
- `FRONTEND_URL`
- `CD_RUN_MANIFEST_PATH`

### Run Manifest Structure

File: `CD/config/manifests/default_v1.json`

```json
{
  "manifest_version": "1.0",
  "run_name": "cd_train_evaluate_promote",
  "stage": "prod",
  "model_name": "HousingModel",
  
  "dataset": {
    "component": "local_csv",
    "version": "v1",
    "config": {
      "path": "assets/data/housing/housing.csv"
    }
  },
  
  "tracker": {
    "component": "mlflow_experiment",
    "version": "v1",
    "config": {
      "experiment_name": "california_housing_experiment"
    }
  },
  
  "splitter": {
    "component": "income_stratified_split",
    "version": "v1",
    "config": {
      "test_size": 0.2,
      "random_state": 42
    }
  },
  
  "preprocessing": {
    "component": "housing_preprocessing",
    "version": "v1",
    "config": {
      "geo_n_clusters": 15,
      "geo_gamma": 1.0
    }
  },
  
  "trainer": {
    "component": "random_forest_pipeline",
    "version": "v1",
    "config": {
      "random_state": 42,
      "max_features": 6
    }
  },
  
  "evaluation": {
    "component": "mlflow_model_eval_register",
    "version": "v1",
    "config": {
      "confidence": 0.95
    }
  },
  
  "promotion": {
    "component": "champion_rmse_policy",
    "version": "v1",
    "config": {
      "metric_name": "test_rmse",
      "lower_is_better": true,
      "champion_alias": "champion"
    }
  }
}
```

**Key Fields**:
- `manifest_version`: Schema version (currently "1.0")
- `run_name`: MLflow run name
- `stage`: "dev" or "prod" (suffixed to experiment name)
- `model_name`: MLflow registered model name
- `dataset/tracker/splitter/preprocessing/trainer/evaluation/promotion`: Each defines:
  - `component`: Name of component (must exist in registry)
  - `version`: Semantic version (e.g., "v1", "v2")
  - `config`: Component-specific settings (merged with env var overrides)

## Prerequisites

- Python 3.11+
- pip
- MLflow tracking backend (local server or remote URI)
- Docker (optional, for containerized serving)

## Setup

### 1. Create a Python Environment

```sh
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment using the command for your shell, or invoke the interpreter directly from `.venv` if you prefer not to activate it.

### 2. Configure the MLflow Backend

**Option A: Local MLflow Server (Docker)**

```bash
docker run -d -p 5000:5000 \
  --name mlflow-server \
  -v "${PWD}/assets:/assets" \
  ghcr.io/mlflow/mlflow:v3.9.0 \
  mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////assets/mlflow/mlflow.db \
  --default-artifact-root /assets/mlflow/artifacts
```

**Option B: Remote MLflow Server**

Set `MLFLOW_TRACKING_URI` in `.env` to your server URL.

### 3. Review the Environment File

Make sure `.env` contains values for the variables listed above. Edit that file whenever you want to change runtime configuration.

## Running the Pipeline

### Training Only

From repo root:

```sh
python CD/main.py
```

This executes the full pipeline:
1. **Load Manifest**: From `CD_RUN_MANIFEST_PATH` in `.env` (default: `CD/config/manifests/default_v1.json`)
2. **Resolve Components**: Look up component implementations in registry by (component_name, version)
3. **Load Data**: Use dataset component to load CSV
4. **Split Data**: Use splitter component (stratified by income percentile)
5. **Build Pipeline**: Use preprocessing component to create sklearn ColumnTransformer
6. **Train Model**: Use trainer component to fit RandomForest
7. **Evaluate**: Use evaluation component to compute RMSE, CI, log to MLflow, register model with `staging` alias
8. **Promote**: Use promotion policy component to conditionally promote to `champion` alias
9. **Log Lineage**: MLflow run includes manifest artifact and component version tags

**Output**:
```
Resolving tracker component
MLflow tracking initialized successfully. http://localhost:5000 california_housing_experiment_prod prod
...
Registered Model Version: 1
...
New test_rmse: 0.7139
Current Champion test_rmse: 0.7254
New model is better. Promoting to Champion.
...
Pipeline finished. test_rmse=0.7139, model_version=1, promoted=True
```

### Use a Custom Manifest

Create a new manifest (e.g., `CD/config/manifests/dev_v1.json`) with different component versions or configs, then:

```sh
CD_RUN_MANIFEST_PATH=CD/config/manifests/dev_v1.json
python CD/main.py
```

### Configuration Precedence

Values in the environment override `.env`, and `.env` overrides manifest defaults for the runtime values below:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_STAGE`
- `MODEL_NAME`

Component-specific config still comes from the manifest. Edit the manifest when you want to change component behavior.

### Expected MLflow Output

After training, check MLflow UI:

```
MLFLOW_TRACKING_URI
├── Experiment: california_housing_experiment_prod
│   └── Run: cd_train_evaluate_promote
│       ├── Parameters:
│       │   ├── dataset_path: assets/data/housing/housing.csv
│       │   ├── model_name: HousingModel
│       │   ├── stage: prod
│       │   └── manifest_path: CD/config/manifests/default_v1.json
│       ├── Metrics:
│       │   ├── test_rmse: 0.7139
│       │   ├── test_rmse_ci_lower: 0.7089
│       │   ├── test_rmse_ci_upper: 0.7189
│       │   └── promoted_to_champion: 1
│       ├── Artifacts:
│       │   ├── run_manifest.json (full manifest)
│       │   └── HousingModel/ (registered model pickle)
│       └── Tags:
│           ├── manifest_version: 1.0
│           ├── dataset_component: local_csv:v1
│           ├── tracker_component: mlflow_experiment:v1
│           ├── splitter_component: income_stratified_split:v1
│           ├── preprocessing_component: housing_preprocessing:v1
│           ├── trainer_component: random_forest_pipeline:v1
│           ├── evaluation_component: mlflow_model_eval_register:v1
│           └── promotion_component: champion_rmse_policy:v1
└── Models:
    └── HousingModel
        ├── Version 1: staging, champion
```

## Running Inference Service

After successful training (model registered in MLflow), start the FastAPI service.

### FastAPI Inference Service

```sh
cd app/ml_service
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Startup**:
- Loads `HousingModel@champion` from MLflow Registry
- On failure, starts without model but `/health` returns `model_loaded=false`

The service reads `MODEL_NAME`, `MODEL_PRODUCTION_ALIAS`, and `MLFLOW_TRACKING_URI` from `.env`.

### Streamlit Frontend (Optional)

In another terminal:

```sh
cd app/server
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open: `http://localhost:8501`

## API Specification

Base URL: `http://localhost:8000`

### GET /

Root endpoint with basic service info and available routes.

**Response**:
```json
{
  "service": "Housing Price Prediction API",
  "status": "ok",
  "model_loaded": true,
  "model_name": "HousingModel",
  "model_version": "123...",
  "docs_url": "/docs",
  "redoc_url": "/redoc",
  "endpoints": ["/", "/health", "/model_info", "/predict", "/predict_batch", "/reload_model"]
}
```

### GET /health

Health check endpoint (useful for load balancers).

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "HousingModel",
  "model_version": "123..."
}
```

### GET /model_info

Detailed model metadata.

**Response**:
```json
{
  "model_name": "HousingModel",
  "model_alias": "champion",
  "model_loaded": true,
  "model_version": "123..."
}
```

### POST /predict

Single housing record prediction.

**Request**:
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

**Response**:
```json
{
  "prediction": 3.75,
  "model_version": "123..."
}
```

**Field Constraints**:
- All numeric fields must be float/int
- `ocean_proximity` must be one of: `"NEAR BAY"`, `"INLAND"`, `"NEAR OCEAN"`, `"ISLAND"`, `"<1H OCEAN"`
- All fields are required

### POST /predict_batch

Batch predictions on multiple records.

**Request**:
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
    },
    {
      "longitude": -118.5,
      "latitude": 34.2,
      "housing_median_age": 25.0,
      "total_rooms": 1500.0,
      "total_bedrooms": 250.0,
      "population": 600.0,
      "households": 200.0,
      "median_income": 2.8,
      "ocean_proximity": "<1H OCEAN"
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [3.75, 2.95],
  "model_version": "123..."
}
```

### POST /reload_model

Force reload the `@champion` model from MLflow Registry (no request body).

**Response**:
```json
{
  "message": "Model reloaded successfully.",
  "model_version": "123..."
}
```

### Interactive API Docs

FastAPI auto-generates interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker & Containerization

### FastAPI Service Container

```bash
docker build -t housing-ml-service -f app/ml_service/Dockerfile app/ml_service

docker run --rm -p 8000:8000 \
  -e MODEL_NAME=HousingModel \
  -e MODEL_PRODUCTION_ALIAS=champion \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  housing-ml-service
```

### Streamlit Frontend Container

```bash
docker build -t housing-streamlit -f app/server/Dockerfile app/server

docker run --rm -p 8501:8501 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  housing-streamlit
```

### Docker Compose (All Services)

*Note: Create `docker-compose.yml` for one-command startup of MLflow + FastAPI + Streamlit*

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ml_service

# Shut down
docker-compose down
```

## Extending with New Component Versions

To add a new version of any component (e.g., a new preprocessing pipeline):

### 1. Add Component Implementation

Edit `CD/components/component_registry.py`:

```python
# Example: Add preprocessing_v2
def _housing_preprocessing_v2(*, config: Dict[str, Any]):
    # New preprocessing logic
    preprocessing = build_preprocessing_pipeline()
    # ... apply different transforms ...
    return preprocessing

PREPROCESSING_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("housing_preprocessing", "v1"): _housing_preprocessing_v1,
    ("housing_preprocessing", "v2"): _housing_preprocessing_v2,  # NEW
}
```

### 2. Create New Manifest

Create `CD/config/manifests/default_v2.json` with:

```json
{
  "manifest_version": "1.0",
  "preprocessing": {
    "component": "housing_preprocessing",
    "version": "v2",
    "config": { ... }
  },
  ... other components ...
}
```

### 3. Run with New Version

```sh
CD_RUN_MANIFEST_PATH=CD/config/manifests/default_v2.json
python CD/main.py
```

MLflow will tag the run with `preprocessing_component:housing_preprocessing:v2` for full lineage.

## Troubleshooting

### Training Pipeline Issues

**"MLFLOW_TRACKING_URI must be set"**
- Set `MLFLOW_TRACKING_URI` in `.env` to your MLflow server URL.
- Verify MLflow server is running: `curl http://localhost:5000/health`

**"Run manifest not found"**
- Default path is `CD/config/manifests/default_v1.json`
- Override by setting `CD_RUN_MANIFEST_PATH` in `.env`.

**"Unsupported component 'xyz' version 'v2'"**
- Component not registered in `CD/components/component_registry.py`
- List available components in error message

**"Dataset not found at: assets/data/housing/housing.csv"**
- Download dataset: implement and run `CD/data/housing_data_ingestion.py:download_housing_dataset()`
- Or specify correct CSV path in manifest `dataset.config.path`

### Inference Service Issues

**"Model is not loaded. Please check MLflow server or reload."** (on /health or /predict)

1. Verify training completed and model registered:
   - Check MLflow UI: `http://localhost:5000`
   - Look for `HousingModel` model with `champion` alias

2. Check service logs:
  Watch the terminal output, or use `docker logs <container_id>` if the service runs in a container.

3. Try manual reload:
   ```bash
   curl -X POST http://localhost:8000/reload_model
   ```

**Service can't connect to MLflow**
- Verify `MLFLOW_TRACKING_URI` in `.env` points to the correct server
- Test connectivity: `curl http://localhost:5000/health`
- If using Docker, use the host address reachable from the container or join both services to the same Docker network

### Frontend Issues

**Streamlit can't connect to backend**
- Verify FastAPI service is running on `http://localhost:8000`
- Check `BACKEND_URL` in `.env`
- Test: `curl http://localhost:8000/health`

**"Cannot connect to backend at http://localhost:8000"**
- If using Docker, use the host address reachable from the container or configure a shared Docker network
- Ensure ports are exposed: `-p 8000:8000` in docker run

## Contributing

Key files to understand before extending:
- `CD/main.py` - Orchestration and component resolution
- `CD/components/component_registry.py` - Registry of all versioned implementations
- `CD/config/run_manifest.py` - Manifest schema and validation
- `CD/config/manifests/*.json` - Example configurations

Best practices:
- Add new component versions, don't edit existing ones (preserves lineage)
- Increment version suffix (v1 → v2) when changing behavior
- Update manifest examples when adding new component versions
- All component functions must match the same signature (kwargs only)

## License

See LICENSE file (if present).

## Contact

For questions, open an issue in the repository.
