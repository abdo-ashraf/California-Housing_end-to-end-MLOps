# Training And Promotion Layer Guide

This folder contains the training and promotion pipeline for the California Housing MLOps project.

The training_and_promotion layer is manifest-driven: component names and versions are selected from a run manifest, then resolved at runtime from the component registry.

## What This Pipeline Does

- Loads a run manifest from `CD_RUN_MANIFEST_PATH`
- Resolves versioned components for dataset, tracker, splitter, preprocessing, trainer, evaluation, and promotion
- Trains the model pipeline defined by the selected trainer component
- Evaluates RMSE and bootstrap confidence interval
- Logs metrics, params, artifacts, and lineage tags to MLflow
- Registers the trained model in MLflow Model Registry
- Assigns `staging` alias and conditionally promotes to `champion`

## Folder Structure

- `main.py`: Pipeline orchestrator
- `components/component_registry.py`: Versioned component registry and resolver functions
- `config/run_manifest.py`: Manifest loader and schema validation
- `config/manifests/default_v1.json`: Default manifest
- `data/`: Data loading and splitting logic
- `pipeline/`: Feature engineering and preprocessing pipeline
- `models/`: Evaluation, registration, and promotion policy
- `tracking/`: MLflow experiment setup
- `assets/data/housing/housing.csv`: Dataset used by default manifest

## Prerequisites

- Python 3.11+
- MLflow tracking backend (remote or local)
- Valid environment variables (see below)

Install dependencies:

```sh
python -m pip install -r training_and_promotion/requirements.txt
```

## Environment Configuration

The training_and_promotion pipeline auto-loads environment variables using `find_dotenv()`.

Recommended setup:

```powershell
Copy-Item training_and_promotion/.env.example training_and_promotion/.env
```

If `training_and_promotion/.env` is not present, it falls back to the nearest parent `.env` (for example, repository root).

Required:

- `MLFLOW_TRACKING_URI`

Common optional overrides:

- `CD_RUN_MANIFEST_PATH` (default: `training_and_promotion/config/manifests/default_v1.json`)
- `MLFLOW_STAGE` (`dev` or `prod`; default comes from manifest)
- `MODEL_NAME` (default comes from manifest)
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

## Run the Pipeline

From repository root:

```sh
python training_and_promotion/main.py
```

Typical console flow:

- Resolves tracker component
- Initializes MLflow experiment
- Trains and evaluates the model selected by the manifest
- Registers model version
- Applies champion promotion policy
- Prints final summary with RMSE, model version, and promotion result

## Manifest-Driven Configuration

Default manifest:

- `training_and_promotion/config/manifests/default_v1.json`

Required top-level sections:

- `dataset`
- `tracker`
- `splitter`
- `preprocessing`
- `trainer`
- `evaluation`
- `promotion`

Each section must define:

- `component`
- `version`
- `config` (object; can be empty)

Example pattern:

```json
{
  "trainer": {
    "component": "random_forest_pipeline",
    "version": "v1",
    "config": {
      "random_state": 42,
      "max_features": 6
    }
  }
}
```

Use another manifest:

```sh
CD_RUN_MANIFEST_PATH=training_and_promotion/config/manifests/default_v1.json
python training_and_promotion/main.py
```

PowerShell:

```powershell
$env:CD_RUN_MANIFEST_PATH = "training_and_promotion/config/manifests/default_v1.json"
python training_and_promotion/main.py
```

## Configuration Precedence

For runtime values below, precedence is:

1. Shell environment variables
2. `training_and_promotion/.env`
3. Parent `.env` (for example, repository root)
4. Manifest defaults

Applies to:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_STAGE`
- `MODEL_NAME`

Component behavior still comes from manifest `config` values.

## Component Catalog (Current)

- Dataset: `local_csv:v1`
- Tracker: `mlflow_experiment:v1`
- Splitter: `income_stratified_split:v1`
- Preprocessing: `housing_preprocessing:v1`
- Trainer: `random_forest_pipeline:v1`
- Evaluation: `mlflow_model_eval_register:v1`
- Promotion: `champion_rmse_policy:v1`

To add a new version, implement it in `components/component_registry.py` and reference it from a new manifest.

## MLflow Outputs

During each run, the pipeline logs:

- Params: dataset path, model name, stage, experiment name, manifest path
- Metrics: `test_rmse`, CI bounds, `promoted_to_champion`
- Artifact: `run_manifest.json`
- Tags: manifest version + selected component versions

Model registry behavior:

- New model version is registered under `MODEL_NAME`
- `staging` alias is updated to new version
- `champion` alias is updated only if promotion policy passes

## Troubleshooting

- `MLFLOW_TRACKING_URI must be set`:
  - Set `MLFLOW_TRACKING_URI` in your shell or `.env`.
- Manifest errors:
  - Verify all required sections and fields exist in manifest JSON.
- Unsupported component/version:
  - Confirm pair exists in `components/component_registry.py`.
- Promotion not happening:
  - Check `test_rmse` on new run versus current champion run.

## Related Docs

- Root project guide: `README.md`
- Runtime app stack (FastAPI + Streamlit + Compose): `app/README.md`
