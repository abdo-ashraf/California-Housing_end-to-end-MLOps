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

- `main.py`: Thin entrypoint (env + manifest + runtime config resolution)
- `services/training_pipeline_service.py`: Training pipeline orchestration service
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

If your current directory is `training_and_promotion/`, use:

```sh
python -m pip install -r requirements.txt
```

## Environment Configuration

The training_and_promotion pipeline auto-loads environment variables from `training_and_promotion/.env`.

Recommended setup:

```powershell
Copy-Item training_and_promotion/.env.example training_and_promotion/.env
```

If `training_and_promotion/.env` is not present, set variables in your shell before running the pipeline.

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
3. Manifest defaults

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

## Add A New Component Version

Follow this workflow when adding a new version (for example, `trainer:v2`).

1. Pick the component section to extend:
   - `dataset`, `tracker`, `splitter`, `preprocessing`, `trainer`, `evaluation`, or `promotion`.
2. Implement a new versioned function in `training_and_promotion/components/component_registry.py`.
3. Register the new function in the matching `*_COMPONENTS` dictionary with a new key tuple.
4. Create a new manifest JSON file (recommended) under `training_and_promotion/config/manifests/`.
5. Point the run to that manifest and execute the pipeline.

Example: add `trainer` version `v2`

```python
# training_and_promotion/components/component_registry.py
def _random_forest_pipeline_v2(*, preprocessing, config: Dict[str, Any]):
    max_features = config.get("max_features", 8)
    random_state = int(config.get("random_state", 42))
    n_estimators = int(config.get("n_estimators", 500))

    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            random_state=random_state,
            max_features=max_features,
            n_estimators=n_estimators,
        ),
    )
    return model


TRAINER_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("random_forest_pipeline", "v1"): _random_forest_pipeline_v1,
    ("random_forest_pipeline", "v2"): _random_forest_pipeline_v2,
}
```

Then create a manifest like `training_and_promotion/config/manifests/trainer_v2.json`:

```json
{
  "trainer": {
    "component": "random_forest_pipeline",
    "version": "v2",
    "config": {
      "random_state": 42,
      "max_features": 8,
      "n_estimators": 500
    }
  }
}
```

Run with the new manifest:

```powershell
$env:CD_RUN_MANIFEST_PATH = "training_and_promotion/config/manifests/trainer_v2.json"
python training_and_promotion/main.py
```

Important notes:

- Keep function signatures compatible with the section type:
  - Dataset: `(*, config, project_root)`
  - Tracker: `(*, tracking_uri, stage, config)`
  - Splitter: `(*, data, config)`
  - Preprocessing: `(*, config)`
  - Trainer: `(*, preprocessing, config)`
  - Evaluation: `(*, X_test, y_test, final_model, model_name, config)`
  - Promotion: `(*, new_version, model_name, config)`
- If you forget to register a new `(component, version)` pair, the pipeline raises an `Unsupported component` error with available options.
- Prefer creating a new manifest file for each experiment version instead of modifying the default manifest in place.

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
  - Set `MLFLOW_TRACKING_URI` in your shell or `training_and_promotion/.env`.
- Manifest errors:
  - Verify all required sections and fields exist in manifest JSON.
- Unsupported component/version:
  - Confirm pair exists in `components/component_registry.py`.
- Promotion not happening:
  - Check `test_rmse` on new run versus current champion run.

## Related Docs

- Root project guide: `README.md`
- Runtime app stack (FastAPI + Streamlit + Compose): `app/README.md`
