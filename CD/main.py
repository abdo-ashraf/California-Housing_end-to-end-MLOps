import os
from pathlib import Path
from typing import Literal, cast

import mlflow
from tracking.experiment_setup import setup_mlflow_experiment
from data.data_splitting import stratified_income_train_test_split
from pipeline.preprocessing_pipeline import build_preprocessing_pipeline
from models import evaluate_and_register_model, promote_model_if_better
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


def resolve_dataset_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "assets" / "data" / "housing" / "housing.csv"


def main():
    # Initialize MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "california_housing_experiment")
    stage_raw = os.getenv("MLFLOW_STAGE", "prod")
    model_name = os.getenv("MODEL_NAME", "HousingModel")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set")

    if stage_raw not in {"dev", "prod"}:
        raise ValueError("MLFLOW_STAGE must be either 'dev' or 'prod'")
    stage = cast(Literal["dev", "prod"], stage_raw)

    print("Initializing MLflow tracking")
    experiment_name = setup_mlflow_experiment(
        tracking_uri=tracking_uri,
        stage=stage,
        exp_name=exp_name
    )
    print("MLflow tracking initialized successfully.", tracking_uri, exp_name, stage)

    # Load the data
    csv_path = resolve_dataset_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    data = pd.read_csv(csv_path)

    # Perform stratified split
    train_set, test_set = stratified_income_train_test_split(data)

    ## DEV: Train on the subset of the training set to speed up development iterations
    # train_set = train_set.sample(frac=0.1, random_state=42)

    X, y = train_set.drop("median_house_value", axis=1), train_set["median_house_value"]

    # Build the preprocessing pipeline
    preprocessing_pipeline = build_preprocessing_pipeline()

    ## TRAIN BEST PARAMS RANDOM FOREST MODEL
    final_model = make_pipeline(
        preprocessing_pipeline,
        RandomForestRegressor(random_state=42, max_features=6)
    )

    # Apply chosen geo-cluster hyperparameter before fitting.
    final_model = final_model.set_params(
        columntransformer__geo__clustersimilarity__n_clusters=15
    )

    # Evaluate the best model on the test set
    X_housing_test, y_housing_test = test_set.drop("median_house_value", axis=1), test_set["median_house_value"]
    with mlflow.start_run(run_name="cd_train_evaluate_promote"):
        mlflow.log_param("dataset_path", str(csv_path))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("stage", stage)
        mlflow.log_param("experiment_name", experiment_name)

        final_model.fit(X, y)
        rmse, model_version = evaluate_and_register_model(
            X_housing_test,
            y_housing_test,
            final_model,
            model_name=model_name
        )

        # Promote the model if it performs better than the current champion model.
        promoted = promote_model_if_better(
            new_version=str(model_version),
            model_name=model_name
        )
        mlflow.log_metric("promoted_to_champion", int(promoted))

    print(
        f"Pipeline finished. test_rmse={rmse:.4f}, "
        f"model_version={model_version}, promoted={promoted}"
    )



if __name__ == "__main__":
    main()