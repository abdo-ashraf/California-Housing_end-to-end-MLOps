import os
from pathlib import Path
from typing import Literal, cast

import mlflow
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

from components import (
    resolve_dataset_component,
    resolve_evaluation_component,
    resolve_preprocessing_component,
    resolve_promotion_component,
    resolve_splitter_component,
    resolve_tracker_component,
    resolve_trainer_component,
)
from config import load_run_manifest


def resolve_project_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root


def resolve_manifest_path(project_root: Path) -> Path:
    manifest_raw = os.getenv("CD_RUN_MANIFEST_PATH", "CD/config/manifests/default_v1.json")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return manifest_path


def main():
    project_root = resolve_project_root()
    manifest_path = resolve_manifest_path(project_root)
    manifest = load_run_manifest(manifest_path)

    # Resolve stage/model values with env taking precedence.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    stage_raw = os.getenv("MLFLOW_STAGE", manifest.stage)
    model_name = os.getenv("MODEL_NAME", manifest.model_name)

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set")

    if stage_raw not in {"dev", "prod"}:
        raise ValueError("MLFLOW_STAGE must be either 'dev' or 'prod'")
    stage = cast(Literal["dev", "prod"], stage_raw)

    print("Resolving tracker component")
    tracker_setup = resolve_tracker_component(
        manifest.tracker["component"],
        manifest.tracker["version"],
    )

    experiment_name = tracker_setup(
        tracking_uri=tracking_uri,
        stage=stage,
        config=manifest.tracker["config"],
    )
    print("MLflow tracking initialized successfully.", tracking_uri, experiment_name, stage)

    dataset_resolver = resolve_dataset_component(
        manifest.dataset["component"],
        manifest.dataset["version"],
    )
    csv_path = dataset_resolver(
        config=manifest.dataset["config"],
        project_root=project_root,
    )

    data = pd.read_csv(csv_path)

    splitter = resolve_splitter_component(
        manifest.splitter["component"],
        manifest.splitter["version"],
    )
    train_set, test_set = splitter(
        data=data,
        config=manifest.splitter["config"],
    )

    ## DEV: Train on the subset of the training set to speed up development iterations
    # train_set = train_set.sample(frac=0.1, random_state=42)

    X, y = train_set.drop("median_house_value", axis=1), train_set["median_house_value"]

    preprocessing_builder = resolve_preprocessing_component(
        manifest.preprocessing["component"],
        manifest.preprocessing["version"],
    )
    preprocessing_pipeline = preprocessing_builder(config=manifest.preprocessing["config"])

    trainer_builder = resolve_trainer_component(
        manifest.trainer["component"],
        manifest.trainer["version"],
    )
    final_model = trainer_builder(
        preprocessing=preprocessing_pipeline,
        config=manifest.trainer["config"],
    )

    evaluation_runner = resolve_evaluation_component(
        manifest.evaluation["component"],
        manifest.evaluation["version"],
    )

    promotion_runner = resolve_promotion_component(
        manifest.promotion["component"],
        manifest.promotion["version"],
    )

    X_housing_test, y_housing_test = test_set.drop("median_house_value", axis=1), test_set["median_house_value"]

    with mlflow.start_run(run_name=manifest.run_name):
        mlflow.log_param("dataset_path", str(csv_path))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("stage", stage)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("manifest_path", str(manifest_path))
        mlflow.log_dict(manifest.raw, "run_manifest.json")

        mlflow.set_tags({
            "manifest_version": manifest.manifest_version,
            "dataset_component": f"{manifest.dataset['component']}:{manifest.dataset['version']}",
            "tracker_component": f"{manifest.tracker['component']}:{manifest.tracker['version']}",
            "splitter_component": f"{manifest.splitter['component']}:{manifest.splitter['version']}",
            "preprocessing_component": f"{manifest.preprocessing['component']}:{manifest.preprocessing['version']}",
            "trainer_component": f"{manifest.trainer['component']}:{manifest.trainer['version']}",
            "evaluation_component": f"{manifest.evaluation['component']}:{manifest.evaluation['version']}",
            "promotion_component": f"{manifest.promotion['component']}:{manifest.promotion['version']}",
        })

        final_model.fit(X, y)
        rmse, model_version = evaluation_runner(
            X_test=X_housing_test,
            y_test=y_housing_test,
            final_model=final_model,
            model_name=model_name,
            config=manifest.evaluation["config"],
        )

        promoted = promotion_runner(
            new_version=str(model_version),
            model_name=model_name,
            config=manifest.promotion["config"],
        )
        mlflow.log_metric("promoted_to_champion", int(promoted))

    print(
        f"Pipeline finished. test_rmse={rmse:.4f}, "
        f"model_version={model_version}, promoted={promoted}"
    )



if __name__ == "__main__":
    main()