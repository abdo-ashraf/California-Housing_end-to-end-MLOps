from pathlib import Path
from typing import Literal, Tuple

import mlflow
import pandas as pd

from components import (
    get_dataset_component,
    get_evaluation_component,
    get_preprocessing_component,
    get_promotion_component,
    get_splitter_component,
    get_tracker_component,
    get_trainer_component,
)
from config import RunManifest


def _resolve_training_data(train_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = train_set.drop("median_house_value", axis=1)
    target = train_set["median_house_value"]
    return features, target


def _resolve_test_data(test_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = test_set.drop("median_house_value", axis=1)
    target = test_set["median_house_value"]
    return features, target


def _build_manifest_tags(manifest: RunManifest) -> dict[str, str]:
    return {
        "manifest_version": manifest.manifest_version,
        "dataset_component": f"{manifest.dataset['component']}:{manifest.dataset['version']}",
        "tracker_component": f"{manifest.tracker['component']}:{manifest.tracker['version']}",
        "splitter_component": f"{manifest.splitter['component']}:{manifest.splitter['version']}",
        "preprocessing_component": f"{manifest.preprocessing['component']}:{manifest.preprocessing['version']}",
        "trainer_component": f"{manifest.trainer['component']}:{manifest.trainer['version']}",
        "evaluation_component": f"{manifest.evaluation['component']}:{manifest.evaluation['version']}",
        "promotion_component": f"{manifest.promotion['component']}:{manifest.promotion['version']}",
    }


def run_training_pipeline(
    *,
    manifest: RunManifest,
    project_root: Path,
    manifest_path: Path,
    tracking_uri: str,
    stage: Literal["dev", "prod"],
    model_name: str,
) -> Tuple[float, int, bool]:
    tracker_setup = get_tracker_component(
        manifest.tracker["component"],
        manifest.tracker["version"],
    )

    experiment_name = tracker_setup(
        tracking_uri=tracking_uri,
        stage=stage,
        config=manifest.tracker["config"],
    )
    print("MLflow tracking initialized successfully.", tracking_uri, experiment_name)

    dataset_resolver = get_dataset_component(
        manifest.dataset["component"],
        manifest.dataset["version"],
    )
    csv_path = dataset_resolver(
        config=manifest.dataset["config"],
        project_root=project_root,
    )

    data = pd.read_csv(csv_path)

    splitter = get_splitter_component(
        manifest.splitter["component"],
        manifest.splitter["version"],
    )
    train_set, test_set = splitter(
        data=data,
        config=manifest.splitter["config"],
    )

    training_features, training_target = _resolve_training_data(train_set)

    preprocessing_builder = get_preprocessing_component(
        manifest.preprocessing["component"],
        manifest.preprocessing["version"],
    )
    preprocessing_pipeline = preprocessing_builder(config=manifest.preprocessing["config"])

    trainer_builder = get_trainer_component(
        manifest.trainer["component"],
        manifest.trainer["version"],
    )
    final_model = trainer_builder(
        preprocessing=preprocessing_pipeline,
        config=manifest.trainer["config"],
    )

    evaluation_runner = get_evaluation_component(
        manifest.evaluation["component"],
        manifest.evaluation["version"],
    )

    promotion_runner = get_promotion_component(
        manifest.promotion["component"],
        manifest.promotion["version"],
    )

    test_features, test_target = _resolve_test_data(test_set)

    with mlflow.start_run():
        mlflow.log_param("dataset_path", str(csv_path))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("stage", stage)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("manifest_path", str(manifest_path))
        mlflow.log_dict(manifest.raw, "run_manifest.json")

        mlflow.set_tags(_build_manifest_tags(manifest))

        final_model.fit(training_features, training_target)
        rmse, model_version = evaluation_runner(
            X_test=test_features,
            y_test=test_target,
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

    return rmse, int(model_version), promoted
