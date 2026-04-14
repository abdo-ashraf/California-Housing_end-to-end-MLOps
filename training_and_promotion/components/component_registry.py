from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple, cast

from data.data_splitting import stratified_income_train_test_split
from models import evaluate_and_register_model, promote_model_if_better
from pipeline.preprocessing_pipeline import build_preprocessing_pipeline
from tracking.experiment_setup import setup_mlflow_experiment

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


ComponentKey = Tuple[str, str]
ComponentFn = Callable[..., Any]


def _require_component(registry: Dict[ComponentKey, ComponentFn], component: str, version: str) -> ComponentFn:
    key = (component, version)
    if key not in registry:
        available = ", ".join([f"{name}:{ver}" for name, ver in sorted(registry.keys())])
        raise ValueError(
            f"Unsupported component '{component}' version '{version}'. "
            f"Available: {available}"
        )
    return registry[key]


# -------------------------
# Dataset components
# -------------------------
def _dataset_local_csv_v1(*, config: Dict[str, Any], project_root: Path) -> Path:
    path_value = config.get("path", "assets/data/housing/housing.csv")
    csv_path = project_root / path_value
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return csv_path


DATASET_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("local_csv", "v1"): _dataset_local_csv_v1,
}


def resolve_dataset_component(component: str, version: str) -> ComponentFn:
    return _require_component(DATASET_COMPONENTS, component, version)


# -------------------------
# Splitter components
# -------------------------
def _income_stratified_split_v1(*, data, config: Dict[str, Any]):
    return stratified_income_train_test_split(
        data,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("random_state", 42)),
    )


SPLITTER_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("income_stratified_split", "v1"): _income_stratified_split_v1,
}


def resolve_splitter_component(component: str, version: str) -> ComponentFn:
    return _require_component(SPLITTER_COMPONENTS, component, version)


# -------------------------
# Preprocessing components
# -------------------------
def _housing_preprocessing_v1(*, config: Dict[str, Any]):
    preprocessing = build_preprocessing_pipeline()

    if "geo_n_clusters" in config:
        preprocessing = preprocessing.set_params(
            geo__clustersimilarity__n_clusters=int(config["geo_n_clusters"])
        )

    if "geo_gamma" in config:
        preprocessing = preprocessing.set_params(
            geo__clustersimilarity__gamma=float(config["geo_gamma"])
        )

    return preprocessing


PREPROCESSING_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("housing_preprocessing", "v1"): _housing_preprocessing_v1,
}


def resolve_preprocessing_component(component: str, version: str) -> ComponentFn:
    return _require_component(PREPROCESSING_COMPONENTS, component, version)


# -------------------------
# Trainer components
# -------------------------
def _random_forest_pipeline_v1(*, preprocessing, config: Dict[str, Any]):
    max_features = config.get("max_features", 6)
    random_state = int(config.get("random_state", 42))

    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            random_state=random_state,
            max_features=max_features,
        ),
    )

    return model


TRAINER_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("random_forest_pipeline", "v1"): _random_forest_pipeline_v1,
}


def resolve_trainer_component(component: str, version: str) -> ComponentFn:
    return _require_component(TRAINER_COMPONENTS, component, version)


# -------------------------
# Evaluation components
# -------------------------
def _mlflow_evaluation_register_v1(*, X_test, y_test, final_model, model_name: str, config: Dict[str, Any]):
    confidence = float(config.get("confidence", 0.95))
    return evaluate_and_register_model(
        X_test,
        y_test,
        final_model,
        model_name=model_name,
        confidence=confidence,
    )


EVALUATION_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("mlflow_model_eval_register", "v1"): _mlflow_evaluation_register_v1,
}


def resolve_evaluation_component(component: str, version: str) -> ComponentFn:
    return _require_component(EVALUATION_COMPONENTS, component, version)


# -------------------------
# Promotion components
# -------------------------
def _champion_rmse_policy_v1(*, new_version: str, model_name: str, config: Dict[str, Any]) -> bool:
    return bool(
        promote_model_if_better(
            new_version=new_version,
            model_name=model_name,
            metric_name=config.get("metric_name", "test_rmse"),
            lower_is_better=bool(config.get("lower_is_better", True)),
            champion_alias=config.get("champion_alias", "champion"),
        )
    )


PROMOTION_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("champion_rmse_policy", "v1"): _champion_rmse_policy_v1,
}


def resolve_promotion_component(component: str, version: str) -> ComponentFn:
    return _require_component(PROMOTION_COMPONENTS, component, version)


# -------------------------
# Tracker components
# -------------------------
def _mlflow_experiment_tracker_v1(*, tracking_uri: str, stage: str, config: Dict[str, Any]) -> str:
    if stage not in {"dev", "prod"}:
        raise ValueError("Tracker stage must be either 'dev' or 'prod'")
    stage_literal = cast(Literal["dev", "prod"], stage)

    exp_name = str(config.get("experiment_name", "california_housing_experiment"))
    return setup_mlflow_experiment(
        tracking_uri=tracking_uri,
        stage=stage_literal,
        exp_name=exp_name,
    )


TRACKER_COMPONENTS: Dict[ComponentKey, ComponentFn] = {
    ("mlflow_experiment", "v1"): _mlflow_experiment_tracker_v1,
}


def resolve_tracker_component(component: str, version: str) -> ComponentFn:
    return _require_component(TRACKER_COMPONENTS, component, version)
