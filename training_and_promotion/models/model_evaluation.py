from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error
from scipy import stats
import numpy as np


def evaluate_and_register_model(
    X_test,
    y_test,
    final_model,
    model_name="HousingModel",
    confidence=0.95
):

    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run found.")

    client = MlflowClient()

    # -------------------------
    # Predictions
    # -------------------------
    final_predictions = final_model.predict(X_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)

    mlflow.log_metric("test_rmse", final_rmse)
    mlflow.log_param("confidence_level", confidence)
    mlflow.log_param("test_set_size", len(X_test))

    # -------------------------
    # Confidence Interval
    # -------------------------
    def rmse(se):
        return np.sqrt(np.mean(se))

    squared_errors = (final_predictions - y_test) ** 2

    boot_result = stats.bootstrap(
        [squared_errors],
        rmse,
        confidence_level=confidence,
        random_state=42 # type: ignore
    )

    rmse_lower, rmse_upper = boot_result.confidence_interval

    mlflow.log_metric("test_rmse_ci_lower", rmse_lower)
    mlflow.log_metric("test_rmse_ci_upper", rmse_upper)

    # -------------------------
    # Log & Register Model
    # -------------------------
    pipeline_code_path = Path(__file__).resolve().parents[1] / "pipeline"
    if not pipeline_code_path.exists():
        raise FileNotFoundError(f"Pipeline code path not found: {pipeline_code_path}")

    model_info = mlflow_sklearn.log_model(
        final_model,
        registered_model_name=model_name,
        serialization_format="cloudpickle",
        code_paths=[str(pipeline_code_path)],
    )

    model_version = model_info.registered_model_version
    print(f"Registered Model Version: {model_version}")

    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=str(model_version)
    )


    return final_rmse, model_version
