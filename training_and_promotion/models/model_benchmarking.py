from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
import numpy as np


def train_and_benchmark_models(X, y, preprocessing):

    df_results = {
        "models": [],
        "full_data_train_rmse": [],
        "cv_mean_rmse": [],
        "cv_std_rmse": []
    }

    models = {
        "lin_reg": LinearRegression(),
        "tree_reg": DecisionTreeRegressor(random_state=42),
        "forest_reg": RandomForestRegressor(random_state=42)
    }

    for model_name, model in models.items():

        print(f"{model_name} training...")

        # Nested run only
        with mlflow.start_run(run_name=model_name, nested=True):

            pipeline = make_pipeline(preprocessing, model)
            pipeline.fit(X, y)

            # Train RMSE
            predictions = pipeline.predict(X)
            train_rmse = root_mean_squared_error(y, predictions)

            # Cross-validation
            cv_scores = -cross_val_score(
                pipeline,
                X,
                y,
                scoring="neg_root_mean_squared_error",
                cv=10
            )

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Log params
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metric("train_rmse", float(train_rmse))
            mlflow.log_metric("cv_mean_rmse", float(cv_mean))
            mlflow.log_metric("cv_std_rmse", float(cv_std))

            # Log model artifact
            mlflow_sklearn.log_model(pipeline, name=model_name,
            serialization_format="cloudpickle")

            # Store results
            df_results["models"].append(model_name)
            df_results["full_data_train_rmse"].append(train_rmse)
            df_results["cv_mean_rmse"].append(cv_mean)
            df_results["cv_std_rmse"].append(cv_std)

    return df_results
