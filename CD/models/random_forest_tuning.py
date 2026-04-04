from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
import numpy as np


def tune_random_forest_with_grid_search(X, y, preprocessing):

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])

    param_grid = [
        {
            'preprocessing__geo__clustersimilarity__n_clusters': [5, 8, 10],
            'random_forest__max_features': [4, 6, 8]
        },
        {
            'preprocessing__geo__clustersimilarity__n_clusters': [10, 15],
            'random_forest__max_features': [6, 8, 10]
        },
    ]

    with mlflow.start_run(run_name="rf_grid_search", nested=True):

        # Log search space (important!)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring", "neg_root_mean_squared_error")
        mlflow.log_dict({"param_grid": param_grid}, "param_grid.json")

        grid_search = GridSearchCV(
            full_pipeline,
            param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            return_train_score=True
        )

        grid_search.fit(X, y)

        # ----------------------------
        # Best Results
        # ----------------------------
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_rmse", best_score)

        # ----------------------------
        # Log full CV results table
        # ----------------------------
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("assets/grid_search_results.csv", index=False)
        mlflow.log_artifact("assets/grid_search_results.csv")

        # ----------------------------
        # Log Best Model
        # ----------------------------
        mlflow_sklearn.log_model(
            grid_search.best_estimator_,
            name="best_random_forest_model",
            serialization_format="cloudpickle"
        )

    return grid_search
