from typing import Literal
import mlflow

def setup_mlflow_experiment(tracking_uri, stage: Literal["dev", "prod"], exp_name="california_housing_experiment"):

    assert stage in ["dev", "prod"], "Stage must be either 'dev' or 'prod'"

    mlflow.set_tracking_uri(tracking_uri)
    exp_name_tag = exp_name+"_"+stage

    # Check if it exists to avoid redundancy or errors
    exp = mlflow.get_experiment_by_name(exp_name_tag)

    if not exp:
        # Set the custom artifact path only once at creation
        mlflow.create_experiment(exp_name_tag, artifact_location=None, tags={"stage": stage})

    # Set the active experiment
    mlflow.set_experiment(exp_name_tag)

    return exp_name_tag
