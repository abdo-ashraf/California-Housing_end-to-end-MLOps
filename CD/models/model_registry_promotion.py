from mlflow.tracking import MlflowClient


def promote_model_if_better(
    new_version,
    model_name="HousingModel",
    metric_name="test_rmse",
    lower_is_better=True,
    champion_alias="champion"
):

    client = MlflowClient()

    # --- Get new model metrics ---
    new_model = client.get_model_version(model_name, new_version)
    new_run = client.get_run(new_model.run_id)
    new_metric = new_run.data.metrics.get(metric_name)

    if new_metric is None:
        raise ValueError(f"Metric '{metric_name}' not found in new model run.")

    # --- Try to get current champion via alias ---
    try:
        champion_version = client.get_model_version_by_alias(
            model_name,
            champion_alias
        )

        champ_run = client.get_run(champion_version.run_id)
        champ_metric = champ_run.data.metrics.get(metric_name)

        if champ_metric is None:
            raise ValueError(f"Metric '{metric_name}' not found in champion run.")

        print(f"New {metric_name}: {new_metric}")
        print(f"Current Champion {metric_name}: {champ_metric}")

        is_better = (
            new_metric < champ_metric
            if lower_is_better
            else new_metric > champ_metric
        )

        if is_better:
            print("New model is better. Promoting to Champion.")
            client.set_registered_model_alias(
                name=model_name,
                alias=champion_alias,
                version=new_version
            )
            return True
        else:
            print("New model is NOT better. Keeping current Champion.")
            return False

    except Exception:
        # No champion exists yet
        print("No Champion found. Assigning new model as Champion.")
        
        # Remove existing staging alias if it exists
        try:
            client.delete_registered_model_alias(
                name=model_name,
                alias=champion_alias
            )
        except Exception:
            pass  # If alias doesn't exist, no need to delete it

        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=new_version
        )
        return True
