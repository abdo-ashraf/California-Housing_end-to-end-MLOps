import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow


def stratified_income_train_test_split(data: pd.DataFrame,
                                       test_size: float = 0.2,
                                       random_state: int = 42,
                                       bins: tuple = (0., 1.5, 3.0, 4.5, 6., np.inf),
                                       labels: tuple = (1, 2, 3, 4, 5),
                                       include_lowest: bool = True):

    working_data = data.copy()
    working_data["income_cat"] = pd.cut(
        working_data["median_income"],
        bins=bins,
        labels=labels,
        include_lowest=include_lowest
    )

    strat_train_set, strat_test_set = train_test_split(
        working_data,
        test_size=test_size,
        random_state=random_state,
        stratify=working_data["income_cat"]
    )

    if mlflow.active_run() is not None:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("stratify", "income_cat")

    ## No need for income_cat anymore
    strat_train_set.drop("income_cat", axis=1, inplace=True)
    strat_test_set.drop("income_cat", axis=1, inplace=True)

    # train_dataset = mlflow.data.from_pandas(
    # strat_train_set, source=None, name="california-housing-train", targets="median_house_value")

    # test_dataset = mlflow.data.from_pandas(
    # strat_test_set, source=None, name="california-housing-test", targets="median_house_value")

    # mlflow.log_input(train_dataset, context="training")
    # mlflow.log_input(test_dataset, context="testing")

    return strat_train_set, strat_test_set