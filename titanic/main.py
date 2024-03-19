"""The entrypoint module."""
from pathlib import Path

import pandas as pd
import typer

import titanic.constants as c
import titanic.operations as op
import titanic.preprocess as pre


def run(input_path: Path, test_path: Path, result_path: Path = c.RESULT_DATA_PATH) -> None:
    """Runs the process including data splitting, preprocessing, training, prediction, and saving
    predictions."""
    titanic_data = pd.read_csv(input_path)
    titanic_test_data = pd.read_csv(test_path)

    strat_train_set, strat_test_set = pre.split_data(titanic_data)

    X_data, Y_data = pre.preprocess_data(strat_train_set)

    final_clf = op.train(X_data, Y_data)
    X_data_test, Y_data_test = pre.preprocess_data(strat_test_set)

    print(f"Prediction score: {final_clf.score(X_data_test, Y_data_test)}")

    X_data_final, y_data_final = pre.preprocess_data(titanic_data)
    prod_final_clf = op.train(X_data_final, y_data_final)
    X_data_final_test = pre.preprocess_prediction_data(titanic_test_data)

    op.predict(prod_final_clf, X_data_final_test, titanic_test_data, result_path)

    titanic_data = pd.read_csv(input_path)
    titanic_test_data = pd.read_csv(test_path)

    strat_train_set, strat_test_set = pre.split_data(titanic_data)

    X_data, Y_data = pre.preprocess_data(strat_train_set)


def start():
    typer.run(run)
