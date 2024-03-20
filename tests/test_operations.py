import os

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from sklearn.ensemble import RandomForestClassifier

from titanic import operations as op
from titanic import preprocess as pre

# @given(
#     st.lists(st.integers(min_value=1, max_value=100), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=1), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=1, max_value=3), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["male", "female"]), min_size=8, max_size=8),
#     st.lists(st.floats(min_value=0.5, max_value=100), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=8), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=6), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.floats(min_value=4, max_value=130), min_size=8, max_size=8), #fare
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["S", "C", "Q"]), min_size=8, max_size=8)
# )
# def test_train(PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):
#     titanic_df = pd.DataFrame(
#         {
#             "PassengerId": PassengerId,
#             "Survived": Survived,
#             "Pclass": Pclass,
#             "Name": Name,
#             "Sex": Sex,
#             "Age": Age,
#             "SibSp": SibSp,
#             "Parch": Parch,
#             "Ticket": Ticket,
#             "Fare": Fare,
#             "Cabin": Cabin,
#             "Embarked": Embarked,
#         }
#     )
#     X, y = pre.preprocess_data(titanic_df)
#     clf = op.train(X, y)
#     assert isinstance(clf, RandomForestClassifier)


def test_train(sample_data: pd.DataFrame):
    X, y = pre.preprocess_data(sample_data)
    clf = op.train(X, y)
    assert isinstance(clf, RandomForestClassifier)


# @given(
#     st.lists(st.integers(min_value=1, max_value=100), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=1), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=1, max_value=3), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["male", "female"]), min_size=8, max_size=8),
#     st.lists(st.floats(min_value=0.5, max_value=100), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=8), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=6), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.floats(min_value=4, max_value=130), min_size=8, max_size=8), #fare
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["S", "C", "Q"]), min_size=8, max_size=8)
# )
# def test_predict(PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, sample_test: pd.DataFrame):
#     titanic_df = pd.DataFrame(
#         {
#             "PassengerId": PassengerId,
#             "Survived": Survived,
#             "Pclass": Pclass,
#             "Name": Name,
#             "Sex": Sex,
#             "Age": Age,
#             "SibSp": SibSp,
#             "Parch": Parch,
#             "Ticket": Ticket,
#             "Fare": Fare,
#             "Cabin": Cabin,
#             "Embarked": Embarked,
#         }
#     )
#     X, y = pre.preprocess_data(titanic_df)
#     clf = op.train(X, y)

#     result_path = "data/result.csv"
#     test = pre.preprocess_prediction_data(sample_test)
#     predictions = op.predict(clf, test, sample_test, result_path)
#     assert len(predictions) > 0
#     assert os.path.exists(result_path)

#     os.remove(result_path)


def test_predict(sample_data: pd.DataFrame, sample_test: pd.DataFrame):
    X, y = pre.preprocess_data(sample_data)
    clf = op.train(X, y)

    result_path = "data/result.csv"
    test = pre.preprocess_prediction_data(sample_test)
    predictions = op.predict(clf, test, sample_test, result_path)
    assert len(predictions) > 0
    assert os.path.exists(result_path)

    os.remove(result_path)
