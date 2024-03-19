import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import shapiro

from titanic import preprocess as pre


def test_split_data(mock_data: pd.DataFrame):
    train_set, test_set = pre.split_data(mock_data)
    input_length = len(mock_data)
    test_length = len(test_set)
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    assert not train_set.empty
    assert not test_set.empty
    assert 0.15 <= test_length / input_length <= 0.25


# @given(
#     st.lists(st.integers(min_value=1, max_value=100), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=1), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=1, max_value=3), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["male", "female"]), min_size=8, max_size=8),
#     st.lists(st.floats(allow_nan=True), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=8), min_size=8, max_size=8),
#     st.lists(st.integers(min_value=0, max_value=6), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.floats(allow_nan=True), min_size=8, max_size=8),
#     st.lists(st.text(), min_size=8, max_size=8),
#     st.lists(st.sampled_from(["S", "C", "Q"]), min_size=8, max_size=8)
# )
# def test_preprocess_data(PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):
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
#     res = shapiro(y)
#     assert res.statistic > 0.05
#     res = shapiro(X)
#     assert res.statistic > 0.05


def test_preprocess_data(mock_data: pd.DateOffset):
    X, y = pre.preprocess_data(mock_data)
    res = shapiro(y)
    assert res.statistic > 0.05
    res = shapiro(X)
    assert res.statistic > 0.05
