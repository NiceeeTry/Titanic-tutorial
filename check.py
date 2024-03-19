import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames


@given(
    st.lists(st.integers(min_value=1, max_value=100), min_size=8, max_size=8),
    st.lists(st.integers(min_value=0, max_value=1), min_size=8, max_size=8),
    st.lists(st.integers(min_value=1, max_value=3), min_size=8, max_size=8),
    st.lists(st.text(), min_size=8, max_size=8),
    st.lists(st.sampled_from(["male", "female"]), min_size=8, max_size=8),
    st.lists(st.floats(min_value=0.5, max_value=100), min_size=8, max_size=8),
    st.lists(st.integers(min_value=0, max_value=8), min_size=8, max_size=8),
    st.lists(st.integers(min_value=0, max_value=6), min_size=8, max_size=8),
    st.lists(st.text(), min_size=8, max_size=8),
    st.lists(st.floats(min_value=4, max_value=100), min_size=8, max_size=8),
    st.lists(st.text(), min_size=8, max_size=8),
    st.lists(st.sampled_from(["S", "C", "Q"]), min_size=8, max_size=8),
)
def test_split_data(*args):
    unique_lists = [
        arg.draw(st.lists(st.unique(arg.example()), min_size=8, max_size=8)) for arg in args
    ]
    df = pd.DataFrame(
        {
            "PassengerId": unique_lists[0],
            "Survived": unique_lists[1],
            "Pclass": unique_lists[2],
            "Name": unique_lists[3],
            "Sex": unique_lists[4],
            "Age": unique_lists[5],
            "SibSp": unique_lists[6],
            "Parch": unique_lists[7],
            "Ticket": unique_lists[8],
            "Fare": unique_lists[9],
            "Cabin": unique_lists[10],
            "Embarked": unique_lists[11],
        }
    )
    print(df)

    # df = pd.DataFrame(
    #     {
    #         "PassengerId": PassengerId,
    #         "Survived": Survived,
    #         "Pclass": Pclass,
    #         "Name": Name,
    #         "Sex": Sex,
    #         "Age": Age,
    #         "SibSp": SibSp,
    #         "Parch": Parch,
    #         "Ticket": Ticket,
    #         "Fare": Fare,
    #         "Cabin": Cabin,
    #         "Embarked": Embarked,
    #     }
    # )
