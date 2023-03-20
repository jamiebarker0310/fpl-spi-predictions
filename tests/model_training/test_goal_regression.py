import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.model_training.goal_regression import (
    create_column_transformer,
    create_pipeline,
    create_search,
    holdout_split,
    train_model,
)


@pytest.fixture
def test_team_df():
    df = pd.DataFrame(
        {"name": ["a", "b", "c"], "spi": [1, 2, 3], "off": [2, 3, 4], "def": [0, 1, 2]}
    )

    return df


@pytest.fixture
def test_match_df():
    df = pd.DataFrame(
        {
            "league": ["x"] * 6,
            "team1": ["a", "a", "b", "b", "c", "c"],
            "team2": ["b", "c", "a", "c", "a", "b"],
            "proj_score1": [1, 2, 3, 1, 2, 3],
            "proj_score2": [0, 1, 2, 0, 1, 2],
        }
    )

    return df


@pytest.fixture
def test_features(test_df):
    return test_df.iloc[:, :-2]


@pytest.fixture
def test_labels(test_df):
    return test_df.iloc[:, -2:]


def test_create_column_transformer():
    response = create_column_transformer()

    assert isinstance(response, ColumnTransformer)


def test_create_pipeline(test_team_df):
    response = create_pipeline(test_team_df)

    assert isinstance(response, Pipeline)


def test_holdout_split(test_match_df):
    X_train, X_test, y_train, y_test = holdout_split(test_match_df, train_size=5 / 6)

    assert y_train.columns.values.tolist() == ["proj_score1", "proj_score2"]

    assert y_test.columns.values.tolist() == ["proj_score1", "proj_score2"]

    assert len(X_train) == len(y_train) == 5

    assert len(X_test) == len(y_test) == 1


def test_create_search(test_team_df):
    response = create_search(test_team_df)

    assert isinstance(response, RandomizedSearchCV)


def test_train_model(test_match_df, test_team_df):
    response = train_model(test_match_df, test_team_df, n_iter=3)

    assert isinstance(response, RandomizedSearchCV)
