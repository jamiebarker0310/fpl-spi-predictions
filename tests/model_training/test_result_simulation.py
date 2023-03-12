import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from src.model_training.result_simulation import create_search, holdout_split, train_model

@pytest.fixture
def test_df():
    df = pd.DataFrame(
        {"proj_score1_pred": [1, 2, 3], 
         "proj_score2_pred": [3, 2, 1],
         "prob1": [0.5,0.4,0.3],
         "prob2": [0.1,0.2,0.3],
         "probtie": [0.4,0.4,0.4]
         })

    return df

def test_create_search():

    response = create_search()

    assert isinstance(response, GridSearchCV)

def test_holdout_split(test_df):

    X_train, X_test, y_train, y_test = holdout_split(test_df, train_size=2/3)

    assert y_train.columns.values.tolist() == y_test.columns.values.tolist() == ["prob1", "prob2", "probtie"]

    assert X_train.columns.values.tolist() == X_test.columns.values.tolist() == ["proj_score1_pred", "proj_score2_pred"]

    assert len(X_train) == len(y_train) == 2

    assert len(X_test) == len(y_test) == 1

def test_train_model(test_df):

    response = train_model(test_df)

    assert isinstance(response, GridSearchCV)

