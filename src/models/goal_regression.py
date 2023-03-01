import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from joblib import dump, load
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split


def train_model(df, feature_cols, target_col):
    # Assume we have a dataframe `df` with columns `numeric_col` and `categorical_col`
    numeric_features = ["spi1", "off1", "def1", "spi2", "off2", "def2"]
    categorical_features = ["league_id"]

    # Define the transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Define the pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", XGBRegressor())])

    param_grid = {
        # 'regressor__max_depth': [3, 5, 7],
        # 'regressor__learning_rate': [0.1, 0.01, 0.001],
        # 'regressor__n_estimators': [100, 500, 1000],
        # 'regressor__gamma': [0, 0.1],
    }

    # Fit the pipeline to the data
    X = df[feature_cols]  # X is the input features
    y = df[target_col]  # y is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    cv = GridSearchCV(
        pipe, param_grid, scoring="neg_root_mean_squared_error", cv=3, verbose=2
    )
    cv.fit(X_train, y_train)

    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    score = cv.score(X_test, y_test)

    logging.info(f"test score: {score}")

    return cv


def main():
    df = pd.read_csv("data/interim/matches.csv")
    feature_cols = ["league_id", "spi1", "off1", "def1", "spi2", "off2", "def2"]

    for target_col in ["proj_score1", "proj_score2"]:
        cv = train_model(df, feature_cols, target_col)
        dump(cv, f"models/{target_col}.joblib")

    for target_col in ["proj_score1", "proj_score2"]:
        cv = load(f"models/{target_col}.joblib")
        df[f"{target_col}_pred"] = cv.predict(df[feature_cols])

    df.to_csv("data/interim/outputted_scores.csv", index=False)


if __name__ == "__main__":
    main()
