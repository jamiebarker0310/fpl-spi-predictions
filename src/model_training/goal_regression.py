import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from joblib import dump, load
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor

from src.models.spimapper import SPIMapper


def train_model(df, df_team):

    feature_cols = ["league", "team1", "team2"]
    target_cols = ["proj_score1", "proj_score2"]

    # Assume we have a dataframe `df` with columns `numeric_col` and `categorical_col`
    numeric_features = ["spi1", "off1", "def1", "spi2", "off2", "def2"]
    categorical_features = ["league"]

    # Define the transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Define the pipeline
    pipe = Pipeline([
        ('spi_mapper', SPIMapper(df_team)),
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(XGBRegressor()))
    ])

    param_grid = {
        # 'regressor__estimator__learning_rate': [0.05, 0.1, 0.15],
        # 'regressor__estimator__max_depth': [3, 4, 5],
        # 'regressor__estimator__n_estimators': [50, 100, 150],
        # 'regressor__estimator__subsample': [0.8, 1.0],
        # 'regressor__estimator__colsample_bytree': [0.8, 1.0],
        # 'regressor__estimator__gamma': [0, 0.1, 0.2],
        # 'regressor__estimator__reg_alpha': [0, 0.1, 0.5],
        # 'regressor__estimator__reg_lambda': [0, 1, 10],
    }

    # Fit the pipeline to the data
    X = df[feature_cols]  # X is the input features
    y = df[target_cols]  # y is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    cv = GridSearchCV(
        pipe, param_grid, scoring="neg_root_mean_squared_error", cv=5, verbose=2
    )
    cv.fit(X_train, y_train)

    logging.getLogger().setLevel(logging.INFO)

    score = cv.score(X_test, y_test)

    logging.info(f"test score: {score}")

    return cv


def main():

    df = pd.read_csv("data/interim/matches.csv")
    df_team = pd.read_csv("data/interim/team.csv")

    cv = train_model(df, df_team)

    dump(cv.best_estimator_, "models/goal_regressor.joblib")

    cv = load("models/goal_regressor.joblib")
    
    df[["proj_score1_pred", "proj_score2_pred"]] = cv.predict(df[["league", "team1", "team2"]])

    df.to_csv("data/interim/outputted_scores.csv", index=False)


if __name__ == "__main__":
    main()
