import pandas as pd
from xgboost import XGBRegressor
from joblib import dump, load
import logging

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from src.models.spimapper import SPIMapper


def create_column_transformer():
    # Assume we have a dataframe `df` with columns `numeric_col` and `categorical_col`
    numeric_features = ["spi1", "off1", "def1", "spi2", "off2", "def2"]
    categorical_features = ["league"]

    # Define the transformers for preprocessing
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return column_transformer


def create_pipeline(df_team):
    column_transformer = create_column_transformer()

    # Define the pipeline
    pipe = Pipeline(
        [
            ("spi_mapper", SPIMapper(df_team)),
            ("preprocessor", column_transformer),
            ("regressor", MultiOutputRegressor(XGBRegressor())),
        ]
    )

    return pipe


def create_search(df_team, n_iter=60):
    pipe = create_pipeline(df_team)

    param_dist = {
        "regressor__estimator__learning_rate": [0.05, 0.1, 0.15],
        "regressor__estimator__max_depth": [3, 4, 5],
        "regressor__estimator__n_estimators": stats.randint(100, 200),
        "regressor__estimator__subsample": stats.uniform(0.8, 0.2),
        "regressor__estimator__colsample_bytree": stats.uniform(0.8, 0.2),
        "regressor__estimator__gamma": stats.uniform(0, 0.2),
        "regressor__estimator__reg_alpha": stats.loguniform(10**-3, 0.5),
        "regressor__estimator__reg_lambda": stats.loguniform(10**-3, 10),
    }

    cv = RandomizedSearchCV(
        pipe, param_dist, scoring="neg_root_mean_squared_error", cv=3, n_iter=n_iter
    )

    return cv


def holdout_split(df, train_size=0.9):
    feature_cols = ["league", "team1", "team2"]
    target_cols = ["proj_score1", "proj_score2"]

    # Fit the pipeline to the data
    X = df[feature_cols]  # X is the input features
    y = df[target_cols]  # y is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train, X_test, y_train, y_test


def train_model(df, df_team, n_iter=60):
    cv = create_search(df_team, n_iter=n_iter)

    X_train, X_test, y_train, y_test = holdout_split(df)

    cv.fit(X_train, y_train)

    logging.getLogger().setLevel(logging.INFO)

    score = cv.score(X_test, y_test)

    logging.info(f"test score: {score}")

    for key, value in cv.best_params_.items():
        logging.info(f"best {key}: {value}")

    return cv


def main():
    df = pd.read_csv("data/interim/matches.csv")
    df_team = pd.read_csv("data/interim/team.csv")

    cv = train_model(df, df_team)

    dump(cv.best_estimator_, "models/goal_regressor.joblib")

    cv = load("models/goal_regressor.joblib")

    df[["proj_score1_pred", "proj_score2_pred"]] = cv.predict(
        df[["league", "team1", "team2"]]
    )

    df.to_csv("data/interim/outputted_scores.csv", index=False)


if __name__ == "__main__":
    main()
