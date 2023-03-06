import logging
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from src.models.poisson_simulator import PoissonSimulator


def train_model(df):
    feature_cols = ["proj_score1_pred", "proj_score2_pred"]

    target_cols = ["prob1", "prob2", "probtie"]

    X = df[feature_cols]
    y = df[target_cols]

    param_grid = {
        "diagonal_inflation": np.linspace(0.9, 1.3, 3),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    cv = GridSearchCV(
        PoissonSimulator(target_cols=target_cols),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=2,
        verbose=2,
    )

    cv.fit(X_train, y_train)

    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    score = cv.score(X_test, y_test)

    logging.info(f"test score: {score}")
    logging.info(f"diagonal inflation: {cv.best_params_}")

    return cv


def main():
    df = pd.read_csv("data/interim/outputted_scores.csv")

    cv = train_model(df)

    cv.best_estimator_.target_cols = None

    dump(cv.best_estimator_, f"models/poisson.joblib")

    cv = load(f"models/poisson.joblib")

    cv.target_cols = None

    df = pd.concat(
        [
            df,
            cv.predict(
                df[["proj_score1_pred", "proj_score2_pred"]]
            ).add_suffix("_probability_pred"),
        ],
        axis=1,
    )

    df.to_csv("data/processed/outputted_predictions.csv", index=False)


if __name__ == "__main__":
    main()
