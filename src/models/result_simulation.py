import logging
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.stats import poisson


def simulate_match(proj_score_1, proj_score_2, n_distributions, diagonal_inflation):
    df_sim = pd.DataFrame(
        {
            "sim_score1": poisson.rvs(proj_score_1, size=(n_distributions)),
            "sim_score2": poisson.rvs(proj_score_2, size=(n_distributions)),
        }
    )

    df_sim = df_sim[["sim_score1", "sim_score2"]].value_counts().reset_index()
    # rename value
    df_sim = df_sim.rename({0: "count"}, axis=1)
    # calculate results
    df_sim.loc[df_sim["sim_score1"] > df_sim["sim_score2"], "result"] = "W"
    df_sim.loc[df_sim["sim_score1"] == df_sim["sim_score2"], "result"] = "D"
    df_sim.loc[df_sim["sim_score1"] < df_sim["sim_score2"], "result"] = "L"
    df_sim["result"] = pd.Categorical(
        df_sim["result"], categories=["W", "D", "L"], ordered=True
    )
    # draw adjustment
    df_sim.loc[
        df_sim["sim_score1"] == df_sim["sim_score2"], "count"
    ] *= diagonal_inflation
    # convert to probability
    df_sim["probability"] = df_sim["count"] / df_sim["count"].sum()
    # get probabilities
    df_sim_dict = df_sim.groupby("result")["probability"].sum().sort_index().to_dict()

    df_sim_dict["home_cs"] = df_sim.loc[df_sim["sim_score1"] == 0, "probability"].sum()
    df_sim_dict["away_cs"] = df_sim.loc[df_sim["sim_score2"] == 0, "probability"].sum()

    return df_sim_dict


class PoissonSimulator(BaseEstimator):
    def __init__(
        self,
        diagonal_inflation=1.09,
        n_distributions=1_000,
        score_cols=("proj_score1", "proj_score2"),
    ) -> None:
        super().__init__()
        self.diagonal_inflation = diagonal_inflation
        self.n_distributions = n_distributions

        self.score_cols = score_cols

    def fit(self, X=None, y=None):
        pass

    def predict(self, X, y=None, key="W"):
        score1_col, score2_col = self.score_cols

        func = lambda x: simulate_match(
            x[score1_col], x[score2_col], self.n_distributions, self.diagonal_inflation
        )[key]

        return X.apply(func, axis=1)

    def full_value_predict(self, X):
        score1_col, score2_col = self.score_cols

        func = lambda x: simulate_match(
            x[score1_col], x[score2_col], self.n_distributions, self.diagonal_inflation
        )

        return X.apply(lambda x: func(x), axis=1).apply(pd.Series)


def train_model(df):
    param_grid = {
        "diagonal_inflation": np.linspace(1, 1.2, 2),
        # "score_cols": [("proj_score1", "proj_score2"), ("proj_score1_pred", "proj_score2_pred")]
    }

    X_train, X_test = train_test_split(df, train_size=0.9)

    cv = GridSearchCV(
        PoissonSimulator(),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=2,
    )

    cv.fit(X_train, X_train["prob1"])

    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    score = cv.score(X_test, X_test["prob1"])

    logging.info(f"test score: {score}")

    return cv


def main():
    df = pd.read_csv("data/interim/outputted_scores.csv")

    cv = train_model(df)
    dump(cv, f"models/poisson.joblib")

    cv = load(f"models/poisson.joblib")

    df = pd.concat(
        [df, cv.estimator.full_value_predict(df).add_suffix("_probability_pred")],
        axis=1,
    )

    df.to_csv("data/processed/outputted_predictions.csv", index=False)


if __name__ == "__main__":
    main()
