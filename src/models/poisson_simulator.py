from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import poisson
import pandas as pd

def simulate_matches(df, n_distributions, diagonal_inflation):
    # intialise arrays
    X = np.ones((len(df), n_distributions, 2))
    # create array of distributions for each parameter
    mus = X * df.values.reshape(len(df), 1, 2)
    # sample from distributions
    samples = poisson.rvs(mus).reshape(mus.shape)

    # initialise multi-index
    index = pd.MultiIndex.from_product(
        [df.index.values, range(n_distributions)], names=["match", "simulation"]
    )
    df1 = pd.DataFrame(index=index, columns=[1, 2])

    # set values with simulations
    df1.loc[:, 1] = samples[:, :, 0].flatten()
    df1.loc[:, 2] = samples[:, :, 1].flatten()

    # calculate result
    df1["result"] = df1.diff(axis=1)[2].apply(np.sign).map({-1: "1", 0: "tie", 1: "2"})

    # calculate clean sheets
    df1["cs1"] = df1[1] == 0
    df1["cs2"] = df1[2] == 0

    # set value for diagonal inflation
    df1["count"] = 1
    df1.loc[df1["result"] == "tie", "count"] = diagonal_inflation

    # reset index
    df1 = df1.reset_index()

    # initialise probabilities
    probabilities = []

    # for results and clean sheets
    for col in ["result", "cs1", "cs2"]:
        # calculate sum of each
        df2 = df1.groupby(["match", col])["count"].sum().unstack()
        # adjust column names to avoid duplicates
        df2.columns = [(col, col1) for col1 in df2.columns]
        # convert sum into probability
        df2 = df2.div(df2.sum(axis=1), axis=0)
        # add to list
        probabilities.append(df2.copy(deep=True))
    # concatenate probabilities
    df1 = pd.concat(probabilities, axis=1)
    # keep only required columns
    df1 = df1[
        [
            ("result", "1"),
            ("result", "2"),
            ("result", "tie"),
            ("cs1", True),
            ("cs2", True),
        ]
    ]
    # rename columns
    df1.columns = ["prob1", "prob2", "probtie", "cs1", "cs2"]
    # return dataframe
    return df1


class PoissonSimulator(BaseEstimator):
    def __init__(
        self,
        target_cols=None,
        diagonal_inflation=1.09,
        n_distributions=1_000,
        score_cols=("proj_score1_pred", "proj_score2_pred"),
    ) -> None:
        super().__init__()
        self.target_cols = target_cols
        self.diagonal_inflation = diagonal_inflation
        self.n_distributions = n_distributions

        self.score_cols = score_cols

    def fit(self, X=None, y=None):
        pass

    def predict(self, X, y=None):
        df = simulate_matches(X, self.n_distributions, self.diagonal_inflation)

        if self.target_cols:
            return df[self.target_cols]

        else:
            return df