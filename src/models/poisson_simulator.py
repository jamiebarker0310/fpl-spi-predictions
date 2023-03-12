from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import poisson
import pandas as pd


def create_simulations(df, n_simulations):
    # intialise arrays
    X = np.ones((len(df), n_simulations, 2))
    # create array of distributions for each parameter
    mus = X * df.values.reshape(len(df), 1, 2)
    # sample from distributions
    samples = poisson.rvs(mus).reshape(mus.shape)

    return samples


def simulations_to_df(df, simulations):
    # initialise multi-index
    index = pd.MultiIndex.from_product(
        [df.index.values, range(simulations.shape[1])], names=["match", "simulation"]
    )
    df_sim = pd.DataFrame(index=index, columns=[1, 2])

    # set values with simulations
    df_sim.loc[:, 1] = simulations[:, :, 0].flatten()
    df_sim.loc[:, 2] = simulations[:, :, 1].flatten()

    return df_sim


def calculate_match_stats(df_sim):
    # calculate result
    df_sim["result"] = (
        df_sim.diff(axis=1)[2].apply(np.sign).map({-1: "1", 0: "tie", 1: "2"})
    )

    # calculate clean sheets
    df_sim["cs1"] = df_sim[2] == 0
    df_sim["cs2"] = df_sim[1] == 0

    return df_sim


def diagonal_inflation(df_sim, draw_inflation):
    # set value for diagonal inflation
    df_sim["count"] = 1
    df_sim.loc[df_sim["result"] == "tie", "count"] = draw_inflation

    # reset index
    df_sim = df_sim.reset_index()

    return df_sim


def calculate_probabilities(df_sim):
    # initialise probabilities
    probabilities = []
    # for results and clean sheets
    for col in ["result", "cs1", "cs2"]:
        # calculate sum of each
        df_sim_agg = df_sim.groupby(["match", col])["count"].sum().unstack()
        # adjust column names to avoid duplicates
        df_sim_agg.columns = [(col, col1) for col1 in df_sim_agg.columns]
        # convert sum into probability
        df_sim_agg = df_sim_agg.div(df_sim_agg.sum(axis=1), axis=0)
        # add to list
        probabilities.append(df_sim_agg.copy(deep=True))
    # concatenate probabilities
    df = pd.concat(probabilities, axis=1)
    # keep only required columns
    df = df[
        [
            ("result", "1"),
            ("result", "2"),
            ("result", "tie"),
            ("cs1", True),
            ("cs2", True),
        ]
    ]

    df.columns = ["prob1", "prob2", "probtie", "cs1", "cs2"]

    return df.fillna(0)


def simulate_matches(df, n_simulations, draw_inflation):
    simulations = create_simulations(df, n_simulations)

    df_sim = simulations_to_df(df, simulations)

    df_sim = calculate_match_stats(df_sim)

    df_sim = diagonal_inflation(df_sim, draw_inflation)

    df_prob = calculate_probabilities(df_sim)

    return df_prob


class PoissonSimulator(BaseEstimator):
    def __init__(
        self,
        target_cols=None,
        diagonal_inflation=1.09,
        n_simulations=1_000,
    ) -> None:
        super().__init__()
        self.target_cols = target_cols
        self.diagonal_inflation = diagonal_inflation
        self.n_simulations = n_simulations

    def fit(self, X=None, y=None):
        pass

    def predict(self, X, y=None):
        df = simulate_matches(X, self.n_simulations, self.diagonal_inflation)

        if self.target_cols:
            return df[self.target_cols]

        else:
            return df
