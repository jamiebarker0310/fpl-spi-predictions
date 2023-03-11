import numpy as np
import pandas as pd
from unittest.mock import patch

import pytest

from src.models.poisson_simulator import (
    PoissonSimulator,
    calculate_match_stats,
    calculate_probabilities,
    create_simulations,
    diagonal_inflation,
    simulate_matches,
    simulations_to_df,
)


@pytest.fixture
def test_df():
    df = pd.DataFrame({"score1": [1, 2, 3], "score2": [3, 2, 1]})

    return df


@pytest.fixture
def test_simulations():
    X = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
            [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
        ]
    )

    return X


@pytest.fixture
def test_sim_df():
    df = pd.DataFrame(
        data={
            1: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            2: [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
            ],
            names=["match", "simulation"],
        ),
    )

    return df


@pytest.fixture
def test_sim_df_match_stats():
    df = pd.DataFrame(
        data={
            1: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            2: [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            "result": ["tie"] * 5 + ["1"] * 5 + ["2"] * 5,
            "cs1": [False] * 5 + [True] * 5 + [False] * 5,
            "cs2": [False] * 10 + [True] * 5,
        },
        index=pd.MultiIndex.from_tuples(
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
            ],
            names=["match", "simulation"],
        ),
    )
    return df


@pytest.fixture
def test_sim_df_inflated():
    expected = pd.DataFrame(
        data={
            "match": [0] * 5 + [1] * 5 + [2] * 5,
            "simulation": list(range(5)) * 3,
            1: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            2: [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            "result": ["tie"] * 5 + ["1"] * 5 + ["2"] * 5,
            "cs1": [False] * 5 + [True] * 5 + [False] * 5,
            "cs2": [False] * 10 + [True] * 5,
            "count": [1.1] * 5 + [1] * 10,
        }
    )

    return expected


def test_create_simulations_mock(test_df):
    n_simulations = 100

    with patch("scipy.stats.poisson.rvs", lambda x: np.ones(x.shape)):
        response = create_simulations(test_df, n_simulations)

        assert np.array_equal(response, np.ones(shape=(3, n_simulations, 2)))


def test_create_simulations_shape(test_df):
    n_simulations = 100

    response = create_simulations(test_df, n_simulations)

    assert response.shape == (3, n_simulations, 2)


def test_simulations_to_df(test_df, test_simulations, test_sim_df):
    response = simulations_to_df(test_df, test_simulations)

    expected = test_sim_df

    pd.testing.assert_frame_equal(response, expected)


def test_calculate_match_stats(test_sim_df, test_sim_df_match_stats):
    response = calculate_match_stats(test_sim_df)

    expected = test_sim_df_match_stats

    pd.testing.assert_frame_equal(response, expected)


def test_diagonal_inflation(test_sim_df_match_stats, test_sim_df_inflated):
    draw_inflation = 1.1
    response = diagonal_inflation(test_sim_df_match_stats, draw_inflation)

    expected = test_sim_df_inflated

    pd.testing.assert_frame_equal(response, expected)


def test_calculate_probabilities(test_sim_df_inflated):
    response = calculate_probabilities(test_sim_df_inflated)

    expected = pd.DataFrame(
        index=pd.RangeIndex(stop=3, name="match"),
        data={
            "prob1": [0.0, 1.0, 0.0],
            "prob2": [0.0, 0.0, 1.0],
            "probtie": [1.0, 0.0, 0.0],
            "cs1": [0.0, 1.0, 0.0],
            "cs2": [0.0, 0.0, 1.0],
        },
    )
    expected.columns = expected.columns.values

    pd.testing.assert_frame_equal(response, expected)


def test_simulate_matches(test_df):
    response = simulate_matches(test_df, 10, 1.1)

    assert isinstance(response, pd.DataFrame)

    assert len(response) == 3

    assert response.columns.values.tolist() == [
        "prob1",
        "prob2",
        "probtie",
        "cs1",
        "cs2",
    ]


def test_poisson_simulator_init():
    response = PoissonSimulator(
        target_cols=["prob1", "prob2", "probtie"],
        diagonal_inflation=1.10,
        n_simulations=100,
    )

    assert response.target_cols == ["prob1", "prob2", "probtie"]

    assert response.diagonal_inflation == 1.10

    assert response.n_simulations == 100


def test_poisson_simulator_fit():
    response = PoissonSimulator(
        target_cols=["prob1", "prob2", "probtie"],
        diagonal_inflation=1.10,
        n_simulations=100,
    )

    X = pd.DataFrame(np.random.uniform(0, 5, size=(100, 2)))

    response.fit(X)

    assert response


def test_poisson_simulator_predict():
    model = PoissonSimulator(
        target_cols=["prob1", "prob2", "probtie"],
        diagonal_inflation=1.10,
        n_simulations=100,
    )

    X = pd.DataFrame(np.random.uniform(0, 5, size=(100, 2)))

    model.fit(X)

    response = model.predict(X)

    assert isinstance(response, pd.DataFrame)

    assert response.shape == (100, 3)

    assert response.columns.values.tolist() == ["prob1", "prob2", "probtie"]
