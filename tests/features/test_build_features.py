import numpy as np
import pytest
import pandas as pd

from src.features.build_features import filter_match_data, filter_team_data


@pytest.fixture
def test_match_data():
    df = pd.DataFrame(
        {
            "league": [f"league{i+1}" for i in range(5)],
            "team1": [f"team1-{i+1}" for i in range(5)],
            "team2": [f"team2-{i+1}" for i in range(5)],
            "proj_score1": [1.1, 2.2, 3.3, 1.1, 2.2],
            "proj_score2": [0.5, 1.1, 2.2, 3.3, 1.1],
            "score1": [0, np.nan, np.nan, np.nan, np.nan],
            "score2": [4, np.nan, np.nan, np.nan, np.nan],
        }
    )

    return df


@pytest.fixture
def test_team_data():
    df = pd.DataFrame(
        {
            "name": sum(
                [f"team1-{i+1}" for i in range(5)], [f"team2-{i+1}" for i in range(5)]
            ),
            "league": [f"league{i}" for i in range(10)],
            "off": [1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5],
            "def": [1.2, 1.3, 1.4, 1.5, 1.6, 2.2, 2.3, 2.4, 2.5, 2.6],
            "spi": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    df = df.loc[~df.name.isin(["team1-2", "team2-3"])]

    return df


def test_filter_match_data(test_match_data, test_team_data):
    response = filter_match_data(test_match_data, test_team_data).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "league": [f"league{i+1}" for i in range(3, 5)],
            "team1": [f"team1-{i+1}" for i in range(3, 5)],
            "team2": [f"team2-{i+1}" for i in range(3, 5)],
            "proj_score1": [1.1, 2.2],
            "proj_score2": [3.3, 1.1],
        }
    )

    pd.testing.assert_frame_equal(response, expected)


def test_filter_team_data(test_team_data):
    response = filter_team_data(test_team_data).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "name": sum(
                [f"team1-{i+1}" for i in range(5)], [f"team2-{i+1}" for i in range(5)]
            ),
            "off": [1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5],
            "def": [1.2, 1.3, 1.4, 1.5, 1.6, 2.2, 2.3, 2.4, 2.5, 2.6],
            "spi": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    expected = expected.loc[~expected.name.isin(["team1-2", "team2-3"])].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(response, expected)
