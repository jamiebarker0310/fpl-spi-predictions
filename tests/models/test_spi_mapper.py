import pandas as pd
import pytest

from src.models.spimapper import SPIMapper


@pytest.fixture
def test_team_df():
    df = pd.DataFrame(
        {"name": ["a", "b", "c"], "spi": [1, 2, 3], "off": [2, 3, 4], "def": [0, 1, 2]}
    )

    return df


@pytest.fixture
def test_match_df():
    df = pd.DataFrame(
        {
            "league": ["x"] * 6,
            "team1": ["a", "a", "b", "b", "c", "c"],
            "team2": ["b", "c", "a", "c", "a", "b"],
        }
    )

    return df


def test_spi_mapper_init(test_team_df):
    response = SPIMapper(test_team_df)

    pd.testing.assert_frame_equal(response.df_team, test_team_df)


def test_spi_mapper_fit(test_team_df, test_match_df):
    response = SPIMapper(test_team_df)

    response.fit(test_match_df)

    assert response


def test_spi_mapper_predict(test_team_df, test_match_df):
    model = SPIMapper(test_team_df)

    model.fit(test_match_df)

    response = model.transform(test_match_df)

    expected = pd.DataFrame(
        {
            "league": ["x"] * 6,
            "spi1": [1, 1, 2, 2, 3, 3],
            "off1": [2, 2, 3, 3, 4, 4],
            "def1": [0, 0, 1, 1, 2, 2],
            "spi2": [2, 3, 1, 3, 1, 2],
            "off2": [3, 4, 2, 4, 2, 3],
            "def2": [1, 2, 0, 2, 0, 1],
        }
    )

    pd.testing.assert_frame_equal(response, expected)
