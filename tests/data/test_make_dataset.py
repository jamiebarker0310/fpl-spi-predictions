import os
from pathlib import Path

import pandas as pd
import pytest

from src.data.make_dataset import initialise_file_url_pairs, download_file


def test_initialise_file_url_pairs():
    base_url = "https://projects.fivethirtyeight.com/soccer-api/club"
    expected_pairs = [
        (
            "matches",
            f"{base_url}/spi_matches_latest.csv",
        ),
        (
            "teams",
            f"{base_url}/spi_global_rankings.csv",
        ),
    ]

    assert initialise_file_url_pairs() == expected_pairs


@pytest.fixture
def test_file():
    name = "test"
    url = "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv"
    path = Path("data/test/")
    path.mkdir(parents=True)
    download_file(name, url, path)
    yield f"{path}{name}.csv"
    os.remove(f"{path}{name}.csv")
    os.rmdir(path)


def test_download_file(test_file):
    assert os.path.exists(test_file)
    df = pd.read_csv(test_file)
    assert not df.empty
