# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd

logger = logging.getLogger(__name__)


def initialise_file_url_pairs():
    file_url_pairs = [
        (
            "matches",
            "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv",
        ),
        (
            "teams",
            "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv",
        ),
    ]

    return file_url_pairs


def download_file(name, url, path="data/external/"):
    logger.info(f"downloading {name} csv from: {url}")
    df = pd.read_csv(url)

    filepath = f"{path}{name}.csv"
    logger.info(f"writing {name} csv to {filepath}")
    df.to_csv(filepath, index=False)


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    file_url_pairs = initialise_file_url_pairs()
    for name, url in file_url_pairs:
        download_file(name, url)


if __name__ == "__main__":
    main()
