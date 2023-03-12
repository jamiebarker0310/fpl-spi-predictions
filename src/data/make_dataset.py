# -*- coding: utf-8 -*-
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def initialise_file_url_pairs():

    base_url = "https://projects.fivethirtyeight.com/soccer-api/club"
    file_url_pairs = [
        (
            "matches",
            f"{base_url}/spi_matches_latest.csv",
        ),
        (
            "teams",
            f"{base_url}/spi_global_rankings.csv",
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
