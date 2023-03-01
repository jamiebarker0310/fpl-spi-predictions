# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    file_url_dict = [
        (
            "matches",
            "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv",
        ),
        (
            "teams",
            "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv",
        ),
    ]
    for name, url in file_url_dict:
        logger.info(f"downloading {name} csv from: {url}")
        df = pd.read_csv(url)

        filepath = f"data/external/{name}.csv"
        logger.info(f"writing {name} csv to {filepath}")
        df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
