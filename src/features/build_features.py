import pandas as pd


def filter_data(df):
    cols = [
        "season",
        "date",
        "league_id",
        "league",
        "team1",
        "team2",
        "spi1",
        "spi2",
        "prob1",
        "prob2",
        "probtie",
        "proj_score1",
        "proj_score2",
        "importance1",
        "importance2",
    ]

    df.loc[df.score1.isna(), cols]

    return df


def merge_team_data(df_match, df_team):
    df = df_match.merge(
        df_team[["name", "off", "def", "spi"]].add_suffix("1"),
        left_on=["team1", "spi1"],
        right_on=["name1", "spi1"],
        how="inner",
    ).merge(
        df_team[["name", "off", "def", "spi"]].add_suffix("2"),
        left_on=["team2", "spi2"],
        right_on=["name2", "spi2"],
        how="inner",
    )

    return df


def main():
    df_match = pd.read_csv("data/external/matches.csv")
    df_team = pd.read_csv("data/external/teams.csv")

    df_match = filter_data(df_match)

    df = merge_team_data(df_match, df_team)

    df.to_csv("data/interim/matches.csv", index=False)


if __name__ == "__main__":
    main()
