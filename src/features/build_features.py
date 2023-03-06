import pandas as pd


def filter_match_data(df_match, df_team):
    cols = ["league", "team1", "team2", "proj_score1", "proj_score2"]

    df_match.loc[df_match.score1.isna(), cols]

    df_team["name"].unique()

    cond1 = df_match["team1"].isin(df_team["name"].unique())
    cond2 = df_match["team2"].isin(df_team["name"].unique())
    cond3 = df_match.score1.isna()

    return df_match.loc[(cond1) & (cond2) & (cond3)]


def filter_team_data(df):
    df = df[["name", "off", "def", "spi"]]

    return df


def main():
    df_match = pd.read_csv("data/external/matches.csv")
    df_team = pd.read_csv("data/external/teams.csv")

    df_match = filter_match_data(df_match, df_team)
    df_team = filter_team_data(df_team)

    df_match.to_csv("data/interim/matches.csv", index=False)
    df_team.to_csv("data/interim/team.csv", index=False)


if __name__ == "__main__":
    main()
