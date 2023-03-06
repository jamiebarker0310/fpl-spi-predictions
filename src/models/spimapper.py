from sklearn.base import BaseEstimator, TransformerMixin


class SPIMapper(BaseEstimator, TransformerMixin):
    def __init__(self, df_team) -> None:
        super().__init__()
        self.df_team = df_team

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = (
            X.merge(
                self.df_team.add_suffix(1),
                left_on="team1",
                right_on="name1",
                how="left",
            )
            .merge(
                self.df_team.add_suffix(2),
                left_on="team2",
                right_on="name2",
                how="left",
            )[["league", "spi1", "off1", "def1", "spi2", "off2", "def2"]]
            .fillna(-1)
        )

        return X
