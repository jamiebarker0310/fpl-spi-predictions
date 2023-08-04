import logging
from enum import Enum
from typing import Union

from fastapi import FastAPI
from joblib import load
import pandas as pd
from pydantic import confloat, BaseModel

goal_regressor = load("models/goal_regressor.joblib")

result_simulator = load("models/poisson.joblib")


class Team(str, Enum):
    bou = "AFC Bournemouth"
    ars = "Arsenal"
    avl = "Aston Villa"
    bre = "Brentford"
    bha = "Brighton and Hove Albion"
    bur = "Burnley"
    che = "Chelsea"
    cry = "Crystal Palace"
    eve = "Everton"
    ful = "Fulham"
    liv = "Liverpool"
    lut = "Luton Town"
    mci = "Manchester City"
    mun = "Manchester United"
    new = "Newcastle"
    nfo = "Nottingham Forest"
    shu = "Sheffield United"
    tot = "Tottenham Hostpur"
    whu = "West Ham United"
    wol = "Wolverhampton"


class PredictedResult(BaseModel):
    team1: Union[Team, None] = None
    team2: Union[Team, None] = None

    proj_score1: confloat(ge=0)
    proj_score2: confloat(ge=0)

    prob1: Union[confloat(ge=0, le=1), None] = None
    prob2: Union[confloat(ge=0, le=1), None] = None
    probtie: Union[confloat(ge=0, le=1), None] = None

    cs1: Union[confloat(ge=0, le=1), None] = None
    cs2: Union[confloat(ge=0, le=1), None] = None


logger = logging.Logger(__name__)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to Your Football Score Predictor FastAPI"}


@app.get(
    "/predict_score", response_model=PredictedResult, response_model_exclude_none=True
)
async def predict_score(team1: Team, team2: Team):
    match = pd.DataFrame(
        {"league": ["Barclays Premier League"], "team1": [team1], "team2": [team2]}
    )

    result = goal_regressor.predict(match)

    result_dict = {
        "team1": team1,
        "team2": team2,
        "proj_score1": float(result[0][0]),
        "proj_score2": float(result[0][1]),
    }

    return result_dict


@app.get(
    "/simulate_match_from_score",
    response_model=PredictedResult,
    response_model_exclude_none=True,
)
async def simulate_match_from_score(score1: confloat(ge=0), score2: confloat(ge=0)):
    match = pd.DataFrame(
        {
            "proj_score1": [score1],
            "proj_score2": [score2],
        }
    )

    result_dict = result_simulator.predict(match).iloc[0].to_dict()

    result_dict["proj_score1"] = score1
    result_dict["proj_score2"] = score2

    print(result_dict)

    return result_dict


@app.get("/simulate_match", response_model=PredictedResult)
async def simulate_match(team1: Team, team2: Team):
    match = pd.DataFrame(
        {"league": ["Barclays Premier League"], "team1": [team1], "team2": [team2]}
    )

    result = goal_regressor.predict(match)

    result_dict = {
        "proj_score1": float(result[0][0]),
        "proj_score2": float(result[0][1]),
    }

    match = pd.DataFrame(
        {
            "proj_score1_pred": [result[0][0]],
            "proj_score2_pred": [result[0][1]],
        }
    )

    result_dict = {**result_dict, **result_simulator.predict(match).iloc[0].to_dict()}

    result_dict["team1"] = team1
    result_dict["team2"] = team2

    return result_dict
