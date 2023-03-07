from typing import Union
import logging
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd

from src.models.spimapper import SPIMapper

goal_regressor = load(f"models/goal_regressor.joblib")

result_simulator = load("models/poisson.joblib")

class Team(str, Enum):
    bou = "AFC Bournemouth"
    ars = "Arsenal"
    avl = "Aston Villa"
    bre = "Brentford"
    bha = "Brighton and Hove Albion"
    che = "Chelsea"
    cry = "Crystal Palace"
    eve = "Everton"
    ful = "Fulham"
    lee = "Leeds United"
    lei = "Leicester City"
    liv = "Liverpool"
    mci = "Manchester City"
    mun = "Manchester United"
    new = "Newcastle"
    nfo = "Nottingham Forest"
    sou = "Southampton"
    tot = "Tottenham Hostpur"
    whu = "West Ham United"
    wol = "Wolverhampton"


logger = logging.Logger(__name__)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to Your Football Score Predictor FastAPI"}


@app.get("/predict_score")
async def predict_score(team1: Team, team2: Team):

    match = pd.DataFrame({
        "league": ["Barclays Premier League"],
        "team1": [team1],
        "team2": [team2]
    })
    
    result = goal_regressor.predict(match)

    result_dict = {"proj_score1": float(result[0][0]),"proj_score2": float(result[0][1])}

    return result_dict

@app.get("/simulate_match_from_score")
async def simulate_match_from_score(score1: float, score2: float):

    match = pd.DataFrame({
        "proj_score1_pred": [score1],
        "proj_score2_pred": [score2],
    })
    
    result_dict = result_simulator.predict(match).iloc[0].to_dict()

    return result_dict

@app.get("/simulate_match")
async def simulate_match(team1: Team, team2: Team):

    match = pd.DataFrame({
        "league": ["Barclays Premier League"],
        "team1": [team1],
        "team2": [team2]
    })
    
    result = goal_regressor.predict(match)

    result_dict = {"proj_score1": float(result[0][0]),"proj_score2": float(result[0][1])}

    match = pd.DataFrame({
        "proj_score1_pred": [result[0][0]],
        "proj_score2_pred": [result[0][1]],
    })

    result_dict = {**result_dict, **result_simulator.predict(match).iloc[0].to_dict()}

    return result_dict