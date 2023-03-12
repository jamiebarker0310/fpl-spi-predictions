from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Welcome to Your Football Score Predictor FastAPI"
    }


def test_predict_score():
    response = client.get(
        "/predict_score", params={"team1": "Chelsea", "team2": "Arsenal"}
    )

    assert response.status_code == 200

    assert "proj_score1" in response.json().keys()
    assert "proj_score2" in response.json().keys()
    assert "team1" in response.json().keys()
    assert "team2" in response.json().keys()

    assert len(response.json().keys()) == 4

    assert isinstance(response.json()["proj_score1"], float)
    assert response.json()["proj_score1"] >= 0
    assert isinstance(response.json()["proj_score2"], float)
    assert response.json()["proj_score2"] >= 0

    assert response.json()["team1"] == "Chelsea"
    assert response.json()["team2"] == "Arsenal"


def test_predict_score_wrong_team():
    response = client.get(
        "/predict_score", params={"team1": "Chelsea", "team2": "No Team"}
    )

    assert response.status_code == 422


def test_simulate_match_from_score():
    response = client.get(
        "/simulate_match_from_score", params={"score1": 1.0, "score2": 1.5}
    )

    assert response.status_code == 200

    assert "proj_score1" in response.json().keys()
    assert "proj_score2" in response.json().keys()
    assert "prob1" in response.json().keys()
    assert "prob2" in response.json().keys()
    assert "probtie" in response.json().keys()
    assert "cs1" in response.json().keys()
    assert "cs2" in response.json().keys()
    assert len(response.json().keys()) == 7

    assert response.json()["proj_score1"] == 1.0
    assert response.json()["proj_score2"] == 1.5

    for key in ["prob1", "prob2", "probtie", "cs1", "cs2"]:
        val = response.json()[key]
        assert isinstance(val, float)

        assert val >= 0
        assert val <= 1


def test_simulate_match_from_negative_score():
    response = client.get(
        "/simulate_match_from_score", params={"score1": 1.0, "score2": -1.5}
    )

    assert response.status_code == 422


def test_simulate_match():
    response = client.get(
        "/simulate_match", params={"team1": "Chelsea", "team2": "Arsenal"}
    )

    assert response.status_code == 200

    for key in [
        "team1",
        "team2",
        "proj_score1",
        "proj_score2",
        "prob1",
        "prob2",
        "probtie",
        "cs1",
        "cs2",
    ]:
        assert key in response.json().keys()

    assert len(response.json().keys()) == 9

    assert response.json()["team1"] == "Chelsea"
    assert response.json()["team2"] == "Arsenal"

    assert isinstance(response.json()["proj_score1"], float)
    assert response.json()["proj_score1"] >= 0
    assert isinstance(response.json()["proj_score2"], float)
    assert response.json()["proj_score2"] >= 0

    for key in ["prob1", "prob2", "probtie", "cs1", "cs2"]:
        val = response.json()[key]
        assert isinstance(val, float)

        assert val >= 0
        assert val <= 1


def test_simulate_match_wrong_team():
    response = client.get(
        "/predict_score", params={"team1": "Chelsea", "team2": "No Team"}
    )

    assert response.status_code == 422
