import json
from pathlib import Path

from backend.app import app

BASE_DIR = Path(__file__).resolve().parents[2]
PAYLOAD_FILE = BASE_DIR / "backend" / "test_payload.json"


def load_payload():
    with open(PAYLOAD_FILE, "r") as file:
        return json.load(file)


def test_health_endpoint():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.json["features"] == 15
    assert response.json["classes"] == 12


def test_valid_prediction():
    client = app.test_client()

    response = client.post(
        "/predict",
        json=load_payload()
    )

    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probabilities" in response.json


def test_missing_feature():
    client = app.test_client()

    payload = load_payload()
    payload.pop("service")

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "error" in response.json


def test_invalid_json():
    client = app.test_client()

    response = client.post(
        "/predict",
        data="this is not valid JSON",
        content_type="application/json"
    )

    assert response.status_code == 400
    assert "error" in response.json