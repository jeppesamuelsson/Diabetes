import os
from fastapi.testclient import TestClient
from diabetes_service.app import app

def _prepare_model(tmp_path):
    from diabetes_service.train import train
    out = tmp_path / "model"
    out.mkdir(parents=True, exist_ok=True)
    res = train("v0.1", out.as_posix())
    os.environ["MODEL_PATH"] = res["model_path"]

def test_health(tmp_path):
    _prepare_model(tmp_path)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in ("ok", "error")

def test_predict_ok(tmp_path):
    _prepare_model(tmp_path)
    client = TestClient(app)
    payload = { "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
                "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
                "s5": 0.02, "s6": -0.001 }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)
