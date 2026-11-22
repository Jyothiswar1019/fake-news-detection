import pytest
from fastapi.testclient import TestClient

from fakenews.config import resolve

pytestmark = pytest.mark.skipif(
    not resolve("models/fakenews_pipeline.joblib").exists(),
    reason="train the model first",
)


def test_health_and_predict():
    from api.main import app
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/predict", json={"text": "The Senate voted on Tuesday to approve the budget " * 3})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"real", "fake", "uncertain"}
        assert 0 <= body["fake_probability"] <= 1


def test_too_short_text_rejected():
    from api.main import app
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": "hi"}).status_code == 422


def test_home_page_served():
    from api.main import app
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Fake News Detector" in r.text