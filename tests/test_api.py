import pytest

from langscout.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m langscout.train` first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from langscout.api import app
    with TestClient(app) as c:          # the context manager runs the start-up hook (loads the model)
        yield c


def test_health_and_languages(client):
    assert client.get("/health").json()["status"] == "ok"
    langs = client.get("/languages").json()["languages"]
    assert "Hinglish" in langs and len(langs) == 18


def test_detect(client):
    r = client.post("/detect", json={"text": "Ceci est une belle journée"})
    body = r.json()
    assert r.status_code == 200 and body["language"] == "French" and body["status"] == "ok"
    assert len(body["top_k"]) == 3


def test_detect_unsupported_script(client):
    body = client.post("/detect", json={"text": "这是一个测试"}).json()
    assert body["status"] == "unsupported_script" and body["language"] is None


def test_batch_and_segments(client):
    out = client.post("/detect/batch", json={"texts": ["Hello there my friend", "Bonjour tout le monde"]}).json()
    assert [r["language"] for r in out["results"]] == ["English", "French"]
    segs = client.post("/detect/segments", json={"text": "I love this city. Ich liebe diese Stadt."}).json()["segments"]
    assert [s["language"] for s in segs] == ["English", "German"]


def test_explain_and_validation(client):
    assert client.post("/explain", json={"text": "Das ist ein Test"}).json()["target"] == "German"
    assert client.post("/explain", json={"text": "123"}).status_code == 422
    assert client.post("/detect", json={"text": "x", "top_k": 0}).status_code == 422
