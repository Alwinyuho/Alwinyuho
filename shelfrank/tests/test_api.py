import pytest
from fastapi.testclient import TestClient

from shelfrank.api import create_app


class FakeRecommender:
    items = ["A", "B"]

    def search(self, q):
        return [{"asin": "A", "title": "Mario Kart", "new_item": False}]

    def recommend(self, history, k):
        if "A" not in history:
            return []
        return [{"asin": "B", "title": "Zelda", "new_item": True, "score": 0.9}][:k]


@pytest.fixture
def client():
    with TestClient(create_app(FakeRecommender())) as test_client:
        yield test_client


def test_recommend(client):
    response = client.post("/recommend", json={"history": ["A"], "k": 5})
    assert response.status_code == 200
    assert response.json()[0]["asin"] == "B"


def test_unknown_history_is_404(client):
    assert client.post("/recommend", json={"history": ["Z"]}).status_code == 404


def test_validation(client):
    assert client.post("/recommend", json={"history": []}).status_code == 422
    assert client.post("/recommend", json={"history": ["A"], "k": 500}).status_code == 422
    assert client.get("/search", params={"q": "m"}).status_code == 422


def test_search(client):
    assert client.get("/search", params={"q": "mario"}).json()[0]["title"] == "Mario Kart"
