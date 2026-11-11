from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bhasharag.api import create_app


class FakePipeline:
    indexes = {"hi": object()}
    profile = SimpleNamespace(name="cpu")
    generator = None

    def answer(self, question, lang):
        return SimpleNamespace(to_dict=lambda: {"answer": question, "refused": False})


@pytest.fixture
def client():
    with TestClient(create_app(FakePipeline())) as test_client:
        yield test_client


def test_health_reports_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["languages"] == ["hi"]
    assert body["generator"] == "extractive"


def test_ask_returns_pipeline_answer(client):
    response = client.post("/ask", json={"question": " प्रश्न ", "lang": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "प्रश्न"


def test_unloaded_language_is_a_clear_error(client):
    response = client.post("/ask", json={"question": "x", "lang": "ta"})
    assert response.status_code == 400
    assert "hi" in response.json()["detail"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_bad_questions_are_rejected(client, question):
    assert client.post("/ask", json={"question": question, "lang": "hi"}).status_code == 422
