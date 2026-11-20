"""The HTTP API through FastAPI's TestClient, with the fake embedding and fake language model."""

import pytest

pytestmark = pytest.mark.integration


def test_health_reports_the_index(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok" and body["chunk_count"] > 0


def test_info_describes_the_models(api_client):
    body = api_client.get("/info").json()
    assert body["llm_provider"] == "fake" and body["demo_mode"] is True
    assert body["embedding_model"] == "fake-hashing-embedding"


def test_chat_returns_answer_citations_and_sources(api_client):
    body = api_client.post("/chat", json={"question": "What is Headline Radar?"}).json()
    assert body["status"] == "answered"
    assert "HDBSCAN" in body["answer"]
    assert body["citations"] and body["sources"] == ["Resume"]


def test_chat_validates_the_body(api_client):
    assert api_client.post("/chat", json={}).status_code == 422
    history = [{"role": "user", "content": "q"}] * 21
    assert api_client.post("/chat", json={"question": "q", "history": history}).status_code == 422


def test_retrieve_returns_ranked_passages(api_client):
    body = api_client.post("/retrieve", json={"query": "Syracuse University", "top_k": 2}).json()
    assert body["results"][0]["section"] == "Education"
    assert len(body["results"]) <= 2


def test_retrieve_rejects_long_queries(api_client):
    assert api_client.post("/retrieve", json={"query": "x" * 600}).status_code == 422


def test_api_key_protects_everything_but_health(protected_client):
    assert protected_client.get("/health").status_code == 200
    assert protected_client.get("/info").status_code == 401
    assert protected_client.post("/chat", json={"question": "q"}).status_code == 401
    assert protected_client.get("/info", headers={"X-API-Key": "test-key"}).status_code == 200
    assert protected_client.get("/info", headers={"Authorization": "Bearer test-key"}).status_code == 200


def test_chat_accepts_ten_turns_of_history(api_client):
    history = []
    for number in range(10):
        history.append({"role": "user", "content": f"question {number}"})
        history.append({"role": "assistant", "content": f"answer {number}"})
    body = api_client.post("/chat", json={"question": "How long has he done that?", "history": history}).json()
    assert body["status"] in {"answered", "not_covered", "declined"}
