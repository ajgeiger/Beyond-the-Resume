"""The client against stub transports: parsing and error handling."""

import httpx
import pytest

from rag_chatbot_client import ChatbotClient, ChatbotError

CHAT_REPLY = {
    "answer": "He works at Signature Aviation.",
    "status": "answered",
    "citations": [
        {
            "number": 1,
            "chunk_id": "resume-001",
            "source": "Resume",
            "section": "Experience",
            "text": "Lead Applied AI Scientist at Signature Aviation.",
            "score": 0.03,
            "dense_score": 0.5,
            "bm25_score": 4.0,
            "rerank_score": None,
        }
    ],
    "sources": ["Resume"],
    "provider": "fake",
    "model": "fake",
    "index_version": "abc",
    "took_ms": 3.2,
}


def client_with(handler, api_key=""):
    http = httpx.Client(base_url="http://chatbot", transport=httpx.MockTransport(handler))
    return ChatbotClient(api_key=api_key, http=http)


def test_chat_parses_the_answer_and_citations():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("X-API-Key")
        seen["path"] = request.url.path
        return httpx.Response(200, json=CHAT_REPLY)

    response = client_with(handler, api_key="k").chat("Where does he work?")
    assert response.is_answer and response.sources == ("Resume",)
    assert response.citations[0].section == "Experience"
    assert seen == {"key": "k", "path": "/chat"}


def test_http_errors_raise_one_error_type():
    with pytest.raises(ChatbotError) as raised:
        client_with(lambda request: httpx.Response(401, json={})).chat("q")
    assert raised.value.status_code == 401 and not raised.value.is_outage


def test_connection_failures_are_outages():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ChatbotError) as raised:
        client_with(handler).health()
    assert raised.value.is_outage


def test_server_errors_are_outages():
    with pytest.raises(ChatbotError) as raised:
        client_with(lambda request: httpx.Response(503)).info()
    assert raised.value.is_outage
