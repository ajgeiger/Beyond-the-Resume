"""The installable client talking to the real API app."""

import pytest

from rag_chatbot_client import ChatbotClient, ChatbotError

pytestmark = pytest.mark.integration


def test_client_chat_round_trip(api_client):
    client = ChatbotClient(http=api_client)
    response = client.chat("Where did he study for his master's degree?")
    assert response.is_answer
    assert "Syracuse University" in response.answer
    assert response.citations[0].number == 1
    assert client.health()["status"] == "ok"
    assert client.retrieve("Headline Radar", top_k=1)["results"]


def test_client_sends_the_api_key(protected_client):
    with pytest.raises(ChatbotError) as raised:
        ChatbotClient(http=protected_client).info()
    assert raised.value.status_code == 401
    assert ChatbotClient(api_key="test-key", http=protected_client).info()["llm_provider"] == "fake"
