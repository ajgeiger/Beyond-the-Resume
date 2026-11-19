"""Providers, the provider factory and pricing. No network calls: providers get stub clients."""

from types import SimpleNamespace

import pytest

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.llm.AnthropicChatProvider import AnthropicChatProvider
from rag_chatbot.llm.FakeChatProvider import FakeChatProvider
from rag_chatbot.llm.OpenAIChatProvider import OpenAIChatProvider
from rag_chatbot.llm.ProviderFactory import ProviderFactory
from rag_chatbot.prompts.PromptBuilder import PromptBuilder

CHUNK = Chunk(
    chunk_id="resume-001",
    source="Resume",
    section="Experience",
    text="Lead Applied AI Scientist at Signature Aviation since Jul 2026, building forecasting systems.",
    position=0,
    token_count=14,
)


def request():
    return PromptBuilder().request("What is his current role?", [], [CHUNK])


def test_auto_falls_back_to_the_fake_provider_without_keys():
    assert isinstance(ProviderFactory.create(LlmSettings()), FakeChatProvider)


def test_auto_prefers_anthropic_then_openai():
    anthropic = ProviderFactory.create(LlmSettings(anthropic_api_key="a", openai_api_key="b"))
    openai = ProviderFactory.create(LlmSettings(openai_api_key="b"))
    assert isinstance(anthropic, AnthropicChatProvider) and anthropic.model == "claude-sonnet-5-5"
    assert isinstance(openai, OpenAIChatProvider) and openai.model == "gpt-6-astra"


def test_a_different_model_is_a_config_change():
    settings = LlmSettings(provider="openai", openai_api_key="b", openai_model="bigger-model", base_url="http://x/v1")
    provider = ProviderFactory.create(settings)
    assert provider.model == "bigger-model"
    assert str(provider.client.base_url).startswith("http://x/v1")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        ProviderFactory.create(LlmSettings(provider="banana"))


def test_a_named_provider_needs_its_key():
    with pytest.raises(ValueError):
        ProviderFactory.create(LlmSettings(provider="anthropic"))


def test_keys_never_appear_in_repr():
    assert "secret-key" not in repr(LlmSettings(anthropic_api_key="secret-key", openai_api_key="secret-key"))


def test_anthropic_payload_caches_the_rules_only():
    provider = AnthropicChatProvider("key")
    payload = provider.payload(request())
    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["system"][1]
    assert "[C1]" in payload["system"][1]["text"]
    assert payload["messages"][-1]["role"] == "user"


def test_anthropic_reply_is_parsed_with_usage():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=" He leads applied AI. [C1] ")],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=800
        ),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_kwargs: response))
    reply = AnthropicChatProvider("", client=client).complete(request())
    assert reply.text == "He leads applied AI. [C1]"
    assert reply.usage.cache_read_tokens == 800


def test_openai_reply_separates_cached_tokens():
    response = SimpleNamespace(
        output_text="He leads applied AI. [C1]",
        usage=SimpleNamespace(
            input_tokens=900, output_tokens=20, input_tokens_details=SimpleNamespace(cached_tokens=800)
        ),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: response))
    reply = OpenAIChatProvider("", client=client).complete(request())
    assert reply.usage.input_tokens == 100 and reply.usage.cache_read_tokens == 800


def test_fake_provider_quotes_and_cites_the_passages():
    reply = FakeChatProvider().complete(request())
    assert "Signature Aviation" in reply.text and "[C1]" in reply.text


def test_extra_pricing_from_config_is_used():
    provider = OpenAIChatProvider(
        "key", model="bigger-model", extra_pricing={"bigger-model": {"input": 1.0, "output": 4.0}}
    )
    assert provider.pricing.cost(input_tokens=1_000_000, output_tokens=1_000_000) == 5.0
