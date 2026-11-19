"""Answers from Claude through the Anthropic Messages API."""

import anthropic

from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.llm.ChatReply import ChatReply
from rag_chatbot.llm.ChatRequest import ChatRequest
from rag_chatbot.llm.TokenUsage import TokenUsage


class AnthropicChatProvider(AbstractChatProvider):
    """Claude Sonnet 5.5 by default, with prompt caching on the stable part of the prompt.

    The rules are the same for every visitor, so they carry the cache
    breakpoint. The retrieved passages change with each question and come
    after it, uncached.
    """

    DEFAULT_MODEL = "claude-sonnet-5-5"
    name = "anthropic"

    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = 700, client=None, extra_pricing=None
    ):
        super().__init__(model, extra_pricing)
        if not api_key and client is None:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def payload(self, request: ChatRequest) -> dict:
        """The exact arguments sent to the API. Kept separate so tests can check it without a network call."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": request.system_rules, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": request.knowledge_context},
            ],
            "messages": [{"role": message["role"], "content": message["content"]} for message in request.messages],
        }

    def complete(self, request: ChatRequest) -> ChatReply:
        response = self.client.messages.create(**self.payload(request))
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = response.usage
        return ChatReply(
            text=text.strip(),
            model=self.model,
            usage=TokenUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
        )
