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

    Sonnet 5.5 thinks before it answers unless told not to, and that thinking
    counts against max_tokens. Short answers do not need it, so it is turned
    off. A model that does not accept the setting is called again without it.
    """

    DEFAULT_MODEL = "claude-sonnet-5-5"
    THINKING_OFF = {"type": "between_tools"}
    name = "anthropic"

    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = 1024, client=None, extra_pricing=None
    ):
        super().__init__(model, extra_pricing)
        if not api_key and client is None:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.max_tokens = max_tokens
        self.thinking = dict(self.THINKING_OFF)
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def payload(self, request: ChatRequest) -> dict:
        """The exact arguments sent to the API. Kept separate so tests can check it without a network call."""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": request.system_rules, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": request.knowledge_context},
            ],
            "messages": [{"role": message["role"], "content": message["content"]} for message in request.messages],
        }
        if self.thinking:
            payload["thinking"] = self.thinking
        return payload

    def complete(self, request: ChatRequest) -> ChatReply:
        try:
            response = self.client.messages.create(**self.payload(request))
        except anthropic.BadRequestError as error:
            if not self.thinking or "thinking" not in str(error):
                raise
            self.thinking = None
            response = self.client.messages.create(**self.payload(request))
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = response.usage
        return ChatReply(
            text=text.strip(),
            model=self.model,
            truncated=getattr(response, "stop_reason", "") == "max_tokens",
            usage=TokenUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
        )
