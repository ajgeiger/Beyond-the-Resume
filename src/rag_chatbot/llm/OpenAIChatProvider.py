"""Answers from an OpenAI model through the Responses API. Optional."""

import openai

from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.llm.ChatReply import ChatReply
from rag_chatbot.llm.ChatRequest import ChatRequest
from rag_chatbot.llm.TokenUsage import TokenUsage


class OpenAIChatProvider(AbstractChatProvider):
    """OpenAI models through the Responses API.

    OpenAI caches a repeated prompt prefix automatically, so the stable
    instructions go first and the conversation goes last. With base_url set,
    the same class talks to any server that implements the OpenAI API, which
    is how a self hosted or larger open model plugs in.
    """

    DEFAULT_MODEL = "gpt-6-astra"
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        client=None,
        base_url: str = "",
        extra_pricing=None,
    ):
        super().__init__(model, extra_pricing)
        if not api_key and client is None:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.max_tokens = max_tokens
        self.client = client or openai.OpenAI(api_key=api_key, base_url=base_url or None)

    def payload(self, request: ChatRequest) -> dict:
        return {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
            "instructions": f"{request.system_rules}\n\n{request.knowledge_context}",
            "input": [{"role": message["role"], "content": message["content"]} for message in request.messages],
        }

    def complete(self, request: ChatRequest) -> ChatReply:
        response = self.client.responses.create(**self.payload(request))
        usage = response.usage
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        incomplete = getattr(response, "incomplete_details", None)
        return ChatReply(
            text=(response.output_text or "").strip(),
            model=self.model,
            truncated=getattr(incomplete, "reason", "") == "max_output_tokens",
            usage=TokenUsage(
                input_tokens=max((usage.input_tokens or 0) - cached, 0),
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=cached,
            ),
        )
