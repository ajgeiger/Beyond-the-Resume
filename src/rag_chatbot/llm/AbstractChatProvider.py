"""The contract every language model provider follows."""

from abc import ABC, abstractmethod

from rag_chatbot.llm.ChatReply import ChatReply
from rag_chatbot.llm.ChatRequest import ChatRequest
from rag_chatbot.llm.ModelPricing import ModelPricing


class AbstractChatProvider(ABC):
    """A provider turns a ChatRequest into a ChatReply.

    Swapping providers is one setting (LLM_PROVIDER), and swapping models is
    another (LLM_MODEL). Everything else, from retrieval to guardrails, stays
    the same.
    """

    name = "abstract"

    def __init__(self, model: str, extra_pricing: dict | None = None):
        self.model = model
        self.extra_pricing = extra_pricing or {}

    @property
    def pricing(self) -> ModelPricing:
        return ModelPricing.for_model(self.model, self.extra_pricing)

    @property
    def is_demo(self) -> bool:
        """True when answers do not come from a real language model."""
        return False

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatReply:
        """Answer the last user message in the request."""
