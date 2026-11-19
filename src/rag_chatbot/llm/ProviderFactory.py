"""Pick the chat provider named in the settings."""

from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.llm.AnthropicChatProvider import AnthropicChatProvider
from rag_chatbot.llm.FakeChatProvider import FakeChatProvider
from rag_chatbot.llm.OpenAIChatProvider import OpenAIChatProvider


class ProviderFactory:
    """anthropic, openai or fake. auto means Anthropic, then OpenAI, by which key is set, else fake."""

    @staticmethod
    def create(settings: LlmSettings) -> AbstractChatProvider:
        provider = settings.resolved_provider
        if provider == "anthropic":
            return AnthropicChatProvider(
                settings.anthropic_api_key,
                settings.anthropic_model,
                settings.max_output_tokens,
                extra_pricing=settings.pricing,
            )
        if provider == "openai":
            return OpenAIChatProvider(
                settings.openai_api_key,
                settings.openai_model,
                settings.max_output_tokens,
                base_url=settings.base_url,
                extra_pricing=settings.pricing,
            )
        if provider == "fake":
            return FakeChatProvider()
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use anthropic, openai, fake or auto.")
