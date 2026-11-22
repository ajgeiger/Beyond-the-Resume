"""Which language model writes the answers."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LlmSettings:
    """provider is anthropic, openai, fake or auto.

    auto picks Anthropic when ANTHROPIC_API_KEY is set, then OpenAI when
    OPENAI_API_KEY is set, then the offline fake. A bigger or different model
    is a config change: set the model name here or LLM_MODEL in the
    environment. rewrite_model (or LLM_REWRITE_MODEL) writes follow up search
    queries. base_url points the OpenAI provider at any OpenAI compatible
    server, such as a Hugging Face Inference Endpoint or vLLM.
    """

    provider: str = "auto"
    anthropic_model: str = "claude-sonnet-5-5"
    rewrite_model: str = "claude-haiku-5-5"
    openai_model: str = "gpt-6-astra"
    base_url: str = ""
    max_output_tokens: int = 1024
    anthropic_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    # Extra prices in US dollars per million tokens, keyed by model name.
    pricing: dict = field(default_factory=dict)

    @property
    def resolved_provider(self) -> str:
        if self.provider != "auto":
            return self.provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "fake"

    @property
    def model(self) -> str:
        """The model the resolved provider will call."""
        provider = self.resolved_provider
        if provider == "anthropic":
            return self.anthropic_model
        if provider == "openai":
            return self.openai_model
        return "fake"
