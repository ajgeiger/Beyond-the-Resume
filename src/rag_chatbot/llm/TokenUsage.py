"""Token counts reported by a provider for one call."""

from dataclasses import dataclass

from rag_chatbot.llm.ModelPricing import ModelPricing


@dataclass(frozen=True)
class TokenUsage:
    """input_tokens excludes cached tokens; cached tokens are counted separately."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self, pricing: ModelPricing) -> float:
        return pricing.cost(self.input_tokens, self.output_tokens, self.cache_write_tokens, self.cache_read_tokens)
