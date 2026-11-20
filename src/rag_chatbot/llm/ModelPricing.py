"""Token prices per million tokens, used by the cost tracker and the spend cap."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """US dollars per million tokens for one model.

    cache_write is the price for writing the prompt cache (5 minute cache),
    cache_read is the price for reading it back.
    """

    input: float
    output: float
    cache_write: float = 0.0
    cache_read: float = 0.0

    @classmethod
    def for_model(cls, model: str, extra: dict | None = None) -> "ModelPricing":
        """Look up a model, with prices from configs/settings.toml first.

        Unknown models get the highest known price, so the cap stays safe.
        """
        table = {**PRICE_TABLE, **{name: cls(**values) for name, values in (extra or {}).items()}}
        if model in table:
            return table[model]
        return max(table.values(), key=lambda pricing: pricing.output)

    def cost(self, input_tokens: int, output_tokens: int, cache_write_tokens: int = 0, cache_read_tokens: int = 0):
        """Cost in dollars of one call."""
        per_million = 1_000_000
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_write_tokens * self.cache_write
            + cache_read_tokens * self.cache_read
        ) / per_million


# Claude prices come from the Anthropic pricing page (November 2026): Sonnet 5.5 is
# $2 in and $10 out, a 5 minute cache write is 1.25 times input, and a cache read is
# 0.05 times input. The GPT-6 Astra cache read rate is not assumed, so cached
# tokens are charged at the full input rate.
PRICE_TABLE = {
    "claude-sonnet-5-5": ModelPricing(input=2.0, output=10.0, cache_write=2.5, cache_read=0.10),
    "claude-haiku-5-5": ModelPricing(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
    "claude-fable-5-2": ModelPricing(input=8.0, output=40.0, cache_write=10.0, cache_read=0.20),
    "gpt-6-astra": ModelPricing(input=10.0, output=50.0, cache_write=10.0, cache_read=10.0),
    "fake": ModelPricing(input=0.0, output=0.0, cache_write=0.0, cache_read=0.0),
}
