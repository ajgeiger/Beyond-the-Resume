from datetime import UTC, datetime

from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.llm.ModelPricing import ModelPricing
from rag_chatbot.llm.TokenUsage import TokenUsage


def test_sonnet_pricing_matches_the_published_rates():
    pricing = ModelPricing.for_model("claude-sonnet-5-5")
    assert pricing.cost(input_tokens=1_000_000, output_tokens=0) == 2.0
    assert pricing.cost(input_tokens=0, output_tokens=1_000_000) == 10.0


def test_usage_cost_counts_cache_reads_cheaply():
    pricing = ModelPricing.for_model("claude-sonnet-5-5")
    cached = TokenUsage(input_tokens=100, output_tokens=400, cache_read_tokens=8_000).cost(pricing)
    uncached = TokenUsage(input_tokens=8_100, output_tokens=400).cost(pricing)
    assert cached < uncached / 3


def test_unknown_model_is_priced_at_the_highest_rate():
    assert ModelPricing.for_model("some-new-model").output == max(
        ModelPricing.for_model(name).output for name in ("claude-sonnet-5-5", "gpt-6-astra")
    )


def test_tracker_stops_at_the_budget_and_resets_next_month(tmp_path):
    now = {"value": datetime(2026, 11, 30, tzinfo=UTC)}
    tracker = CostTracker(tmp_path / "usage.json", budget_usd=1.0, clock=lambda: now["value"])
    tracker.record(0.6)
    assert not tracker.is_exhausted()
    tracker.record(0.5)
    assert tracker.is_exhausted() and tracker.remaining() == 0.0
    now["value"] = datetime(2026, 12, 1, tzinfo=UTC)
    assert not tracker.is_exhausted() and tracker.spent() == 0.0
