"""Pick a query rewriter from the language model settings."""

from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.rewrite.AbstractQueryRewriter import AbstractQueryRewriter
from rag_chatbot.rewrite.ClaudeQueryRewriter import ClaudeQueryRewriter
from rag_chatbot.rewrite.GuardedQueryRewriter import GuardedQueryRewriter
from rag_chatbot.rewrite.HeuristicQueryRewriter import HeuristicQueryRewriter


class QueryRewriterFactory:
    """A Claude rewriter when an Anthropic key is set, otherwise the free heuristic."""

    @staticmethod
    def create(settings: LlmSettings, tracker: CostTracker | None = None) -> AbstractQueryRewriter:
        fallback = HeuristicQueryRewriter()
        if settings.resolved_provider == "anthropic" and settings.anthropic_api_key:
            paid = ClaudeQueryRewriter(
                settings.anthropic_api_key,
                settings.rewrite_model,
                extra_pricing=settings.pricing,
            )
            return GuardedQueryRewriter(paid, fallback, tracker)
        return fallback
