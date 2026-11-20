"""Use a paid rewriter for follow ups, and fall back to a free heuristic when it cannot run."""

from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.rewrite.AbstractQueryRewriter import AbstractQueryRewriter
from rag_chatbot.rewrite.HeuristicQueryRewriter import HeuristicQueryRewriter
from rag_chatbot.rewrite.RewriteResult import RewriteResult


class GuardedQueryRewriter(AbstractQueryRewriter):
    """Skip the paid call when there is no history, the question is already clear, or the budget is spent."""

    def __init__(
        self,
        paid: AbstractQueryRewriter,
        fallback: HeuristicQueryRewriter | None = None,
        tracker: CostTracker | None = None,
    ):
        self.paid = paid
        self.fallback = fallback or HeuristicQueryRewriter()
        self.tracker = tracker

    def rewrite(self, question: str, history: list[dict]) -> RewriteResult:
        if not history or not self.fallback.is_follow_up(question):
            return self.fallback.rewrite(question, history)
        if self.tracker is not None and self.tracker.is_exhausted():
            return self.fallback.rewrite(question, history)
        try:
            result = self.paid.rewrite(question, history)
        except Exception:
            return self.fallback.rewrite(question, history)
        if not result.query.strip():
            return self.fallback.rewrite(question, history)
        return result
