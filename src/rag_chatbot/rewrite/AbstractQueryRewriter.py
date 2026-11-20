"""Turn a follow up question into a standalone retrieval query."""

from abc import ABC, abstractmethod

from rag_chatbot.rewrite.RewriteResult import RewriteResult


class AbstractQueryRewriter(ABC):
    """Rewrite is only for retrieval. The language model still sees the original question and history."""

    @abstractmethod
    def rewrite(self, question: str, history: list[dict]) -> RewriteResult:
        """Return a self contained query when earlier turns are needed, otherwise the question as given."""
