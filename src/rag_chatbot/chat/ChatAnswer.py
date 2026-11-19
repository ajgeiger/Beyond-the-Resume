"""What the page shows for one question."""

from dataclasses import dataclass

from rag_chatbot.retrieval.RetrievedChunk import RetrievedChunk


@dataclass(frozen=True)
class ChatAnswer:
    """status is one of: answered, not_covered, declined, blocked, budget, error.

    sources lists the chunks the answer cites, in citation order, with their retrieval scores.
    """

    text: str
    status: str
    sources: tuple[RetrievedChunk, ...] = ()
    cost_usd: float = 0.0

    @property
    def is_answer(self) -> bool:
        return self.status == "answered"
