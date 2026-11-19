"""The body POST /chat returns."""

from pydantic import BaseModel

from rag_chatbot.api.schemas.CitationResult import CitationResult


class ChatResponseBody(BaseModel):
    """status is answered, not_covered, declined, blocked, budget or error.

    citations lists the cited passages in order. sources lists the source
    documents they come from, once each.
    """

    answer: str
    status: str
    citations: list[CitationResult]
    sources: list[str]
    provider: str
    model: str
    index_version: str
    took_ms: float
