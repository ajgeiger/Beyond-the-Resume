"""The body of POST /chat."""

from pydantic import BaseModel, Field

from rag_chatbot.api.schemas.HistoryTurn import HistoryTurn


class ChatRequestBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Earlier turns, oldest first. Only the last few are used.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=50)
