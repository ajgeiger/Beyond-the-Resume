"""The body POST /retrieve returns."""

from pydantic import BaseModel

from rag_chatbot.api.schemas.ChunkResult import ChunkResult


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    index_version: str
    took_ms: float
    results: list[ChunkResult]
