"""One retrieved chunk as the API returns it."""

from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_id: str
    source: str
    section: str
    text: str
    token_count: int
    position: int
    rank: int
    score: float
    dense_score: float
    bm25_score: float
    rerank_score: float | None = None
