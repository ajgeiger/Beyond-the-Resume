"""The body GET /info returns."""

from pydantic import BaseModel


class InfoResponse(BaseModel):
    embedding_model: str
    dimension: int
    chunk_count: int
    index_version: str
    built_at: str
    chunking: dict
    sources: dict
    reranker: str | None
    retrieval: dict
    llm_provider: str
    llm_model: str
    demo_mode: bool
    spend_cap_reached: bool
