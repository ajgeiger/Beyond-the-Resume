"""The body of POST /retrieve."""

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    # None uses the reranker when the service has one turned on.
    rerank: bool | None = None
