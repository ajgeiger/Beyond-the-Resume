"""The body GET /health returns."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    index_version: str
    chunk_count: int
