"""One earlier turn of the conversation."""

from typing import Literal

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
