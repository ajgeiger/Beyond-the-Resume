"""The chatbot's answer to one question."""

from dataclasses import dataclass

from rag_chatbot_client.Citation import Citation


@dataclass(frozen=True)
class ChatResponse:
    """status is answered, not_covered, declined, blocked, budget or error."""

    answer: str
    status: str
    citations: tuple[Citation, ...]
    sources: tuple[str, ...]
    provider: str
    model: str
    index_version: str
    took_ms: float

    @property
    def is_answer(self) -> bool:
        return self.status == "answered"

    @classmethod
    def from_dict(cls, values: dict) -> "ChatResponse":
        return cls(
            answer=values["answer"],
            status=values["status"],
            citations=tuple(Citation.from_dict(item) for item in values.get("citations", [])),
            sources=tuple(values.get("sources", [])),
            provider=values.get("provider", ""),
            model=values.get("model", ""),
            index_version=values.get("index_version", ""),
            took_ms=float(values.get("took_ms", 0.0)),
        )
