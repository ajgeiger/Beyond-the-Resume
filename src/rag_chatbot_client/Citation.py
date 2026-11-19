"""One passage an answer cites."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    source: str
    section: str
    text: str
    score: float

    @classmethod
    def from_dict(cls, values: dict) -> "Citation":
        return cls(
            values["number"], values["chunk_id"], values["source"], values["section"], values["text"], values["score"]
        )
