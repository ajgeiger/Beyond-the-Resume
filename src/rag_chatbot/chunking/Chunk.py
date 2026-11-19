"""One retrievable passage with the metadata needed to cite it."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    """chunk_id is stable for a given build, for example resume-003."""

    chunk_id: str
    source: str
    section: str
    text: str
    token_count: int
    position: int

    @property
    def embedding_text(self) -> str:
        """The document and section path go in front, so the vector knows what the passage is about."""
        return f"{self.source}: {self.section}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Chunk":
        return cls(**values)
