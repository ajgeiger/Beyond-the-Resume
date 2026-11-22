"""How documents are cut into chunks."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChunkSettings:
    chunk_size_tokens: int = 160
    chunk_overlap_tokens: int = 16
    min_chunk_tokens: int = 8
    duplicate_overlap: float = 0.8

    def __post_init__(self):
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be at least 0 and smaller than chunk_size_tokens.")

    def to_dict(self) -> dict:
        return asdict(self)
