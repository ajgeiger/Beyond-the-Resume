"""Which embedding model turns text into vectors."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""
    batch_size: int = 32
    # CPU threads for torch. 0 leaves the torch default (one per core).
    torch_threads: int = 2

    def to_dict(self) -> dict:
        return asdict(self)
