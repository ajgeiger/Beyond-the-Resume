"""How dense and keyword results are combined."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetrievalSettings:
    default_top_k: int = 8
    max_top_k: int = 20
    candidate_pool: int = 30
    rrf_k: int = 60
    min_dense_similarity: float = 0.32
    min_bm25_score: float = 3.5

    def to_dict(self) -> dict:
        return asdict(self)
