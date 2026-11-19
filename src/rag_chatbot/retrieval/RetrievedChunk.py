"""One search result with every score that led to it."""

from dataclasses import dataclass

from rag_chatbot.chunking.Chunk import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    """score is the value results are sorted by: the rerank score when reranking, else the fused score."""

    chunk: Chunk
    rank: int
    score: float
    dense_score: float
    bm25_score: float
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "rank": self.rank,
            "score": round(self.score, 6),
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }
