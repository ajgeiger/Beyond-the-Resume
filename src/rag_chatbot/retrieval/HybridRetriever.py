"""Dense vectors plus BM25, fused with reciprocal rank fusion, with an optional rerank."""

from rag_chatbot.config.RetrievalSettings import RetrievalSettings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.index.VectorStore import VectorStore
from rag_chatbot.retrieval.AbstractReranker import AbstractReranker
from rag_chatbot.retrieval.Bm25Index import Bm25Index
from rag_chatbot.retrieval.ReciprocalRankFusion import ReciprocalRankFusion
from rag_chatbot.retrieval.RetrievedChunk import RetrievedChunk


class HybridRetriever:
    """Rank chunks for a query.

    mode is hybrid, dense or bm25. Dense search finds paraphrases, BM25 finds
    exact names and numbers, and fusion keeps the strengths of both. A chunk
    only takes part when it clears min_dense_similarity or min_bm25_score, so
    a question the sources do not cover returns an empty list.
    """

    MODES = ("hybrid", "dense", "bm25")

    def __init__(
        self,
        store: VectorStore,
        model: AbstractEmbeddingModel,
        settings: RetrievalSettings,
        reranker: AbstractReranker | None = None,
    ):
        self.store = store
        self.model = model
        self.settings = settings
        self.reranker = reranker
        self.bm25 = Bm25Index(store.chunks)
        self.fusion = ReciprocalRankFusion(settings.rrf_k)

    def retrieve(
        self, query: str, top_k: int = 5, mode: str = "hybrid", rerank: bool | None = None
    ) -> list[RetrievedChunk]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {', '.join(self.MODES)}")
        if not query.strip() or not len(self.store):
            return []
        dense = [float(value) for value in self.store.similarities(self.model.encode_query(query))]
        keyword = self.bm25.scores(query)
        rankings = []
        if mode in ("hybrid", "dense"):
            rankings.append(self.ranking(dense, self.settings.min_dense_similarity))
        if mode in ("hybrid", "bm25"):
            rankings.append(self.ranking(keyword, self.settings.min_bm25_score))
        fused = self.fusion.fuse(rankings)[: max(top_k, self.settings.candidate_pool)]

        use_reranker = self.reranker is not None if rerank is None else (rerank and self.reranker is not None)
        rerank_scores: list[float | None] = [None] * len(fused)
        if use_reranker and fused:
            passages = [self.store.chunks[position].embedding_text for position, _ in fused]
            rerank_scores = list(self.reranker.scores(query, passages))
            order = sorted(range(len(fused)), key=lambda item: rerank_scores[item], reverse=True)
            fused = [fused[item] for item in order]
            rerank_scores = [rerank_scores[item] for item in order]

        results = []
        for rank, ((position, fused_score), rerank_score) in enumerate(zip(fused, rerank_scores, strict=True), start=1):
            if rank > top_k:
                break
            score = fused_score if rerank_score is None else rerank_score
            results.append(
                RetrievedChunk(
                    self.store.chunks[position], rank, score, dense[position], keyword[position], rerank_score
                )
            )
        return results

    def ranking(self, scores: list[float], minimum: float) -> list[int]:
        """Positions that clear the bar, best first, cut to the candidate pool."""
        passing = [position for position, score in enumerate(scores) if score >= minimum and score > 0]
        passing.sort(key=lambda position: scores[position], reverse=True)
        return passing[: self.settings.candidate_pool]
