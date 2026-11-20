"""A sentence-transformers cross encoder that reads the query and passage together."""

from rag_chatbot.retrieval.AbstractReranker import AbstractReranker

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class CrossEncoderReranker(AbstractReranker):
    """More accurate than the bi encoder, but it runs once per candidate, so it only sees the fused shortlist."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is not installed. The reranker needs the torch extra.")
        self.model = CrossEncoder(model_name, device="cpu")

    def scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self.model.predict([(query, passage) for passage in passages])]
