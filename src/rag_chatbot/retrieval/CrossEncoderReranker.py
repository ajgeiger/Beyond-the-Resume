"""A sentence-transformers cross encoder that reads the query and passage together."""

from sentence_transformers import CrossEncoder

from rag_chatbot.retrieval.AbstractReranker import AbstractReranker


class CrossEncoderReranker(AbstractReranker):
    """More accurate than the bi encoder, but it runs once per candidate, so it only sees the fused shortlist."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = CrossEncoder(model_name, device="cpu")

    def scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self.model.predict([(query, passage) for passage in passages])]
