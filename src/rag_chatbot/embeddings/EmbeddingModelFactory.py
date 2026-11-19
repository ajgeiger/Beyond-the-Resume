"""Pick the embedding model named in the settings."""

from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.embeddings.FakeEmbeddingModel import FakeEmbeddingModel
from rag_chatbot.embeddings.SentenceTransformerEmbeddingModel import SentenceTransformerEmbeddingModel


class EmbeddingModelFactory:
    @staticmethod
    def create(settings: EmbeddingSettings) -> AbstractEmbeddingModel:
        if settings.provider == "sentence-transformers":
            return SentenceTransformerEmbeddingModel(settings)
        if settings.provider == "fake":
            return FakeEmbeddingModel()
        raise ValueError(f"Unknown embedding provider: {settings.provider}")
