"""Pick the embedding model named in the settings."""

from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.embeddings.FakeEmbeddingModel import FakeEmbeddingModel
from rag_chatbot.embeddings.OnnxEmbeddingModel import OnnxEmbeddingModel

try:
    from rag_chatbot.embeddings.SentenceTransformerEmbeddingModel import SentenceTransformerEmbeddingModel
except ImportError:
    SentenceTransformerEmbeddingModel = None


class EmbeddingModelFactory:
    @staticmethod
    def create(settings: EmbeddingSettings) -> AbstractEmbeddingModel:
        if settings.provider == "onnx":
            return OnnxEmbeddingModel(settings)
        if settings.provider == "sentence-transformers":
            if SentenceTransformerEmbeddingModel is None:
                raise ImportError(
                    "sentence-transformers is not installed. Use provider 'onnx', or pip install the torch extra."
                )
            return SentenceTransformerEmbeddingModel(settings)
        if settings.provider == "fake":
            return FakeEmbeddingModel()
        raise ValueError(f"Unknown embedding provider: {settings.provider}")
