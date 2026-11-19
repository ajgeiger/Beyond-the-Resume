"""A local sentence-transformers model, such as all-MiniLM-L6-v2."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel


class SentenceTransformerEmbeddingModel(AbstractEmbeddingModel):
    """Runs on CPU. all-MiniLM-L6-v2 is 384 dimensions, about 90 MB, and reads up to 256 tokens."""

    def __init__(self, settings: EmbeddingSettings):
        super().__init__(settings.model_name)
        self.settings = settings
        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)
        self.model = SentenceTransformer(settings.model_name, device="cpu")

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.settings.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return self.normalize(vectors)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([self.settings.query_prefix + text], normalize_embeddings=True)
        return self.normalize(vector)[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))
