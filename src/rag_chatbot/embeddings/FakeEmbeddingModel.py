"""A tiny deterministic model for tests. No download and no torch."""

import hashlib
import re

import numpy as np

from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel


class FakeEmbeddingModel(AbstractEmbeddingModel):
    """Hash each word into one of a few buckets and count. Texts that share words point the same way."""

    WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 512):
        super().__init__("fake-hashing-embedding")
        self.size = dimension

    @property
    def dimension(self) -> int:
        return self.size

    def vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.size, dtype=np.float32)
        for word in self.WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size
            vector[bucket] += 1.0
        return vector

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.size), dtype=np.float32)
        return self.normalize(np.stack([self.vector(text) for text in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.normalize(self.vector(text))

    def count_tokens(self, text: str) -> int:
        return len(text.split())
