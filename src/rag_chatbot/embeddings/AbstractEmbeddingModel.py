"""The contract for a model that turns text into vectors."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractEmbeddingModel(ABC):
    """Vectors come back as float32 rows with unit length, so a dot product is cosine similarity.

    count_tokens uses the model's own tokenizer, which lets the chunker size
    chunks so the model never truncates them.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of each vector."""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """One row per text."""

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """One vector for a search query."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Tokens in text, without special tokens."""

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
