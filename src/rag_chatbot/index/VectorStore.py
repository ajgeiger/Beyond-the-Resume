"""Chunks and their vectors, saved as a JSON file and a numpy array."""

import json
from pathlib import Path

import numpy as np

from rag_chatbot.chunking.Chunk import Chunk


class VectorStore:
    """Exact cosine search with one matrix product.

    A few hundred chunks fit in well under a megabyte, so an exact numpy
    search is faster to load and simpler to ship than an approximate index.
    """

    CHUNKS_FILE = "chunks.json"
    VECTORS_FILE = "vectors.npy"

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk needs exactly one vector.")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk ids must be unique.")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to every chunk, in chunk order."""
        if not len(self.chunks):
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ np.asarray(query_vector, dtype=np.float32)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = [chunk.to_dict() for chunk in self.chunks]
        (directory / self.CHUNKS_FILE).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", "utf-8")
        np.save(directory / self.VECTORS_FILE, self.vectors)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        rows = json.loads((directory / cls.CHUNKS_FILE).read_text(encoding="utf-8"))
        vectors = np.load(directory / cls.VECTORS_FILE)
        return cls([Chunk.from_dict(row) for row in rows], vectors)

    @classmethod
    def exists(cls, directory: Path) -> bool:
        directory = Path(directory)
        return (directory / cls.CHUNKS_FILE).is_file() and (directory / cls.VECTORS_FILE).is_file()
