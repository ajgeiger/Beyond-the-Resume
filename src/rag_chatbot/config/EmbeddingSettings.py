"""Which embedding model turns text into vectors."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "onnx"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prefix: str = ""
    batch_size: int = 32
    # Local folder with the ONNX graph and tokenizer. Empty uses models/all-MiniLM-L6-v2.
    model_dir: str = ""
    # CPU threads for the embedding runtime. 0 leaves the library default.
    torch_threads: int = 2

    def to_dict(self) -> dict:
        return asdict(self)
