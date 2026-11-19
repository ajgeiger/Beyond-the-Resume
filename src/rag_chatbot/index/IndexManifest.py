"""What an index was built from, so a rebuild only happens when something changed."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IndexManifest:
    """fingerprint covers the documents, the model, the chunk settings and the pipeline version.

    index_version is the first twelve characters of the fingerprint, so two
    builds from the same inputs share a version.
    """

    fingerprint: str
    documents_hash: str
    documents: list[dict]
    model_name: str
    dimension: int
    chunking: dict
    chunk_count: int
    pipeline_version: str
    built_at: str
    source_counts: dict = field(default_factory=dict)

    FILE = "manifest.json"

    @property
    def index_version(self) -> str:
        return self.fingerprint[:12]

    @staticmethod
    def hash_documents(documents: list[dict]) -> str:
        text = json.dumps([[item["name"], item["sha256"]] for item in documents])
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def make_fingerprint(documents_hash: str, model_name: str, chunking: dict, extras: dict) -> str:
        payload = {"documents": documents_hash, "model": model_name, "chunking": chunking, **extras}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"index_version": self.index_version, **asdict(self)}

    def save(self, directory: Path) -> None:
        path = Path(directory) / self.FILE
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "IndexManifest | None":
        path = Path(directory) / cls.FILE
        if not path.is_file():
            return None
        values = json.loads(path.read_text(encoding="utf-8"))
        values.pop("index_version", None)
        return cls(**values)
