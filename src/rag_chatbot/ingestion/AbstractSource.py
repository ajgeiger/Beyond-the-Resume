"""The contract for a document the index can read."""

from abc import ABC, abstractmethod
from pathlib import Path

from rag_chatbot.ingestion.Section import Section


class AbstractSource(ABC):
    """A source reads one file and returns its text as titled sections."""

    def __init__(self, path: Path, name: str):
        self.path = Path(path)
        self.name = name

    def exists(self) -> bool:
        return self.path.is_file()

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    @abstractmethod
    def sections(self) -> list[Section]:
        """Read the file and return its sections in reading order."""
