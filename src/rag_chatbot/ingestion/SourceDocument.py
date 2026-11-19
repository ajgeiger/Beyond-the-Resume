"""A source that was read, with the hash used to detect changes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    name: str
    path: str
    sha256: str
