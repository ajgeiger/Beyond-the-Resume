"""One document listed in configs/sources.toml."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    """path is the copy in data/raw that the index is built from. origin is where it is copied from."""

    name: str
    kind: str
    path: Path
    origin: Path | None = None
    assignments: tuple[str, ...] = ()
