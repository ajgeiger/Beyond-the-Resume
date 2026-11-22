"""The list of source documents, read from configs/sources.toml."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.config.Correction import Correction
from rag_chatbot.config.SourceSpec import SourceSpec


@dataclass(frozen=True)
class SourceCatalog:
    sources: tuple[SourceSpec, ...]
    skipped_sections: tuple[str, ...] = ()
    corrections: tuple[Correction, ...] = ()

    @classmethod
    def load(cls, path: Path, root: Path | None = None) -> "SourceCatalog":
        """root is the repository root. SOURCE_ROOT overrides origin_root, the folder origins are read from."""
        path = Path(path)
        root = Path(root) if root else path.parent.parent
        values = tomllib.loads(path.read_text(encoding="utf-8"))
        origin_root = Path(os.environ.get("SOURCE_ROOT") or root / values.get("origin_root", ".."))
        raw_dir = root / values.get("raw_dir", "data/raw")
        sources = tuple(
            SourceSpec(
                item["name"],
                item["kind"],
                raw_dir / item["file"],
                origin_root / item["origin"] if item.get("origin") else None,
                tuple(item.get("assignments", ())),
            )
            for item in values["sources"]
        )
        corrections = tuple(Correction(item["find"], item["replace"]) for item in values.get("corrections", []))
        return cls(sources, tuple(values.get("skipped_sections", [])), corrections)
