"""What a call to the index builder did."""

from dataclasses import dataclass

from rag_chatbot.index.IndexManifest import IndexManifest


@dataclass(frozen=True)
class BuildResult:
    manifest: IndexManifest
    rebuilt: bool
    skipped_sources: tuple[str, ...] = ()
