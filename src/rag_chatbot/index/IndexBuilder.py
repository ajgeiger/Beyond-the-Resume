"""Build the index, or leave it alone when nothing changed."""

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.index.BuildResult import BuildResult
from rag_chatbot.index.IndexManifest import IndexManifest
from rag_chatbot.index.VectorStore import VectorStore
from rag_chatbot.ingestion.IngestionPipeline import IngestionPipeline


class IndexBuilder:
    """Hash the sources first. If the fingerprint matches the saved manifest, skip the work.

    Bump PIPELINE_VERSION when a change to the parsing or cleaning code
    should force every index to rebuild.
    """

    PIPELINE_VERSION = "4"

    def __init__(self, pipeline: IngestionPipeline, model: AbstractEmbeddingModel, catalog: SourceCatalog):
        self.pipeline = pipeline
        self.model = model
        self.catalog = catalog

    def fingerprint(self, documents: list[dict]) -> tuple[str, str]:
        documents_hash = IndexManifest.hash_documents(documents)
        extras = {
            "pipeline": self.PIPELINE_VERSION,
            "corrections": [asdict(item) for item in self.catalog.corrections],
            "skipped_sections": list(self.catalog.skipped_sections),
        }
        chunking = self.pipeline.chunker.settings.to_dict()
        return IndexManifest.make_fingerprint(documents_hash, self.model.name, chunking, extras), documents_hash

    def build(self, directory: Path, force: bool = False) -> BuildResult:
        documents = [asdict(document) for document in self.pipeline.documents()]
        fingerprint, documents_hash = self.fingerprint(documents)
        current = IndexManifest.load(directory)
        if not force and current and current.fingerprint == fingerprint and VectorStore.exists(directory):
            return BuildResult(current, rebuilt=False)

        chunks = self.pipeline.chunks()
        if not chunks:
            raise ValueError("No chunks were produced. Check the source paths before replacing the index.")
        vectors = self.model.encode_documents([chunk.embedding_text for chunk in chunks])
        VectorStore(chunks, vectors).save(directory)
        manifest = IndexManifest(
            fingerprint=fingerprint,
            documents_hash=documents_hash,
            documents=documents,
            model_name=self.model.name,
            dimension=self.model.dimension,
            chunking=self.pipeline.chunker.settings.to_dict(),
            chunk_count=len(chunks),
            pipeline_version=self.PIPELINE_VERSION,
            built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source_counts=dict(Counter(chunk.source for chunk in chunks)),
        )
        manifest.save(directory)
        return BuildResult(manifest, rebuilt=True, skipped_sources=tuple(self.pipeline.skipped))
