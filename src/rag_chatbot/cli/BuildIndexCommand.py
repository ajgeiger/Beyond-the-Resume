"""Chunk the sources, embed them and save the index."""

from rag_chatbot.chunking.HeadingAwareChunker import HeadingAwareChunker
from rag_chatbot.config.Settings import Settings
from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.embeddings.EmbeddingModelFactory import EmbeddingModelFactory
from rag_chatbot.index.IndexBuilder import IndexBuilder
from rag_chatbot.ingestion.IngestionPipeline import IngestionPipeline
from rag_chatbot.ingestion.SourceFactory import SourceFactory


class BuildIndexCommand:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, force: bool = False) -> int:
        catalog = SourceCatalog.load(self.settings.sources_file, self.settings.root)
        model = EmbeddingModelFactory.create(self.settings.embedding)
        chunker = HeadingAwareChunker(self.settings.chunking, model.count_tokens)
        pipeline = IngestionPipeline(SourceFactory.create(catalog), catalog, chunker)
        result = IndexBuilder(pipeline, model, catalog).build(self.settings.index_dir, force)
        manifest = result.manifest
        verb = "Built" if result.rebuilt else "Up to date, kept"
        print(f"{verb} index {manifest.index_version} in {self.settings.index_dir}")
        print(f"  model: {manifest.model_name} ({manifest.dimension} dimensions)")
        print(f"  chunking: {manifest.chunking}")
        print(f"  chunks: {manifest.chunk_count}")
        for source, count in manifest.source_counts.items():
            print(f"    {source}: {count}")
        for path in result.skipped_sources:
            print(f"  skipped missing source: {path}")
        return 0
