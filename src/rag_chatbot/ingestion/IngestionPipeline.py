"""Read every source, clean it, drop repeats and chunk it."""

import hashlib

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.chunking.HeadingAwareChunker import HeadingAwareChunker
from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.guardrails.ContactScrubber import ContactScrubber
from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.ParagraphDeduplicator import ParagraphDeduplicator
from rag_chatbot.ingestion.Section import Section
from rag_chatbot.ingestion.SourceDocument import SourceDocument
from rag_chatbot.ingestion.TextCleaner import TextCleaner


class IngestionPipeline:
    """Turn source documents into chunks.

    Missing sources are reported and skipped, so one moved file does not stop
    the build. The last step refuses to return any chunk that still looks like
    it holds a phone number or an email address.
    """

    def __init__(self, sources: list[AbstractSource], catalog: SourceCatalog, chunker: HeadingAwareChunker):
        self.sources = sources
        self.cleaner = TextCleaner(catalog.corrections)
        self.chunker = chunker
        self.duplicate_overlap = chunker.settings.duplicate_overlap
        self.skipped: list[str] = []

    def documents(self) -> list[SourceDocument]:
        """Every source that exists, with its content hash. Cheap, so it runs before deciding to rebuild."""
        found = []
        for source in self.sources:
            if source.exists():
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                found.append(SourceDocument(source.name, source.path.name, digest))
        return found

    def sections(self) -> list[Section]:
        self.skipped = []
        deduplicator = ParagraphDeduplicator(self.duplicate_overlap)
        sections = []
        for source in self.sources:
            if not source.exists():
                self.skipped.append(str(source.path))
                continue
            for section in source.sections():
                cleaned = (self.cleaner.clean(paragraph) for paragraph in section.paragraphs)
                paragraphs = tuple(text for text in cleaned if text and deduplicator.is_new(text))
                if paragraphs:
                    header = self.cleaner.clean(section.header) if section.header else ""
                    title = self.cleaner.clean(section.title)
                    sections.append(Section(section.source, title, paragraphs, header=header))
        return sections

    def chunks(self) -> list[Chunk]:
        chunks = self.chunker.chunk(self.sections())
        leaked = [chunk.chunk_id for chunk in chunks if ContactScrubber.contains_contact(chunk.embedding_text)]
        if leaked:
            raise ValueError(f"Contact details survived cleaning in chunks: {', '.join(leaked)}")
        return chunks
