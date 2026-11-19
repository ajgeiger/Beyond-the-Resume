"""Cut sections into token sized chunks that never cross a heading."""

import re
from collections.abc import Callable

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.config.ChunkSettings import ChunkSettings
from rag_chatbot.ingestion.Section import Section


class HeadingAwareChunker:
    """Pack sentences into chunks of at most chunk_size_tokens.

    Each section is chunked on its own, so a chunk belongs to exactly one
    heading. Text is packed a sentence at a time, and a sentence longer than a
    whole chunk is split at words. Consecutive chunks of a section share up to
    chunk_overlap_tokens of text: whole trailing sentences when they fit,
    otherwise the end of the last sentence. Chunks under min_chunk_tokens are
    dropped.
    """

    SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, settings: ChunkSettings, count_tokens: Callable[[str], int]):
        self.settings = settings
        self.count_tokens = count_tokens

    def chunk(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        per_source: dict[str, int] = {}
        for section in sections:
            for text in self.pack(self.units(section.paragraphs)):
                if self.count_tokens(text) < self.settings.min_chunk_tokens:
                    continue
                number = per_source.get(section.source, 0) + 1
                per_source[section.source] = number
                chunk_id = f"{self.slug(section.source)}-{number:03d}"
                chunks.append(
                    Chunk(chunk_id, section.source, section.title, text, self.count_tokens(text), len(chunks))
                )
        return chunks

    def units(self, paragraphs: tuple[str, ...]) -> list[tuple[str, int]]:
        """Sentences that each fit in one chunk, with their token counts."""
        size = self.settings.chunk_size_tokens
        units = []
        for paragraph in paragraphs:
            for sentence in self.SENTENCE_END.split(paragraph):
                sentence_tokens = self.count_tokens(sentence)
                if sentence_tokens <= size:
                    units.append((sentence, sentence_tokens))
                else:
                    units += [(piece, self.count_tokens(piece)) for piece in self.split_words(sentence, size)]
        return units

    def pack(self, units: list[tuple[str, int]]) -> list[str]:
        size, overlap = self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens
        packed: list[str] = []
        current: list[tuple[str, int]] = []
        for unit in units:
            if current and self.total(current) + unit[1] > size:
                packed.append(" ".join(text for text, _ in current))
                current = self.tail(current, overlap)
                while current and self.total(current) + unit[1] > size:
                    current.pop(0)
            current.append(unit)
        if current:
            packed.append(" ".join(text for text, _ in current))
        return packed

    def tail(self, units: list[tuple[str, int]], budget: int) -> list[tuple[str, int]]:
        """The trailing text of a chunk that fits in the overlap budget."""
        if budget <= 0:
            return []
        kept: list[tuple[str, int]] = []
        for text, tokens in reversed(units):
            if self.total(kept) + tokens > budget:
                break
            kept.insert(0, (text, tokens))
        if kept:
            return kept
        words = units[-1][0].split()
        for start in range(len(words)):
            piece = " ".join(words[start:])
            tokens = self.count_tokens(piece)
            if tokens <= budget:
                return [(piece, tokens)]
        return []

    def split_words(self, text: str, size: int) -> list[str]:
        pieces, current = [], []
        for word in text.split():
            if current and self.count_tokens(" ".join(current + [word])) > size:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces

    @staticmethod
    def total(units: list[tuple[str, int]]) -> int:
        return sum(tokens for _, tokens in units)

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
