"""Cut sections into token sized chunks that never cross a heading."""

import re
from collections.abc import Callable

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.config.ChunkSettings import ChunkSettings
from rag_chatbot.ingestion.Section import Section


class HeadingAwareChunker:
    """Pack text into chunks of at most chunk_size_tokens.

    Each section is chunked on its own, so a chunk belongs to exactly one
    heading. A paragraph that fits is kept together. Longer paragraphs are
    split on sentences, and a sentence longer than a whole chunk is split at
    words. When a section has a header, that header is prefixed onto every
    chunk and reserved in the size budget. Consecutive chunks of a section
    share up to chunk_overlap_tokens of body text, only when a section has
    to be split. Chunks under min_chunk_tokens are dropped.
    """

    SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, settings: ChunkSettings, count_tokens: Callable[[str], int]):
        self.settings = settings
        self.count_tokens = count_tokens

    def chunk(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        per_source: dict[str, int] = {}
        for section in sections:
            budget = self.body_budget(section.header)
            for group in self.groups(section.paragraphs, budget, isolate_lists=bool(section.header)):
                for body in self.pack(group, budget):
                    text = self.with_header(section.header, body)
                    if self.count_tokens(text) < self.settings.min_chunk_tokens:
                        continue
                    number = per_source.get(section.source, 0) + 1
                    per_source[section.source] = number
                    chunk_id = f"{self.slug(section.source)}-{number:03d}"
                    chunks.append(
                        Chunk(chunk_id, section.source, section.title, text, self.count_tokens(text), len(chunks))
                    )
        return chunks

    def body_budget(self, header: str) -> int:
        """Tokens left for the body after reserving the header."""
        reserved = self.count_tokens(header) + 1 if header else 0
        return max(self.settings.min_chunk_tokens, self.settings.chunk_size_tokens - reserved)

    def groups(
        self, paragraphs: tuple[str, ...], budget: int, isolate_lists: bool = False
    ) -> list[list[tuple[str, int]]]:
        """Units that may be packed together, grouped by paragraph kind.

        Ordinary paragraphs that fit are packed with their neighbors. When
        isolate_lists is set, a parent bullet that introduces a list of
        sub-bullets is split so each sub-bullet stays with the parent lead-in
        and is not packed with its siblings, even when the whole list would
        fit in one chunk. That is for resume role sections, not site pages.
        """
        groups: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []

        def flush() -> None:
            if current:
                groups.append(list(current))
                current.clear()

        for paragraph in paragraphs:
            lead, rest = self.colon_lead_in(paragraph, budget)
            sentences = [item for item in self.SENTENCE_END.split(rest) if item.strip()] if lead else []
            if isolate_lists and lead and len(sentences) > 1:
                flush()
                for unit in self.split_paragraph(paragraph, budget):
                    groups.append([unit])
                continue
            paragraph_tokens = self.count_tokens(paragraph)
            if paragraph_tokens <= budget:
                current.append((paragraph, paragraph_tokens))
                continue
            flush()
            current.extend(self.split_paragraph(paragraph, budget))
        flush()
        return groups

    def split_paragraph(self, paragraph: str, budget: int) -> list[tuple[str, int]]:
        """Split a long paragraph on sentences, keeping a list lead-in on each piece."""
        lead, rest = self.colon_lead_in(paragraph, budget)
        units: list[tuple[str, int]] = []
        for sentence in self.SENTENCE_END.split(rest):
            piece = f"{lead} {sentence}".strip() if lead else sentence
            tokens = self.count_tokens(piece)
            if tokens <= budget:
                units.append((piece, tokens))
            else:
                units += [(part, self.count_tokens(part)) for part in self.split_words(piece, budget)]
        return units

    MAX_LEAD_WORDS = 12

    def colon_lead_in(self, paragraph: str, budget: int) -> tuple[str, str]:
        """If a parent sentence ends in a colon, keep a short lead-in with every piece."""
        if ": " not in paragraph:
            return "", paragraph
        lead, rest = paragraph.split(": ", 1)
        words = f"{lead}:".split()
        if len(words) > self.MAX_LEAD_WORDS:
            words = words[-self.MAX_LEAD_WORDS :]
        lead = " ".join(words)
        # The lead-in has to leave room for at least one following sentence.
        if self.count_tokens(lead) > budget // 2:
            return "", paragraph
        return lead, rest

    def pack(self, units: list[tuple[str, int]], budget: int) -> list[str]:
        overlap = self.settings.chunk_overlap_tokens
        packed: list[str] = []
        current: list[tuple[str, int]] = []
        for unit in units:
            if current and self.total(current) + unit[1] > budget:
                packed.append(" ".join(text for text, _ in current))
                current = self.tail(current, overlap)
                while current and self.total(current) + unit[1] > budget:
                    current.pop(0)
            current.append(unit)
        if current:
            packed.append(" ".join(text for text, _ in current))
        return packed

    def tail(self, units: list[tuple[str, int]], budget: int) -> list[tuple[str, int]]:
        """Trailing whole units that fit in the overlap budget.

        If the last unit is larger than the overlap, skip overlap rather than
        starting the next chunk in the middle of a sentence.
        """
        if budget <= 0:
            return []
        kept: list[tuple[str, int]] = []
        for text, tokens in reversed(units):
            if self.total(kept) + tokens > budget:
                break
            kept.insert(0, (text, tokens))
        return kept

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
    def with_header(header: str, body: str) -> str:
        if not header:
            return body
        if body.startswith(header):
            return body
        return f"{header} {body}"

    @staticmethod
    def total(units: list[tuple[str, int]]) -> int:
        return sum(tokens for _, tokens in units)

    @staticmethod
    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
