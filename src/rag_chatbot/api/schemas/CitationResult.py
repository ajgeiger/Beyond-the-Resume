"""One passage an answer cites."""

from rag_chatbot.api.schemas.ChunkResult import ChunkResult


class CitationResult(ChunkResult):
    """number is the citation's position in the answer, starting at 1."""

    number: int
