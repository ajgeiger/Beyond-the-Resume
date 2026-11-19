"""Everything a provider needs to answer one question."""

from dataclasses import dataclass

from rag_chatbot.chunking.Chunk import Chunk


@dataclass(frozen=True)
class ChatRequest:
    """system_rules never change between visitors, so providers cache them.

    knowledge_context holds only the passages retrieved for this question,
    each labeled with a citation id such as C1. hits lists the same chunks
    in label order, so C1 is hits[0].

    messages is the short conversation in provider neutral form:
    a list of {"role": "user" or "assistant", "content": text}.
    """

    system_rules: str
    knowledge_context: str
    messages: tuple[dict, ...]
    hits: tuple[Chunk, ...] = ()
