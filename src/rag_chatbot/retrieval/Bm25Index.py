"""Okapi BM25 keyword scoring over the chunks."""

import math
from collections import Counter

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.retrieval.QueryExpander import QueryExpander
from rag_chatbot.retrieval.Tokenizer import Tokenizer


class Bm25Index:
    """Scores every chunk for a query. Words in the section title count double, and the query is expanded."""

    K1 = 1.5
    B = 0.75
    TITLE_WEIGHT = 2

    def __init__(self, chunks: list[Chunk]):
        self.documents = [self.terms_for(chunk) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        document_frequency = Counter(term for counts in self.documents for term in counts)
        total = len(self.documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def terms_for(self, chunk: Chunk) -> Counter:
        counts = Counter(Tokenizer.terms(chunk.text))
        for term in Tokenizer.terms(chunk.section):
            counts[term] += self.TITLE_WEIGHT
        return counts

    def scores(self, query: str) -> list[float]:
        """One score per chunk, in chunk order. Zero means no shared terms."""
        query_terms = set(QueryExpander.expand(query))
        return [self.score(query_terms, position) for position in range(len(self.documents))]

    def score(self, query_terms: set[str], position: int) -> float:
        counts = self.documents[position]
        length_ratio = self.lengths[position] / self.average_length if self.average_length else 1.0
        total = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency:
                weight = frequency * (self.K1 + 1) / (frequency + self.K1 * (1 - self.B + self.B * length_ratio))
                total += self.idf[term] * weight
        return total
