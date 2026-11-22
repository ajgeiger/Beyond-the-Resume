"""Add the words the documents use for common question words, for keyword search only."""

from rag_chatbot.retrieval.Tokenizer import Tokenizer


class QueryExpander:
    """Visitors ask 'where did he study', the resume says 'Education' and 'University'.

    A short hand written list bridges those gaps for BM25. Dense search does
    not need it, but it misses some of these on short resume lines.
    """

    EDUCATION = ("education", "university", "degree", "master", "bachelor")
    RELATED = {
        "study": EDUCATION,
        "studied": EDUCATION,
        "school": EDUCATION,
        "college": EDUCATION,
        "undergraduate": ("education", "bachelor"),
        "undergrad": ("education", "bachelor"),
        "graduate": ("education", "master"),
        "degree": ("education", "university", "science"),
        "education": ("university", "degree"),
        "current": ("present",),
        "now": ("present",),
        "role": ("title", "present"),
        "title": ("role", "present"),
        "background": ("summary", "experience"),
        "tool": ("language", "framework", "cloud"),
        "stack": ("language", "framework", "cloud"),
        "certification": ("certified",),
        "award": ("achievement", "competition", "ranked"),
        "kaggle": ("competition", "ranked"),
        "job": ("experience",),
        "career": ("experience",),
        "built": ("build", "developed", "designed"),
        "build": ("built", "developed", "designed"),
        "pricing": ("price",),
        "champion": ("challenger", "compete", "mlops"),
        "challenger": ("champion", "compete", "mlops"),
    }

    @classmethod
    def expand(cls, query: str) -> list[str]:
        terms = Tokenizer.terms(query)
        extra = [related for term in terms for related in cls.RELATED.get(term, ())]
        return terms + [Tokenizer.stem(word) for word in extra]
