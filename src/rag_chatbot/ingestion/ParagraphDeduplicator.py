"""Drop paragraphs that repeat an earlier one."""

import re


class ParagraphDeduplicator:
    """The resume text appears in three sources. The first copy in priority order wins.

    Two paragraphs count as the same when most of the later one's words
    already appear in an earlier one.
    """

    WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, overlap: float = 0.8):
        self.overlap = overlap
        self.seen: list[set[str]] = []

    def is_new(self, paragraph: str) -> bool:
        terms = set(self.WORD.findall(paragraph.lower()))
        if not terms:
            return False
        for other in self.seen:
            if len(terms & other) / len(terms) >= self.overlap:
                return False
        self.seen.append(terms)
        return True
