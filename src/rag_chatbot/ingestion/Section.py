"""A titled block of text read from one source document."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """One section of a source, before it is cut into chunks.

    title is the heading path, for example "News Topic Explorer > Why it matters".
    paragraphs holds the text split at natural breaks (bullets, blank lines),
    so the chunker prefers to cut between thoughts rather than inside one.
    """

    source: str
    title: str
    paragraphs: tuple[str, ...]
