"""A titled block of text read from one source document."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """One section of a source, before it is cut into chunks.

    title is the heading path, for example "News Topic Explorer > Why it matters".
    paragraphs holds the text split at natural breaks (bullets, blank lines),
    so the chunker prefers to cut between thoughts rather than inside one.
    header is optional context, such as a job's company, title and dates, that
    the chunker prefixes onto every chunk of this section.
    """

    source: str
    title: str
    paragraphs: tuple[str, ...]
    header: str = ""
