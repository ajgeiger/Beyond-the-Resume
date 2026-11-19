"""Read a resume style PDF into sections."""

import re

from pypdf import PdfReader

from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.Section import Section


class PdfSource(AbstractSource):
    """Split a PDF at its upper case headings and at employer lines.

    A line in capitals (PROFESSIONAL EXPERIENCE) starts a section. A line with a
    pipe (Signature Aviation | Orlando, FL) starts a subsection inside it, and a
    role line that ends in a date range (Lead Applied AI Scientist Jul 2026 to
    Present) starts a subsection inside that, so each role is titled on its own.
    A bullet starts a new paragraph; other lines continue the current one.
    """

    BULLET = "\u2022"
    HEADING_PATTERN = re.compile(r"^[A-Z][A-Z &/]{3,}$")
    # The block above the first heading holds the name, current title and links.
    HEADER_TITLE = "Name and current title"

    def text(self) -> str:
        reader = PdfReader(str(self.path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def sections(self) -> list[Section]:
        sections = []
        heading, subheading, role, paragraphs = self.HEADER_TITLE, "", "", []

        def close() -> None:
            if paragraphs:
                title = " > ".join(part for part in (heading, subheading, role) if part)
                sections.append(Section(self.name, title, tuple(paragraphs)))

        for raw_line in self.text().splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if self.HEADING_PATTERN.match(line):
                close()
                heading, subheading, role, paragraphs = line.capitalize(), "", "", []
            elif "|" in line and not line.startswith(self.BULLET):
                close()
                subheading, role = re.sub(r"\s*\|\s*", ", ", line), ""
                paragraphs = [subheading]
            elif subheading and heading != self.HEADER_TITLE and self.looks_like_title_line(line):
                if paragraphs != [subheading]:
                    close()
                    paragraphs = []
                role = line
                paragraphs.append(line)
            elif line.startswith(self.BULLET) or not paragraphs:
                paragraphs.append(line.lstrip(self.BULLET).strip())
            elif heading == self.HEADER_TITLE or self.looks_like_title_line(line):
                paragraphs.append(line)
            elif paragraphs[-1].endswith("-"):
                paragraphs[-1] = f"{paragraphs[-1]}{line}"
            else:
                paragraphs[-1] = f"{paragraphs[-1]} {line}"
        close()
        return sections

    @staticmethod
    def looks_like_title_line(line: str) -> bool:
        """Role lines end with a date range, for example 'Senior Data Scientist May 2022 to Jul 2026'."""
        return bool(re.search(r"(\d{4}|Present)$", line)) and len(line.split()) <= 12
