"""Read a Markdown README into sections that follow its headings."""

import re

from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.Section import Section


class MarkdownSource(AbstractSource):
    """Split a README at its headings and keep only readable prose.

    Each section is titled with its heading path below the top level title.
    A skipped heading skips everything nested under it. Code blocks, images
    and table separators are dropped, links keep their text, and table rows
    become plain sentences.
    """

    HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(self, path, name: str, skipped_sections: tuple[str, ...] = ()):
        super().__init__(path, name)
        self.skipped_sections = {title.lower() for title in skipped_sections}

    def sections(self) -> list[Section]:
        sections = []
        headings: list[tuple[int, str]] = []
        paragraphs: list[str] = []
        buffer: list[str] = []
        table_header: list[str] = []
        in_code = False

        def flush_paragraph() -> None:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer.clear()

        def close() -> None:
            flush_paragraph()
            if paragraphs and not self.is_skipped(headings):
                sections.append(Section(self.name, self.title_for(headings), tuple(paragraphs)))
            paragraphs.clear()

        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            heading = self.HEADING.match(raw_line)
            if heading:
                close()
                level = len(heading.group(1))
                headings = [item for item in headings if item[0] < level] + [(level, self.clean(heading.group(2)))]
                table_header = []
                continue
            if self.is_link_bar(raw_line):
                continue
            if raw_line.strip().startswith("|"):
                cells = self.table_cells(raw_line)
                if not cells:
                    continue
                if not table_header:
                    table_header = cells
                    continue
                flush_paragraph()
                paragraphs.append(self.table_row(table_header, cells))
                continue
            table_header = []
            line = self.clean(raw_line)
            if not line:
                flush_paragraph()
            elif raw_line.lstrip().startswith(("* ", "- ")) or re.match(r"^\s*\d+\.\s", raw_line):
                flush_paragraph()
                buffer.append(line)
                flush_paragraph()
            else:
                buffer.append(line)
        close()
        return sections

    def is_skipped(self, headings: list[tuple[int, str]]) -> bool:
        return any(title.lower() in self.skipped_sections for _, title in headings)

    def title_for(self, headings: list[tuple[int, str]]) -> str:
        """The heading path without the document title, or the document title for its intro."""
        nested = [title for level, title in headings if level > 1]
        if nested:
            return " > ".join(nested)
        return headings[0][1] if headings else self.name

    @staticmethod
    def is_link_bar(line: str) -> bool:
        """A line of in page links such as 'Run it | The website | Sources'."""
        return len(re.findall(r"\]\(#", line)) >= 2

    def table_cells(self, line: str) -> list[str]:
        """Cells of a table row, or an empty list for the separator row."""
        cells = [self.clean(cell) for cell in line.strip().strip("|").replace("\\|", "/").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells):
            return []
        return cells

    @staticmethod
    def table_row(header: list[str], cells: list[str]) -> str:
        """Turn a row into a sentence such as 'Home. What it does: ...'."""
        parts = [cells[0]] if cells else []
        for label, value in zip(header[1:], cells[1:], strict=False):
            if value:
                parts.append(f"{label}: {value}")
        return ". ".join(part.rstrip(".") for part in parts if part) + "."

    @staticmethod
    def clean(line: str) -> str:
        """Turn one Markdown line into plain text."""
        text = line.strip()
        if text.startswith("!["):
            return ""
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&nbsp;", " ").replace("\\|", "")
        text = re.sub(r"^\s*([*\-]|\d+\.)\s+", "", text)
        text = text.replace("**", "").replace("`", "")
        text = re.sub(r"(^|\s)\*([^*]+)\*", r"\1\2", text)
        return re.sub(r"\s+", " ", text).strip()
