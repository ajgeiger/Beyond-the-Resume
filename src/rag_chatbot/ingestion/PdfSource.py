"""Read a resume style PDF into sections."""

import re

from pypdf import PdfReader

from rag_chatbot.ingestion.AbstractSource import AbstractSource
from rag_chatbot.ingestion.Section import Section


class PdfSource(AbstractSource):
    """Split a PDF at its upper case headings and at employer and role lines.

    A line in capitals (PROFESSIONAL EXPERIENCE) starts a section. Under
    experience, a company line with one pipe starts a role block, and the
    title line (with or without a date) stays with that company. Earlier
    Experience rows that already carry the title in the company line are a
    role on their own. A parent bullet that ends in a colon keeps the
    following sub-bullets in the same paragraph. Other wrapped lines
    continue the current paragraph.

    Each experience role stores a short header of company, title and dates so
    the chunker can put that context on every chunk of the role.
    """

    BULLET = "\u2022"
    HEADING_PATTERN = re.compile(r"^[A-Z][A-Z &/]{3,}$")
    DATE_SPAN = re.compile(
        r"(?:"
        r"(?:Issued\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
        r"(?:\s+to\s+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+)?(?:\d{4}|Present))?"
        r"(?:,\s*expires\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})?"
        r"|Spring\s+\d{4}"
        r"|Completed\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
        r"|Current"
        r"|\d{4}(?:\s+to\s+\d{4})?"
        r")$"
    )
    NEW_BULLET_START = re.compile(
        r"^(Built|Build|Designed|Led|Lead|Made|Implemented|Developed|Ran|Compared|Owned|"
        r"Reduced|Architected|Established|Put|Engineered|Delivered|Founded|Analyzed|"
        r"Forecasted|Created|Wrote|Managed|Deployed|Promoted|Reports)\b",
        re.IGNORECASE,
    )
    HEADER_TITLE = "Name and current title"
    EARLIER_EXPERIENCE = "Earlier Experience"

    def text(self) -> str:
        reader = PdfReader(str(self.path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def sections(self) -> list[Section]:
        sections: list[Section] = []
        heading, group, company, role, dates = self.HEADER_TITLE, "", "", "", ""
        paragraphs: list[str] = []
        collecting_subs = False

        def in_experience() -> bool:
            return heading.lower().startswith("professional experience")

        def role_header() -> str:
            return ". ".join(part for part in (company, role, dates) if part)

        def role_title() -> str:
            return " > ".join(part for part in (heading, group, company, role) if part)

        def close(*, keep_company: bool = False) -> None:
            nonlocal paragraphs, collecting_subs, company, role, dates
            header = role_header() if in_experience() else ""
            body = tuple(paragraphs)
            # Education, certs, competitions, and roles with no bullets still need a paragraph.
            if not body and role_header():
                item = role_header()
                if heading and heading not in item:
                    item = f"{heading}. {item}"
                body, header = (item,), ""
            elif not header and heading != self.HEADER_TITLE:
                header = heading
            if body:
                title = role_title() if (company or role or group) else heading
                sections.append(Section(self.name, title, body, header=header))
            paragraphs = []
            collecting_subs = False
            if keep_company:
                role = ""
                dates = ""
            else:
                company, role, dates = "", "", ""

        for raw_line in self.text().splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if self.HEADING_PATTERN.match(line):
                close()
                heading, group, company, role, dates = line.capitalize(), "", "", "", ""
                continue
            if line == self.EARLIER_EXPERIENCE and in_experience():
                close()
                group, company, role, dates = line, "", "", ""
                continue
            if "|" in line and not line.startswith(self.BULLET):
                close()
                company, role, dates, group = self.pipe_fields(line, in_experience(), group)
                continue
            if self.is_title_line(line, heading, company, role):
                if role or (paragraphs and not in_experience()):
                    close(keep_company=in_experience() and bool(company))
                title, title_dates = self.split_dates(line)
                role = title
                if title_dates:
                    dates = title_dates
                continue
            if line.startswith(self.BULLET):
                text = line.lstrip(self.BULLET).strip()
                if (
                    paragraphs
                    and (collecting_subs or paragraphs[-1].endswith(":"))
                    and not self.NEW_BULLET_START.match(text)
                ):
                    paragraphs[-1] = f"{paragraphs[-1]} {text}"
                    collecting_subs = True
                else:
                    collecting_subs = False
                    paragraphs.append(text)
                continue
            if not paragraphs:
                paragraphs.append(line)
            elif heading == self.HEADER_TITLE:
                paragraphs.append(line)
            elif paragraphs[-1].endswith("-"):
                paragraphs[-1] = f"{paragraphs[-1]}{line}"
            else:
                paragraphs[-1] = f"{paragraphs[-1]} {line}"
        close()
        return sections

    def pipe_fields(self, line: str, experience: bool, group: str) -> tuple[str, str, str, str]:
        """Turn a pipe line into company, role, dates and the current group name."""
        parts = [part.strip() for part in re.split(r"\s*\|\s*", line)]
        if experience and len(parts) >= 3:
            location, dates = self.split_dates(parts[-1])
            company = f"{parts[0]}, {location}" if location else parts[0]
            role = parts[1] if len(parts) == 3 else " | ".join(parts[1:-1])
            return company, role, dates, group
        if experience:
            joined = ", ".join(parts)
            company, dates = self.split_dates(joined)
            return company, "", dates, group
        last, dates = self.split_dates(parts[-1])
        company = parts[0]
        role = last if len(parts) > 1 else ""
        return company, role, dates, group

    @classmethod
    def split_dates(cls, text: str) -> tuple[str, str]:
        """Peel a trailing date range off a company, title or award line."""
        text = text.strip()
        match = cls.DATE_SPAN.search(text)
        if not match or match.start() == 0:
            return text, ""
        return text[: match.start()].strip(), match.group(0)

    def is_title_line(self, line: str, heading: str, company: str, role: str) -> bool:
        """A short role or award line, with or without a date at the end."""
        if line.startswith(self.BULLET) or heading == self.HEADER_TITLE:
            return False
        rest, dates = self.split_dates(line)
        if dates and len(line.split()) <= 14:
            return True
        if not company or role or not heading.lower().startswith("professional experience"):
            return False
        return len(rest.split()) <= 8 and "." not in rest
