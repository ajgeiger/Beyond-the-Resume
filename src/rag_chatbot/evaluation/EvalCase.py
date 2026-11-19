"""One evaluation question and the passages that should answer it."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    """A result counts as relevant when its source matches and its section contains one of expected_sections.

    A case with no expected_source is off topic: the right outcome is no results.
    """

    question: str
    expected_source: str | None = None
    expected_sections: tuple[str, ...] = ()

    @property
    def is_off_topic(self) -> bool:
        return self.expected_source is None

    def matches(self, source: str, section: str) -> bool:
        if self.is_off_topic or source != self.expected_source:
            return False
        if not self.expected_sections:
            return True
        return any(expected.lower() in section.lower() for expected in self.expected_sections)

    @classmethod
    def load_all(cls, path: Path) -> list["EvalCase"]:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return [
            cls(row["question"], row.get("expected_source"), tuple(row.get("expected_sections", ()))) for row in rows
        ]
