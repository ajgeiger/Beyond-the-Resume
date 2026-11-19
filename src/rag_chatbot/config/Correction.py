"""A text fix applied to every source before chunking."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    find: str
    replace: str

    def apply(self, text: str) -> str:
        return text.replace(self.find, self.replace)
