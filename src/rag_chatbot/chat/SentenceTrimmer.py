"""End an answer that hit the token limit at its last full sentence."""

import re


class SentenceTrimmer:
    """Cut a cut off answer back to the last sentence that finished.

    When no sentence finished, the partial text is kept and closed with an
    ellipsis so it does not stop mid word.
    """

    SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")

    def trim(self, text: str) -> str:
        text = text.strip()
        ends = list(self.SENTENCE_END.finditer(text))
        if ends:
            return text[: ends[-1].end()].strip()
        words = text.rsplit(" ", 1)[0] if " " in text else text
        return f"{words.rstrip(',;:')}..." if words else text
