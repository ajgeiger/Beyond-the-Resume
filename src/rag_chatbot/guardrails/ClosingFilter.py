"""Keep the thank you and LinkedIn closing for greetings and thanks only."""

import re


class ClosingFilter:
    """Remove closing lines, such as thanking the visitor or pointing to LinkedIn, from ordinary answers.

    The model is told to add the closing only when the visitor greets or
    thanks it, but it sometimes adds one anyway. This check removes those
    sentences unless the question invites them.
    """

    INVITES = re.compile(
        r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b|\b(thanks|thank you|thx|appreciate it)\b|"
        r"\b(linkedin|connect|get in touch|reach out)\b",
        re.IGNORECASE,
    )
    CLOSING = re.compile(
        r"(thanks?( you)? (so much )?for (taking the time to )?(visit\w*|stopping by|your interest|reaching out)|"
        r"(glad|happy|love|welcome) to connect|connect (with him )?(on|through|via) linkedin)",
        re.IGNORECASE,
    )
    SENTENCES = re.compile(r"[^.!?]*[.!?]+[\"')\]]*\s*|[^.!?]+$")

    def invites_closing(self, question: str) -> bool:
        return bool(self.INVITES.search(question))

    def apply(self, question: str, text: str) -> str:
        """The answer without closing sentences, unless the question is a greeting or a thanks."""
        if self.invites_closing(question):
            return text
        paragraphs = []
        for paragraph in text.split("\n"):
            sentences = [part for part in self.SENTENCES.findall(paragraph) if part.strip()]
            kept = "".join(part for part in sentences if not self.CLOSING.search(part)).strip()
            if kept or not paragraph.strip():
                paragraphs.append(kept)
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(paragraphs)).strip()
        return cleaned or text
