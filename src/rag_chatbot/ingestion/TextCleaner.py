"""Normalize text before it is chunked."""

from rag_chatbot.config.Correction import Correction
from rag_chatbot.guardrails.ContactScrubber import ContactScrubber


class TextCleaner:
    """Apply the configured corrections, replace long dashes and strip contact details."""

    DASHES = {"\u2014": ", ", "\u2013": " to "}

    def __init__(self, corrections: tuple[Correction, ...] = ()):
        self.corrections = corrections

    def clean(self, text: str) -> str:
        for correction in self.corrections:
            text = correction.apply(text)
        for dash, replacement in self.DASHES.items():
            text = text.replace(dash, replacement)
        return ContactScrubber.scrub(text)
