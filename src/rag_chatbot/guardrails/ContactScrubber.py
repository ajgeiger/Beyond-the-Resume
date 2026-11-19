"""Remove phone numbers and email addresses from text."""

import re


class ContactScrubber:
    """The resume carries a phone number and a personal email. Neither may reach the index."""

    EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
    PHONE = re.compile(r"(\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b")
    LEFTOVER_SEPARATORS = re.compile(r"(\s*\u2022\s*){2,}")

    @classmethod
    def scrub(cls, text: str) -> str:
        text = cls.EMAIL.sub("", text)
        text = cls.PHONE.sub("", text)
        text = cls.LEFTOVER_SEPARATORS.sub(" \u2022 ", text)
        return re.sub(r"\s+", " ", text).strip(" \u2022,")

    @classmethod
    def contains_contact(cls, text: str) -> bool:
        return bool(cls.EMAIL.search(text) or cls.PHONE.search(text))
