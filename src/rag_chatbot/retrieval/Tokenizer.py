"""Split text into keyword search terms."""

import re


class Tokenizer:
    """Lower case words with common filler words removed and simple plural endings trimmed."""

    STOP_WORDS = frozenset(
        """a about an and are as at be been but by can could did do does for from had has have he her his how i
        if in into is it its me my of on or our she so than that the their them then there these they this to
        us was we were what when where which who whom why will with would you your alex geiger alexander
        tell more much many any some""".split()
    )
    WORD = re.compile(r"[a-z0-9]+")
    # "A/B testing" would otherwise become the stop words "a" and "b".
    SLASHED_INITIALS = re.compile(r"\b([a-z])/([a-z])\b")

    @classmethod
    def terms(cls, text: str) -> list[str]:
        normalized = cls.SLASHED_INITIALS.sub(r"\1\2", text.lower())
        words = cls.WORD.findall(normalized)
        return [cls.stem(word) for word in words if word not in cls.STOP_WORDS and len(word) > 1]

    @staticmethod
    def stem(word: str) -> str:
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
