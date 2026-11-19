"""Check every answer before a visitor sees it."""

import re

from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.guardrails.ContactScrubber import ContactScrubber
from rag_chatbot.guardrails.GuardResult import GuardResult


class OutputGuard:
    """Block answers that break Alex's public rules, and tidy the ones that pass.

    Blocked: anything that says or suggests he is open to work, available or
    for hire, any pay talk, and any phone number or email address.
    Tidied: long dashes become commas, emoji are removed, and any mention of
    the iSchool becomes Syracuse University.
    """

    OPEN_TO_WORK = re.compile(
        r"\b(open to (work|new|opportunit\w*|offers?|roles?|conversations?)|"
        r"(actively |currently )?(looking|searching|seeking) for (a |his |new |the )*"
        r"(job|role|position|opportunit\w*|challenge|chapter)|"
        r"seeking (a |new )*(job|role|position|opportunit\w*|employment)|"
        r"job (search|hunt\w*)|available (for|to) (hire|work|start|interview|join|new)|for hire|"
        r"(hire|hiring|recruit) (him|alex)|(he|alex) (is|would be) (a )?(great|strong|perfect|ideal) "
        r"(hire|fit|candidate|addition)|next (role|opportunit\w*|career move|position)|"
        r"(reach out|contact him|get in touch) (about|for|regarding) (a |an )?(role|opportunit\w*|position|job)|"
        r"(his|alex's) availability|notice period|ready to (join|start|move)|"
        r"salary|compensation|pay range|expected pay)\b",
        re.IGNORECASE,
    )
    ISCHOOL = re.compile(r"\b(the )?(Syracuse (University('s)? )?)?iSchool\b", re.IGNORECASE)
    EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff\ufe0f]")
    DASHES = {" \u2014 ": ", ", "\u2014": ", ", " \u2013 ": " to ", "\u2013": " to "}

    def review(self, text: str) -> GuardResult:
        if ContactScrubber.contains_contact(text):
            return GuardResult(False, ChatCopy.BLOCKED, "contact_details")
        if self.OPEN_TO_WORK.search(text):
            return GuardResult(False, ChatCopy.BLOCKED, "open_to_work")
        return GuardResult(True, self.tidy(text))

    def tidy(self, text: str) -> str:
        for dash, replacement in self.DASHES.items():
            text = text.replace(dash, replacement)
        text = self.ISCHOOL.sub("Syracuse University", text)
        text = self.EMOJI.sub("", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()
