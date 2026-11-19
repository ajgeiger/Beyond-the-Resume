"""Check a question before any tokens are spent on it."""

import re

from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.guardrails.GuardResult import GuardResult


class QuestionGuard:
    """Answer off limits questions with a fixed reply, without calling the model.

    Job status, availability and pay are declined. Requests for contact details
    get a pointer to the public links on the site.
    """

    JOB_STATUS = re.compile(
        r"\b(open to (work|new|opportunit\w*|offers?|roles?)|looking for (a )?(new )?(job|role|position|work)|"
        r"job (search|hunt\w*)|(is|are) (he|alex) available|(his|alex's) availability|"
        r"available (for|to) (hire|work|start|interview|join)|for hire|hire (him|alex)|hiring him|recruit (him|alex)|"
        r"(leave|leaving|quit\w*) (his|signature)|notice period|start date|willing to (relocate|move)|"
        r"interested in (a )?(new )?(job|role|position|opportunit\w*)|salary|compensation|pay(ing)? (him|range)|"
        r"how much (does|did|would) he (make|earn)|rate per hour|hourly rate)\b",
        re.IGNORECASE,
    )
    CONTACT = re.compile(
        r"\b(phone|cell|mobile|email|e-mail|contact (him|alex|details|info\w*)|reach (him|alex)|home address)\b",
        re.IGNORECASE,
    )

    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def check(self, question: str) -> GuardResult:
        text = " ".join(question.split())
        if not text:
            return GuardResult(False, ChatCopy.EMPTY, "empty")
        if len(text) > self.max_chars:
            return GuardResult(False, ChatCopy.TOO_LONG.format(limit=self.max_chars), "too_long")
        if self.JOB_STATUS.search(text):
            return GuardResult(False, ChatCopy.DECLINED, "job_status")
        if self.CONTACT.search(text):
            return GuardResult(False, ChatCopy.CONTACT, "contact")
        return GuardResult(True, text)
