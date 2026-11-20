"""Answer one question from start to finish: guard, retrieve, generate and guard."""

import logging
import re

from rag_chatbot.chat.ChatAnswer import ChatAnswer
from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.guardrails.OutputGuard import OutputGuard
from rag_chatbot.guardrails.QuestionGuard import QuestionGuard
from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.prompts.PromptBuilder import PromptBuilder
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever
from rag_chatbot.retrieval.RetrievedChunk import RetrievedChunk
from rag_chatbot.retrieval.Tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)


class ChatService:
    """The retrieval augmented generation loop.

    1. QuestionGuard answers off limits questions with fixed text.
    2. HybridRetriever finds passages, or none when nothing clears the relevance bar.
    3. CostTracker stops calls once the monthly cap is reached.
    4. The provider writes an answer from the retrieved passages and what is known about his experience.
    5. OutputGuard blocks answers that break the content rules.
    6. The citations field lists the retrieved passages (and any C ids still present in the answer).

    Steps 1 to 3 never call the model, so they never cost anything.
    """

    CITATION = re.compile(r"\s*\[(C\d+(?:\s*,\s*C\d+)*)\]")
    GREETING = re.compile(
        r"^(hi|hello|hey|thanks|thank you|thanks for having me|good (morning|afternoon|evening))"
        r"([,!. ]+(hi|hello|hey|thanks|thank you|thanks for having me))*[,!. ]*$",
        re.IGNORECASE,
    )
    WORK_STYLE = re.compile(
        r"\b(like to work with|to work with him|what (is he|he's) like|what (does he|he) care about|"
        r"his values|his (working )?style|as a teammate|as a colleague|soft skills|personality)\b",
        re.IGNORECASE,
    )
    HIRING_FIT = re.compile(
        r"\b(why (should|would) (i|we|you|someone|a (company|team|manager)) "
        r"(hire|want|choose|consider) (him|alex)|"
        r"why (hire|choose|consider) (him|alex)|"
        r"(hire|hiring) (him|alex)|"
        r"what would he bring|bring to (my|our|a|the) (team|company|org\w*|group)|"
        r"on (my|our|a) team|(his )?(strengths|leadership|technical depth|culture fit)|"
        r"good fit|great fit|why (do i|would i) want (alex|him)|want alex to work|"
        r"as a (hire|candidate)|"
        r"open to work|open to (new )?(opportunit\w*|roles?|offers?)|"
        r"(is|are) (he|alex) available)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever: HybridRetriever,
        provider: AbstractChatProvider,
        tracker: CostTracker,
        top_k: int = 5,
        history_turns: int = 3,
        max_question_chars: int = 500,
    ):
        self.retriever = retriever
        self.provider = provider
        self.tracker = tracker
        self.prompts = PromptBuilder()
        self.question_guard = QuestionGuard(max_question_chars)
        self.output_guard = OutputGuard()
        self.top_k = top_k
        self.history_turns = history_turns

    def ask(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> ChatAnswer:
        history = history or []
        checked = self.question_guard.check(question)
        if not checked.allowed:
            return ChatAnswer(checked.text, "declined")
        if self.is_greeting(checked.text):
            return ChatAnswer(ChatCopy.GREETING, "answered")

        hits = self.retrieve(checked.text, history, top_k or self.top_k)
        if not hits and not self.may_answer_without_hits(checked.text):
            return ChatAnswer(ChatCopy.NOT_COVERED, "not_covered")

        if self.tracker.is_exhausted():
            return ChatAnswer(ChatCopy.BUDGET_REACHED, "budget")

        chunks = [hit.chunk for hit in hits]
        request = self.prompts.request(checked.text, self.recent(history), chunks)
        try:
            reply = self.provider.complete(request)
        except Exception:
            LOGGER.exception("The chat provider failed.")
            return ChatAnswer(ChatCopy.PROVIDER_ERROR, "error")
        cost = reply.usage.cost(self.provider.pricing)
        self.tracker.record(cost)

        reviewed = self.output_guard.review(reply.text)
        if not reviewed.allowed:
            LOGGER.warning("Answer blocked by the output guard: %s", reviewed.reason)
            return ChatAnswer(reviewed.text, "blocked", cost_usd=cost)
        if self.is_not_covered(reviewed.text):
            return ChatAnswer(self.strip_citations(reviewed.text), "not_covered", cost_usd=cost)

        sources = self.cited(reviewed.text, hits) or tuple(hits[:3])
        return ChatAnswer(self.strip_citations(reviewed.text), "answered", sources, cost)

    def retrieve(self, question: str, history: list[dict], top_k: int) -> list[RetrievedChunk]:
        """Retrieve for the question, and for a bare follow up retry with the previous question."""
        hits = self.retriever.retrieve(question, top_k)
        if not hits and history and self.is_follow_up(question):
            hits = self.retriever.retrieve(f"{self.last_user_question(history)} {question}", top_k)
        return hits

    def cited(self, text: str, hits: list[RetrievedChunk]) -> tuple[RetrievedChunk, ...]:
        """Map the C ids in the answer back to the chunks they label, in citation order."""
        labels = {self.prompts.label(position): hit for position, hit in enumerate(hits)}
        ids = []
        for group in self.CITATION.findall(text):
            ids += [label.strip() for label in group.split(",")]
        return tuple(labels[label] for label in dict.fromkeys(ids) if label in labels)

    def strip_citations(self, text: str) -> str:
        return self.CITATION.sub("", text).strip()

    def recent(self, history: list[dict]) -> list[dict]:
        """The last few turns, so follow up questions have context."""
        turns = [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("content") and item.get("role") in ("user", "assistant")
        ]
        return turns[-2 * self.history_turns :]

    @staticmethod
    def is_greeting(question: str) -> bool:
        return bool(ChatService.GREETING.fullmatch(question.strip()))

    @staticmethod
    def is_work_style(question: str) -> bool:
        return bool(ChatService.WORK_STYLE.search(question))

    @staticmethod
    def is_hiring_fit(question: str) -> bool:
        return bool(ChatService.HIRING_FIT.search(question))

    @staticmethod
    def may_answer_without_hits(question: str) -> bool:
        return ChatService.is_work_style(question) or ChatService.is_hiring_fit(question)

    @staticmethod
    def is_not_covered(text: str) -> bool:
        start = text.strip().lower()
        return start.startswith("that is not covered") or start.startswith("i don't have details")

    @staticmethod
    def is_follow_up(question: str) -> bool:
        """A question with no search terms of its own, such as 'Tell me more' or 'Why?'."""
        return not Tokenizer.terms(question)

    @staticmethod
    def last_user_question(history: list[dict]) -> str:
        for item in reversed(history):
            if item.get("role") == "user":
                return item.get("content", "")
        return ""
