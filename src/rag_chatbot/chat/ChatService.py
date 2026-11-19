"""Answer one question from start to finish: guard, retrieve, generate, guard, cite."""

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
    4. The provider writes an answer from the retrieved passages only, citing C1, C2 and so on.
    5. OutputGuard blocks answers that break the content rules.
    6. Citations are mapped back to the chunks they label.

    Steps 1 to 3 never call the model, so they never cost anything.
    """

    CITATION = re.compile(r"\s*\[(C\d+(?:\s*,\s*C\d+)*)\]")
    NOT_COVERED_START = "That is not covered"

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

        hits = self.retrieve(checked.text, history, top_k or self.top_k)
        if not hits:
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
        if reviewed.text.startswith(self.NOT_COVERED_START):
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
    def is_follow_up(question: str) -> bool:
        """A question with no search terms of its own, such as 'Tell me more' or 'Why?'."""
        return not Tokenizer.terms(question)

    @staticmethod
    def last_user_question(history: list[dict]) -> str:
        for item in reversed(history):
            if item.get("role") == "user":
                return item.get("content", "")
        return ""
