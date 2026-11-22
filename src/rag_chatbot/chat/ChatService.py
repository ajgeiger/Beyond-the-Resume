"""Answer one question from start to finish: guard, retrieve, generate and guard."""

import logging
import re

from rag_chatbot.chat.ChatAnswer import ChatAnswer
from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.chat.SentenceTrimmer import SentenceTrimmer
from rag_chatbot.guardrails.ClosingFilter import ClosingFilter
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.guardrails.OutputGuard import OutputGuard
from rag_chatbot.guardrails.QuestionGuard import QuestionGuard
from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.prompts.PromptBuilder import PromptBuilder
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever
from rag_chatbot.retrieval.RetrievedChunk import RetrievedChunk
from rag_chatbot.retrieval.Tokenizer import Tokenizer
from rag_chatbot.rewrite.AbstractQueryRewriter import AbstractQueryRewriter
from rag_chatbot.rewrite.HeuristicQueryRewriter import HeuristicQueryRewriter

LOGGER = logging.getLogger(__name__)


class ChatService:
    """The retrieval augmented generation loop.

    1. QuestionGuard answers off limits questions with fixed text.
    2. Follow up questions are rewritten into a standalone retrieval query.
    3. HybridRetriever finds passages, or none when nothing clears the relevance bar.
    4. CostTracker stops answer calls once the monthly cap is reached.
    5. The provider writes an answer from the retrieved passages, the conversation
       history and what is known about his experience.
    6. An answer cut off at the token limit ends at its last full sentence.
    7. OutputGuard blocks answers that break the content rules, and ClosingFilter
       drops a thank you or LinkedIn line the question did not invite.
    8. The citations field lists the retrieved passages (and any C ids still present in the answer).
    """

    CITATION = re.compile(r"\s*\[(C\d+(?:\s*,\s*C\d+)*)\]")
    GREETING = re.compile(
        r"^(hi|hello|hey|thanks|thank you|thanks for having me|good (morning|afternoon|evening))"
        r"([,!. ]+(hi|hello|hey|thanks|thank you|thanks for having me))*[,!. ]*$",
        re.IGNORECASE,
    )
    THANKS = re.compile(r"^\s*(thanks|thank you|thx|much appreciated|appreciate (it|that|the help))\b", re.IGNORECASE)
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
        top_k: int = 8,
        history_turns: int = 10,
        max_question_chars: int = 500,
        rewriter: AbstractQueryRewriter | None = None,
        max_history_chars: int = 2000,
    ):
        self.retriever = retriever
        self.provider = provider
        self.tracker = tracker
        self.prompts = PromptBuilder()
        self.question_guard = QuestionGuard(max_question_chars)
        self.output_guard = OutputGuard()
        self.closing_filter = ClosingFilter()
        self.trimmer = SentenceTrimmer()
        self.rewriter = rewriter or HeuristicQueryRewriter()
        self.top_k = top_k
        self.history_turns = history_turns
        self.max_history_chars = max_history_chars

    def ask(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> ChatAnswer:
        history = history or []
        checked = self.question_guard.check(question)
        if not checked.allowed:
            return ChatAnswer(checked.text, "declined")
        if self.is_greeting(checked.text):
            return ChatAnswer(ChatCopy.GREETING, "answered")

        rewritten = self.rewriter.rewrite(checked.text, history)
        hits = self.retrieve(checked.text, history, top_k or self.top_k, rewritten.query)
        if not hits and self.is_thanks(checked.text):
            self.charge(rewritten.cost_usd)
            return ChatAnswer(ChatCopy.GREETING, "answered", cost_usd=rewritten.cost_usd)
        if not hits and not self.may_answer_without_hits(checked.text):
            self.charge(rewritten.cost_usd)
            return ChatAnswer(ChatCopy.NOT_COVERED, "not_covered", cost_usd=rewritten.cost_usd)

        if self.tracker.is_exhausted():
            self.charge(rewritten.cost_usd)
            return ChatAnswer(ChatCopy.BUDGET_REACHED, "budget", cost_usd=rewritten.cost_usd)

        chunks = [hit.chunk for hit in hits]
        request = self.prompts.request(checked.text, self.recent(history), chunks)
        try:
            reply = self.provider.complete(request)
        except Exception:
            LOGGER.exception("The chat provider failed.")
            self.charge(rewritten.cost_usd)
            return ChatAnswer(ChatCopy.PROVIDER_ERROR, "error", cost_usd=rewritten.cost_usd)
        cost = reply.usage.cost(self.provider.pricing) + rewritten.cost_usd
        self.tracker.record(cost)

        text = reply.text
        if reply.truncated:
            LOGGER.warning("Answer reached the output token limit and was trimmed to its last sentence.")
            text = self.trimmer.trim(text)
        reviewed = self.output_guard.review(text)
        if not reviewed.allowed:
            LOGGER.warning("Answer blocked by the output guard: %s", reviewed.reason)
            return ChatAnswer(reviewed.text, "blocked", cost_usd=cost)
        text = self.closing_filter.apply(checked.text, reviewed.text)
        if self.is_not_covered(text):
            return ChatAnswer(self.strip_citations(text), "not_covered", cost_usd=cost)

        sources = self.cited(text, hits) or tuple(hits[:3])
        return ChatAnswer(self.strip_citations(text), "answered", sources, cost)

    def retrieve(
        self, question: str, history: list[dict], top_k: int, search_query: str | None = None
    ) -> list[RetrievedChunk]:
        """Search with the rewritten query when there is one, then fall back to the question and last user turn."""
        query = search_query or question
        hits = self.retriever.retrieve(query, top_k)
        if not hits and query != question:
            hits = self.retriever.retrieve(question, top_k)
        if not hits and history:
            last = self.last_user_question(history)
            combined = f"{last} {question}".strip()
            if last and combined not in {query, question}:
                hits = self.retriever.retrieve(combined, top_k)
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
        """The last ten turns, with each message capped, so follow up questions have context."""
        turns = []
        for item in history:
            if item.get("content") and item.get("role") in ("user", "assistant"):
                turns.append({"role": item["role"], "content": str(item["content"])[: self.max_history_chars]})
        return turns[-2 * self.history_turns :]

    def charge(self, cost_usd: float) -> None:
        if cost_usd:
            self.tracker.record(cost_usd)

    @staticmethod
    def is_greeting(question: str) -> bool:
        return bool(ChatService.GREETING.fullmatch(question.strip()))

    @staticmethod
    def is_thanks(question: str) -> bool:
        """A message that opens with thanks, such as 'Thanks, this was helpful!'."""
        return bool(ChatService.THANKS.match(question))

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
