"""A free rewrite that expands thin questions and prepends the previous user question to follow ups."""

import re

from rag_chatbot.retrieval.Tokenizer import Tokenizer
from rag_chatbot.rewrite.AbstractQueryRewriter import AbstractQueryRewriter
from rag_chatbot.rewrite.RewriteResult import RewriteResult


class HeuristicQueryRewriter(AbstractQueryRewriter):
    """Used on its own when no language model is configured, and as a fallback when a paid rewrite fails."""

    FOLLOW_UP = re.compile(
        r"\b(that|this|those|these|it|they|them|then|also|too|the same|there|"
        r"earlier|previous|above|the latter)\b|"
        r"^(tell me more|why\??$|how long|what about|and (what|how|why)|what else)\b",
        re.IGNORECASE,
    )
    THIN = re.compile(
        r"\b(what does (he|alex) do|what is his (job|work)|who is (he|alex)|tell me about him)\b",
        re.IGNORECASE,
    )
    EXPANDED = "Alex Geiger current role work experience"

    def rewrite(self, question: str, history: list[dict]) -> RewriteResult:
        last = self.last_user_question(history)
        if last and self.is_follow_up(question):
            return RewriteResult(f"{last} {question}")
        if self.is_thin(question):
            return RewriteResult(f"{self.EXPANDED} {question}")
        return RewriteResult(question)

    def is_follow_up(self, question: str) -> bool:
        return bool(self.FOLLOW_UP.search(question.strip()))

    def is_thin(self, question: str) -> bool:
        return not Tokenizer.terms(question) or bool(self.THIN.search(question))

    @staticmethod
    def last_user_question(history: list[dict]) -> str:
        for item in reversed(history):
            if item.get("role") == "user" and item.get("content"):
                return str(item["content"]).strip()
        return ""
