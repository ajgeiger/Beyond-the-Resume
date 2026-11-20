"""An offline provider for tests and for running the app without an API key."""

import re

from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.llm.ChatReply import ChatReply
from rag_chatbot.llm.ChatRequest import ChatRequest
from rag_chatbot.llm.TokenUsage import TokenUsage
from rag_chatbot.retrieval.Tokenizer import Tokenizer


class FakeChatProvider(AbstractChatProvider):
    """Answer by quoting the sentences that best match the question.

    Pass scripted_reply to return a fixed text instead, which is how the tests
    check that the guardrails catch a bad answer.
    """

    MAX_SENTENCES = 3
    MIN_SENTENCE_WORDS = 6

    name = "fake"

    def __init__(self, scripted_reply: str | None = None):
        super().__init__("fake")
        self.scripted_reply = scripted_reply
        self.calls: list[ChatRequest] = []

    @property
    def is_demo(self) -> bool:
        return True

    def complete(self, request: ChatRequest) -> ChatReply:
        self.calls.append(request)
        text = self.scripted_reply if self.scripted_reply is not None else self.quote(request)
        rules_words = len(request.system_rules.split())
        input_words = len(request.knowledge_context.split()) + sum(
            len(message["content"].split()) for message in request.messages
        )
        usage = TokenUsage(input_tokens=input_words, output_tokens=len(text.split()), cache_read_tokens=rules_words)
        return ChatReply(text=text, model=self.model, usage=usage)

    def quote(self, request: ChatRequest) -> str:
        if not request.hits:
            return "That is not covered in the material I have."
        question = request.messages[-1]["content"].split("\n")[0].removeprefix("Question:").strip()
        terms = set(Tokenizer.terms(question))
        candidates = []
        for rank, passage in enumerate(request.hits):
            label = f"C{rank + 1}"
            for sentence in re.split(r"(?<=[.!?;])\s+", passage.text):
                words = sentence.split()
                if len(words) < self.MIN_SENTENCE_WORDS:
                    continue
                overlap = len(terms & set(Tokenizer.terms(sentence)))
                candidates.append((-overlap, rank, sentence, label))
        top_overlap = -min(candidates)[0] if candidates else 0
        strong = [item for item in candidates if -item[0] >= max(1, top_overlap / 2)]
        best = sorted(strong)[: self.MAX_SENTENCES]
        if not best:
            best = [(0, 0, request.hits[0].text, "C1")]
        best.sort(key=lambda item: item[1])
        return " ".join(sentence.rstrip(";") for _, _, sentence, _label in best)
