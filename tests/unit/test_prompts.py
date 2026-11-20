"""The prompt holds only the retrieved passages, labeled in rank order."""

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.prompts.PromptBuilder import PromptBuilder


def chunk(number: int) -> Chunk:
    return Chunk(f"doc-{number:03d}", "Resume", f"Section {number}", f"Passage number {number}.", number, 4)


def test_passages_are_labeled_in_rank_order():
    request = PromptBuilder().request("question", [], [chunk(1), chunk(2)])
    assert request.knowledge_context.index("[C1]") < request.knowledge_context.index("[C2]")
    assert "Section 2" in request.knowledge_context
    assert len(request.hits) == 2


def test_history_comes_before_the_question():
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}]
    request = PromptBuilder().request("second", history, [chunk(1)])
    assert [message["role"] for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages[-1]["content"].startswith("Question: second")


def test_rules_carry_the_content_rules():
    rules = PromptBuilder.SYSTEM_RULES
    assert "Syracuse University" in rules
    assert "Lead Applied AI Scientist" in rules
    assert "LinkedIn" in rules and "GitHub" in rules
    assert "Do not show sources or citation tags" in rules
    assert "open to work" in rules
    assert "Question: " not in rules
