"""Clean endings: answers cut at the token limit, and closing lines the visitor did not invite."""

from rag_chatbot.chat.SentenceTrimmer import SentenceTrimmer
from rag_chatbot.guardrails.ClosingFilter import ClosingFilter
from rag_chatbot.llm.ChatReply import ChatReply
from rag_chatbot.llm.FakeChatProvider import FakeChatProvider

CLOSING = "Thanks for taking the time to visit. Alex would be glad to connect on LinkedIn."


def test_a_cut_off_answer_ends_at_its_last_sentence():
    text = "He used Databricks and MLflow. He also used Optuna for tuning, and"
    assert SentenceTrimmer().trim(text) == "He used Databricks and MLflow."


def test_a_cut_off_answer_with_no_full_sentence_ends_cleanly():
    assert SentenceTrimmer().trim("First, a correction: he used tools like") == "First, a correction: he used tools..."


def test_closing_lines_are_removed_from_ordinary_answers():
    text = f"He builds forecasting models at Signature Aviation.\n\n{CLOSING}"
    cleaned = ClosingFilter().apply("What does he work on?", text)
    assert cleaned == "He builds forecasting models at Signature Aviation."


def test_closing_lines_stay_for_greetings_and_thanks():
    for question in ("Hello there, who is Alex?", "Thanks, that was helpful", "How can I connect with him?"):
        assert ClosingFilter().apply(question, CLOSING) == CLOSING


def test_an_answer_that_is_only_a_closing_is_kept():
    assert ClosingFilter().apply("Tell me about him", CLOSING) == CLOSING


class TruncatingProvider(FakeChatProvider):
    def complete(self, request):
        self.calls.append(request)
        return ChatReply(
            "He built the forecasting platform at Signature Aviation. He used tools like", "fake", truncated=True
        )


class ClosingProvider(FakeChatProvider):
    def complete(self, request):
        self.calls.append(request)
        return ChatReply(f"He built forecasting models at Signature Aviation. {CLOSING}", "fake")


def test_the_chat_loop_trims_a_truncated_answer(make_runtime):
    payload = make_runtime(TruncatingProvider()).chat("What did he build at Signature Aviation?")
    assert payload["status"] == "answered"
    assert payload["answer"] == "He built the forecasting platform at Signature Aviation."


def test_the_chat_loop_drops_an_uninvited_linkedin_line(make_runtime):
    payload = make_runtime(ClosingProvider()).chat("What did he build at Signature Aviation?")
    assert payload["answer"] == "He built forecasting models at Signature Aviation."
    assert "LinkedIn" not in payload["answer"]


def test_the_chat_loop_keeps_the_linkedin_line_after_a_thanks(make_runtime):
    payload = make_runtime(ClosingProvider()).chat("Thanks! What did he build at Signature Aviation?")
    assert "LinkedIn" in payload["answer"]
