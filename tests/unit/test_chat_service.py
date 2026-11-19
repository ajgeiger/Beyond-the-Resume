"""The chat loop end to end on the small test index, with the fake model."""

from rag_chatbot.chat.ChatCopy import ChatCopy
from rag_chatbot.llm.FakeChatProvider import FakeChatProvider


def test_answers_with_citations_from_the_index(make_runtime):
    payload = make_runtime().chat("Where did he study for his master's degree?")
    assert payload["status"] == "answered"
    assert "Syracuse University" in payload["answer"]
    assert "[C" not in payload["answer"]
    assert payload["citations"][0]["number"] == 1
    assert payload["sources"] == ["Resume"]


def test_off_topic_questions_never_reach_the_model(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("lasagna recipe tomorrow weather")
    assert payload["status"] == "not_covered" and payload["answer"] == ChatCopy.NOT_COVERED
    assert provider.calls == []


def test_job_status_questions_are_declined_without_a_call(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("Is he open to work?")
    assert payload["status"] == "declined"
    assert provider.calls == []


def test_answers_with_contact_details_are_blocked(make_runtime):
    provider = FakeChatProvider("Call him at 555 123 4567 about Signature Aviation. [C1]")
    payload = make_runtime(provider).chat("What did he build at Signature Aviation?")
    assert payload["status"] == "blocked" and payload["answer"] == ChatCopy.BLOCKED


def test_the_spend_cap_stops_model_calls(make_runtime):
    provider = FakeChatProvider()
    runtime = make_runtime(provider)
    runtime.tracker.record(runtime.settings.guardrails.monthly_budget_usd)
    payload = runtime.chat("What did he build at Signature Aviation?")
    assert payload["status"] == "budget"
    assert provider.calls == []


def test_a_provider_failure_returns_the_fixed_error_text(make_runtime):
    class BrokenProvider(FakeChatProvider):
        def complete(self, request):
            raise RuntimeError("down")

    payload = make_runtime(BrokenProvider()).chat("What did he build at Signature Aviation?")
    assert payload["status"] == "error" and payload["answer"] == ChatCopy.PROVIDER_ERROR


def test_a_bare_follow_up_reuses_the_previous_question(make_runtime):
    history = [
        {"role": "user", "content": "What is Headline Radar?"},
        {"role": "assistant", "content": "It clusters news headlines."},
    ]
    payload = make_runtime().chat("Tell me more", history)
    assert payload["status"] == "answered"
    assert any(item["section"] == "Projects" for item in payload["citations"])
