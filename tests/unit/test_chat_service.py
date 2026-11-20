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


def test_pay_questions_are_declined_without_a_call(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("What salary does he expect?")
    assert payload["status"] == "declined"
    assert provider.calls == []


def test_hiring_fit_questions_reach_the_model(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("Why should I hire Alex?")
    assert provider.calls
    assert payload["status"] in {"answered", "not_covered"}


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


def test_a_greeting_uses_the_fixed_thanks(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("Hi, thanks for having me")
    assert payload["status"] == "answered"
    assert payload["answer"] == ChatCopy.GREETING
    assert provider.calls == []


def test_a_work_style_question_still_reaches_the_model(make_runtime):
    provider = FakeChatProvider()
    payload = make_runtime(provider).chat("What is he like to work with?")
    assert provider.calls
    assert payload["status"] in {"answered", "not_covered"}


def test_follow_up_questions_send_history_to_the_model(make_runtime):
    provider = FakeChatProvider()
    runtime = make_runtime(provider)
    first = runtime.chat("What did he build at Signature Aviation?")
    history = [
        {"role": "user", "content": "What did he build at Signature Aviation?"},
        {"role": "assistant", "content": first["answer"]},
    ]
    second = runtime.chat("How long has he done that?", history)
    request = provider.calls[-1]
    assert [message["role"] for message in request.messages] == ["user", "assistant", "user"]
    assert "Signature Aviation" in request.messages[0]["content"]
    assert "How long has he done that?" in request.messages[-1]["content"]
    assert second["status"] in {"answered", "not_covered"}


def test_recent_history_keeps_the_last_ten_turns_and_caps_length(make_runtime):
    service = make_runtime().chat_service
    service.history_turns = 10
    service.max_history_chars = 20
    history = []
    for number in range(12):
        history.append({"role": "user", "content": f"question {number} " + "x" * 40})
        history.append({"role": "assistant", "content": f"answer {number} " + "y" * 40})
    recent = service.recent(history)
    assert len(recent) == 20
    assert recent[0]["content"].startswith("question 2")
    assert all(len(item["content"]) <= 20 for item in recent)
