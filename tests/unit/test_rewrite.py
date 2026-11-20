"""Follow up questions are rewritten into standalone retrieval queries."""

from types import SimpleNamespace

from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.rewrite.ClaudeQueryRewriter import ClaudeQueryRewriter
from rag_chatbot.rewrite.GuardedQueryRewriter import GuardedQueryRewriter
from rag_chatbot.rewrite.HeuristicQueryRewriter import HeuristicQueryRewriter
from rag_chatbot.rewrite.QueryRewriterFactory import QueryRewriterFactory

HISTORY = [
    {"role": "user", "content": "What does he do?"},
    {"role": "assistant", "content": "He is Lead Applied AI Scientist at Signature Aviation."},
]


def test_heuristic_rewrites_a_follow_up_with_the_previous_question():
    result = HeuristicQueryRewriter().rewrite("How long has he done that?", HISTORY)
    assert result.query.startswith("What does he do?")
    assert "How long has he done that?" in result.query
    assert result.cost_usd == 0


def test_heuristic_leaves_a_standalone_question_alone():
    result = HeuristicQueryRewriter().rewrite("Where did he study?", HISTORY)
    assert result.query == "Where did he study?"


def test_heuristic_leaves_a_first_question_alone():
    result = HeuristicQueryRewriter().rewrite("How long has he done that?", [])
    assert result.query == "How long has he done that?"


def test_claude_payload_asks_for_a_standalone_query():
    rewriter = ClaudeQueryRewriter("key")
    payload = rewriter.payload("How long has he done that?", HISTORY)
    assert payload["model"] == "claude-haiku-5-5"
    assert payload["messages"][0]["role"] == "user"
    assert "What does he do?" in payload["messages"][0]["content"]
    assert "How long has he done that?" in payload["messages"][0]["content"]


def test_claude_rewrite_parses_the_query_and_cost():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=" How long has Alex been Lead Applied AI Scientist? ")],
        usage=SimpleNamespace(
            input_tokens=200, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0
        ),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_kwargs: response))
    result = ClaudeQueryRewriter("", client=client).rewrite("How long has he done that?", HISTORY)
    assert result.query == "How long has Alex been Lead Applied AI Scientist?"
    assert result.cost_usd > 0


def test_guarded_rewriter_falls_back_when_the_paid_call_fails():
    class BrokenRewriter:
        def rewrite(self, question, history):
            raise RuntimeError("down")

    result = GuardedQueryRewriter(BrokenRewriter()).rewrite("How long has he done that?", HISTORY)
    assert "What does he do?" in result.query
    assert result.cost_usd == 0


def test_guarded_rewriter_uses_the_heuristic_when_the_budget_is_spent(tmp_path):
    class PaidRewriter:
        def rewrite(self, question, history):
            raise AssertionError("should not be called")

    tracker = CostTracker(tmp_path / "usage.json", budget_usd=1.0)
    tracker.record(1.0)
    result = GuardedQueryRewriter(PaidRewriter(), tracker=tracker).rewrite("How long has he done that?", HISTORY)
    assert "What does he do?" in result.query


def test_factory_uses_the_heuristic_without_a_key():
    rewriter = QueryRewriterFactory.create(LlmSettings())
    assert isinstance(rewriter, HeuristicQueryRewriter)



def test_heuristic_expands_a_thin_first_question():
    result = HeuristicQueryRewriter().rewrite("What does he do?", [])
    assert "Alex Geiger" in result.query
    assert "What does he do?" in result.query
