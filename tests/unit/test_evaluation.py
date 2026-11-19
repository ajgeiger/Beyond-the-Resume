"""Recall at k and MRR, on a tiny index where the answers are known."""

import json

from rag_chatbot.evaluation.EvalCase import EvalCase
from rag_chatbot.evaluation.RetrievalEvaluator import RetrievalEvaluator


def test_case_matching_rules():
    case = EvalCase("q", "Resume", ("Education",))
    assert case.matches("Resume", "Education > Syracuse University")
    assert not case.matches("Resume", "Experience")
    assert not case.matches("README", "Education")
    assert EvalCase("q", "Resume").matches("Resume", "Anything")
    assert EvalCase("q").is_off_topic


def test_evaluator_reports_recall_mrr_and_off_topic(runtime):
    cases = [
        EvalCase("Where did he study at university?", "Resume", ("Education",)),
        EvalCase("Headline Radar HDBSCAN headlines", "Resume", ("Projects",)),
        EvalCase("lasagna recipe tomorrow weather"),
    ]
    report = RetrievalEvaluator(runtime.retriever, ks=(1, 3)).evaluate(cases, "hybrid")
    assert report.questions == 2 and report.off_topic_questions == 1
    assert report.recall_at_k == {1: 1.0, 3: 1.0}
    assert report.mrr == 1.0
    assert report.off_topic_empty == 1.0
    assert "R@1 1.000" in report.line()


def test_load_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"question": "q", "expected_source": "Resume", "expected_sections": ["Education"]}]))
    assert EvalCase.load_all(path) == [EvalCase("q", "Resume", ("Education",))]
