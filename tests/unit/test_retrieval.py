"""BM25, fusion and the hybrid retriever."""

import pytest

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.retrieval.AbstractReranker import AbstractReranker
from rag_chatbot.retrieval.Bm25Index import Bm25Index
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever
from rag_chatbot.retrieval.QueryExpander import QueryExpander
from rag_chatbot.retrieval.ReciprocalRankFusion import ReciprocalRankFusion


class ReverseReranker(AbstractReranker):
    """Prefers the passage the fused order put last."""

    def __init__(self):
        super().__init__("reverse")

    def scores(self, query, passages):
        return [float(position) for position in range(len(passages))]


def test_fusion_rewards_agreement():
    fused = ReciprocalRankFusion(k=60).fuse([[1, 2, 3], [2, 1, 4]])
    assert {position for position, _ in fused[:2]} == {1, 2}
    assert fused[-1][0] in (3, 4)
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_bm25_prefers_matching_terms_and_section_titles():
    chunks = [
        Chunk("a-001", "A", "Education", "Syracuse University master's degree.", 4, 0),
        Chunk("a-002", "A", "Experience", "Forecasting and pricing models.", 4, 1),
    ]
    scores = Bm25Index(chunks).scores("Where did he study?")
    assert scores[0] > 0 and scores[1] == 0


def test_query_expander_bridges_question_words():
    assert "educ" not in QueryExpander.expand("study")
    assert "education" in QueryExpander.expand("Where did he study?")


def test_hybrid_finds_the_right_section(runtime):
    top = runtime.retriever.retrieve("Where did he study at university?", top_k=3)[0]
    assert top.chunk.section == "Education"
    assert top.rank == 1 and top.score > 0


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_every_mode_returns_ranked_results(runtime, mode):
    results = runtime.retriever.retrieve("Headline Radar news headlines HDBSCAN", top_k=2, mode=mode)
    assert results and results[0].chunk.section == "Projects"
    assert [item.rank for item in results] == list(range(1, len(results) + 1))


def test_off_topic_question_returns_nothing(runtime):
    assert runtime.retriever.retrieve("lasagna recipe tomorrow weather", top_k=5) == []


def test_unknown_mode_is_rejected(runtime):
    with pytest.raises(ValueError):
        runtime.retriever.retrieve("anything", mode="magic")


def test_reranker_reorders_and_reports_its_scores(runtime):
    retriever = HybridRetriever(runtime.store, runtime.model, runtime.settings.retrieval, ReverseReranker())
    plain = retriever.retrieve("Signature Aviation forecasting Python Syracuse", top_k=5, rerank=False)
    reranked = retriever.retrieve("Signature Aviation forecasting Python Syracuse", top_k=5)
    assert len(plain) > 1
    assert reranked[0].chunk.chunk_id == plain[-1].chunk.chunk_id
    assert reranked[0].rerank_score is not None and plain[0].rerank_score is None


def test_top_k_is_capped_by_settings(runtime):
    response = runtime.retrieve("Signature Aviation", top_k=999)
    assert len(response["results"]) <= runtime.settings.retrieval.max_top_k


def test_tokenizer_keeps_slash_initials():
    from rag_chatbot.retrieval.Tokenizer import Tokenizer

    terms = Tokenizer.terms("What A/B testing has he done?")
    assert "ab" in terms
    assert "testing" in terms
    assert "a" not in terms and "b" not in terms
