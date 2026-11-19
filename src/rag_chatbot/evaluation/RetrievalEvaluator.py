"""Measure recall at k and MRR for a retriever on the evaluation set."""

from rag_chatbot.evaluation.EvalCase import EvalCase
from rag_chatbot.evaluation.EvalReport import EvalReport
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever


class RetrievalEvaluator:
    def __init__(self, retriever: HybridRetriever, ks: tuple[int, ...] = (1, 3, 5)):
        self.retriever = retriever
        self.ks = tuple(sorted(ks))

    def evaluate(self, cases: list[EvalCase], label: str, mode: str = "hybrid", rerank: bool = False) -> EvalReport:
        depth = max(self.ks)
        on_topic = [case for case in cases if not case.is_off_topic]
        off_topic = [case for case in cases if case.is_off_topic]
        first_hits: list[int | None] = []
        misses = []
        for case in on_topic:
            results = self.retriever.retrieve(case.question, depth, mode, rerank)
            rank = next((item.rank for item in results if case.matches(item.chunk.source, item.chunk.section)), None)
            first_hits.append(rank)
            if rank is None:
                misses.append(case.question)
        empty = sum(1 for case in off_topic if not self.retriever.retrieve(case.question, depth, mode, rerank))
        count = len(on_topic) or 1
        return EvalReport(
            label=label,
            questions=len(on_topic),
            recall_at_k={k: round(sum(1 for r in first_hits if r and r <= k) / count, 4) for k in self.ks},
            mrr=round(sum(1.0 / r for r in first_hits if r) / count, 4),
            off_topic_questions=len(off_topic),
            off_topic_empty=round(empty / len(off_topic), 4) if off_topic else 1.0,
            misses=misses,
        )
