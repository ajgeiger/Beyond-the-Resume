"""Report recall at k and MRR on the evaluation set."""

import json
from dataclasses import replace
from pathlib import Path

from rag_chatbot.chat.RagRuntime import RagRuntime
from rag_chatbot.config.Settings import Settings
from rag_chatbot.evaluation.EvalCase import EvalCase
from rag_chatbot.evaluation.RetrievalEvaluator import RetrievalEvaluator
from rag_chatbot.retrieval.CrossEncoderReranker import CrossEncoderReranker


class EvaluateCommand:
    """Compare BM25 alone, dense alone, hybrid and, when asked, hybrid with the reranker.

    Retrieval only, so it runs with the fake language model and costs nothing.
    """

    def __init__(self, settings: Settings):
        self.settings = replace(settings, llm=replace(settings.llm, provider="fake"))

    def run(self, cases_file: Path, output: Path | None, with_rerank: bool) -> int:
        runtime = RagRuntime.from_settings(self.settings)
        retriever = runtime.retriever
        if with_rerank and retriever.reranker is None:
            retriever.reranker = CrossEncoderReranker(self.settings.reranker.model_name)
        cases = EvalCase.load_all(cases_file)
        evaluator = RetrievalEvaluator(retriever)
        setups = [("bm25", "bm25", False), ("dense", "dense", False), ("hybrid", "hybrid", False)]
        if with_rerank:
            setups.append(("hybrid + rerank", "hybrid", True))
        reports = [evaluator.evaluate(cases, label, mode, rerank) for label, mode, rerank in setups]

        manifest = runtime.manifest
        print(f"Index {manifest.index_version}, {len(runtime.store)} chunks, {manifest.model_name}")
        print(f"{len(cases)} questions, {reports[0].off_topic_questions} of them off topic")
        for report in reports:
            print(report.line())
        for report in reports:
            for question in report.misses:
                print(f"  {report.label} missed: {question}")
        if output:
            payload = {
                "index_version": manifest.index_version,
                "model": manifest.model_name,
                "chunk_count": len(runtime.store),
                "reports": [report.to_dict() for report in reports],
            }
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {output}")
        return 0
