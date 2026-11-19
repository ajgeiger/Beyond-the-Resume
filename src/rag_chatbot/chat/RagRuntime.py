"""Load the index and models once, and serve retrieval and chat. The API and the CLI both use it."""

import time
from pathlib import Path

from rag_chatbot.chat.ChatAnswer import ChatAnswer
from rag_chatbot.chat.ChatService import ChatService
from rag_chatbot.config.Settings import Settings
from rag_chatbot.embeddings.AbstractEmbeddingModel import AbstractEmbeddingModel
from rag_chatbot.embeddings.EmbeddingModelFactory import EmbeddingModelFactory
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.index.IndexManifest import IndexManifest
from rag_chatbot.index.VectorStore import VectorStore
from rag_chatbot.llm.AbstractChatProvider import AbstractChatProvider
from rag_chatbot.llm.ProviderFactory import ProviderFactory
from rag_chatbot.retrieval.AbstractReranker import AbstractReranker
from rag_chatbot.retrieval.CrossEncoderReranker import CrossEncoderReranker
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever


class RagRuntime:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        manifest: IndexManifest,
        model: AbstractEmbeddingModel,
        provider: AbstractChatProvider,
        reranker: AbstractReranker | None = None,
    ):
        if manifest.model_name != model.name:
            raise ValueError(f"The index was built with {manifest.model_name}, but {model.name} is loaded.")
        self.settings = settings
        self.store = store
        self.manifest = manifest
        self.model = model
        self.provider = provider
        self.retriever = HybridRetriever(store, model, settings.retrieval, reranker)
        guardrails = settings.guardrails
        self.tracker = CostTracker(guardrails.usage_file, guardrails.monthly_budget_usd)
        self.chat_service = ChatService(
            self.retriever,
            provider,
            self.tracker,
            settings.retrieval.default_top_k,
            guardrails.history_turns,
            guardrails.max_question_chars,
        )

    @classmethod
    def from_settings(cls, settings: Settings, index_dir: Path | None = None) -> "RagRuntime":
        directory = Path(index_dir or settings.index_dir)
        manifest = IndexManifest.load(directory)
        if manifest is None or not VectorStore.exists(directory):
            raise FileNotFoundError(f"No index in {directory}. Run: make index")
        model = EmbeddingModelFactory.create(settings.embedding)
        reranker = CrossEncoderReranker(settings.reranker.model_name) if settings.reranker.enabled else None
        provider = ProviderFactory.create(settings.llm)
        return cls(settings, VectorStore.load(directory), manifest, model, provider, reranker)

    def top_k(self, requested: int | None) -> int:
        limits = self.settings.retrieval
        return min(max(1, requested or limits.default_top_k), limits.max_top_k)

    def retrieve(self, query: str, top_k: int | None = None, mode: str = "hybrid", rerank: bool | None = None) -> dict:
        started = time.perf_counter()
        results = self.retriever.retrieve(query, self.top_k(top_k), mode, rerank)
        return {
            "query": query,
            "mode": mode,
            "reranked": any(item.rerank_score is not None for item in results),
            "index_version": self.manifest.index_version,
            "took_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": [item.to_dict() for item in results],
        }

    def chat(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> dict:
        started = time.perf_counter()
        answer = self.chat_service.ask(question, history, self.top_k(top_k))
        return self.chat_payload(answer, started)

    def chat_payload(self, answer: ChatAnswer, started: float) -> dict:
        citations = [{"number": number, **item.to_dict()} for number, item in enumerate(answer.sources, start=1)]
        return {
            "answer": answer.text,
            "status": answer.status,
            "citations": citations,
            "sources": list(dict.fromkeys(item.chunk.source for item in answer.sources)),
            "provider": self.provider.name,
            "model": self.provider.model,
            "index_version": self.manifest.index_version,
            "took_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def info(self) -> dict:
        return {
            "embedding_model": self.manifest.model_name,
            "dimension": self.manifest.dimension,
            "chunk_count": len(self.store),
            "index_version": self.manifest.index_version,
            "built_at": self.manifest.built_at,
            "chunking": self.manifest.chunking,
            "sources": self.manifest.source_counts,
            "reranker": self.retriever.reranker.name if self.retriever.reranker else None,
            "retrieval": self.settings.retrieval.to_dict(),
            "llm_provider": self.provider.name,
            "llm_model": self.provider.model,
            "demo_mode": self.provider.is_demo,
            "spend_cap_reached": self.tracker.is_exhausted(),
        }
