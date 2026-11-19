"""Shared fixtures. Tests use the fake embedding model and the fake language model: no downloads, no network."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from rag_chatbot.api.RagApi import RagApi
from rag_chatbot.chat.RagRuntime import RagRuntime
from rag_chatbot.chunking.HeadingAwareChunker import HeadingAwareChunker
from rag_chatbot.config.ApiSettings import ApiSettings
from rag_chatbot.config.ChunkSettings import ChunkSettings
from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.config.GuardrailSettings import GuardrailSettings
from rag_chatbot.config.LlmSettings import LlmSettings
from rag_chatbot.config.RetrievalSettings import RetrievalSettings
from rag_chatbot.config.Settings import Settings
from rag_chatbot.config.SourceCatalog import SourceCatalog
from rag_chatbot.config.SourceSpec import SourceSpec
from rag_chatbot.embeddings.FakeEmbeddingModel import FakeEmbeddingModel
from rag_chatbot.guardrails.CostTracker import CostTracker
from rag_chatbot.index.IndexBuilder import IndexBuilder
from rag_chatbot.ingestion.IngestionPipeline import IngestionPipeline
from rag_chatbot.ingestion.SourceFactory import SourceFactory
from rag_chatbot.llm.FakeChatProvider import FakeChatProvider

SOURCE_TEXT = (
    "# Resume\n\n## Experience\n\n"
    "Lead Applied AI Scientist at Signature Aviation since Jul 2026. "
    "Built forecasting models and dynamic pricing systems with Python.\n\n"
    "## Education\n\nSyracuse University, Master of Science in Applied Data Science.\n\n"
    "## Projects\n\nHeadline Radar clusters news headlines with MiniLM, UMAP and HDBSCAN.\n"
)


@pytest.fixture
def model():
    return FakeEmbeddingModel()


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "source.md"
    path.write_text(SOURCE_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def catalog(source_file):
    return SourceCatalog((SourceSpec("Resume", "markdown", source_file),))


@pytest.fixture
def settings(tmp_path):
    return Settings(
        chunking=ChunkSettings(40, 8),
        embedding=EmbeddingSettings("fake", "fake-hashing-embedding"),
        retrieval=RetrievalSettings(min_dense_similarity=0.3, min_bm25_score=0.1),
        llm=LlmSettings(provider="fake"),
        guardrails=GuardrailSettings(usage_file=tmp_path / "usage.json"),
        api=ApiSettings(),
        index_dir=tmp_path / "index",
        root=tmp_path,
    )


@pytest.fixture
def pipeline(model, catalog, settings):
    chunker = HeadingAwareChunker(settings.chunking, model.count_tokens)
    return IngestionPipeline(SourceFactory.create(catalog), catalog, chunker)


@pytest.fixture
def builder(pipeline, model, catalog):
    return IndexBuilder(pipeline, model, catalog)


@pytest.fixture
def built_index(builder, settings):
    return builder.build(settings.index_dir)


@pytest.fixture
def runtime(settings, built_index):
    return RagRuntime.from_settings(settings)


@pytest.fixture
def make_runtime(settings, built_index):
    """A runtime with a chosen provider, settings changes and a fixed clock for the spend cap."""

    def build(provider=None, **changes) -> RagRuntime:
        loaded = RagRuntime.from_settings(replace(settings, **changes))
        if provider is not None:
            loaded.provider = provider
            loaded.chat_service.provider = provider
        loaded.tracker = CostTracker(
            settings.guardrails.usage_file,
            settings.guardrails.monthly_budget_usd,
            clock=lambda: datetime(2026, 11, 19, tzinfo=UTC),
        )
        loaded.chat_service.tracker = loaded.tracker
        return loaded

    return build


@pytest.fixture
def api_client(runtime):
    with TestClient(RagApi(runtime).build()) as client:
        yield client


@pytest.fixture
def protected_client(make_runtime, settings):
    protected = make_runtime(api=replace(settings.api, api_key="test-key"))
    with TestClient(RagApi(protected).build()) as client:
        yield client


@pytest.fixture
def fake_provider():
    return FakeChatProvider()
