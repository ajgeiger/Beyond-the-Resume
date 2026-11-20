"""The embedding factory and, when the MiniLM files are present, the ONNX model."""

from pathlib import Path

import numpy as np
import pytest

from rag_chatbot.config.EmbeddingSettings import EmbeddingSettings
from rag_chatbot.config.Settings import Settings
from rag_chatbot.embeddings.EmbeddingModelFactory import EmbeddingModelFactory
from rag_chatbot.embeddings.FakeEmbeddingModel import FakeEmbeddingModel
from rag_chatbot.embeddings.OnnxEmbeddingModel import OnnxEmbeddingModel
from rag_chatbot.embeddings.OnnxModelCache import OnnxModelCache
from rag_chatbot.index.VectorStore import VectorStore
from rag_chatbot.retrieval.HybridRetriever import HybridRetriever

try:
    from rag_chatbot.embeddings.SentenceTransformerEmbeddingModel import SentenceTransformerEmbeddingModel
except ImportError:
    SentenceTransformerEmbeddingModel = None

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_DIR = ROOT / "data" / "processed" / "index"
CONFIG = ROOT / "configs" / "settings.toml"
SAMPLE_QUERIES = [
    "What is his current role?",
    "Where did he get his master's degree?",
    "What did he work on at GHD?",
    "What is Golden Oak Research?",
    "How did he place in the NFL Big Data Bowl?",
    "Where does Headline Radar get its news from?",
    "What pages does the portfolio website have?",
    "What does he work on at Signature Aviation?",
    "What is his favorite pizza topping?",
    "How tall is Mount Everest?",
]


def test_the_factory_builds_the_fake_model():
    model = EmbeddingModelFactory.create(EmbeddingSettings(provider="fake"))
    assert isinstance(model, FakeEmbeddingModel)
    vector = model.encode_query("Signature Aviation")
    assert vector.shape == (model.dimension,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="banana"):
        EmbeddingModelFactory.create(EmbeddingSettings(provider="banana"))


def test_the_config_defaults_to_the_onnx_provider():
    settings = Settings.load(CONFIG, env={})
    assert settings.embedding.provider == "onnx"
    assert settings.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_environment_overrides_the_embedding_folder():
    settings = Settings.load(CONFIG, env={"EMBEDDING_PROVIDER": "fake", "EMBEDDING_MODEL_DIR": "/tmp/model"})
    assert settings.embedding.provider == "fake"
    assert settings.embedding.model_dir == "/tmp/model"


def onnx_settings() -> EmbeddingSettings:
    if not OnnxModelCache().is_complete(MODEL_DIR):
        pytest.skip("The MiniLM ONNX files are not in models/all-MiniLM-L6-v2.")
    return EmbeddingSettings(model_dir=str(MODEL_DIR))


def test_onnx_query_vectors_are_unit_length_and_384_wide():
    model = EmbeddingModelFactory.create(onnx_settings())
    assert isinstance(model, OnnxEmbeddingModel)
    assert model.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert model.dimension == 384
    vector = model.encode_query("What is his current role?")
    assert vector.shape == (384,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
    assert model.count_tokens("What is his current role?") == 6


def test_onnx_query_vectors_match_sentence_transformers():
    if SentenceTransformerEmbeddingModel is None:
        pytest.skip("sentence-transformers is not installed.")
    settings = onnx_settings()
    onnx = OnnxEmbeddingModel(settings)
    baseline = SentenceTransformerEmbeddingModel(
        EmbeddingSettings(provider="sentence-transformers", model_name=settings.model_name)
    )
    similarities = []
    for query in SAMPLE_QUERIES:
        similarity = float(np.dot(onnx.encode_query(query), baseline.encode_query(query)))
        similarities.append(similarity)
    assert min(similarities) >= 0.999, similarities


def test_onnx_and_sentence_transformers_return_the_same_top_chunks():
    if SentenceTransformerEmbeddingModel is None:
        pytest.skip("sentence-transformers is not installed.")
    if not (INDEX_DIR / "vectors.npy").is_file():
        pytest.skip("No committed index.")
    settings = onnx_settings()
    store = VectorStore.load(INDEX_DIR)
    retrieval = Settings.load(CONFIG, env={}).retrieval
    onnx = HybridRetriever(store, OnnxEmbeddingModel(settings), retrieval)
    baseline = HybridRetriever(
        store,
        SentenceTransformerEmbeddingModel(
            EmbeddingSettings(provider="sentence-transformers", model_name=settings.model_name)
        ),
        retrieval,
    )
    for query in SAMPLE_QUERIES:
        left = [item.chunk.chunk_id for item in onnx.retrieve(query, top_k=5)]
        right = [item.chunk.chunk_id for item in baseline.retrieve(query, top_k=5)]
        assert left == right, query
