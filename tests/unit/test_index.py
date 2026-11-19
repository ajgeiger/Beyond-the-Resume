"""Saving, loading and only rebuilding when something changed."""

import numpy as np
import pytest

from rag_chatbot.chunking.Chunk import Chunk
from rag_chatbot.index.IndexManifest import IndexManifest
from rag_chatbot.index.VectorStore import VectorStore


def test_build_writes_store_and_manifest(built_index, settings):
    manifest = built_index.manifest
    assert built_index.rebuilt
    assert VectorStore.exists(settings.index_dir)
    assert manifest.model_name == "fake-hashing-embedding"
    assert manifest.chunk_count == len(VectorStore.load(settings.index_dir)) > 0
    assert manifest.chunking["chunk_size_tokens"] == 40
    assert len(manifest.index_version) == 12
    assert manifest.documents[0]["sha256"]


def test_second_build_is_skipped_when_nothing_changed(builder, built_index, settings):
    again = builder.build(settings.index_dir)
    assert not again.rebuilt
    assert again.manifest.fingerprint == built_index.manifest.fingerprint


def test_changed_document_triggers_rebuild(builder, built_index, settings, source_file):
    source_file.write_text(source_file.read_text() + "\n## Awards\n\nRanked 24th in the NFL Big Data Bowl.\n")
    again = builder.build(settings.index_dir)
    assert again.rebuilt
    assert again.manifest.index_version != built_index.manifest.index_version


def test_force_rebuilds_with_the_same_version(builder, built_index, settings):
    again = builder.build(settings.index_dir, force=True)
    assert again.rebuilt
    assert again.manifest.index_version == built_index.manifest.index_version


def test_manifest_round_trip(built_index, settings):
    loaded = IndexManifest.load(settings.index_dir)
    assert loaded == built_index.manifest


def test_store_rejects_mismatched_rows():
    chunk = Chunk("a-001", "A", "S", "text", 1, 0)
    with pytest.raises(ValueError):
        VectorStore([chunk], np.zeros((2, 4)))
    with pytest.raises(ValueError):
        VectorStore([chunk, chunk], np.zeros((2, 4)))


def test_vectors_have_unit_length(settings, built_index):
    vectors = VectorStore.load(settings.index_dir).vectors
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
