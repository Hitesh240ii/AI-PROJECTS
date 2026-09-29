import math
import os

import pytest

from app.embeddings.embedding_service import EmbeddingService


def test_dimension_and_shapes(embedder, fake_model):
    docs = embedder.embed_documents(["annual leave policy", "cafeteria lunch menu"])
    query = embedder.embed_query("annual leave")
    assert embedder.dimension == fake_model.DIM
    assert len(docs) == 2 and all(len(v) == embedder.dimension for v in docs)
    assert len(query) == embedder.dimension


def test_vectors_are_normalised(embedder):
    vec = embedder.embed_query("annual leave days")
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-5)


def test_empty_input_returns_empty_list(embedder):
    assert embedder.embed_documents([]) == []


def test_model_is_injected_not_reloaded(embedder, fake_model):
    embedder.embed_query("a b")
    embedder.embed_query("c d")
    assert embedder.model is fake_model  # same object every time


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="downloads the real model; set RUN_MODEL_TESTS=1")
def test_real_model_dimension():
    assert EmbeddingService("all-MiniLM-L6-v2").dimension == 384
