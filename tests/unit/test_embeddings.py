import pytest

from app.models.embeddings import get_text_embedder

pytestmark = pytest.mark.slow


def test_embed_returns_dense_and_sparse():
    embedder = get_text_embedder()
    results = embedder.embed(["检索服务", "retrieval service"])
    assert len(results) == 2
    for r in results:
        assert len(r.dense) == 1024
        assert isinstance(r.sparse, dict)
        assert len(r.sparse) > 0


def test_get_text_embedder_is_singleton():
    assert get_text_embedder() is get_text_embedder()
