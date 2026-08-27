import pytest

from app.models.reranker import Reranker


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="reranker_dtype"):
        Reranker("BAAI/bge-reranker-v2-m3", max_concurrency=2, dtype="fp8")
