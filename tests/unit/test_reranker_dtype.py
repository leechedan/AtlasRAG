import pytest

from app.models.reranker import Reranker


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="reranker_dtype"):
        Reranker("BAAI/bge-reranker-v2-m3", dtype="fp8")
