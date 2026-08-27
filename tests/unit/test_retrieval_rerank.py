from app.retrieval.rerank import apply_rerank
from app.store.qdrant_client import SearchHit


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, query, texts):
        return self._scores


def test_apply_rerank_reorders_by_score_and_truncates():
    hits = [
        SearchHit(score=0.1, payload={"doc_id": "a", "text": "a"}),
        SearchHit(score=0.9, payload={"doc_id": "b", "text": "b"}),
        SearchHit(score=0.5, payload={"doc_id": "c", "text": "c"}),
    ]
    reranker = FakeReranker([0.2, 0.8, 0.6])

    result = apply_rerank("query", hits, reranker, top_k=2)

    assert [h.payload["doc_id"] for h in result] == ["b", "c"]
    assert [h.score for h in result] == [0.8, 0.6]


def test_apply_rerank_empty_hits_returns_empty():
    reranker = FakeReranker([])
    assert apply_rerank("query", [], reranker, top_k=5) == []
