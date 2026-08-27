"""Regression test for the tokenizer race documented in
docs/concurrency-benchmark.md: HF's fast tokenizer is not thread-safe, so
concurrent compute_score() calls could raise "RuntimeError: Already
borrowed". Reranker.score() must fully serialize access to the model
instead of allowing >1 concurrent caller in.
"""

import threading
import time

from app.models.reranker import Reranker


class _FakeModel:
    """Stands in for FlagReranker: records whether it was ever entered by
    more than one thread at a time."""

    def __init__(self):
        self._active = 0
        self._lock = threading.Lock()
        self.max_concurrent_seen = 0

    def compute_score(self, pairs):
        with self._lock:
            self._active += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._active)
        time.sleep(0.02)
        with self._lock:
            self._active -= 1
        return [0.5 for _ in pairs]


def _make_reranker_with_fake_model() -> tuple[Reranker, _FakeModel]:
    reranker = object.__new__(Reranker)
    fake_model = _FakeModel()
    reranker._model = fake_model
    reranker._lock = threading.Lock()
    return reranker, fake_model


def test_score_serializes_concurrent_calls():
    reranker, fake_model = _make_reranker_with_fake_model()

    threads = [
        threading.Thread(target=reranker.score, args=("query", ["text"]))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake_model.max_concurrent_seen == 1
