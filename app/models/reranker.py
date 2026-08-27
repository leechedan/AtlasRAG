"""Cross-encoder reranker wrapper for Stage 2 of the two-stage rerank
pipeline. See docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md.
"""

import threading
from functools import lru_cache

from app.config import settings


class Reranker:
    def __init__(self, model_name: str, max_concurrency: int):
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=True, normalize=True)
        self._semaphore = threading.Semaphore(max_concurrency)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        with self._semaphore:
            scores = self._model.compute_score(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model_name, settings.reranker_max_concurrency)
