"""Cross-encoder reranker wrapper for Stage 2 of the two-stage rerank
pipeline. See docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md.
"""

import threading
from functools import lru_cache

import torch

from app.config import settings

_DTYPE_CASTS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class Reranker:
    def __init__(self, model_name: str, dtype: str = "bf16"):
        if dtype not in _DTYPE_CASTS:
            raise ValueError(f"Unsupported reranker_dtype: {dtype!r} (expected one of {sorted(_DTYPE_CASTS)})")

        from FlagEmbedding import FlagReranker

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # use_fp16=False: FlagEmbedding's own fp16 switch just calls
        # model.half() unconditionally on every compute_score call, which
        # would stomp the bf16 cast below. We cast once here instead and
        # FlagEmbedding never touches dtype again after construction.
        self._model = FlagReranker(model_name, use_fp16=False, normalize=True, device=device)
        self._model.model.to(dtype=_DTYPE_CASTS[dtype])
        # A plain Lock (not a Semaphore admitting >1 caller), because the
        # HF fast tokenizer compute_score() calls into is not thread-safe:
        # concurrent calls hit "RuntimeError: Already borrowed" from the
        # Rust tokenizer's internal RefCell (see docs/concurrency-benchmark.md).
        # Serializing costs nothing in practice -- that same benchmark found
        # a single request already saturates GPU compute (88%+ utilization
        # at concurrency=1), so concurrent inference wasn't buying any real
        # throughput to begin with.
        self._lock = threading.Lock()

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        with self._lock:
            scores = self._model.compute_score(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model_name, settings.reranker_dtype)
