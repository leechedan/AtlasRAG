import threading
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings


@dataclass
class DenseSparseVector:
    dense: list[float]
    sparse: dict[int, float]


class TextEmbedder:
    def __init__(self, model_name: str):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        # EXPERIMENT: same fix as app/models/reranker.py's _lock -- FlagEmbedding's
        # encode() is not thread-safe under concurrent calls from FastAPI's
        # threadpool, see 2026-08-27 debugging session.
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[DenseSparseVector]:
        with self._lock:
            output = self._model.encode(
                texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
        results = []
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
            sparse_int_keys = {int(token_id): float(weight) for token_id, weight in sparse.items()}
            results.append(DenseSparseVector(dense=dense.tolist(), sparse=sparse_int_keys))
        return results


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder(settings.text_embed_model)
