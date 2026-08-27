"""Stage 2 of the two-stage rerank pipeline: reorders a Stage 1 candidate
pool by cross-encoder relevance and truncates to top_k. See
docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md.
"""

from app.store.qdrant_client import SearchHit


def apply_rerank(query: str, hits: list[SearchHit], reranker, top_k: int) -> list[SearchHit]:
    if not hits:
        return hits
    texts = [hit.payload.get("text", "") for hit in hits]
    scores = reranker.score(query, texts)
    reranked = sorted(
        (SearchHit(score=score, payload=hit.payload) for hit, score in zip(hits, scores)),
        key=lambda hit: hit.score,
        reverse=True,
    )
    return reranked[:top_k]
