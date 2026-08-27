import logging

from fastapi import APIRouter, Response

from app.config import settings
from app.jobs.queue import get_redis_connection
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.models.reranker import get_reranker
from app.store.qdrant_client import get_default_client

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health(response: Response):
    # get_text_embedder/get_clip_embedder/get_reranker are @lru_cache(maxsize=1)
    # singletons (Important Finding I1). The lifespan hook in
    # app/api/main.py loads all of them (reranker only if enabled) before
    # the app finishes starting, so in normal operation this endpoint is
    # only ever reached once they're resident. cache_info().currsize == 1
    # tells us whether a given singleton has actually been constructed,
    # without needing any new state. If /health is somehow reached before
    # that (e.g. a future deployment mode where lifespan blocking doesn't
    # apply), report not-ready rather than falsely claiming "ok".
    models_ready = (
        get_text_embedder.cache_info().currsize == 1
        and get_clip_embedder.cache_info().currsize == 1
        and (not settings.reranker_enabled or get_reranker.cache_info().currsize == 1)
    )
    if not models_ready:
        response.status_code = 503
        return {"status": "loading"}

    # Models loaded is necessary but not sufficient for "ready to serve
    # traffic" -- Qdrant and Redis are external processes (see
    # docker-compose.yml) that can be down or unreachable independently of
    # this process's own state. Probe both with a cheap call so /health
    # actually reflects whether search/ingest requests can succeed, not
    # just whether model loading finished.
    checks: dict[str, str] = {}

    try:
        get_default_client().get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        logger.exception("Health check: Qdrant unreachable")
        checks["qdrant"] = "unreachable"

    try:
        get_redis_connection().ping()
        checks["redis"] = "ok"
    except Exception:
        logger.exception("Health check: Redis unreachable")
        checks["redis"] = "unreachable"

    healthy = all(status == "ok" for status in checks.values())
    if not healthy:
        response.status_code = 503
    return {"status": "ok" if healthy else "degraded", "checks": checks}
