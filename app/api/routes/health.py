from fastapi import APIRouter, Response

from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder

router = APIRouter()


@router.get("/health")
def health(response: Response):
    # get_text_embedder/get_clip_embedder are @lru_cache(maxsize=1)
    # singletons (Important Finding I1). The lifespan hook in
    # app/api/main.py loads both before the app finishes starting, so in
    # normal operation this endpoint is only ever reached once both are
    # resident. cache_info().currsize == 1 tells us whether a given
    # singleton has actually been constructed, without needing any new
    # state. If /health is somehow reached before that (e.g. a future
    # deployment mode where lifespan blocking doesn't apply), report
    # not-ready rather than falsely claiming "ok".
    models_ready = (
        get_text_embedder.cache_info().currsize == 1
        and get_clip_embedder.cache_info().currsize == 1
    )
    if not models_ready:
        response.status_code = 503
        return {"status": "loading"}
    return {"status": "ok"}
