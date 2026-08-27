import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import chunks, collections, directories, documents, endpoints, health, jobs, search
from app.config import settings
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.models.reranker import get_reranker

# Without this, `logger.info(...)` calls below (and everywhere else in this
# process) are silently dropped: Python's logging module only emits INFO
# through a "last resort" handler for WARNING+ when no handler is
# configured, so nothing below INFO ever reaches `docker logs` unless a
# handler is attached somewhere. app/worker.py already does this for the
# worker process; the api process never did, so the model-loading log
# lines below existed in code but were invisible in practice -- which is
# exactly why "docker logs" showed nothing but uvicorn's own startup lines
# during a several-minute-long cold model load, with no way to tell
# "still loading" from "stuck".
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both @lru_cache(maxsize=1) model singletons before yielding, so
    # the app is only considered "started" (and uvicorn only starts
    # accepting requests) once models are actually resident (Important
    # Finding I1). Verified against the installed starlette==0.41.3
    # (routing.py Lifespan.__call__: `lifespan.startup.complete` is sent
    # only after the code before `yield` in this context manager returns)
    # and uvicorn==0.32.1 (lifespan/on.py: a `lifespan.startup.failed`
    # message -- sent when this block raises before yielding -- sets
    # `should_exit = True` and logs "Application startup failed. Exiting.",
    # so an exception here is not swallowed and the process exits non-zero
    # instead of silently serving requests with no models loaded.
    #
    # Logged as separate per-model stages (not one "loading models..."
    # line) so `docker logs -f` shows visible progress through a slow cold
    # start (e.g. downloading BGE-M3/Chinese-CLIP weights on first run)
    # instead of one silent multi-minute gap between "starting" and "done".
    logger.info("Startup: loading text embedder (BGE-M3)... this can take several minutes on first run before weights are cached")
    start = time.monotonic()
    get_text_embedder()
    logger.info("Startup: text embedder loaded (%.1fs)", time.monotonic() - start)

    logger.info("Startup: loading CLIP embedder (Chinese-CLIP)...")
    start = time.monotonic()
    get_clip_embedder()
    logger.info("Startup: CLIP embedder loaded (%.1fs)", time.monotonic() - start)

    if settings.reranker_enabled:
        logger.info("Startup: loading reranker (%s)...", settings.reranker_model_name)
        start = time.monotonic()
        get_reranker()
        logger.info("Startup: reranker loaded (%.1fs)", time.monotonic() - start)
    else:
        logger.info("Startup: reranker disabled (RERANKER_ENABLED=false); skipping preload")

    logger.info("Startup complete; ready to accept requests")
    yield


app = FastAPI(title="Multimodal Retrieval Service", lifespan=lifespan)
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(directories.router)
app.include_router(jobs.router)
app.include_router(collections.router)
app.include_router(endpoints.router)
