import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import collections, directories, documents, health, jobs, search
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder

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
    logger.info("Loading models (text embedder, CLIP embedder)...")
    get_text_embedder()
    get_clip_embedder()
    logger.info("Models loaded; ready to accept requests")
    yield


app = FastAPI(title="Multimodal Retrieval Service", lifespan=lifespan)
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(directories.router)
app.include_router(jobs.router)
app.include_router(collections.router)
