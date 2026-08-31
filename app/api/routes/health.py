import logging
import time

from fastapi import APIRouter, Response

from app.config import settings
from app.ingest.parsers.pdf_parser import MINERU_INFLIGHT_KEY, MINERU_LAST_CALL_AT_KEY
from app.jobs.queue import get_redis_connection
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.models.reranker import get_reranker
from app.store.qdrant_client import get_default_client

router = APIRouter()
logger = logging.getLogger(__name__)

# 重启前的空闲防抖窗口:两个 RQ 任务交接的瞬间,inflight 计数器会短暂归零,
# 单看计数器容易把"队列里还有活但恰好在交接间隙"误判成空闲,所以再叠加一个
# "距离上一次调用是否已经过去足够久"的信号一起判断。
MINERU_IDLE_RECENCY_THRESHOLD_SECONDS = 120


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


@router.get("/health/mineru-idle")
def mineru_idle_status():
    """判断 mineru-api 是否空闲,供重启前的判断路径调用。两个角度都满足才算
    空闲:1) 计数器为 0(当前没有请求在跟 mineru-api 交互);2) 距离上一次调用
    已经过去足够久(给 RQ 任务交接间隙留出防抖窗口,见上面常量注释)。"""
    conn = get_redis_connection()

    inflight_raw = conn.get(MINERU_INFLIGHT_KEY)
    inflight = int(inflight_raw) if inflight_raw is not None else 0
    inflight_idle = inflight <= 0

    last_call_raw = conn.get(MINERU_LAST_CALL_AT_KEY)
    if last_call_raw is None:
        seconds_since_last_call = None
        recency_idle = True
    else:
        seconds_since_last_call = time.time() - float(last_call_raw)
        recency_idle = seconds_since_last_call >= MINERU_IDLE_RECENCY_THRESHOLD_SECONDS

    return {
        "idle": inflight_idle and recency_idle,
        "reasons": [
            {
                "check": "计数器值",
                "idle": inflight_idle,
                "value": inflight,
            },
            {
                "check": "最近转换时间",
                "idle": recency_idle,
                "seconds_since_last_call": seconds_since_last_call,
                "threshold_seconds": MINERU_IDLE_RECENCY_THRESHOLD_SECONDS,
            },
        ],
    }
