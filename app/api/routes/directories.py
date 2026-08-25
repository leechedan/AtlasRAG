import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DeleteResponse,
    IndexDirectoryRequest,
    RemoveWatchRequest,
    UploadResponse,
)
from app.ingest.watcher import unregister_watch
from app.jobs.queue import get_queue
from app.jobs.tasks import ingest_directory_job

router = APIRouter()


@router.post("/directories/index", response_model=UploadResponse)
def index_directory(request: IndexDirectoryRequest):
    # Without this, a typo'd path silently produced a 0/0 job status
    # indistinguishable from success, and (with watch=True) got persisted
    # into Redis as a watched directory that would later crash the worker
    # on start (Critical Finding 3).
    if not Path(request.path).is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {request.path}")

    queue = get_queue()
    job_id = str(uuid.uuid4())
    job = queue.enqueue(
        ingest_directory_job,
        request.path, request.recursive, request.watch, job_id,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        separators=request.separators,
        job_id=job_id,
        # A directory ingest can process arbitrarily many files, so it gets
        # an effectively unbounded timeout rather than RQ's 180s default --
        # otherwise any directory taking longer than 3 minutes gets
        # SIGALRM-killed mid-batch, leaving a partial index with no retry.
        # job_timeout=-1 means "never times out" (verified against the
        # installed rq==2.1.0: rq/worker.py special-cases job.timeout == -1
        # in both its heartbeat-TTL and horse-monitor logic, and
        # signal.alarm(-1) -- what the death-penalty timeout ultimately
        # calls -- was empirically confirmed not to fire). job_timeout is an
        # RQ-reserved kwarg popped out of kwargs by Queue.parse_args before
        # the rest are forwarded to the target function, so it doesn't
        # collide with ingest_directory_job's own kwargs, the same way
        # job_id= above doesn't.
        job_timeout=-1,
    )
    return UploadResponse(job_id=job.id)


@router.delete("/directories/watch", response_model=DeleteResponse)
def remove_watch(request: RemoveWatchRequest):
    """Remove a watch entry already persisted in Redis. This is the
    recovery path for a bad/stale entry (registered before the path
    validation above existed, or a directory later deleted) without
    resorting to manual Redis surgery (Critical Finding 3)."""
    redis_conn = get_queue().connection
    removed = unregister_watch(redis_conn, request.path)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no watch registered for path: {request.path}")
    return DeleteResponse(status="deleted")
