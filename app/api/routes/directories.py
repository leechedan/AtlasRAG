from fastapi import APIRouter

from app.api.schemas import IndexDirectoryRequest, UploadResponse
from app.jobs.queue import get_queue

router = APIRouter()


@router.post("/directories/index", response_model=UploadResponse)
def index_directory(request: IndexDirectoryRequest):
    from app.jobs.tasks import ingest_directory_job

    queue = get_queue()

    # RQ assigns job.id only once enqueue() returns, but ingest_directory_job
    # needs that id as its own job_id argument (the key it writes per-file
    # status under) so that GET /jobs/{job_id} can find it later. We can't
    # know job.id before enqueueing, so: enqueue first (job_id=None lets RQ
    # generate one), then append the resulting job.id onto the already-
    # enqueued job's args and job.save() to persist the corrected args back
    # to Redis before a worker picks it up. See rq/job.py Job.args setter
    # (marks _data stale) and Job.save() (re-serializes via to_dict()/data
    # property and HSETs it) — verified this round-trips correctly with a
    # live fakeredis + rq.Queue check.
    job = queue.enqueue(
        ingest_directory_job,
        args=(request.path, request.recursive, request.watch),
        kwargs={
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "separators": request.separators,
        },
        job_id=None,
    )
    job.args = (*job.args, job.id)
    job.save()
    return UploadResponse(job_id=job.id)
