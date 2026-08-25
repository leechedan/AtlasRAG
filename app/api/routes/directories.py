import uuid

from fastapi import APIRouter

from app.api.schemas import IndexDirectoryRequest, UploadResponse
from app.jobs.queue import get_queue
from app.jobs.tasks import ingest_directory_job

router = APIRouter()


@router.post("/directories/index", response_model=UploadResponse)
def index_directory(request: IndexDirectoryRequest):
    queue = get_queue()
    job_id = str(uuid.uuid4())
    job = queue.enqueue(
        ingest_directory_job,
        request.path, request.recursive, request.watch, job_id,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        separators=request.separators,
        job_id=job_id,
    )
    return UploadResponse(job_id=job.id)
