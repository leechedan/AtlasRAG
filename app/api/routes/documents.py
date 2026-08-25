import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import DeleteResponse, UploadResponse
from app.config import settings
from app.ingest.hashing import upload_source_ref
from app.jobs.queue import get_queue
from app.jobs.status import create_job_status
from app.jobs.tasks import delete_document_job, ingest_file_job

router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
    separators: list[str] | None = Form(None),
):
    # Sanitize the client-supplied filename to just its final path segment.
    # `file.filename` is untrusted multipart input; without this, a filename
    # like "../../../etc/passthrough" would let the `dest_dir / filename`
    # join below escape settings.uploads_dir via pathlib honoring ".."
    # segments, letting a client write to an arbitrary path on disk.
    safe_filename = Path(file.filename).name

    source_ref = upload_source_ref(file.filename)
    dest_dir = settings.uploads_dir / source_ref.replace(":", "_").replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / safe_filename

    content = await file.read()
    dest_path.write_bytes(content)

    queue = get_queue()

    # Generate the job id ourselves (instead of letting RQ auto-generate one)
    # so it can be used consistently both as the key ingest_file_job writes
    # its status under, and as the id the client gets back to poll with via
    # GET /jobs/{job_id}. RQ's Queue.enqueue accepts a reserved `job_id=`
    # kwarg that forces the created Job to use this id rather than a random
    # one (confirmed by reading rq/queue.py: Queue.parse_args pops 'job_id'
    # out of kwargs before the rest are forwarded to the target function, so
    # it never collides with ingest_file_job's own `job_id` positional arg).
    job_id = str(uuid.uuid4())
    create_job_status(queue.connection, job_id, total_files=1)

    job = queue.enqueue(
        ingest_file_job,
        str(dest_path),
        source_ref,
        job_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        job_id=job_id,
    )
    return UploadResponse(job_id=job.id)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    queue = get_queue()
    queue.enqueue(delete_document_job, doc_id)
    return DeleteResponse(status="deleted")
