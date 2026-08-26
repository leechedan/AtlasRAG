import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_store_client
from app.api.schemas import ChunkOut, DeleteResponse, DocumentSummaryOut, UploadResponse
from app.config import settings
from app.ingest.hashing import upload_source_ref
from app.jobs.queue import get_queue
from app.jobs.status import create_job_status
from app.jobs.tasks import delete_document_job, ingest_file_job
from app.store.qdrant_client import (
    compute_chunk_id,
    find_image_by_doc_id,
    get_existing_content_hash,
    list_all_documents,
    list_chunks_for_document,
)

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
        # Without this, RQ's default job_timeout (180s, rq.queue.Queue.
        # DEFAULT_TIMEOUT) applies and any file whose parsing/OCR/embedding
        # takes longer than 3 minutes gets SIGALRM-killed mid-ingest.
        # job_timeout is an RQ-reserved kwarg (confirmed by reading
        # rq/queue.py: Queue.parse_args pops 'job_timeout' out of kwargs
        # before the rest are forwarded to the target function), so it
        # doesn't collide with ingest_file_job's own kwargs, the same way
        # job_id= above doesn't.
        job_timeout=3600,
    )
    return UploadResponse(job_id=job.id)


@router.get("/documents", response_model=list[DocumentSummaryOut])
def list_documents(store=Depends(get_store_client)):
    return [DocumentSummaryOut(**d) for d in list_all_documents(store)]


@router.get("/documents/{doc_id}/chunks", response_model=list[ChunkOut])
def get_document_chunks(doc_id: str, store=Depends(get_store_client)):
    if get_existing_content_hash(store, doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    chunks = list_chunks_for_document(store, doc_id)
    return [
        ChunkOut(
            chunk_id=compute_chunk_id(c["doc_id"], c["chunk_index"]),
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            text=c["text"],
            page=c.get("page"),
            source_type=c["source_type"],
            source_path=c["source_path"],
            file_type=c["file_type"],
            heading_path=c.get("heading_path"),
            has_table=c.get("has_table", False),
            has_code_block=c.get("has_code_block", False),
        )
        for c in chunks
    ]


@router.get("/documents/{doc_id}/thumbnail")
def get_document_thumbnail(doc_id: str, store=Depends(get_store_client)):
    # The Admin UI (and any other external caller) only ever talks to this
    # service over HTTP -- it has no access to the api/worker containers'
    # /srv/data filesystem (see docker-compose.yml: admin-ui has no volume
    # mounts). Without this endpoint, an image record's thumbnail_path is a
    # path that only resolves inside api/worker's own containers, so
    # cross-modal text->image search hits have no way to actually be
    # displayed anywhere outside those two containers -- the search finds
    # the image, but nothing can render it.
    image = find_image_by_doc_id(store, doc_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    thumbnail_path = Path(image["thumbnail_path"])
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="thumbnail file missing")
    return FileResponse(thumbnail_path, media_type="image/jpeg")


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    queue = get_queue()
    queue.enqueue(delete_document_job, doc_id)
    return DeleteResponse(status="deleted")
