import io
import tempfile
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.api.deps import get_store_client
from app.api.schemas import ChunkOut, DeleteResponse, DocumentSummaryOut, UploadResponse
from app.config import settings
from app.ingest.convert import IMAGES_SUBDIR, SUPPORTED_CONVERT_EXTS, convert_file_to_markdown
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


@router.post("/documents/convert-to-markdown")
def convert_document_to_markdown(file: UploadFile = File(...)):
    """Synchronous document -> Markdown conversion: parse the uploaded file
    with the same parser `POST /documents/upload` would use for its
    extension (see app.ingest.convert), but return the whole document as
    one Markdown string instead of chunking/embedding/writing it to
    Qdrant. No job is enqueued and no doc_id is minted -- the response
    body *is* the result.

    Any images the document embeds (PDF/docx/pptx/xlsx figures via MinerU,
    or the file itself for a standalone image upload) are referenced from
    the Markdown as relative paths ("images/<filename>", matching
    MinerU's own convention) rather than inlined -- an ordinary
    filesystem-based Markdown reader (VS Code, Typora, Obsidian, ...) only
    renders those once the referenced files actually exist next to the
    .md on disk, so they're persisted under settings.converted_images_dir
    keyed by a convert_id, fetchable as one archive via GET
    /documents/convert-to-markdown/{convert_id}/images.zip
    (download_converted_images below) -- unzipping it next to the saved
    .md file reproduces the "images/" layout the Markdown expects. That
    zip URL is exposed both as the `X-Images-Zip-Url` response header
    *and* as a leading HTML comment in the Markdown body itself -- a
    header alone is easy to miss for a caller that just saves/reads the
    plain-text body (e.g. `curl -o out.md`), and an HTML comment renders
    invisibly in any Markdown viewer while still being visible in the raw
    text every caller already has.

    Defined as a plain `def` (not `async def`) so Starlette runs it in its
    threadpool (see starlette.routing: a sync endpoint is dispatched via
    `run_in_threadpool`), keeping the blocking `httpx.post` inside
    parse_pdf_to_markdown (used for pdf/docx/pptx/xlsx, see
    app.ingest.parsers.pdf_parser) from stalling this worker's asyncio
    event loop for however long MinerU takes.
    """
    safe_filename = Path(file.filename).name
    ext = Path(safe_filename).suffix.lower()
    if ext not in SUPPORTED_CONVERT_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file extension: {ext!r}; supported: {sorted(SUPPORTED_CONVERT_EXTS)}",
        )

    # `file.file` (the underlying SpooledTemporaryFile) rather than
    # `await file.read()` -- this route is a sync `def`, so there is no
    # event loop here to await on.
    content = file.file.read()

    # A scratch dir, not settings.uploads_dir: this conversion is not an
    # ingest, so the *uploaded source file* is never persisted, unlike
    # /documents/upload's dest_path which deliberately stays around for
    # the ingest job to read. Only the converted output (Markdown text +
    # any extracted images, both built from this input, not the input
    # itself) is kept.
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / safe_filename
        dest_path.write_bytes(content)
        try:
            converted = convert_file_to_markdown(dest_path)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except RuntimeError as exc:
            # parse_pdf_to_markdown raises RuntimeError for a MinerU-side
            # failure (non-200, empty markdown) -- that's an upstream
            # dependency failure, not a bad request from this caller.
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    convert_id = uuid.uuid4().hex
    md_filename = f"{Path(safe_filename).stem}.md"
    # Response headers must be latin-1 encodable (Starlette encodes them as
    # such), but an uploaded filename is arbitrary Unicode (e.g. Chinese
    # characters) -- encoding it straight into Content-Disposition raised
    # UnicodeEncodeError for any non-latin-1 filename. RFC 6266 fixes this
    # with a dual filename: a plain ASCII fallback for parsers that only
    # understand `filename=`, plus RFC 5987 percent-encoded UTF-8 in
    # `filename*=` for parsers that support the real name.
    ascii_filename = md_filename.encode("ascii", "ignore").decode("ascii") or "converted.md"
    encoded_filename = quote(md_filename)
    headers = {
        "Content-Disposition": (
            f'inline; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        ),
        "X-Convert-Id": convert_id,
    }
    markdown_text = converted.markdown_text
    if converted.images:
        images_dir = settings.converted_images_dir / convert_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in converted.images:
            img.image.convert("RGB").save(images_dir / img.filename, "JPEG")
        zip_url = f"/documents/convert-to-markdown/{convert_id}/images.zip"
        headers["X-Images-Zip-Url"] = zip_url
        # A response header alone is easy to miss -- most callers of a
        # "plain Markdown body" endpoint just save/read the body and never
        # look at headers (e.g. `curl -o out.md`, pasting the body
        # somewhere). An HTML comment is invisible when the Markdown is
        # rendered but still right there in the raw text every caller
        # already has, so the zip URL is discoverable without requiring
        # header access.
        markdown_text = f"<!-- images zip: {zip_url} (convert_id: {convert_id}) -->\n{markdown_text}"

    return Response(content=markdown_text, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/documents/convert-to-markdown/{convert_id}/images.zip")
def download_converted_images(convert_id: str):
    """Batch download for the images referenced by a prior
    POST /documents/convert-to-markdown response (its `X-Convert-Id`).
    Returns a zip with each image under "images/<filename>" -- extracting
    it next to the saved .md reproduces the relative paths that Markdown
    already references, so a plain filesystem-based Markdown reader shows
    the images with no further edits."""
    images_dir = settings.converted_images_dir / convert_id
    if not images_dir.is_dir():
        raise HTTPException(status_code=404, detail="unknown convert_id or it had no images")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_path in sorted(images_dir.iterdir()):
            zf.write(image_path, arcname=f"{IMAGES_SUBDIR}/{image_path.name}")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{convert_id}-images.zip"'},
    )


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


@router.get("/documents/{doc_id}/image")
def get_document_image(doc_id: str, store=Depends(get_store_client)):
    # 和 get_document_thumbnail 同样的理由：MinerU 抽取的原图只在 worker
    # 容器本地磁盘上（settings.extracted_images_dir），Admin UI 等外部调用
    # 方必须经由这个 HTTP 接口才能显示。
    image = find_image_by_doc_id(store, doc_id)
    if image is None or not image.get("image_path"):
        raise HTTPException(status_code=404, detail="original image not available")
    image_path = Path(image["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(image_path, media_type="image/jpeg")


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    queue = get_queue()
    queue.enqueue(delete_document_job, doc_id)
    return DeleteResponse(status="deleted")
