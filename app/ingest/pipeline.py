"""Per-file ingestion orchestration: detect file type, parse, chunk, embed,
upsert into Qdrant, and skip re-ingestion of unchanged content.

`ingest_file` is the single entry point later tasks (RQ job queue, directory
watcher, upload/directory API routes) call once per file. It must never let
an exception escape: a batch job calls it once per file, and one bad file
must not abort the rest of the batch, so failures are captured and reported
via `IngestResult(status="failed", ...)` instead.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.config import settings
from app.ingest.chunking import chunk_csv_rows, chunk_text
from app.ingest.hashing import make_doc_id, sha256_file
from app.ingest.markdown_chunker import chunk_markdown_structured
from app.ingest.parsers.csv_parser import parse_csv_file
from app.ingest.parsers.image_parser import parse_image
from app.ingest.parsers.pdf_parser import parse_pdf_to_markdown
from app.ingest.parsers.text_parser import parse_text_file
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import (
    ImageRecord,
    TextChunkRecord,
    delete_document,
    find_all_doc_ids_by_source_path,
    get_default_client,
    get_existing_content_hash,
    upsert_image,
    upsert_text_chunk,
)

SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# MinerU's office backend only understands the OOXML formats -- not the
# legacy binary .doc/.ppt/.xls.
SUPPORTED_OFFICE_EXTS = {".docx", ".pptx", ".xlsx"}

# File types whose MinerU output is markdown, same as a PDF's -- these need
# the heading/table-aware chunk_markdown_structured() chunker rather than
# the plain char-level one (see _write_text_chunks below).
_MINERU_MARKDOWN_FILE_TYPES = {"pdf"} | {ext.lstrip(".") for ext in SUPPORTED_OFFICE_EXTS}


@dataclass
class IngestResult:
    doc_id: str
    status: str  # "ingested" | "skipped_unchanged" | "failed"
    chunks_written: int
    error: str | None = None


@dataclass
class ChunkOptions:
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    separators: list[str] | None = None


def _write_text_chunks(
    client,
    doc_id: str,
    source_path: str,
    file_type: str,
    text: str,
    page_number: int | None,
    source_type: str,
    content_hash: str,
    chunk_options: ChunkOptions,
    origin: str,
    created_at: str,
    start_index: int = 0,
) -> int:
    """Chunk, embed, and upsert one text blob (a whole file, or a single PDF
    page). `start_index` lets callers keep chunk_index monotonically
    increasing across several calls for the same doc_id (e.g. one call per
    PDF page) so chunk_index values never collide and silently overwrite
    each other in Qdrant.

    Markdown files (`file_type == "md"`) and files whose text came from
    MinerU (`file_type` in `_MINERU_MARKDOWN_FILE_TYPES` -- pdf/docx/pptx/
    xlsx, all returned as markdown by MinerU) use the heading-AST-aware
    `chunk_markdown_structured()` instead of the char-level `chunk_text()`,
    so chunks respect section boundaries and never truncate a table/code
    block mid-element; each chunk's `heading_path`/`has_table`/
    `has_code_block` are carried into the stored record. Every other file
    type keeps the existing char-level chunker with those three fields left
    at their defaults (`None`/`False`/`False`)."""
    if file_type == "md" or file_type in _MINERU_MARKDOWN_FILE_TYPES:
        structured_chunks = chunk_markdown_structured(
            text,
            chunk_size=chunk_options.chunk_size,
            chunk_overlap=chunk_options.chunk_overlap,
        )
        chunks = [c.content for c in structured_chunks]
        heading_paths: list[str | None] = [c.heading_path for c in structured_chunks]
        has_tables = [c.has_table for c in structured_chunks]
        has_code_blocks = [c.has_code_block for c in structured_chunks]
    else:
        chunks = chunk_text(
            text,
            chunk_size=chunk_options.chunk_size,
            chunk_overlap=chunk_options.chunk_overlap,
            separators=chunk_options.separators,
        )
        heading_paths = [None] * len(chunks)
        has_tables = [False] * len(chunks)
        has_code_blocks = [False] * len(chunks)

    if not chunks:
        return 0
    vectors = get_text_embedder().embed(chunks)
    for idx, (chunk, vector, heading_path, has_table, has_code_block) in enumerate(
        zip(chunks, vectors, heading_paths, has_tables, has_code_blocks)
    ):
        record = TextChunkRecord(
            doc_id=doc_id,
            chunk_index=start_index + idx,
            text=chunk,
            source_path=source_path,
            file_type=file_type,
            page=page_number,
            source_type=source_type,
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
            heading_path=heading_path,
            has_table=has_table,
            has_code_block=has_code_block,
        )
        upsert_text_chunk(client, record, vector)
    return len(chunks)


def _write_csv_chunks(
    client, doc_id: str, source_path: str, chunks: list[str], content_hash: str, origin: str, created_at: str
) -> int:
    if not chunks:
        return 0
    vectors = get_text_embedder().embed(chunks)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        record = TextChunkRecord(
            doc_id=doc_id,
            chunk_index=idx,
            text=chunk,
            source_path=source_path,
            file_type="csv",
            page=None,
            source_type="normal",
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        )
        upsert_text_chunk(client, record, vector)
    return len(chunks)


def _cleanup_stale_doc_ids(client, doc_id: str, source_path: str) -> None:
    """Delete Qdrant points left behind by earlier ingests of the same
    source_path under a different (older-content) doc_id.

    doc_id is content-addressed (make_doc_id(source_ref, content_hash)), so
    when a watched/re-uploaded file's content changes, the next ingest
    computes a *new* doc_id -- the previous doc_id's points would otherwise
    remain in Qdrant forever, growing the index unbounded and making search
    return stale content indefinitely (Critical Finding 2).

    Scope: only finds stale doc_ids via the plain source_path case
    (source_path == str(path) for every top-level ingestion branch --
    txt/md/csv/image, and a PDF's own text chunks). PDF-extracted images use
    a composite source_path (f"{pdf_path}#img={idx}") so they aren't found
    directly here -- but each delete_document(client, stale_doc_id) call
    below cascades to that stale doc_id's own composite-doc_id image records
    (f"{stale_doc_id}:img{N}") via qdrant_client._delete_related_pdf_images,
    so they aren't left behind either."""
    for stale_doc_id in find_all_doc_ids_by_source_path(client, source_path):
        if stale_doc_id != doc_id:
            delete_document(client, stale_doc_id)


def _save_thumbnail(image: Image.Image, doc_id: str) -> str:
    thumbnail = image.copy()
    thumbnail.thumbnail((256, 256))
    thumbnail_path = settings.thumbnails_dir / f"{doc_id}.jpg"
    thumbnail.convert("RGB").save(thumbnail_path, "JPEG")
    return str(thumbnail_path)


def _ingest_image_file(
    client, doc_id: str, path: Path, content_hash: str, chunk_options: ChunkOptions, origin: str, created_at: str
) -> int:
    image = Image.open(path).convert("RGB")
    parsed = parse_image(image, doc_id)
    upsert_image(
        client,
        ImageRecord(
            doc_id=doc_id,
            source_path=str(path),
            thumbnail_path=parsed.thumbnail_path,
            width=parsed.width,
            height=parsed.height,
            has_ocr_text=bool(parsed.ocr_text),
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        ),
        parsed.clip_vector,
    )
    written = 1
    if parsed.ocr_text:
        written += _write_text_chunks(
            client,
            doc_id,
            str(path),
            path.suffix.lstrip("."),
            parsed.ocr_text,
            None,
            "ocr_image",
            content_hash,
            chunk_options,
            origin=origin,
            created_at=created_at,
        )
    return written


def _ingest_mineru_extracted_image(
    client, doc_id: str, doc_path: Path, idx: int, image: Image.Image, content_hash: str, origin: str, created_at: str
) -> str:
    """入库 MinerU 从 PDF/docx/pptx/xlsx 里抽出的一张内嵌图片：持久化原图、
    生成缩略图、做 CLIP 向量化。不调用 parse_image()——那个函数会跑一次不需要的
    OCR，风格上延续 _ingest_scanned_pdf_page 原本"直接内联 CLIP+缩略图"的做法。
    """
    img_doc_id = f"{doc_id}:img{idx}"
    persisted_path = settings.extracted_images_dir / f"{img_doc_id}.jpg"
    image.save(persisted_path, "JPEG")
    thumbnail_path = _save_thumbnail(image, img_doc_id)
    clip_vector = get_clip_embedder().embed_image(image)
    upsert_image(
        client,
        ImageRecord(
            doc_id=img_doc_id,
            source_path=f"{doc_path}#img={idx}",
            thumbnail_path=thumbnail_path,
            width=image.width,
            height=image.height,
            has_ocr_text=False,
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
            image_path=str(persisted_path),
        ),
        clip_vector,
    )
    return img_doc_id


def _ingest_mineru_document(
    client, doc_id: str, path: Path, file_type: str, content_hash: str, chunk_options: ChunkOptions, origin: str, created_at: str
) -> int:
    """Ingest a file whose content MinerU turns into one markdown document:
    pdf, or (via SUPPORTED_OFFICE_EXTS) docx/pptx/xlsx."""
    parsed = parse_pdf_to_markdown(path)
    text = parsed.markdown_text
    for idx, doc_image in enumerate(parsed.images, start=1):
        img_doc_id = _ingest_mineru_extracted_image(
            client, doc_id, path, idx, doc_image.image, content_hash, origin=origin, created_at=created_at
        )
        text = text.replace(f"images/{doc_image.filename}", f"/documents/{img_doc_id}/image")
    return _write_text_chunks(
        client, doc_id, str(path), file_type, text, None, "normal", content_hash, chunk_options,
        origin=origin, created_at=created_at,
    )


def ingest_file(
    path: Path, source_ref: str, client=None, chunk_options: ChunkOptions | None = None
) -> IngestResult:
    client = client or get_default_client()
    chunk_options = chunk_options or ChunkOptions()
    doc_id: str | None = None
    origin = "upload" if source_ref.startswith("upload:") else "watch"
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        content_hash = sha256_file(path)
        doc_id = make_doc_id(source_ref, content_hash)

        existing_hash = get_existing_content_hash(client, doc_id)
        if existing_hash == content_hash:
            return IngestResult(doc_id=doc_id, status="skipped_unchanged", chunks_written=0)

        # About to write new chunks/records under `doc_id` -- clean up any
        # older doc_id(s) left behind by a previous ingest of this same
        # source_path (see _cleanup_stale_doc_ids). source_path is str(path)
        # consistently across every branch below.
        _cleanup_stale_doc_ids(client, doc_id, str(path))

        ext = path.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTS:
            text = parse_text_file(path)
            written = _write_text_chunks(
                client, doc_id, str(path), ext.lstrip("."), text, None, "normal", content_hash, chunk_options,
                origin=origin, created_at=created_at,
            )
        elif ext == ".csv":
            rows, header = parse_csv_file(path)
            chunks = chunk_csv_rows(rows, header, max_chars=chunk_options.chunk_size)
            written = _write_csv_chunks(client, doc_id, str(path), chunks, content_hash, origin=origin, created_at=created_at)
        elif ext in SUPPORTED_IMAGE_EXTS:
            written = _ingest_image_file(client, doc_id, path, content_hash, chunk_options, origin=origin, created_at=created_at)
        elif ext == ".pdf" or ext in SUPPORTED_OFFICE_EXTS:
            written = _ingest_mineru_document(
                client, doc_id, path, ext.lstrip("."), content_hash, chunk_options, origin=origin, created_at=created_at
            )
        else:
            raise ValueError(f"unsupported file extension: {ext}")

        return IngestResult(doc_id=doc_id, status="ingested", chunks_written=written)
    except Exception as exc:  # noqa: BLE001 - a single failed file must not abort a batch ingest job
        return IngestResult(doc_id=doc_id or source_ref, status="failed", chunks_written=0, error=str(exc))


def delete_ingested_document(doc_id: str, client=None) -> None:
    client = client or get_default_client()
    delete_document(client, doc_id)
