"""RQ job functions: per-file ingest, per-directory ingest, and document
delete. These are the functions enqueued by the upload/directory-index API
routes (Task 17/18) and executed by the RQ worker (Task 19).
"""

from pathlib import Path

from app.ingest.pipeline import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_TEXT_EXTS,
    ChunkOptions,
    delete_ingested_document,
    ingest_file,
)
from app.jobs.queue import get_redis_connection
from app.jobs.status import create_job_status, update_job_file_result

SUPPORTED_EXTS = SUPPORTED_TEXT_EXTS | SUPPORTED_IMAGE_EXTS | {".csv", ".pdf"}


def ingest_file_job(
    file_path: str,
    source_ref: str,
    job_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    redis_conn = get_redis_connection()
    chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    result = ingest_file(Path(file_path), source_ref, chunk_options=chunk_options)
    update_job_file_result(redis_conn, job_id, file_path, result.status, result.error)


def ingest_directory_job(
    dir_path: str,
    recursive: bool,
    watch: bool,
    job_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    redis_conn = get_redis_connection()
    chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    directory = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    create_job_status(redis_conn, job_id, total_files=len(files))
    for file_path in files:
        result = ingest_file(file_path, source_ref=str(file_path), chunk_options=chunk_options)
        update_job_file_result(redis_conn, job_id, str(file_path), result.status, result.error)

    if watch:
        from app.ingest.watcher import register_watch

        register_watch(
            redis_conn,
            dir_path,
            recursive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )


def delete_document_job(doc_id: str) -> None:
    delete_ingested_document(doc_id)
