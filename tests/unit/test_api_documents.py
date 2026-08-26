import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import fakeredis
from fastapi.testclient import TestClient
from PIL import Image
from rq import Queue

from app.api.deps import get_store_client
from app.api.main import app
from app.jobs.status import get_job_status


@patch("app.api.routes.documents.get_queue")
def test_upload_document_enqueues_job_and_returns_job_id(mock_get_queue):
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-123")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/documents/upload", files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-job-123"
    mock_queue.enqueue.assert_called_once()
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["chunk_size"] is None
    assert kwargs["separators"] is None


@patch("app.api.routes.documents.get_queue")
def test_upload_document_with_custom_chunk_options(mock_get_queue):
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-456")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    # NOTE: with httpx==0.28.1 (pinned in requirements.txt), `data=` combined
    # with `files=` must be a Mapping, or httpx's encode_request treats the
    # whole `data` as raw byte content and silently drops `files` entirely
    # (see httpx/_content.py: `if data is not None and not isinstance(data,
    # Mapping): ... return encode_content(data)`), which yields a 422 "file
    # field required" response with no exception raised. The brief's original
    # `data=[("chunk_size", "500"), ...]` list-of-tuples form (a common older
    # httpx/requests idiom for repeated keys) reproduces that 422 in this
    # repo's pinned httpx. The correct current idiom for repeated form keys
    # is a dict whose value for that key is a list -- httpx's MultipartStream
    # yields one DataField per list item under the same field name.
    response = client.post(
        "/documents/upload",
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"chunk_size": "500", "chunk_overlap": "50", "separators": ["\n\n", "---"]},
    )

    assert response.status_code == 200
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["chunk_size"] == 500
    assert kwargs["chunk_overlap"] == 50
    assert kwargs["separators"] == ["\n\n", "---"]


@patch("app.api.routes.documents.get_queue")
def test_upload_document_sets_job_timeout(mock_get_queue):
    """Regression test for Critical Finding 1: without an explicit
    job_timeout, RQ's default of 180s (rq.queue.Queue.DEFAULT_TIMEOUT)
    applies to every enqueued job. A single-file ingest can involve slow
    parsing/OCR/embedding for large files, so it gets a generous but
    bounded timeout rather than the 180s default."""
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-timeout")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    client.post(
        "/documents/upload", files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
    )

    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["job_timeout"] == 3600


def test_get_document_thumbnail_returns_image_bytes(tmp_path, monkeypatch):
    thumb_path = tmp_path / "thumb.jpg"
    Image.new("RGB", (10, 10), color="red").save(thumb_path, "JPEG")

    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.find_image_by_doc_id",
        lambda store, doc_id: {"doc_id": doc_id, "thumbnail_path": str(thumb_path)},
    )

    client = TestClient(app)
    response = client.get("/documents/img1/thumbnail")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == thumb_path.read_bytes()


def test_get_document_thumbnail_404_when_no_image_record(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.find_image_by_doc_id", lambda store, doc_id: None
    )

    client = TestClient(app)
    response = client.get("/documents/unknown/thumbnail")
    app.dependency_overrides.clear()

    assert response.status_code == 404


@patch("app.api.routes.documents.get_queue")
def test_delete_document_enqueues_delete_job(mock_get_queue):
    mock_queue = MagicMock()
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.delete("/documents/some-doc-id")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_queue.enqueue.assert_called_once()


@patch("app.jobs.tasks.ingest_file")
def test_upload_document_returned_job_id_matches_real_status_key(mock_ingest_file):
    """Regression test for the job_id mismatch bug: the reference implementation
    passed a made-up job_id (the dest file path) into ingest_file_job's status
    writes, but returned RQ's own separately auto-generated job.id to the
    caller. A client polling GET /jobs/{job_id} (Task 18) with the returned
    job_id would then find nothing, because status was written under a
    different key.

    Here we run a *real* synchronous RQ queue backed by fakeredis (following
    the pattern in tests/unit/test_jobs.py's test_ingest_file_job_updates_status)
    instead of mocking get_queue, so the job actually executes inline and
    writes real status. We only mock ingest_file itself (Task 13's pattern)
    to avoid running real embedding/Qdrant code.
    """
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d1", status="ingested", chunks_written=1)

    redis_conn = fakeredis.FakeStrictRedis()
    real_queue = Queue(connection=redis_conn, is_async=False)

    with (
        patch("app.api.routes.documents.get_queue", return_value=real_queue),
        patch("app.jobs.tasks.get_redis_connection", return_value=redis_conn),
    ):
        client = TestClient(app)
        response = client.post(
            "/documents/upload", files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = get_job_status(redis_conn, job_id)
    assert status["total"] == 1
    assert status["processed"] == 1
    # The file's status was written keyed under this exact job_id -> proves
    # the returned job_id is the real key the worker wrote status under.
    assert len(status["files"]) == 1
    file_status = next(iter(status["files"].values()))
    assert file_status["status"] == "ingested"


@patch("app.api.routes.documents.get_queue")
def test_upload_document_sanitizes_path_traversal_filename(mock_get_queue, tmp_path, monkeypatch):
    """A malicious filename like '../../../etc/passthrough' must not let the
    upload destination path escape settings.uploads_dir via pathlib's `/`
    join honoring '..' segments.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-789")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("../../../etc/passthrough", io.BytesIO(b"evil"), "text/plain")},
    )

    assert response.status_code == 200
    # No file should have been written outside of settings.uploads_dir (i.e.
    # outside of tmp_path, since uploads_dir = data_dir / "uploads").
    escaped_path = tmp_path.parent / "etc" / "passthrough"
    assert not escaped_path.exists()

    # The dest_path used for the write, and passed as ingest_file_job's
    # file_path arg (1st positional arg), must resolve to somewhere inside
    # settings.uploads_dir. Note: the *directory component* (derived from
    # source_ref, which embeds the raw unsanitized filename as a label) can
    # legitimately contain a literal ".." substring -- that's harmless
    # because "/" was already replaced with "_" there, so it's just
    # characters in a single path segment, not a traversal-capable ".."
    # segment. What actually matters is that the final filename component is
    # sanitized (so it can't reintroduce a real ".." segment) and that the
    # resolved absolute path never leaves uploads_dir.
    args, _ = mock_queue.enqueue.call_args
    dest_path_arg = args[1]
    resolved = Path(dest_path_arg).resolve()
    assert resolved.is_relative_to(settings.uploads_dir.resolve())
    assert Path(dest_path_arg).name == "passthrough"


def test_list_documents_returns_aggregated_documents(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.list_all_documents",
        lambda store: [{
            "doc_id": "d1", "source_path": "/a.txt", "filename": "a.txt", "file_type": "txt",
            "origin": "watch", "chunk_count": 2, "source_types": ["normal"], "created_at": None,
            "content_hash": "h1", "related_images": [],
        }],
    )

    client = TestClient(app)
    response = client.get("/documents")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["doc_id"] == "d1"
    assert body[0]["chunk_count"] == 2
    assert body[0]["related_images"] == []


def test_list_documents_empty_when_no_documents(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.list_all_documents", lambda store: [])

    client = TestClient(app)
    response = client.get("/documents")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []


def test_get_document_chunks_returns_ordered_chunks(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.get_existing_content_hash", lambda store, doc_id: "h1")
    monkeypatch.setattr(
        "app.api.routes.documents.list_chunks_for_document",
        lambda store, doc_id: [
            {
                "doc_id": doc_id, "chunk_index": 0, "text": "c0", "page": None,
                "source_type": "normal", "source_path": "/a.md", "file_type": "md",
                "heading_path": "一、简介", "has_table": False, "has_code_block": True,
            },
        ],
    )

    client = TestClient(app)
    response = client.get("/documents/d1/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body[0]["chunk_index"] == 0
    assert body[0]["chunk_id"]
    assert body[0]["heading_path"] == "一、简介"
    assert body[0]["has_table"] is False
    assert body[0]["has_code_block"] is True


def test_get_document_chunks_404_when_document_missing(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.get_existing_content_hash", lambda store, doc_id: None)

    client = TestClient(app)
    response = client.get("/documents/unknown/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 404
