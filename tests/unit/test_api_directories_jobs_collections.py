from unittest.mock import MagicMock, patch

import fakeredis
from fastapi.testclient import TestClient
from rq import Queue

from app.api.deps import get_store_client
from app.api.main import app
from app.jobs.status import get_job_status


@patch("app.api.routes.directories.get_queue")
def test_index_directory_enqueues_job(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job")
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = mock_job
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post("/directories/index", json={"path": "/data/docs", "recursive": True, "watch": True})

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-dir-job"

    args, kwargs = mock_queue.enqueue.call_args
    assert args[1:4] == ("/data/docs", True, True)
    generated_job_id = args[4]
    assert generated_job_id == kwargs["job_id"]
    assert kwargs["chunk_size"] is None
    assert kwargs["chunk_overlap"] is None
    assert kwargs["separators"] is None


@patch("app.api.routes.directories.get_queue")
def test_index_directory_passes_custom_chunk_options(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job-2")
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = mock_job
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/directories/index",
        json={
            "path": "/data/docs", "recursive": True, "watch": False,
            "chunk_size": 800, "chunk_overlap": 100, "separators": ["\n\n", "---"],
        },
    )

    assert response.status_code == 200
    args, kwargs = mock_queue.enqueue.call_args
    assert args[1:4] == ("/data/docs", True, False)
    generated_job_id = args[4]
    assert generated_job_id == kwargs["job_id"]
    assert kwargs["chunk_size"] == 800
    assert kwargs["chunk_overlap"] == 100
    assert kwargs["separators"] == ["\n\n", "---"]


@patch("app.jobs.tasks.ingest_file")
def test_index_directory_returned_job_id_matches_real_status_key(mock_ingest_file, tmp_path):
    """Regression test for the enqueue/save race: the previous implementation
    enqueued the job with job_id=None first (immediately visible to any idle
    worker), then patched job.id onto job.args and called job.save() to fix
    it up afterward. A worker that dequeued between those two steps would
    call ingest_directory_job with only 3 positional args (missing job_id),
    raising a TypeError before create_job_status ever ran -- leaving
    GET /jobs/{job_id} permanently showing an empty/zero status.

    This runs a *real* synchronous RQ queue backed by fakeredis so the job
    actually executes inline (as SimpleWorker/is_async=False does, collapsing
    the race window to zero for this test while still exercising the exact
    same code path), and asserts the job_id returned to the caller is the
    real key ingest_directory_job's own create_job_status/
    update_job_file_result calls wrote status under.
    """
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d1", status="ingested", chunks_written=1)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")

    redis_conn = fakeredis.FakeStrictRedis()
    real_queue = Queue(connection=redis_conn, is_async=False)

    with (
        patch("app.api.routes.directories.get_queue", return_value=real_queue),
        patch("app.jobs.tasks.get_redis_connection", return_value=redis_conn),
    ):
        client = TestClient(app)
        response = client.post(
            "/directories/index", json={"path": str(tmp_path), "recursive": False, "watch": False}
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = get_job_status(redis_conn, job_id)
    assert status["total"] == 2
    assert status["processed"] == 2
    assert len(status["files"]) == 2
    for file_status in status["files"].values():
        assert file_status["status"] == "ingested"


@patch("app.api.routes.jobs.get_job_status")
@patch("app.api.routes.jobs.get_redis_connection")
def test_get_job_status_returns_progress(mock_get_conn, mock_get_status):
    mock_get_status.return_value = {"total": 2, "processed": 1, "files": {"/a.txt": {"status": "ingested", "error": None}}}

    client = TestClient(app)
    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["processed"] == 1


def test_get_collections_stats():
    app.dependency_overrides[get_store_client] = lambda: object()
    with patch("app.api.routes.collections.get_stats", return_value={"text_chunks_count": 5, "images_count": 2}):
        client = TestClient(app)
        response = client.get("/collections/stats")

    assert response.status_code == 200
    assert response.json() == {"text_chunks_count": 5, "images_count": 2}
    app.dependency_overrides.clear()
