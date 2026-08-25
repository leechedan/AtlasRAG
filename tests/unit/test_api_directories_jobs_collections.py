from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.deps import get_store_client
from app.api.main import app


@patch("app.api.routes.directories.get_queue")
def test_index_directory_enqueues_job(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job", args=("/data/docs", True, True))
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = mock_job
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post("/directories/index", json={"path": "/data/docs", "recursive": True, "watch": True})

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-dir-job"
    mock_job.save.assert_called_once()
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["kwargs"] == {"chunk_size": None, "chunk_overlap": None, "separators": None}


@patch("app.api.routes.directories.get_queue")
def test_index_directory_passes_custom_chunk_options(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job-2", args=("/data/docs", True, False))
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
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["kwargs"] == {"chunk_size": 800, "chunk_overlap": 100, "separators": ["\n\n", "---"]}


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
