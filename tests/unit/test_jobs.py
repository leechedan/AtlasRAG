from unittest.mock import patch

import fakeredis
from rq import Queue, SimpleWorker

from app.jobs.status import create_job_status, get_job_status, update_job_file_result
from app.jobs.tasks import ingest_directory_job, ingest_file_job


def make_redis():
    return fakeredis.FakeStrictRedis()


def test_job_status_lifecycle():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-1", total_files=2)
    update_job_file_result(redis_conn, "job-1", "/a.txt", "ingested")
    update_job_file_result(redis_conn, "job-1", "/b.txt", "failed", error="bad file")

    status = get_job_status(redis_conn, "job-1")

    assert status["total"] == 2
    assert status["processed"] == 2
    assert status["files"]["/a.txt"]["status"] == "ingested"
    assert status["files"]["/b.txt"]["error"] == "bad file"


@patch("app.jobs.tasks.ingest_file")
def test_ingest_file_job_updates_status(mock_ingest_file, tmp_path):
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d1", status="ingested", chunks_written=3)

    redis_conn = make_redis()
    queue = Queue(connection=redis_conn, is_async=False)
    create_job_status(redis_conn, "job-2", total_files=1)

    # ingest_file_job resolves its own Redis connection via get_redis_connection()
    # (real code path: worker + queue share settings.redis_url). In this test the
    # queue runs on a fakeredis instance, so the job's own connection must be
    # patched to the same fake instance or its status write lands on whatever
    # real Redis get_redis_connection() would otherwise reach.
    with patch("app.jobs.tasks.get_redis_connection", return_value=redis_conn):
        queue.enqueue(ingest_file_job, str(tmp_path / "a.txt"), str(tmp_path / "a.txt"), "job-2")

    status = get_job_status(redis_conn, "job-2")
    assert status["files"][str(tmp_path / "a.txt")]["status"] == "ingested"


@patch("app.jobs.tasks.ingest_file")
def test_ingest_directory_job_enumerates_supported_files(mock_ingest_file, tmp_path):
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d", status="ingested", chunks_written=1)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.unsupported").write_text("y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("z")

    redis_conn = make_redis()
    with patch("app.jobs.tasks.get_redis_connection", return_value=redis_conn):
        ingest_directory_job(str(tmp_path), recursive=True, watch=False, job_id="job-3")

    status = get_job_status(redis_conn, "job-3")
    assert status["total"] == 2
    assert mock_ingest_file.call_count == 2
