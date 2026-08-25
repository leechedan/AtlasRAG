import fakeredis

from app.jobs.status import create_job_status, delete_job_status, list_all_job_statuses, update_job_file_result


def make_redis():
    return fakeredis.FakeStrictRedis()


def test_list_all_job_statuses_returns_summary_per_job():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-a", total_files=2)
    update_job_file_result(redis_conn, "job-a", "/a.txt", "ingested")
    update_job_file_result(redis_conn, "job-a", "/b.txt", "failed", error="bad file")
    create_job_status(redis_conn, "job-b", total_files=1)
    update_job_file_result(redis_conn, "job-b", "/c.txt", "ingested")

    summaries = {s["job_id"]: s for s in list_all_job_statuses(redis_conn)}

    assert summaries["job-a"] == {
        "job_id": "job-a",
        "total": 2,
        "processed": 2,
        "success_count": 1,
        "failed_count": 1,
    }
    assert summaries["job-b"] == {
        "job_id": "job-b",
        "total": 1,
        "processed": 1,
        "success_count": 1,
        "failed_count": 0,
    }


def test_list_all_job_statuses_returns_empty_list_when_no_jobs():
    redis_conn = make_redis()
    assert list_all_job_statuses(redis_conn) == []


def test_delete_job_status_removes_both_keys():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-x", total_files=1)
    update_job_file_result(redis_conn, "job-x", "/a.txt", "ingested")

    delete_job_status(redis_conn, "job-x")

    assert list_all_job_statuses(redis_conn) == []
    assert redis_conn.exists("job:job-x:total") == 0
    assert redis_conn.exists("job:job-x:files") == 0


def test_delete_job_status_is_a_noop_for_unknown_job_id():
    redis_conn = make_redis()
    delete_job_status(redis_conn, "does-not-exist")  # 不应抛异常
