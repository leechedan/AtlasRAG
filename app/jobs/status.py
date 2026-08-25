"""Per-job ingestion status tracking in Redis.

Each job's per-file results live in a Redis hash (one field per file path)
rather than a single JSON blob updated via read-modify-write. A directory
ingest job currently calls `update_job_file_result` sequentially for each
file within one RQ job, so no race exists for that call pattern today. But
a future per-file job queue (e.g. FastAPI enqueuing one `ingest_file_job`
per file under a shared `job_id`, picked up by several RQ workers at once)
would race on a read-modify-write-whole-document approach: worker A reads,
worker B reads before A writes, A writes, B writes and clobbers A's update,
silently losing that file's status. `HSET` on a per-file hash field is a
single atomic Redis command, so concurrent updates to different files (or
even the same file) can never clobber each other regardless of how many
workers call it in parallel. The total file count is kept in a separate
key so creating a job doesn't require read-modify-write either.
"""

import json

JOB_KEY_PREFIX = "job:"


def _total_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}:total"


def _files_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}:files"


def create_job_status(redis_conn, job_id: str, total_files: int) -> None:
    redis_conn.set(_total_key(job_id), total_files)
    redis_conn.delete(_files_key(job_id))


def update_job_file_result(redis_conn, job_id: str, file_path: str, status: str, error: str | None = None) -> None:
    redis_conn.hset(_files_key(job_id), file_path, json.dumps({"status": status, "error": error}))


def get_job_status(redis_conn, job_id: str) -> dict:
    total_raw = redis_conn.get(_total_key(job_id))
    total = int(total_raw) if total_raw is not None else 0

    files_raw = redis_conn.hgetall(_files_key(job_id))
    files: dict = {}
    for key, value in files_raw.items():
        path = key.decode() if isinstance(key, bytes) else key
        raw_value = value.decode() if isinstance(value, bytes) else value
        files[path] = json.loads(raw_value)

    return {"total": total, "processed": len(files), "files": files}


def list_all_job_statuses(redis_conn) -> list[dict]:
    job_ids = set()
    for raw_key in redis_conn.scan_iter(f"{JOB_KEY_PREFIX}*:total"):
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        job_ids.add(key[len(JOB_KEY_PREFIX) : -len(":total")])
    for raw_key in redis_conn.scan_iter(f"{JOB_KEY_PREFIX}*:files"):
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        job_ids.add(key[len(JOB_KEY_PREFIX) : -len(":files")])

    summaries = []
    for job_id in sorted(job_ids):
        status = get_job_status(redis_conn, job_id)
        success_count = sum(1 for f in status["files"].values() if f["status"] != "failed")
        failed_count = sum(1 for f in status["files"].values() if f["status"] == "failed")
        summaries.append(
            {
                "job_id": job_id,
                "total": status["total"],
                "processed": status["processed"],
                "success_count": success_count,
                "failed_count": failed_count,
            }
        )
    return summaries


def delete_job_status(redis_conn, job_id: str) -> None:
    redis_conn.delete(_total_key(job_id), _files_key(job_id))
