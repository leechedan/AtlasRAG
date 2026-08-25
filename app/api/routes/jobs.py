from fastapi import APIRouter

from app.api.schemas import JobStatusResponse, JobSummaryOut
from app.jobs.queue import get_redis_connection
from app.jobs.status import delete_job_status, get_job_status, list_all_job_statuses

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    redis_conn = get_redis_connection()
    status = get_job_status(redis_conn, job_id)
    return JobStatusResponse(**status)


@router.get("/jobs", response_model=list[JobSummaryOut])
def list_jobs():
    redis_conn = get_redis_connection()
    return [JobSummaryOut(**summary) for summary in list_all_job_statuses(redis_conn)]


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    redis_conn = get_redis_connection()
    delete_job_status(redis_conn, job_id)
