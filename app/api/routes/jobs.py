from fastapi import APIRouter

from app.api.schemas import JobStatusResponse
from app.jobs.queue import get_redis_connection
from app.jobs.status import get_job_status

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    redis_conn = get_redis_connection()
    status = get_job_status(redis_conn, job_id)
    return JobStatusResponse(**status)
