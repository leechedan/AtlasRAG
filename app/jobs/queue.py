"""Redis connection and RQ queue helpers for the ingestion job queue."""

import redis
from rq import Queue

from app.config import settings


def get_redis_connection():
    return redis.from_url(settings.redis_url)


def get_queue(connection=None) -> Queue:
    return Queue(connection=connection or get_redis_connection())
