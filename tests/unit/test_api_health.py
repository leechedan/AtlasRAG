import time
from unittest.mock import MagicMock

import fakeredis
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import settings
from app.ingest.parsers.pdf_parser import MINERU_INFLIGHT_KEY, MINERU_LAST_CALL_AT_KEY


def _fake_cached_getter(currsize: int) -> MagicMock:
    """A stand-in for get_text_embedder/get_clip_embedder/get_reranker that
    only needs to answer .cache_info().currsize -- health() never calls the
    getter itself, so we don't need to (and must not) actually load a real
    GPU model here."""
    fake = MagicMock()
    fake.cache_info.return_value = MagicMock(currsize=currsize)
    return fake


def _fake_qdrant_client(*, reachable: bool) -> MagicMock:
    fake = MagicMock()
    if reachable:
        fake.get_collections.return_value = MagicMock()
    else:
        fake.get_collections.side_effect = RuntimeError("connection refused")
    return fake


def _fake_redis_connection(*, reachable: bool) -> MagicMock:
    fake = MagicMock()
    if reachable:
        fake.ping.return_value = True
    else:
        fake.ping.side_effect = RuntimeError("connection refused")
    return fake


def _patch_all_ready(monkeypatch, *, reranker_enabled: bool = True):
    monkeypatch.setattr("app.api.routes.health.get_text_embedder", _fake_cached_getter(1))
    monkeypatch.setattr("app.api.routes.health.get_clip_embedder", _fake_cached_getter(1))
    monkeypatch.setattr(settings, "reranker_enabled", reranker_enabled)
    if reranker_enabled:
        monkeypatch.setattr("app.api.routes.health.get_reranker", _fake_cached_getter(1))
    monkeypatch.setattr(
        "app.api.routes.health.get_default_client", lambda: _fake_qdrant_client(reachable=True)
    )
    monkeypatch.setattr(
        "app.api.routes.health.get_redis_connection", lambda: _fake_redis_connection(reachable=True)
    )


def test_health_returns_ok_when_everything_healthy(monkeypatch):
    _patch_all_ready(monkeypatch)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"qdrant": "ok", "redis": "ok"}}


def test_health_returns_503_when_text_embedder_not_yet_loaded(monkeypatch):
    """Regression test for Important Finding I1: /health used to return
    {"status": "ok"} unconditionally and immediately, even before the
    lazily-constructed @lru_cache model singletons had ever been built,
    which didn't reflect real readiness."""
    _patch_all_ready(monkeypatch)
    monkeypatch.setattr("app.api.routes.health.get_text_embedder", _fake_cached_getter(0))
    # Not loaded yet means the dependency checks below must never run.
    monkeypatch.setattr("app.api.routes.health.get_default_client", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr("app.api.routes.health.get_redis_connection", MagicMock(side_effect=AssertionError))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_returns_503_when_clip_embedder_not_yet_loaded(monkeypatch):
    _patch_all_ready(monkeypatch)
    monkeypatch.setattr("app.api.routes.health.get_clip_embedder", _fake_cached_getter(0))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_returns_503_when_reranker_enabled_but_not_yet_loaded(monkeypatch):
    _patch_all_ready(monkeypatch, reranker_enabled=True)
    monkeypatch.setattr("app.api.routes.health.get_reranker", _fake_cached_getter(0))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_skips_reranker_check_when_disabled(monkeypatch):
    _patch_all_ready(monkeypatch, reranker_enabled=False)
    # If reranker_enabled=False and health() still called get_reranker, this
    # would blow up instead of silently reporting readiness.
    monkeypatch.setattr("app.api.routes.health.get_reranker", MagicMock(side_effect=AssertionError))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"qdrant": "ok", "redis": "ok"}}


def test_health_returns_503_degraded_when_qdrant_unreachable(monkeypatch):
    _patch_all_ready(monkeypatch)
    monkeypatch.setattr(
        "app.api.routes.health.get_default_client", lambda: _fake_qdrant_client(reachable=False)
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "checks": {"qdrant": "unreachable", "redis": "ok"}}


def test_health_returns_503_degraded_when_redis_unreachable(monkeypatch):
    _patch_all_ready(monkeypatch)
    monkeypatch.setattr(
        "app.api.routes.health.get_redis_connection", lambda: _fake_redis_connection(reachable=False)
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "checks": {"qdrant": "ok", "redis": "unreachable"}}


def test_mineru_idle_status_true_when_no_calls_ever_recorded(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.health.get_redis_connection", lambda: fakeredis.FakeStrictRedis()
    )

    client = TestClient(app)
    response = client.get("/health/mineru-idle")

    assert response.status_code == 200
    body = response.json()
    assert body["idle"] is True
    assert body["reasons"][0] == {"check": "计数器值", "idle": True, "value": 0}
    assert body["reasons"][1]["check"] == "最近转换时间"
    assert body["reasons"][1]["idle"] is True
    assert body["reasons"][1]["seconds_since_last_call"] is None


def test_mineru_idle_status_false_when_inflight_counter_nonzero(monkeypatch):
    fake_conn = fakeredis.FakeStrictRedis()
    fake_conn.incr(MINERU_INFLIGHT_KEY)
    monkeypatch.setattr("app.api.routes.health.get_redis_connection", lambda: fake_conn)

    client = TestClient(app)
    response = client.get("/health/mineru-idle")

    body = response.json()
    assert body["idle"] is False
    assert body["reasons"][0] == {"check": "计数器值", "idle": False, "value": 1}


def test_mineru_idle_status_false_when_last_call_too_recent(monkeypatch):
    fake_conn = fakeredis.FakeStrictRedis()
    fake_conn.set(MINERU_LAST_CALL_AT_KEY, time.time())
    monkeypatch.setattr("app.api.routes.health.get_redis_connection", lambda: fake_conn)

    client = TestClient(app)
    response = client.get("/health/mineru-idle")

    body = response.json()
    assert body["idle"] is False
    assert body["reasons"][1]["check"] == "最近转换时间"
    assert body["reasons"][1]["idle"] is False


def test_mineru_idle_status_true_when_last_call_old_enough(monkeypatch):
    fake_conn = fakeredis.FakeStrictRedis()
    fake_conn.set(MINERU_LAST_CALL_AT_KEY, time.time() - 3600)
    monkeypatch.setattr("app.api.routes.health.get_redis_connection", lambda: fake_conn)

    client = TestClient(app)
    response = client.get("/health/mineru-idle")

    body = response.json()
    assert body["idle"] is True
    assert body["reasons"][1]["idle"] is True
