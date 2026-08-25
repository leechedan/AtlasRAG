from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import app


def _fake_embedder_getter(currsize: int) -> MagicMock:
    """A stand-in for get_text_embedder/get_clip_embedder that only needs
    to answer .cache_info().currsize -- health() never calls it, so we
    don't need to (and must not) actually load a real GPU model here."""
    fake = MagicMock()
    fake.cache_info.return_value = MagicMock(currsize=currsize)
    return fake


def test_health_returns_ok_when_both_models_loaded(monkeypatch):
    monkeypatch.setattr("app.api.routes.health.get_text_embedder", _fake_embedder_getter(1))
    monkeypatch.setattr("app.api.routes.health.get_clip_embedder", _fake_embedder_getter(1))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_models_not_yet_loaded(monkeypatch):
    """Regression test for Important Finding I1: /health used to return
    {"status": "ok"} unconditionally and immediately, even before the
    lazily-constructed @lru_cache model singletons had ever been built,
    which didn't reflect real readiness."""
    monkeypatch.setattr("app.api.routes.health.get_text_embedder", _fake_embedder_getter(0))
    monkeypatch.setattr("app.api.routes.health.get_clip_embedder", _fake_embedder_getter(0))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_returns_503_when_only_one_model_loaded(monkeypatch):
    monkeypatch.setattr("app.api.routes.health.get_text_embedder", _fake_embedder_getter(1))
    monkeypatch.setattr("app.api.routes.health.get_clip_embedder", _fake_embedder_getter(0))

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "loading"}
