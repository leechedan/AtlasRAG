from fastapi.testclient import TestClient

from app.api.deps import get_store_client
from app.api.main import app


def test_list_chunks_returns_page(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.chunks.scroll_chunks",
        lambda store, cursor, limit: (
            [
                {
                    "chunk_id": "cid1", "doc_id": "d1", "chunk_index": 0, "text": "t0",
                    "page": None, "source_type": "normal", "source_path": "/a.md", "file_type": "md",
                    "heading_path": "一、简介", "has_table": True, "has_code_block": False,
                },
            ],
            "next-cursor-value",
        ),
    )

    client = TestClient(app)
    response = client.get("/chunks?limit=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["chunk_id"] == "cid1"
    assert body["chunks"][0]["heading_path"] == "一、简介"
    assert body["chunks"][0]["has_table"] is True
    assert body["chunks"][0]["has_code_block"] is False
    assert body["next_cursor"] == "next-cursor-value"


def test_list_chunks_defaults_heading_metadata_when_missing_from_payload(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.chunks.scroll_chunks",
        lambda store, cursor, limit: (
            [
                {
                    "chunk_id": "cid2", "doc_id": "d1", "chunk_index": 0, "text": "t0",
                    "page": None, "source_type": "normal", "source_path": "/a.txt", "file_type": "txt",
                },
            ],
            None,
        ),
    )

    client = TestClient(app)
    response = client.get("/chunks?limit=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["heading_path"] is None
    assert body["chunks"][0]["has_table"] is False
    assert body["chunks"][0]["has_code_block"] is False


def test_list_chunks_last_page_has_null_next_cursor(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.chunks.scroll_chunks", lambda store, cursor, limit: ([], None)
    )

    client = TestClient(app)
    response = client.get("/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"chunks": [], "next_cursor": None}
