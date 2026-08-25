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
                    "page": None, "source_type": "normal", "source_path": "/a.txt", "file_type": "txt",
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
    assert body["next_cursor"] == "next-cursor-value"


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
