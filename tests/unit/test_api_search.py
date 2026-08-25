import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_clip_embedder_dep, get_store_client, get_text_embedder_dep
from app.api.main import app
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import SearchHit


class FakeTextEmbedder:
    def embed(self, texts):
        return [DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts]


class FakeClipEmbedder:
    def embed_text(self, text):
        return [0.2] * 512

    def embed_image(self, image):
        return [0.3] * 512


class FakeStoreClient:
    pass


def override_search_text_hybrid(*args, **kwargs):
    return [SearchHit(score=0.9, payload={"doc_id": "d1", "text": "命中文本"})]


def override_search_images(*args, **kwargs):
    return [SearchHit(score=0.8, payload={"doc_id": "img1", "source_path": "/a.png"})]


def override_list_chunks_for_document(*args, **kwargs):
    return [{"doc_id": "img1", "chunk_index": 0, "text": "关联OCR文本", "source_type": "ocr_image"}]


def make_client(monkeypatch):
    app.dependency_overrides[get_text_embedder_dep] = lambda: FakeTextEmbedder()
    app.dependency_overrides[get_clip_embedder_dep] = lambda: FakeClipEmbedder()
    app.dependency_overrides[get_store_client] = lambda: FakeStoreClient()
    monkeypatch.setattr("app.api.routes.search.search_text_hybrid", override_search_text_hybrid)
    monkeypatch.setattr("app.api.routes.search.search_images", override_search_images)
    monkeypatch.setattr(
        "app.api.routes.search.list_chunks_for_document", override_list_chunks_for_document
    )
    return TestClient(app)


def test_search_text_returns_text_hits(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/search/text", json={"query": "你好", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["text_hits"][0]["payload"]["doc_id"] == "d1"
    assert body["image_hits"] == []
    app.dependency_overrides.clear()


def test_search_text_with_image_modality_returns_both(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/search/text", json={"query": "你好", "top_k": 5, "modalities": ["text", "image"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["text_hits"]) == 1
    assert len(body["image_hits"]) == 1
    app.dependency_overrides.clear()


def test_search_image_returns_image_hits(monkeypatch):
    client = make_client(monkeypatch)
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/search/image", files={"image": ("q.png", buf, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["image_hits"][0]["payload"]["doc_id"] == "img1"
    app.dependency_overrides.clear()


def test_search_image_returns_related_text_hits(monkeypatch):
    client = make_client(monkeypatch)
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/search/image", files={"image": ("q.png", buf, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["related_text_hits"]) == 1
    assert body["related_text_hits"][0]["payload"]["text"] == "关联OCR文本"
    app.dependency_overrides.clear()
