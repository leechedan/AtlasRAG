from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import (
    ImageRecord,
    TextChunkRecord,
    delete_document,
    ensure_collections,
    get_existing_content_hash,
    get_qdrant_client,
    get_stats,
    list_chunks_for_document,
    search_images,
    search_text_hybrid,
    upsert_image,
    upsert_text_chunk,
)


def make_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client


def test_upsert_and_search_text_chunk():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5, 2: 0.3})
    record = TextChunkRecord(
        doc_id="doc1", chunk_index=0, text="你好世界", source_path="/a.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h1",
    )
    upsert_text_chunk(client, record, vec)

    hits = search_text_hybrid(client, vec, top_k=5)
    assert len(hits) == 1
    assert hits[0].payload["doc_id"] == "doc1"
    assert hits[0].payload["text"] == "你好世界"


def test_upsert_and_search_image():
    client = make_client()
    vector = [0.2] * 512
    record = ImageRecord(
        doc_id="img1", source_path="/a.png", thumbnail_path="/thumb/a.jpg",
        width=100, height=100, has_ocr_text=False, content_hash="h2",
    )
    upsert_image(client, record, vector)

    hits = search_images(client, vector, top_k=5)
    assert len(hits) == 1
    assert hits[0].payload["doc_id"] == "img1"


def test_get_existing_content_hash_none_when_missing():
    client = make_client()
    assert get_existing_content_hash(client, "unknown-doc") is None


def test_get_existing_content_hash_returns_hash_after_upsert():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc2", chunk_index=0, text="text", source_path="/b.txt",
        file_type="txt", page=None, source_type="normal", content_hash="hash-b",
    )
    upsert_text_chunk(client, record, vec)
    assert get_existing_content_hash(client, "doc2") == "hash-b"


def test_delete_document_removes_chunks():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc3", chunk_index=0, text="text", source_path="/c.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h3",
    )
    upsert_text_chunk(client, record, vec)
    delete_document(client, "doc3")
    assert get_existing_content_hash(client, "doc3") is None


def test_list_chunks_for_document_returns_ordered_payloads():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for idx in (2, 0, 1):
        record = TextChunkRecord(
            doc_id="doc5", chunk_index=idx, text=f"chunk-{idx}", source_path="/e.txt",
            file_type="txt", page=None, source_type="normal", content_hash="h5",
        )
        upsert_text_chunk(client, record, vec)

    chunks = list_chunks_for_document(client, "doc5")

    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
    assert [c["text"] for c in chunks] == ["chunk-0", "chunk-1", "chunk-2"]


def test_list_chunks_for_document_empty_when_missing():
    client = make_client()
    assert list_chunks_for_document(client, "unknown-doc") == []


def test_get_stats_reports_counts():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc4", chunk_index=0, text="text", source_path="/d.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h4",
    )
    upsert_text_chunk(client, record, vec)
    stats = get_stats(client)
    assert stats["text_chunks_count"] == 1
    assert stats["images_count"] == 0


def test_upsert_text_chunk_overwrites_same_doc_id_and_chunk_index():
    # 后续入库管线（Task 12+）依赖这个前提：同一个 (doc_id, chunk_index) 重复 upsert
    # 要覆盖旧记录，不能产生重复的点——这是增量重新入库不产生脏数据的基础。
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record_v1 = TextChunkRecord(
        doc_id="dupdoc", chunk_index=0, text="old text", source_path="/f.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h-old",
    )
    upsert_text_chunk(client, record_v1, vec)

    record_v2 = TextChunkRecord(
        doc_id="dupdoc", chunk_index=0, text="new text", source_path="/f.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h-new",
    )
    upsert_text_chunk(client, record_v2, vec)

    chunks = list_chunks_for_document(client, "dupdoc")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "new text"
    assert get_existing_content_hash(client, "dupdoc") == "h-new"
