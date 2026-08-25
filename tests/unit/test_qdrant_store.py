from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import (
    ImageRecord,
    TextChunkRecord,
    delete_document,
    ensure_collections,
    find_all_doc_ids_by_source_path,
    find_doc_id_by_source_path,
    find_image_by_doc_id,
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


def test_find_doc_id_by_source_path_none_when_missing():
    client = make_client()
    assert find_doc_id_by_source_path(client, "/no/such/path.txt") is None


def test_find_doc_id_by_source_path_returns_doc_id_after_upsert():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc6", chunk_index=0, text="text", source_path="/g.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h6",
    )
    upsert_text_chunk(client, record, vec)
    assert find_doc_id_by_source_path(client, "/g.txt") == "doc6"


def test_find_all_doc_ids_by_source_path_empty_when_missing():
    client = make_client()
    assert find_all_doc_ids_by_source_path(client, "/no/such/path.txt") == set()


def test_find_all_doc_ids_by_source_path_returns_every_stale_and_current_doc_id():
    # Regression coverage for Critical Finding 2: find_doc_id_by_source_path
    # uses limit=1, so it only ever sees one doc_id for a given source_path
    # even when several content-addressed doc_ids have accumulated under it
    # (e.g. from repeatedly editing and re-ingesting the same file path).
    # find_all_doc_ids_by_source_path must paginate through every match and
    # return the full set.
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for doc_id in ("doc-v1", "doc-v2", "doc-v3"):
        record = TextChunkRecord(
            doc_id=doc_id, chunk_index=0, text=f"text for {doc_id}", source_path="/h.txt",
            file_type="txt", page=None, source_type="normal", content_hash=f"hash-{doc_id}",
        )
        upsert_text_chunk(client, record, vec)
    # Also cover the image collection.
    upsert_image(
        client,
        ImageRecord(
            doc_id="doc-v4-image", source_path="/h.txt", thumbnail_path="/thumb/h.jpg",
            width=10, height=10, has_ocr_text=False, content_hash="hash-doc-v4-image",
        ),
        [0.2] * 512,
    )

    result = find_all_doc_ids_by_source_path(client, "/h.txt")

    assert result == {"doc-v1", "doc-v2", "doc-v3", "doc-v4-image"}


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


def test_find_image_by_doc_id_returns_payload_after_upsert():
    client = make_client()
    record = ImageRecord(
        doc_id="img-thumb", source_path="/x.png", thumbnail_path="/thumb/x.jpg",
        width=50, height=60, has_ocr_text=False, content_hash="hx",
    )
    upsert_image(client, record, [0.2] * 512)

    payload = find_image_by_doc_id(client, "img-thumb")

    assert payload["thumbnail_path"] == "/thumb/x.jpg"
    assert payload["doc_id"] == "img-thumb"


def test_find_image_by_doc_id_none_when_missing():
    client = make_client()
    assert find_image_by_doc_id(client, "no-such-doc") is None


def test_upsert_text_chunk_writes_origin_and_created_at():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc-origin", chunk_index=0, text="text", source_path="/o.txt",
        file_type="txt", page=None, source_type="normal", content_hash="ho",
        origin="upload", created_at="2026-08-25T00:00:00+00:00",
    )
    upsert_text_chunk(client, record, vec)

    chunks = list_chunks_for_document(client, "doc-origin")
    assert chunks[0]["origin"] == "upload"
    assert chunks[0]["created_at"] == "2026-08-25T00:00:00+00:00"


def test_upsert_image_writes_origin_and_created_at():
    client = make_client()
    record = ImageRecord(
        doc_id="img-origin", source_path="/o.png", thumbnail_path="/thumb/o.jpg",
        width=10, height=10, has_ocr_text=False, content_hash="ho2",
        origin="watch", created_at="2026-08-25T00:00:00+00:00",
    )
    upsert_image(client, record, [0.2] * 512)

    payload = find_image_by_doc_id(client, "img-origin")
    assert payload["origin"] == "watch"
    assert payload["created_at"] == "2026-08-25T00:00:00+00:00"


def test_compute_chunk_id_is_deterministic_and_public():
    from app.store.qdrant_client import compute_chunk_id
    assert compute_chunk_id("d1", 0) == compute_chunk_id("d1", 0)
    assert compute_chunk_id("d1", 0) != compute_chunk_id("d1", 1)


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
