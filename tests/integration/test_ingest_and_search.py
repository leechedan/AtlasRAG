from pathlib import Path

import pytest

from app.ingest.pipeline import ingest_file
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import list_all_documents, scroll_chunks, search_text_hybrid

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_txt_and_search_finds_it(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    assert result.status == "ingested"

    query_vector = get_text_embedder().embed(["GB10 是什么芯片"])[0]
    hits = search_text_hybrid(qdrant_test_client, query_vector, top_k=3)

    assert any("Grace Blackwell" in h.payload["text"] for h in hits)


def test_ingest_csv_and_search_finds_row(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.csv", source_ref=str(FIXTURES / "sample.csv"), client=qdrant_test_client)
    assert result.status == "ingested"

    query_vector = get_text_embedder().embed(["谁是产品经理"])[0]
    hits = search_text_hybrid(qdrant_test_client, query_vector, top_k=3)

    assert any("李四" in h.payload["text"] for h in hits)


def test_list_documents_after_real_ingest(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    assert result.status == "ingested"

    docs = list_all_documents(qdrant_test_client)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == result.doc_id
    assert doc["chunk_count"] == result.chunks_written
    assert doc["origin"] == "watch"
    assert doc["source_types"] == ["normal"]


def test_scroll_chunks_paginates_across_multiple_real_ingests(qdrant_test_client):
    txt_result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    csv_result = ingest_file(FIXTURES / "sample.csv", source_ref=str(FIXTURES / "sample.csv"), client=qdrant_test_client)
    total_expected = txt_result.chunks_written + csv_result.chunks_written

    seen_chunk_ids = set()
    cursor = None
    while True:
        chunks, cursor = scroll_chunks(qdrant_test_client, cursor=cursor, limit=2)
        seen_chunk_ids.update(c["chunk_id"] for c in chunks)
        if cursor is None:
            break

    assert len(seen_chunk_ids) == total_expected
