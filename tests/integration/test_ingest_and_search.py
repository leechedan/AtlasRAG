from pathlib import Path

import pytest

from app.ingest.pipeline import ingest_file
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import search_text_hybrid

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
