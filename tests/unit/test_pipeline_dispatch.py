from pathlib import Path
from unittest.mock import patch

from app.ingest.pipeline import ingest_file
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import ensure_collections, get_qdrant_client


def make_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_txt_file_writes_chunks(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world " * 100, encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "ingested"
    assert result.chunks_written > 0


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_skips_unchanged_file(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("stable content", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    first = ingest_file(f, source_ref=str(f), client=client)
    second = ingest_file(f, source_ref=str(f), client=client)

    assert first.status == "ingested"
    assert second.status == "skipped_unchanged"
    assert first.doc_id == second.doc_id


def test_ingest_unsupported_extension_fails_gracefully(tmp_path):
    f = tmp_path / "note.xyz"
    f.write_text("data", encoding="utf-8")

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "failed"
    assert result.error is not None


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_honors_custom_chunk_options(mock_get_embedder, tmp_path):
    from app.ingest.pipeline import ChunkOptions

    f = tmp_path / "note.txt"
    f.write_text("word " * 400, encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    small_chunks = ingest_file(f, source_ref=str(f) + ":a", client=client, chunk_options=ChunkOptions(chunk_size=50, chunk_overlap=0))
    large_chunks = ingest_file(f, source_ref=str(f) + ":b", client=client, chunk_options=ChunkOptions(chunk_size=5000, chunk_overlap=0))

    assert small_chunks.chunks_written > large_chunks.chunks_written
    assert large_chunks.chunks_written == 1
