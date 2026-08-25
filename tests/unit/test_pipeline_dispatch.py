from pathlib import Path
from unittest.mock import patch

import fitz

from app.ingest.pipeline import ChunkOptions, ingest_file
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import ensure_collections, get_qdrant_client, list_chunks_for_document


def make_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client


def make_multipage_text_pdf(path, page_texts):
    """Build a PDF where every page carries real (non-OCR) text-layer
    content, each page's text distinct so collisions across pages would be
    detectable. Uses the china-s builtin font like test_pdf_parser.py's
    make_test_pdf, since the default helv font renders CJK as placeholder
    glyphs that get_text() cannot extract."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(page.rect, text, fontname="china-s", fontsize=11)
    doc.save(path)
    doc.close()


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


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_pdf_multipage_chunks_do_not_collide(mock_get_embedder, tmp_path):
    """Regression test for the chunk_index collision bug: _write_text_chunks
    used to derive chunk_index from a per-call enumerate() that reset to 0
    on every page, so every page's chunks landed on the same Qdrant point
    ids as page 1's and silently overwrote each other. With the fix
    (threading a running start_index across pages), every chunk from every
    page must end up as a distinct point in Qdrant."""
    pdf_path = tmp_path / "multipage.pdf"
    page_texts = [
        "第一页的可提取文字层内容。" * 15,
        "第二页的可提取文字层内容，与第一页完全不同。" * 15,
        "第三页的可提取文字层内容，同样与前两页不同。" * 15,
    ]
    make_multipage_text_pdf(pdf_path, page_texts)

    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(
        pdf_path,
        source_ref=str(pdf_path),
        client=client,
        chunk_options=ChunkOptions(chunk_size=50, chunk_overlap=0),
    )

    assert result.status == "ingested"
    # Sanity check the fixture actually exercises multiple chunks per page,
    # not just one chunk-per-page (which wouldn't have caught the bug).
    assert result.chunks_written > len(page_texts) * 2

    stored = list_chunks_for_document(client, result.doc_id)
    # This is the assertion that catches the collision bug: if later pages'
    # chunks overwrote earlier pages' points at reused chunk_index values,
    # the number of distinct points actually stored would be less than
    # chunks_written.
    assert len(stored) == result.chunks_written

    stored_indices = [p["chunk_index"] for p in stored]
    assert len(stored_indices) == len(set(stored_indices))
    assert sorted(stored_indices) == list(range(result.chunks_written))

    pages_represented = {p["page"] for p in stored}
    assert pages_represented == {1, 2, 3}
