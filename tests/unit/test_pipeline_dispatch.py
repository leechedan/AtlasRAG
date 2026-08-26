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
def test_ingest_editing_file_removes_stale_previous_version(mock_get_embedder, tmp_path):
    """Regression test for Critical Finding 2: doc_id is content-addressed
    (make_doc_id(source_ref, content_hash)), so when a watched/re-uploaded
    file's content changes, the new content gets a *new* doc_id and the old
    doc_id's points were never removed -- the index grew unbounded and
    search kept returning stale content indefinitely. Ingesting v1, then v2
    (different content, same path/source_ref), must leave only v2's chunks
    in Qdrant."""
    f = tmp_path / "note.txt"
    f.write_text("version one content " * 50, encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    first = ingest_file(f, source_ref=str(f), client=client)
    assert first.status == "ingested"

    f.write_text("version two totally different content " * 50, encoding="utf-8")
    second = ingest_file(f, source_ref=str(f), client=client)
    assert second.status == "ingested"
    assert second.doc_id != first.doc_id

    assert list_chunks_for_document(client, first.doc_id) == []
    stored = list_chunks_for_document(client, second.doc_id)
    assert len(stored) == second.chunks_written
    assert all("version two" in c["text"] for c in stored)


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


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_upload_source_ref_sets_origin_upload(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=f"upload:2026-08-25T00:00:00+00:00:{f.name}", client=client)

    chunks = list_chunks_for_document(client, result.doc_id)
    assert chunks[0]["origin"] == "upload"
    assert chunks[0]["created_at"] is not None


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_plain_path_source_ref_sets_origin_watch(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    chunks = list_chunks_for_document(client, result.doc_id)
    assert chunks[0]["origin"] == "watch"


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_md_file_uses_structured_chunker_with_heading_metadata(mock_get_embedder, tmp_path):
    """Regression coverage for wiring chunk_markdown_structured() into the
    real ingestion path: a .md file with two H2 sections must be split by
    heading (not by the old char-level chunker's arbitrary 1000-char
    windows), and each stored chunk must carry heading_path/has_table/
    has_code_block instead of leaving them at their defaults."""
    f = tmp_path / "doc.md"
    f.write_text(
        "# 标题\n\n"
        "## 一、简介\n\n"
        "这是简介内容。\n\n"
        "## 二、数据表\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "ingested"
    assert result.chunks_written == 2

    chunks = list_chunks_for_document(client, result.doc_id)
    heading_paths = [c["heading_path"] for c in chunks]
    assert "标题 > 一、简介" in heading_paths
    assert "标题 > 二、数据表" in heading_paths
    table_chunk = next(c for c in chunks if c["heading_path"] == "标题 > 二、数据表")
    assert table_chunk["has_table"] is True
    assert table_chunk["has_code_block"] is False
    assert "| a | b |" in table_chunk["text"]


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_txt_file_still_leaves_heading_metadata_at_defaults(mock_get_embedder, tmp_path):
    """Non-Markdown files must keep using the old char-level chunker
    unchanged -- only file_type == "md" switches to the structured one."""
    f = tmp_path / "note.txt"
    f.write_text("plain text content, no markdown structure at all.", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    chunks = list_chunks_for_document(client, result.doc_id)
    assert chunks[0]["heading_path"] is None
    assert chunks[0]["has_table"] is False
    assert chunks[0]["has_code_block"] is False
