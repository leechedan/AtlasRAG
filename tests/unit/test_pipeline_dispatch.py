from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app.ingest.parsers.pdf_parser import ParsedPdfDocument, ParsedPdfImage
from app.ingest.pipeline import ChunkOptions, ingest_file
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import ensure_collections, find_image_by_doc_id, get_qdrant_client, list_chunks_for_document


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
    has_code_block instead of leaving them at their defaults.

    两节正文都写得比 settings.chunk_min_chars 长：这个用例测的是"按标题
    切分 + 每片带 heading 元数据"的接线，正文太短的话会被跨 section 小片
    合并成一片，测不到分节。合并本身另有专门用例覆盖。"""
    f = tmp_path / "doc.md"
    f.write_text(
        "# 标题\n\n"
        "## 一、简介\n\n"
        + ("这是简介内容。" * 40) + "\n\n"
        "## 二、数据表\n\n"
        + ("以下是本节的数据表。" * 30) + "\n\n"
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


@patch("app.ingest.pipeline.get_clip_embedder")
@patch("app.ingest.pipeline.get_text_embedder")
@patch("app.ingest.pipeline.parse_pdf_to_markdown")
def test_ingest_pdf_writes_structured_chunks_and_extracted_images(
    mock_parse_pdf, mock_get_text_embedder, mock_get_clip_embedder, tmp_path, monkeypatch
):
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = (
        "# 第一章 概述\n\n"
        "这是正文，配一张图 ![示例](images/pic1.png) 结尾。\n\n"
        "## 1.1 数据表\n\n"
        "| A | B |\n| - | - |\n| 1 | 2 |\n\n"
        "另配一张图 ![示例2](images/pic2.png) 补充说明。\n"
    )
    fake_images = [
        ParsedPdfImage(filename="pic1.png", image=Image.new("RGB", (20, 20), color="red")),
        ParsedPdfImage(filename="pic2.png", image=Image.new("RGB", (30, 30), color="blue")),
    ]
    mock_parse_pdf.return_value = ParsedPdfDocument(markdown_text=markdown_text, images=fake_images)

    mock_get_text_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]
    mock_get_clip_embedder.return_value.embed_image.return_value = [0.2] * 512

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    client = make_client()
    result = ingest_file(pdf_path, source_ref=str(pdf_path), client=client)

    assert result.status == "ingested"
    mock_parse_pdf.assert_called_once_with(pdf_path)

    stored_chunks = list_chunks_for_document(client, result.doc_id)
    assert any(c["heading_path"] for c in stored_chunks)
    assert any(c["has_table"] for c in stored_chunks)
    assert all(c["page"] is None for c in stored_chunks)
    assert all(c["source_type"] == "normal" for c in stored_chunks)

    joined_text = " ".join(c["text"] for c in stored_chunks)
    assert "images/pic1.png" not in joined_text
    assert f"/documents/{result.doc_id}:img1/image" in joined_text
    assert f"/documents/{result.doc_id}:img2/image" in joined_text

    img1 = find_image_by_doc_id(client, f"{result.doc_id}:img1")
    img2 = find_image_by_doc_id(client, f"{result.doc_id}:img2")
    assert img1 is not None and img1["width"] == 20 and img1["has_ocr_text"] is False
    assert img2 is not None and img2["width"] == 30
    assert img1["image_path"] is not None and Path(img1["image_path"]).exists()


@pytest.mark.parametrize("ext", [".docx", ".pptx", ".xlsx"])
@patch("app.ingest.pipeline.get_clip_embedder")
@patch("app.ingest.pipeline.get_text_embedder")
@patch("app.ingest.pipeline.parse_pdf_to_markdown")
def test_ingest_office_doc_routes_through_mineru_with_structured_chunks(
    mock_parse_pdf, mock_get_text_embedder, mock_get_clip_embedder, tmp_path, monkeypatch, ext
):
    """docx/pptx/xlsx must go through the same MinerU call as PDF
    (parse_pdf_to_markdown), be tagged with their own file_type (not "pdf"),
    and use the heading/table-aware structured chunker -- MinerU returns
    markdown for these formats too, same as PDF."""
    from app.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = "# 标题\n\n## 一、简介\n\n正文内容。\n"
    mock_parse_pdf.return_value = ParsedPdfDocument(markdown_text=markdown_text, images=[])

    mock_get_text_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    doc_path = tmp_path / f"report{ext}"
    doc_path.write_bytes(b"fake office bytes")

    client = make_client()
    result = ingest_file(doc_path, source_ref=str(doc_path), client=client)

    assert result.status == "ingested"
    mock_parse_pdf.assert_called_once_with(doc_path)

    stored_chunks = list_chunks_for_document(client, result.doc_id)
    assert stored_chunks
    assert all(c["file_type"] == ext.lstrip(".") for c in stored_chunks)
    assert any(c["heading_path"] for c in stored_chunks)


def test_ingest_legacy_office_extension_is_unsupported(tmp_path):
    """MinerU's office backend only understands OOXML (.docx/.pptx/.xlsx) --
    the legacy binary .doc must stay unsupported, not silently accepted."""
    f = tmp_path / "old.doc"
    f.write_bytes(b"not ooxml")

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "failed"
    assert result.error is not None
