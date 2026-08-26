import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from app.ingest.parsers.pdf_parser import parse_pdf_to_markdown


def _completed(returncode, stderr=""):
    return subprocess.CompletedProcess(args=["mineru"], returncode=returncode, stdout="", stderr=stderr)


@patch("app.ingest.parsers.pdf_parser.subprocess.run")
@patch("app.ingest.parsers.pdf_parser.tempfile.mkdtemp")
def test_parse_pdf_to_markdown_returns_text_and_images(mock_mkdtemp, mock_run, tmp_path):
    out_dir = tmp_path / "out"
    nested = out_dir / "doc" / "auto"
    nested.mkdir(parents=True)
    (nested / "doc.md").write_text("# 标题\n\n正文 ![图](images/a.jpg)。", encoding="utf-8")
    images_dir = nested / "images"
    images_dir.mkdir()
    Image.new("RGB", (5, 5), color="red").save(images_dir / "a.jpg")
    Image.new("RGB", (6, 6), color="blue").save(images_dir / "b.jpg")

    mock_mkdtemp.return_value = str(out_dir)
    mock_run.return_value = _completed(0)

    result = parse_pdf_to_markdown(Path("/fake/input.pdf"))

    assert "标题" in result.markdown_text
    assert [img.filename for img in result.images] == ["a.jpg", "b.jpg"]
    assert result.images[0].image.size == (5, 5)
    assert result.images[1].image.size == (6, 6)
    # 处理完必须清理临时目录，不能把 MinerU 的中间产物留在磁盘上。
    assert not out_dir.exists()


@patch("app.ingest.parsers.pdf_parser.subprocess.run")
@patch("app.ingest.parsers.pdf_parser.tempfile.mkdtemp")
def test_parse_pdf_to_markdown_raises_on_nonzero_exit(mock_mkdtemp, mock_run, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_mkdtemp.return_value = str(out_dir)
    mock_run.return_value = _completed(1, stderr="model weights missing")

    with pytest.raises(RuntimeError, match="model weights missing"):
        parse_pdf_to_markdown(Path("/fake/input.pdf"))

    assert not out_dir.exists()


@patch("app.ingest.parsers.pdf_parser.subprocess.run")
@patch("app.ingest.parsers.pdf_parser.tempfile.mkdtemp")
def test_parse_pdf_to_markdown_raises_when_no_markdown_produced(mock_mkdtemp, mock_run, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mock_mkdtemp.return_value = str(out_dir)
    mock_run.return_value = _completed(0)

    with pytest.raises(RuntimeError, match="no markdown output"):
        parse_pdf_to_markdown(Path("/fake/input.pdf"))

    assert not out_dir.exists()
