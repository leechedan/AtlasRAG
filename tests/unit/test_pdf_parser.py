from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw

from app.ingest.parsers.pdf_parser import parse_pdf


def make_test_pdf(path):
    doc = fitz.open()

    page1 = doc.new_page()
    # Default PyMuPDF font (helv) has no CJK glyphs and silently renders
    # unsupported characters as placeholders, so use a builtin CJK font.
    # The sentence must also clear parse_pdf's default text_threshold=20.
    page1.insert_text(
        (72, 72), "这是可提取的文字层，用于测试PDF解析器的文字提取功能。", fontname="china-s"
    )

    page2 = doc.new_page()
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "scanned text", fill="black")
    img_path = path.parent / "scan.png"
    img.save(img_path)
    rect = fitz.Rect(0, 0, page2.rect.width, page2.rect.height)
    page2.insert_image(rect, filename=str(img_path))

    doc.save(path)
    doc.close()


@patch("app.ingest.parsers.pdf_parser.run_ocr", return_value="scanned text")
def test_parse_pdf_extracts_text_layer_and_ocr_fallback(mock_ocr, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    make_test_pdf(pdf_path)

    pages = parse_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0].is_scanned is False
    assert "可提取的文字层" in pages[0].text

    assert pages[1].is_scanned is True
    assert pages[1].text == "scanned text"
    assert pages[1].page_image is not None
    mock_ocr.assert_called_once()
