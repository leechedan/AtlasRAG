from io import BytesIO
from unittest.mock import patch

import httpx
import pytest
import respx
from PIL import Image

from app.config import settings
from app.ingest.convert import SUPPORTED_CONVERT_EXTS, convert_file_to_markdown


def test_txt_and_md_returned_verbatim_with_no_images(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# 标题\n正文内容", encoding="utf-8")

    result = convert_file_to_markdown(f)

    assert result.markdown_text == "# 标题\n正文内容"
    assert result.images == []


def test_csv_converted_to_markdown_table(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nalice,30\nbob|x,25\n", encoding="utf-8")

    result = convert_file_to_markdown(f)

    lines = result.markdown_text.splitlines()
    assert lines[0] == "| name | age |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| alice | 30 |"
    # "|" in cell content is escaped so it doesn't break out of the cell.
    assert lines[3] == "| bob\\|x | 25 |"
    assert result.images == []


def test_csv_with_header_only_returns_header_and_separator_rows(tmp_path):
    f = tmp_path / "header_only.csv"
    f.write_text("name,age\n", encoding="utf-8")
    result = convert_file_to_markdown(f)
    assert result.markdown_text == "| name | age |\n| --- | --- |"


@patch("app.ingest.convert.run_ocr")
def test_image_referenced_by_relative_path_with_ocr_text(mock_run_ocr, tmp_path):
    mock_run_ocr.return_value = "识别出的文字"
    f = tmp_path / "photo.png"
    Image.new("RGB", (4, 4), color="green").save(f, "PNG")

    result = convert_file_to_markdown(f)

    assert result.markdown_text == "![photo.png](images/photo.png)\n\n识别出的文字"
    assert [img.filename for img in result.images] == ["photo.png"]
    assert result.images[0].image.size == (4, 4)
    mock_run_ocr.assert_called_once()


@patch("app.ingest.convert.run_ocr")
def test_image_without_ocr_text_has_no_trailing_section(mock_run_ocr, tmp_path):
    mock_run_ocr.return_value = ""
    f = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 4), color="blue").save(f, "JPEG")

    result = convert_file_to_markdown(f)

    assert result.markdown_text == "![photo.jpg](images/photo.jpg)"


@respx.mock
def test_pdf_images_kept_as_mineru_relative_placeholders(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    buf = BytesIO()
    Image.new("RGB", (5, 5), color="red").save(buf, format="JPEG")
    import base64

    data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": {
                    "input": {
                        "md_content": "# 标题\n\n正文 ![图](images/a.jpg)。",
                        "images": {"a.jpg": data_uri},
                    }
                },
            },
        )
    )

    result = convert_file_to_markdown(pdf_path)

    # Left exactly as MinerU emitted it -- the route persists the image
    # under that same filename, so this relative reference resolves once
    # the images zip is extracted alongside the .md.
    assert "images/a.jpg" in result.markdown_text
    assert "标题" in result.markdown_text
    assert [img.filename for img in result.images] == ["a.jpg"]
    assert result.images[0].image.size == (5, 5)


@respx.mock
def test_docx_uses_same_mineru_path_as_pdf(tmp_path):
    docx_path = tmp_path / "input.docx"
    docx_path.write_bytes(b"fake docx bytes")

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(
            200, json={"results": {"input": {"md_content": "office 全文", "images": {}}}}
        )
    )

    result = convert_file_to_markdown(docx_path)
    assert result.markdown_text == "office 全文"
    assert result.images == []


def test_unsupported_extension_raises_value_error(tmp_path):
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError, match="unsupported file extension"):
        convert_file_to_markdown(f)


def test_supported_convert_exts_matches_pipeline_supported_exts():
    from app.ingest.pipeline import SUPPORTED_IMAGE_EXTS, SUPPORTED_OFFICE_EXTS, SUPPORTED_TEXT_EXTS

    assert SUPPORTED_CONVERT_EXTS == (
        SUPPORTED_TEXT_EXTS | {".csv"} | SUPPORTED_IMAGE_EXTS | SUPPORTED_OFFICE_EXTS | {".pdf"}
    )
