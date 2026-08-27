import base64
from io import BytesIO

import httpx
import pytest
import respx
from PIL import Image

from app.config import settings
from app.ingest.parsers.pdf_parser import parse_pdf_to_markdown


def _image_data_uri(color: str, size: tuple[int, int]) -> str:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@respx.mock
def test_parse_pdf_to_markdown_returns_text_and_images(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "pipeline",
                "version": "3.4.5",
                "results": {
                    "input": {
                        "md_content": "# 标题\n\n正文 ![图](images/a.jpg)。",
                        "images": {
                            "a.jpg": _image_data_uri("red", (5, 5)),
                            "b.jpg": _image_data_uri("blue", (6, 6)),
                        },
                    }
                },
            },
        )
    )

    result = parse_pdf_to_markdown(pdf_path)

    assert "标题" in result.markdown_text
    assert [img.filename for img in result.images] == ["a.jpg", "b.jpg"]
    assert result.images[0].image.size == (5, 5)
    assert result.images[1].image.size == (6, 6)


@respx.mock
def test_parse_pdf_to_markdown_strips_sub_sup_tags(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "pipeline",
                "version": "3.4.5",
                "results": {
                    "input": {
                        "md_content": "分子式 H<sub>2</sub>O，注释<sup class=\"ref\">1</sup>。",
                        "images": {},
                    }
                },
            },
        )
    )

    result = parse_pdf_to_markdown(pdf_path)

    assert result.markdown_text == "分子式 H2O，注释1。"


@respx.mock
def test_parse_pdf_to_markdown_raises_on_non_200(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(409, text="model weights missing")
    )

    with pytest.raises(RuntimeError, match="model weights missing"):
        parse_pdf_to_markdown(pdf_path)


@respx.mock
def test_parse_pdf_to_markdown_raises_when_no_markdown_produced(tmp_path):
    pdf_path = tmp_path / "input.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    respx.post(f"{settings.mineru_api_url}/file_parse").mock(
        return_value=httpx.Response(
            200,
            json={
                "backend": "pipeline",
                "version": "3.4.5",
                "results": {"input": {"md_content": "", "images": {}}},
            },
        )
    )

    with pytest.raises(RuntimeError, match="no markdown output"):
        parse_pdf_to_markdown(pdf_path)
