from unittest.mock import patch

from PIL import Image, ImageDraw

from app.ingest.parsers.image_parser import parse_image


def make_text_image() -> Image.Image:
    img = Image.new("RGB", (200, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "hello", fill="black")
    return img


@patch("app.ingest.parsers.image_parser.run_ocr", return_value="hello")
@patch("app.ingest.parsers.image_parser.get_clip_embedder")
def test_parse_image_returns_ocr_text_and_thumbnail(mock_get_clip, mock_ocr, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.ingest.parsers.image_parser.settings",
        SimpleNamespace(thumbnails_dir=tmp_path),
    )
    mock_get_clip.return_value.embed_image.return_value = [0.1] * 512

    result = parse_image(make_text_image(), doc_id="img-doc-1")

    assert result.ocr_text == "hello"
    assert len(result.clip_vector) == 512
    assert result.width == 200
    assert result.height == 80
    assert (tmp_path / "img-doc-1.jpg").exists()
    assert result.thumbnail_path == str(tmp_path / "img-doc-1.jpg")
