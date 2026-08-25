from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from app.config import settings
from app.models.clip_model import get_clip_embedder


@lru_cache(maxsize=1)
def _get_ocr_engine():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang=settings.ocr_lang, use_gpu=False, show_log=False)


def run_ocr(image: Image.Image) -> str:
    import numpy as np

    engine = _get_ocr_engine()
    result = engine.ocr(np.array(image), cls=True)
    if not result or result[0] is None:
        return ""
    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)


@dataclass
class ParsedImage:
    ocr_text: str
    clip_vector: list[float]
    thumbnail_path: str
    width: int
    height: int


def parse_image(image: Image.Image, doc_id: str) -> ParsedImage:
    width, height = image.size
    ocr_text = run_ocr(image)
    clip_vector = get_clip_embedder().embed_image(image)

    thumbnail = image.copy()
    thumbnail.thumbnail((256, 256))
    thumbnail_path = settings.thumbnails_dir / f"{doc_id}.jpg"
    thumbnail.convert("RGB").save(thumbnail_path, "JPEG")

    return ParsedImage(
        ocr_text=ocr_text,
        clip_vector=clip_vector,
        thumbnail_path=str(thumbnail_path),
        width=width,
        height=height,
    )
