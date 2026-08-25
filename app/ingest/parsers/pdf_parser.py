from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from app.ingest.parsers.image_parser import run_ocr


@dataclass
class ParsedPdfPage:
    page_number: int
    text: str
    is_scanned: bool
    page_image: Image.Image | None


def parse_pdf(path: Path, text_threshold: int = 20) -> list[ParsedPdfPage]:
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) >= text_threshold:
                pages.append(ParsedPdfPage(page_number=i + 1, text=text, is_scanned=False, page_image=None))
                continue

            pixmap = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            ocr_text = run_ocr(image)
            pages.append(
                ParsedPdfPage(page_number=i + 1, text=ocr_text, is_scanned=True, page_image=image)
            )
    return pages
