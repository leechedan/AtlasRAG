"""MinerU CLI 封装：把整份 PDF 转成一份 Markdown 全文 + 内嵌图片列表。

调用外部 `mineru` 命令（pip 包 `mineru[pipeline]` 提供），解析失败（子进程
非零退出码、或者输出目录里找不到生成的 .md 文件）直接抛 RuntimeError，交给
调用方（app.ingest.pipeline.ingest_file）已有的统一 try/except 处理，本模块
不做二次兜底。

返回的图片在函数返回前就已经用 PIL 完整解码进内存（Image.convert("RGB")
会强制加载像素数据），所以调用方可以在这个函数返回、临时目录已被清理之后
继续安全地使用这些 Image 对象。
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import settings


@dataclass
class ParsedPdfImage:
    filename: str
    image: Image.Image


@dataclass
class ParsedPdfDocument:
    markdown_text: str
    images: list[ParsedPdfImage]


def parse_pdf_to_markdown(path: Path) -> ParsedPdfDocument:
    tmp_dir = Path(tempfile.mkdtemp(prefix="mineru_"))
    try:
        result = subprocess.run(
            [
                "mineru",
                "-p", str(path),
                "-o", str(tmp_dir),
                "-b", settings.mineru_backend,
                "-l", settings.ocr_lang,
                "-m", "auto",
            ],
            capture_output=True,
            text=True,
            timeout=settings.mineru_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mineru failed for {path} (exit {result.returncode}): {result.stderr[-2000:]}"
            )

        md_files = sorted(tmp_dir.rglob("*.md"))
        if not md_files:
            raise RuntimeError(f"mineru produced no markdown output for {path}")
        md_path = md_files[0]
        markdown_text = md_path.read_text(encoding="utf-8")

        images_dir = md_path.parent / "images"
        images: list[ParsedPdfImage] = []
        if images_dir.is_dir():
            for image_path in sorted(images_dir.iterdir()):
                images.append(
                    ParsedPdfImage(
                        filename=image_path.name,
                        image=Image.open(image_path).convert("RGB"),
                    )
                )

        return ParsedPdfDocument(markdown_text=markdown_text, images=images)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
