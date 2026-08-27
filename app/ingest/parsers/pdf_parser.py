"""MinerU HTTP 客户端封装：把整份文档转成一份 Markdown 全文 + 内嵌图片列表。

MinerU 的 /file_parse 接口本身按文件扩展名/内容分流处理，不只是 PDF —— 调用方
（app.ingest.pipeline._ingest_mineru_document）也用这同一个函数处理
docx/pptx/xlsx（见 app.ingest.pipeline.SUPPORTED_OFFICE_EXTS），MinerU 对这些
office 格式同样会返回 markdown 全文，函数名保留 parse_pdf_to_markdown 是因为
PDF 摄入路径最先落地，实际输入不限于 PDF。

调用常驻的 `mineru-api` 服务（docker-compose 里的 `mineru` 服务，命令为
`mineru-api`，模型只在那边加载一次），而不是像之前那样每次都起一个新的
`mineru` CLI 子进程重新加载模型。解析失败（HTTP 非 200、响应里没有
markdown 内容）直接抛 RuntimeError，交给调用方（app.ingest.pipeline.ingest_file）
已有的统一 try/except 处理，本模块不做二次兜底。
"""

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from app.config import settings

# MinerU 对化学式/脚注等场景会在 md_content 里插入原生 <sub>/<sup> 标签
# （如 "H<sub>2</sub>O"）。这些标签本身不是正文语义的一部分，留在 chunk 里
# 会污染 embedding 和 sparse 关键词匹配，所以在解析边界这里就去标签、保留
# 标签内的字符（"H<sub>2</sub>O" -> "H2O"）。
_SUB_SUP_TAG_RE = re.compile(r"</?(?:sub|sup)(?:\s[^>]*)?>", re.IGNORECASE)


def _strip_sub_sup_tags(markdown_text: str) -> str:
    return _SUB_SUP_TAG_RE.sub("", markdown_text)


@dataclass
class ParsedPdfImage:
    filename: str
    image: Image.Image


@dataclass
class ParsedPdfDocument:
    markdown_text: str
    images: list[ParsedPdfImage]


def _decode_data_uri_image(filename: str, data_uri: str) -> ParsedPdfImage:
    _, _, encoded = data_uri.partition(",")
    raw_bytes = base64.b64decode(encoded)
    return ParsedPdfImage(filename=filename, image=Image.open(BytesIO(raw_bytes)).convert("RGB"))


def parse_pdf_to_markdown(path: Path) -> ParsedPdfDocument:
    with path.open("rb") as fh:
        response = httpx.post(
            f"{settings.mineru_api_url}/file_parse",
            files={"files": (path.name, fh, "application/pdf")},
            data={
                "backend": settings.mineru_backend,
                "lang_list": settings.ocr_lang,
                "parse_method": "auto",
                "return_md": "true",
                "return_images": "true",
            },
            timeout=settings.mineru_timeout_seconds,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"mineru-api failed for {path} (status {response.status_code}): {response.text[-2000:]}"
        )

    results = response.json().get("results", {})
    if not results:
        raise RuntimeError(f"mineru-api produced no results for {path}")
    doc_result = next(iter(results.values()))

    markdown_text = doc_result.get("md_content")
    if not markdown_text:
        raise RuntimeError(f"mineru-api produced no markdown output for {path}")
    markdown_text = _strip_sub_sup_tags(markdown_text)

    images = [
        _decode_data_uri_image(filename, data_uri)
        for filename, data_uri in sorted(doc_result.get("images", {}).items())
    ]

    return ParsedPdfDocument(markdown_text=markdown_text, images=images)
