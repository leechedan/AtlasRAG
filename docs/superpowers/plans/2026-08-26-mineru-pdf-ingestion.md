# MinerU PDF 摄取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 MinerU 完全替换现有 fitz+PaddleOCR 的 PDF 摄取路径，让 PDF 正文复用结构化 Markdown 切片器（`heading_path`/`has_table`/`has_code_block`），并让 MinerU 抽取的内嵌图片持久化、CLIP 向量化，且在正文里保留可访问的链接。

**Architecture:** `app/ingest/parsers/pdf_parser.py` 重写为一个薄的 MinerU CLI 子进程封装（`parse_pdf_to_markdown`），一次性返回整篇 Markdown 文本和内存中的图片对象列表；`app/ingest/pipeline.py::_ingest_pdf_file` 用这份数据驱动：先把每张图片持久化+CLIP 向量化并把正文里对应的相对路径引用改写成 `/documents/{doc_id}:img{n}/image`，再把改写后的正文交给已有的 `chunk_markdown_structured()` 切片、写入 Qdrant。存储层（`ImageRecord`）和 API 层各加一个小字段/接口配合图片持久化。

**Tech Stack:** Python 3.12, FastAPI, Qdrant, RQ, MinerU CLI（`mineru[pipeline]` pip 包，pipeline backend），Pillow，pytest + unittest.mock。

## Global Constraints

- 完全替换现有 PDF 路径，不保留 fitz+PaddleOCR 双引擎/回退开关；解析失败直接让异常沿 `ingest_file` 既有的 `try/except Exception` 冒泡，记成 `IngestResult(status="failed")`。
- PDF 文本 chunk 的 `page` 字段统一为 `None`；`source_type` 统一为 `"normal"`。
- MinerU 抽取的内嵌图片只做 CLIP 向量化，不跑 OCR（不调用 `parse_image()`）。
- 图片原图必须持久化（不只是缩略图），Markdown 正文里的图片引用必须改写成可访问链接，不能直接删除。
- 模型源用 MinerU 默认的 `huggingface`，复用 worker 现有的 `HF_HOME` 挂载，本轮不写自动下载/预热逻辑。
- MinerU backend 用 `pipeline`（不用默认的 `hybrid-engine`），语言参数复用现有 `settings.ocr_lang`。
- `mineru[pipeline]` 锁定版本 `3.4.5`；移除 `pymupdf` 依赖。
- 不新增真实调用 MinerU（下载权重、GPU 推理）的自动化测试；所有单元测试通过 mock `subprocess.run` 验证。

参考设计文档：`docs/superpowers/specs/2026-08-26-mineru-pdf-ingestion-design.md`

---

### Task 1: 配置项与依赖

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `Dockerfile`
- Test: `tests/unit/test_config.py`（新建）

**Interfaces:**
- Produces: `settings.mineru_backend: str`（默认 `"pipeline"`）、`settings.mineru_timeout_seconds: int`（默认 `600`）、`settings.extracted_images_dir: Path`（属性，`data_dir / "extracted_images"`，启动时自动创建目录）。后续任务都依赖这三个。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_config.py`：

```python
from app.config import settings


def test_mineru_settings_have_expected_defaults():
    assert settings.mineru_backend == "pipeline"
    assert settings.mineru_timeout_seconds == 600


def test_extracted_images_dir_is_under_data_dir_and_created():
    assert settings.extracted_images_dir == settings.data_dir / "extracted_images"
    assert settings.extracted_images_dir.is_dir()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL，`AttributeError: 'Settings' object has no attribute 'mineru_backend'`

- [ ] **Step 3: 实现配置项**

编辑 `app/config.py`，在 `ocr_lang: str = "ch"` 之后加两个字段：

```python
    ocr_lang: str = "ch"
    mineru_backend: str = "pipeline"
    mineru_timeout_seconds: int = 600
    watch_reconcile_interval_seconds: int = 3600
```

在 `thumbnails_dir` 属性之后加一个新属性：

```python
    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def extracted_images_dir(self) -> Path:
        return self.data_dir / "extracted_images"
```

文件末尾的启动期建目录逻辑加一行：

```python
settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 更新 `.env.example`**

在 `OCR_LANG=ch` 那一行之后加：

```
MINERU_BACKEND=pipeline
MINERU_TIMEOUT_SECONDS=600
```

- [ ] **Step 6: 更新依赖清单**

编辑 `requirements.txt`：删除 `pymupdf==1.25.1` 这一行，在 `paddlepaddle==2.6.2` 之后加一行：

```
mineru[pipeline]==3.4.5
```

- [ ] **Step 7: 更新 Dockerfile 注释**

`Dockerfile` 里 `RUN pip install --no-cache-dir -r requirements.txt` 前加一行注释，说明 `mineru[pipeline]` 依赖 `libgl1`/`libglib2.0-0`（已在同一 Dockerfile 靠前的 apt 安装步骤里存在，无需新增系统包）：

```dockerfile
# mineru[pipeline] 依赖的 opencv 需要 libgl1/libglib2.0-0，上面 apt 安装步骤已包含，
# 若未来升级 mineru 版本发现新的系统依赖缺失，在这里补充 apt 包。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

- [ ] **Step 8: 提交**

```bash
git add app/config.py .env.example requirements.txt Dockerfile tests/unit/test_config.py
git commit -m "$(cat <<'EOF'
feat: add MinerU config settings and swap pymupdf for mineru[pipeline]

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 存储层 — ImageRecord 原图路径 + 父文档挂靠逻辑通用化

**Files:**
- Modify: `app/store/qdrant_client.py`
- Test: `tests/unit/test_qdrant_store.py`

**Interfaces:**
- Consumes: 无（独立于 Task 1）。
- Produces: `ImageRecord(..., image_path: str | None = None)` 新字段，`upsert_image()` payload 带 `image_path`；`list_all_documents()` 的父文档挂靠逻辑从只认 `:page` 前缀改成认任意 `:` 分隔的前缀（`:img{n}` 也能正确挂靠）。后续 Task 3（写入 `image_path`）和 Task 4（API 读取 `image_path`）依赖这个字段名。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_qdrant_store.py` 里，紧跟在 `test_upsert_and_search_image` 之后加：

```python
def test_upsert_image_stores_image_path():
    client = make_client()
    record = ImageRecord(
        doc_id="img-full", source_path="/full.png", thumbnail_path="/thumb/full.jpg",
        width=100, height=100, has_ocr_text=False, content_hash="hfull",
        image_path="/originals/full.jpg",
    )
    upsert_image(client, record, [0.2] * 512)

    payload = find_image_by_doc_id(client, "img-full")
    assert payload["image_path"] == "/originals/full.jpg"


def test_upsert_image_defaults_image_path_to_none_when_omitted():
    client = make_client()
    record = ImageRecord(
        doc_id="img-nopath", source_path="/x.png", thumbnail_path="/thumb/x.jpg",
        width=10, height=10, has_ocr_text=False, content_hash="hx2",
    )
    upsert_image(client, record, [0.2] * 512)

    payload = find_image_by_doc_id(client, "img-nopath")
    assert payload["image_path"] is None
```

在 `test_list_all_documents_attaches_scanned_pdf_page_image_by_prefix` 之后加：

```python
def test_list_all_documents_attaches_extracted_pdf_image_by_prefix():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    upsert_text_chunk(
        client,
        TextChunkRecord(
            doc_id="doc-pdf2", chunk_index=0, text="text with figure", source_path="/report.pdf",
            file_type="pdf", page=None, source_type="normal", content_hash="hpdf2",
            origin="upload", created_at="2026-08-25T00:00:00+00:00",
        ),
        vec,
    )
    upsert_image(
        client,
        ImageRecord(
            doc_id="doc-pdf2:img1", source_path="/report.pdf#img=1",
            image_path="/originals/doc-pdf2:img1.jpg",
            thumbnail_path="/thumb/doc-pdf2:img1.jpg", width=400, height=300, has_ocr_text=False,
            content_hash="hpdf2", origin="upload", created_at="2026-08-25T00:00:00+00:00",
        ),
        [0.2] * 512,
    )

    docs = list_all_documents(client)

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc-pdf2"
    assert len(docs[0]["related_images"]) == 1
    assert docs[0]["related_images"][0]["doc_id"] == "doc-pdf2:img1"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_qdrant_store.py -k "image_path or extracted_pdf_image" -v`
Expected: FAIL —— `test_upsert_image_stores_image_path` 报 `TypeError: __init__() got an unexpected keyword argument 'image_path'`；`test_list_all_documents_attaches_extracted_pdf_image_by_prefix` 报 `len(docs) == 1` 断言失败（图片被当成独立文档，`len(docs) == 2`）。

- [ ] **Step 3: 实现**

编辑 `app/store/qdrant_client.py`，`ImageRecord` 加字段（放在最后，不影响任何现有位置参数调用）：

```python
@dataclass
class ImageRecord:
    doc_id: str
    source_path: str
    thumbnail_path: str
    width: int
    height: int
    has_ocr_text: bool
    content_hash: str
    origin: str | None = None
    created_at: str | None = None
    image_path: str | None = None
```

`upsert_image()` 的 payload 字典里加一行：

```python
                payload={
                    "doc_id": record.doc_id,
                    "source_path": record.source_path,
                    "thumbnail_path": record.thumbnail_path,
                    "width": record.width,
                    "height": record.height,
                    "has_ocr_text": record.has_ocr_text,
                    "content_hash": record.content_hash,
                    "origin": record.origin,
                    "created_at": record.created_at,
                    "image_path": record.image_path,
                },
```

`list_all_documents()` 里把：

```python
        parent_id = None
        if ":page" in img_doc_id:
            candidate = img_doc_id.split(":page")[0]
            if candidate in text_docs:
                parent_id = candidate
        elif img_doc_id in text_docs:
            parent_id = img_doc_id
```

改成：

```python
        parent_id = None
        if ":" in img_doc_id:
            candidate = img_doc_id.split(":", 1)[0]
            if candidate in text_docs:
                parent_id = candidate
        elif img_doc_id in text_docs:
            parent_id = img_doc_id
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_qdrant_store.py -v`
Expected: PASS（全部用例，包括改动前就有的 `test_list_all_documents_attaches_scanned_pdf_page_image_by_prefix` 依然通过，证明改动是纯泛化、没破坏旧行为）

- [ ] **Step 5: 提交**

```bash
git add app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "$(cat <<'EOF'
feat: add ImageRecord.image_path and generalize document/image linking

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: MinerU PDF 解析 + 摄取管线重写

**Files:**
- Modify: `app/ingest/parsers/pdf_parser.py`（整体重写）
- Modify: `app/ingest/pipeline.py`
- Test: `tests/unit/test_pdf_parser.py`（整体重写）
- Test: `tests/unit/test_pipeline_dispatch.py`（删除一个过时用例，新增一个用例）

**Interfaces:**
- Consumes: Task 1 的 `settings.mineru_backend`/`settings.mineru_timeout_seconds`/`settings.ocr_lang`/`settings.extracted_images_dir`；Task 2 的 `ImageRecord(image_path=...)`、`upsert_image`。
- Produces: `ParsedPdfImage(filename: str, image: PIL.Image.Image)`、`ParsedPdfDocument(markdown_text: str, images: list[ParsedPdfImage])`、`parse_pdf_to_markdown(path: Path) -> ParsedPdfDocument`（`app/ingest/parsers/pdf_parser.py`）。`app/ingest/pipeline.py` 里 `_ingest_pdf_file` 和 `_ingest_pdf_extracted_image` 是内部私有函数，不对外暴露。

#### 第一部分：`pdf_parser.py`

- [ ] **Step 1: 写失败测试（整体替换 `test_pdf_parser.py`）**

用以下内容整体替换 `tests/unit/test_pdf_parser.py`（旧的基于 fitz 造 PDF 的用例连同 `make_test_pdf` 一起删除——它测的是被本任务整体替换掉的旧解析逻辑）：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_pdf_parser.py -v`
Expected: FAIL —— `ImportError: cannot import name 'parse_pdf_to_markdown'`

- [ ] **Step 3: 实现（整体替换 `pdf_parser.py`）**

用以下内容整体替换 `app/ingest/parsers/pdf_parser.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_pdf_parser.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/ingest/parsers/pdf_parser.py tests/unit/test_pdf_parser.py
git commit -m "$(cat <<'EOF'
feat: replace fitz+PaddleOCR PDF parser with MinerU CLI wrapper

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

#### 第二部分：`pipeline.py`

- [ ] **Step 6: 写失败测试**

在 `tests/unit/test_pipeline_dispatch.py` 里，删除整个 `test_ingest_pdf_multipage_chunks_do_not_collide` 函数和它上面用不到之后就没有其它调用者的 `make_multipage_text_pdf` 函数（这个用例测的是逐页循环重置 `chunk_index` 的旧 bug，新架构整篇一次性切片，不再有逐页循环，这类 bug 已经架构性消除）。

同时把文件顶部这两行删除（新测试不再需要造真实 PDF）：

```python
import fitz
```

（保留 `from pathlib import Path` 和 `from unittest.mock import patch`，这两个后面还用得到。）

在文件顶部的 import 块里加两行（放在 `from app.models.embeddings import DenseSparseVector` 附近）：

```python
from PIL import Image

from app.ingest.parsers.pdf_parser import ParsedPdfDocument, ParsedPdfImage
```

并把已有的这一行：

```python
from app.store.qdrant_client import ensure_collections, get_qdrant_client, list_chunks_for_document
```

改成：

```python
from app.store.qdrant_client import ensure_collections, find_image_by_doc_id, get_qdrant_client, list_chunks_for_document
```

然后在文件末尾加一个新用例：

```python
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
        "| A | B |\n| - | - |\n| 1 | 2 |\n"
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
```

- [ ] **Step 7: 运行测试确认失败**

Run: `pytest tests/unit/test_pipeline_dispatch.py -v`
Expected: FAIL —— `ImportError: cannot import name 'parse_pdf_to_markdown'` from `app.ingest.pipeline`（`pipeline.py` 还在 import 已经被 Task 3 第一部分删除的旧 `parse_pdf`/`ParsedPdfPage`）

- [ ] **Step 8: 实现**

编辑 `app/ingest/pipeline.py`。

把导入行：

```python
from app.ingest.parsers.pdf_parser import ParsedPdfPage, parse_pdf
```

改成：

```python
from app.ingest.parsers.pdf_parser import parse_pdf_to_markdown
```

删除整个 `_ingest_scanned_pdf_page` 函数（第 222-247 行那段，函数体见文件当前内容）。

把 `_ingest_pdf_file` 整个函数替换成：

```python
def _ingest_pdf_extracted_image(
    client, doc_id: str, pdf_path: Path, idx: int, image: Image.Image, content_hash: str, origin: str, created_at: str
) -> str:
    """入库 MinerU 从 PDF 里抽出的一张内嵌图片：持久化原图、生成缩略图、
    做 CLIP 向量化。不调用 parse_image()——那个函数会跑一次不需要的 OCR，
    风格上延续 _ingest_scanned_pdf_page 原本"直接内联 CLIP+缩略图"的做法。
    """
    img_doc_id = f"{doc_id}:img{idx}"
    persisted_path = settings.extracted_images_dir / f"{img_doc_id}.jpg"
    image.save(persisted_path, "JPEG")
    thumbnail_path = _save_thumbnail(image, img_doc_id)
    clip_vector = get_clip_embedder().embed_image(image)
    upsert_image(
        client,
        ImageRecord(
            doc_id=img_doc_id,
            source_path=f"{pdf_path}#img={idx}",
            thumbnail_path=thumbnail_path,
            width=image.width,
            height=image.height,
            has_ocr_text=False,
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
            image_path=str(persisted_path),
        ),
        clip_vector,
    )
    return img_doc_id


def _ingest_pdf_file(
    client, doc_id: str, path: Path, content_hash: str, chunk_options: ChunkOptions, origin: str, created_at: str
) -> int:
    parsed = parse_pdf_to_markdown(path)
    text = parsed.markdown_text
    for idx, pdf_image in enumerate(parsed.images, start=1):
        img_doc_id = _ingest_pdf_extracted_image(
            client, doc_id, path, idx, pdf_image.image, content_hash, origin=origin, created_at=created_at
        )
        text = text.replace(f"images/{pdf_image.filename}", f"/documents/{img_doc_id}/image")
    return _write_text_chunks(
        client, doc_id, str(path), "pdf", text, None, "normal", content_hash, chunk_options,
        origin=origin, created_at=created_at,
    )
```

注意：`written` 不统计抽取出来的图片数量，只统计文本 chunk 数——这和改动前的行为一致（旧代码里 `_ingest_scanned_pdf_page` 写整页图片记录时也不计入 `written`），也和 `list_all_documents()` 里 `chunk_count` 只从 `TEXT_COLLECTION` 统计的口径保持一致（`tests/integration/test_ingest_and_search.py` 里 `doc["chunk_count"] == result.chunks_written` 这条既有断言依赖这个口径，不能破坏）。

在 `_write_text_chunks` 里把判断条件：

```python
    if file_type == "md":
```

改成：

```python
    if file_type in ("md", "pdf"):
```

- [ ] **Step 9: 运行测试确认通过**

Run: `pytest tests/unit/test_pipeline_dispatch.py tests/unit/test_pdf_parser.py tests/unit/test_qdrant_store.py tests/unit/test_config.py -v`
Expected: PASS，全部用例

- [ ] **Step 10: 跑一次全量单测，确认没有破坏别的模块**

Run: `pytest tests/unit -v`
Expected: PASS（`test_image_parser.py`/`test_markdown_chunker.py` 等不受影响的模块应保持原有通过状态）

- [ ] **Step 11: 提交**

```bash
git add app/ingest/pipeline.py tests/unit/test_pipeline_dispatch.py
git commit -m "$(cat <<'EOF'
feat: wire MinerU PDF parsing into ingestion pipeline

PDF text now goes through the structured markdown chunker (heading_path/
has_table/has_code_block) and extracted images are persisted, CLIP-embedded,
and linked from the chunk text instead of being dropped.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: API — `GET /documents/{doc_id}/image`

**Files:**
- Modify: `app/api/routes/documents.py`
- Test: `tests/unit/test_api_documents.py`

**Interfaces:**
- Consumes: Task 2 的 `ImageRecord.image_path` / `find_image_by_doc_id` payload 里的 `image_path` 字段。
- Produces: 无（这是本功能面向外部调用方的最终出口，Admin UI 之类的外部消费者直接用这个 URL）。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_api_documents.py` 里，紧跟在 `test_get_document_thumbnail_404_when_no_image_record` 之后加：

```python
def test_get_document_image_returns_image_bytes(tmp_path, monkeypatch):
    img_path = tmp_path / "full.jpg"
    Image.new("RGB", (10, 10), color="blue").save(img_path, "JPEG")

    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.find_image_by_doc_id",
        lambda store, doc_id: {"doc_id": doc_id, "image_path": str(img_path)},
    )

    client = TestClient(app)
    response = client.get("/documents/img1/image")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == img_path.read_bytes()


def test_get_document_image_404_when_no_image_record(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.find_image_by_doc_id", lambda store, doc_id: None)

    client = TestClient(app)
    response = client.get("/documents/unknown/image")
    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_get_document_image_404_when_image_path_missing_on_record(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.find_image_by_doc_id",
        lambda store, doc_id: {"doc_id": doc_id, "thumbnail_path": "/thumb/x.jpg"},
    )

    client = TestClient(app)
    response = client.get("/documents/img2/image")
    app.dependency_overrides.clear()

    assert response.status_code == 404
```

（`Image` 已经在文件顶部 `from PIL import Image` 导入过，`get_store_client`/`app`/`TestClient` 同样已导入，不用重复加。）

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_api_documents.py -k document_image -v`
Expected: FAIL，`404 Not Found`（路由不存在）

- [ ] **Step 3: 实现**

编辑 `app/api/routes/documents.py`，在 `get_document_thumbnail` 函数之后加：

```python
@router.get("/documents/{doc_id}/image")
def get_document_image(doc_id: str, store=Depends(get_store_client)):
    # 和 get_document_thumbnail 同样的理由：MinerU 抽取的原图只在 worker
    # 容器本地磁盘上（settings.extracted_images_dir），Admin UI 等外部调用
    # 方必须经由这个 HTTP 接口才能显示。
    image = find_image_by_doc_id(store, doc_id)
    if image is None or not image.get("image_path"):
        raise HTTPException(status_code=404, detail="original image not available")
    image_path = Path(image["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(image_path, media_type="image/jpeg")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_api_documents.py -v`
Expected: PASS，全部用例

- [ ] **Step 5: 提交**

```bash
git add app/api/routes/documents.py tests/unit/test_api_documents.py
git commit -m "$(cat <<'EOF'
feat: add GET /documents/{doc_id}/image endpoint for extracted PDF images

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 手动端到端验证（非自动化）

这一步依赖真实安装的 `mineru` 命令、下载好的模型权重、以及一块可用 GPU（或至少能跑 pipeline backend 的 CPU），不适合放进自动化测试套件，由人工在实际环境里跑一遍确认整条链路真实可用。

**Files:** 无代码改动。

- [ ] **Step 1: 确认 MinerU 命令可用**

Run: `mineru --version`
Expected: 打印版本号，不报 `command not found`（如果这一步失败，先 `pip install mineru[pipeline]==3.4.5` 到对应环境，再确认模型权重已经放在 `$HOME/.cache/huggingface` 下）

- [ ] **Step 2: 准备一份真实测试 PDF**

找一份包含标题层级、至少一个表格、至少一张图片的 PDF（比如公司现成的说明书/报告），放到 `data/uploads/` 下。

- [ ] **Step 3: 启动完整服务栈**

Run: `docker compose up -d qdrant redis api worker`
Expected: 四个容器都变成 `Up` / `healthy`

- [ ] **Step 4: 通过 API 上传 PDF**

Run:
```bash
curl -F "file=@/path/to/test.pdf" http://localhost:8000/documents/upload
```
Expected: 返回 `{"job_id": "..."}`

- [ ] **Step 5: 轮询任务状态直到完成**

Run: `curl http://localhost:8000/jobs/<job_id>`
Expected: `status` 最终变成完成状态，且没有 `"failed"`

- [ ] **Step 6: 检查切片带上了标题结构**

Run: `curl http://localhost:8000/documents/<doc_id>/chunks | python3 -m json.tool | grep heading_path`
Expected: 至少一个 chunk 的 `heading_path` 不是 `null`，且能看到 PDF 里真实的章节标题文字

- [ ] **Step 7: 检查抽取的图片能正常访问**

Run: `curl http://localhost:8000/documents/<doc_id>/chunks | python3 -m json.tool | grep -o '/documents/[^"]*image'`
拿到其中一个 `img_doc_id`，然后：
```bash
curl -o /tmp/check.jpg http://localhost:8000/documents/<doc_id>:img1/image
file /tmp/check.jpg
```
Expected: `file` 命令识别出这是一个合法的 JPEG 图片文件，不是 404 错误页

- [ ] **Step 8: 记录耗时，评估 mineru_timeout_seconds 是否够用**

观察 Step 4 到 Step 5 之间实际耗时，如果经常接近或超过配置的 600 秒，回到 `app/config.py` 调大 `mineru_timeout_seconds` 默认值。

---

## Self-Review 记录

- **Spec 覆盖**：设计文档里的"范围 - 本轮做"六项分别对应 Task 3（替换解析路径+结构化切片）、Task 3+4（图片持久化+CLIP+链接改写+新接口）、Task 2（父文档挂靠逻辑通用化）；"范围 - 本轮不做"的每一项在 Global Constraints 和各任务描述里都显式保留了对应行为（page=None、source_type="normal"、无回退引擎、独立图片路径不变）。
- **占位符检查**：全文没有 TBD/TODO，MinerU 版本号（3.4.5）、pip extra 名（pipeline）都是从 PyPI 元数据实际查证的具体值，不是占位符。
- **类型一致性**：`ParsedPdfImage`/`ParsedPdfDocument`（Task 3 第一部分定义）与 Task 3 第二部分 `pipeline.py` 里的用法（`parsed.markdown_text`、`parsed.images`、`pdf_image.filename`、`pdf_image.image`）字段名一致；`ImageRecord.image_path`（Task 2 定义）与 Task 3 的 `_ingest_pdf_extracted_image` 构造调用、Task 4 的 API 读取（`image.get("image_path")`）字段名一致；`_ingest_pdf_extracted_image` 返回值 `img_doc_id: str`，与 `_ingest_pdf_file` 里 `text.replace(f"images/{pdf_image.filename}", f"/documents/{img_doc_id}/image")` 的用法一致。
