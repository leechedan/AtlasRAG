# MinerU 接入 PDF 摄取 — 设计文档

日期：2026-08-26

## 背景与问题

现有 PDF 摄取（`app/ingest/parsers/pdf_parser.py` + `app/ingest/pipeline.py::_ingest_pdf_file`）
是逐页处理：PyMuPDF(`fitz`) 抽取每页文本层，抽不到文字的页判定为扫描页，
栅格化整页图像后跑 PaddleOCR 兜底，扫描页的整页图像本身也存成一条
`ImageRecord`。这条路径完全丢失章节/标题层级、列表、表格结构——每页文本
直接进字符级 `chunk_text()`，PDF 的 chunk 天生拿不到
`[[2026-08-26-structured-markdown-chunker-design]]` 里给 `.md` 文件加的
`heading_path`/`has_table`/`has_code_block` 元数据。

[MinerU](https://github.com/opendatalab/mineru) 是一个开源 PDF 转 Markdown
工具，自带版面分析 + OCR，能把整份 PDF（不管是文字层 PDF 还是扫描 PDF）
转成一份保留标题层级、列表、表格（HTML 表格）的 Markdown，并把内嵌图片
单独抽取出来。本设计把 PDF 摄取整体切到 MinerU，同时让 PDF 正文复用现有
的结构化 Markdown 切片器，抽出的图片也正常入库并在 Markdown 正文里保留可
访问的链接。

## 范围

**本轮做**：
- 用 MinerU 完全替换现有 PDF 解析路径（fitz 逐页文本 + PaddleOCR 扫描兜底
  + 整页栅格化），删除 `pymupdf` 依赖。
- PDF 正文改走 `chunk_markdown_structured()`（与 `.md` 文件同一套切片
  器），获得 `heading_path`/`has_table`/`has_code_block`。
- MinerU 抽取的内嵌图片：CLIP 向量化 + 缩略图 + 原图持久化，写入
  `ImageRecord`；Markdown 正文里的图片引用改写成指向新增图片接口的可访问
  链接（而不是删除或保留失效的相对路径）。
- 新增 `GET /documents/{doc_id}/image` 接口返回原图。
- 修正 `list_all_documents()` 里图片挂靠父文档的判断逻辑，使其同时覆盖新
  的 `:img{n}` 命名和已有的 `:page{n}` 命名。

**本轮不做**：
- 不做页码级别的定位——PDF chunk 的 `page` 字段统一为 `None`（MinerU 整篇
  一次性输出合并 Markdown，不再按页处理；页级溯源可以后续用
  `content_list.json` 的 `page_idx` 字段补，本轮不做）。
- 不区分"这段文字来自扫描页还是文字层"——PDF 文本 chunk 的 `source_type`
  统一写 `"normal"`，不再有 `"ocr_image"` 分支。
- 不改独立图片文件上传路径（`app/ingest/parsers/image_parser.py`）——那条
  路径的 OCR+CLIP 逻辑不受影响。
- 不新增 MinerU 之外的回退路径——解析失败直接让异常沿现有的
  `ingest_file` 统一 `try/except` 冒泡，记成 `IngestResult(status="failed")`，
  不做二次兜底、不引入双引擎切换配置。
- 不处理已入库文档的磁盘文件生命周期问题（`delete_document` 目前只删
  Qdrant 记录、不删缩略图文件，这是已有的既存 gap，本轮新增的原图持久化
  文件遵循同样的既存行为，不额外修复）。

## 架构

```
app/ingest/parsers/pdf_parser.py     # 重写：MinerU 子进程调用 + 输出解析
  ParsedPdfDocument (dataclass)
  parse_pdf_to_markdown(path: Path) -> ParsedPdfDocument

app/ingest/pipeline.py               # 改：PDF 摄取流程
  _ingest_pdf_file()                 # 重写：整篇处理，不再逐页循环
  _ingest_pdf_extracted_image()      # 新增：单张抽取图片入库
  _write_text_chunks()               # 改：分支条件扩展到 file_type in ("md", "pdf")

app/config.py                        # 改：新增 mineru_backend / mineru_timeout_seconds / extracted_images_dir
app/store/qdrant_client.py           # 改：ImageRecord 新增 image_path 字段；list_all_documents() 挂靠逻辑通用化
app/api/routes/documents.py          # 改：新增 GET /documents/{doc_id}/image

requirements.txt                     # 改：移除 pymupdf，新增 mineru[pipeline]
Dockerfile                           # 改：如实现时发现需要额外 apt 包再补充

tests/unit/test_pdf_parser.py        # 重写：mock subprocess，不再跑真实 fitz PDF
tests/unit/test_pipeline_dispatch.py # 改：PDF 相关用例改为 mock parse_pdf_to_markdown
```

### 数据流

```
PDF 文件
  → subprocess: mineru -p <pdf> -o <tmp_dir> -b pipeline -l ch -m auto
  → 在 <tmp_dir> 下定位生成的 {stem}.md 全文 + images/ 目录下的图片文件
  → 对每张图片：
      - 持久化原图到 extracted_images_dir/{doc_id}:img{n}.jpg
      - 生成缩略图（复用现有 _save_thumbnail）
      - CLIP 向量化（不跑 OCR）
      - 写入 ImageRecord（doc_id={doc_id}:img{n}, source_path={pdf路径}#img={n}, image_path=持久化路径）
      - 在 markdown 全文里把 "images/{原文件名}" 替换成 "/documents/{doc_id}:img{n}/image"
  → 替换后的正文 → chunk_markdown_structured() → _write_text_chunks(file_type="pdf")
  → 清理临时目录（原图已持久化，临时目录内容不再需要）
```

## 核心改动

### `parse_pdf_to_markdown`

```python
@dataclass
class ParsedPdfDocument:
    markdown_text: str
    image_paths: list[Path]  # MinerU 输出的 images/ 目录下的图片文件，按文件名排序

def parse_pdf_to_markdown(path: Path) -> ParsedPdfDocument:
    ...
```

- 用 `tempfile.mkdtemp()` 建临时输出目录，`subprocess.run(["mineru", "-p", str(path),
  "-o", tmp_dir, "-b", settings.mineru_backend, "-l", settings.ocr_lang, "-m", "auto"],
  timeout=settings.mineru_timeout_seconds, capture_output=True, text=True)`。
- 子进程非零退出码，或者遍历 `tmp_dir` 找不到生成的 `.md` 文件：`raise
  RuntimeError`，带上 stderr 尾部方便排查。异常直接冒泡给 `ingest_file` 现
  有的 broad `except Exception`，记成 `status="failed"`。
- `image_paths` 从 MinerU 输出的 `images/` 子目录里按文件名排序收集（确定
  性顺序，仅用于生成 `:img{n}` 序号，序号本身对结果无强含义）。
- 精确的输出目录嵌套结构（`{stem}/{method}/{stem}.md` 之类）以实现时实际
  跑一次 MinerU 的真实输出为准，用 `tmp_dir.rglob("*.md")` 之类的方式查找
  而不是硬编码固定层级，避免版本差异导致路径假设失效。

### `app/ingest/pipeline.py`

- 删除 `_ingest_scanned_pdf_page`、对 `ParsedPdfPage` 的依赖。
- `_write_text_chunks` 里判断是否走结构化切片器的条件从
  `if file_type == "md":` 改成 `if file_type in ("md", "pdf"):`。
- `_ingest_pdf_file` 重写为整篇处理（不再是 per-page 循环）：
  ```python
  def _ingest_pdf_file(client, doc_id, path, content_hash, chunk_options, origin, created_at) -> int:
      parsed = parse_pdf_to_markdown(path)
      text = parsed.markdown_text
      written = 0
      for idx, image_path in enumerate(parsed.image_paths, start=1):
          img_written, img_doc_id = _ingest_pdf_extracted_image(
              client, doc_id, path, idx, image_path, content_hash, origin, created_at
          )
          written += img_written
          text = text.replace(f"images/{image_path.name}", f"/documents/{img_doc_id}/image")
      written += _write_text_chunks(
          client, doc_id, str(path), "pdf", text, None, "normal", content_hash,
          chunk_options, origin=origin, created_at=created_at,
      )
      return written
  ```
- 新增 `_ingest_pdf_extracted_image`：
  ```python
  def _ingest_pdf_extracted_image(
      client, doc_id, pdf_path, idx, image_path, content_hash, origin, created_at
  ) -> tuple[int, str]:
      image = Image.open(image_path).convert("RGB")
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
              image_path=str(persisted_path),
              thumbnail_path=thumbnail_path,
              width=image.width,
              height=image.height,
              has_ocr_text=False,
              content_hash=content_hash,
              origin=origin,
              created_at=created_at,
          ),
          clip_vector,
      )
      return 1, img_doc_id
  ```
  不调用 `parse_image()`（会跑一次不需要的 OCR），直接内联 CLIP+缩略图，
  风格上延续原 `_ingest_scanned_pdf_page` 的做法。

### `app/store/qdrant_client.py`

- `ImageRecord` 新增字段 `image_path: str | None = None`，`upsert_image()`
  的 payload 里带上这个字段。只有 MinerU 抽图这条路径会填值，其余现有调
  用点（独立图片上传）不受影响，保持默认 `None`。
- `list_all_documents()` 里判断图片是否属于某个父文档的逻辑，从硬编码
  `":page" in img_doc_id` 改成通用的：
  ```python
  if ":" in img_doc_id:
      candidate = img_doc_id.split(":", 1)[0]
      if candidate in text_docs:
          parent_id = candidate
  elif img_doc_id in text_docs:
      parent_id = img_doc_id
  ```
  （doc_id 本身来自 `make_doc_id()` 的 sha256 十六进制串，不含冒号，见
  `compute_chunk_id` 旁的既有注释，所以 `split(":", 1)[0]` 对 `:page{n}`
  和 `:img{n}` 都能正确取回原始 doc_id。）

### `app/config.py`

新增：
```python
mineru_backend: str = "pipeline"
mineru_timeout_seconds: int = 600

@property
def extracted_images_dir(self) -> Path:
    return self.data_dir / "extracted_images"
```
启动时和 `uploads_dir`/`thumbnails_dir` 一样 `mkdir(parents=True, exist_ok=True)`。

模型源：不新增配置项，沿用 MinerU 默认的 `huggingface` 模型源，复用
`docker-compose.yml` 里 worker 服务已有的 `HF_HOME` 挂载
（`${HOME}/.cache/huggingface`），权重由用户自行下载到该目录，本轮不写自
动下载/预热逻辑。

`mineru_timeout_seconds` 默认 600 秒，在现有 RQ job 的 `job_timeout=3600`
（`app/api/routes/documents.py` upload 接口里设置）之内，不冲突。

### `app/api/routes/documents.py`

新增（紧跟在 `get_document_thumbnail` 之后）：
```python
@router.get("/documents/{doc_id}/image")
def get_document_image(doc_id: str, store=Depends(get_store_client)):
    image = find_image_by_doc_id(store, doc_id)
    if image is None or not image.get("image_path"):
        raise HTTPException(status_code=404, detail="original image not available")
    image_path = Path(image["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(image_path, media_type="image/jpeg")
```
与现有 `get_document_thumbnail` 同样的理由：Admin UI 等外部调用方看不到
worker/api 容器内的文件系统，图片必须经由 HTTP 接口才能显示。

## 依赖与部署改动

- `requirements.txt`：移除 `pymupdf==1.25.1`（`fitz` 只有旧 `pdf_parser.py`
  用到，删除后无残留引用），新增 `mineru[pipeline]`（具体版本号实现时确
  认当前可用的最新版本并锁定）。
- `Dockerfile`：现有 `libgl1 libglib2.0-0` 预计够用；实现阶段实际跑一次
  `pip install mineru[pipeline]` + 真实 `mineru -p` 调用，确认是否需要补
  充系统包，如需要再补进 apt 安装列表。
- `paddleocr`/`paddlepaddle` 依赖保留——独立图片上传路径
  （`image_parser.py::parse_image`）仍然用它做 OCR。

## 边界情况

- **MinerU 未生成任何图片**（纯文字 PDF）：`image_paths` 为空列表，跳过
  图片处理循环，正文里没有图片引用可替换，直接走结构化切片。
- **MinerU 子进程超时/崩溃/未安装**：`parse_pdf_to_markdown` 抛出的异常
  沿 `ingest_file` 现有 `try/except Exception` 冒泡，整份文件记为
  `IngestResult(status="failed", error=...)`，不影响同批次其它文件（与现
  有单文件失败隔离行为一致，无需新代码）。
- **PDF 完全没有可提取内容**（如空白页 PDF）：`chunk_markdown_structured`
  对空/纯空白文本已经返回空列表（继承自 `.md` 路径的既有行为），
  `_write_text_chunks` 对空 chunk 列表返回 0，`written` 可能为 0——这与
  现状一致，不是新问题。
- **同一 PDF 重复摄取（内容未变）**：`ingest_file` 在解析前就先比较
  `content_hash`，未变则直接 `skipped_unchanged`，根本不会跑 MinerU，避
  免重复触发这个相对昂贵的子进程调用。
- **同一 PDF 内容变化后重新摄取**：`_cleanup_stale_doc_ids` 只处理
  `source_path` 本身的旧 doc_id，不处理旧的 `:img{n}` 复合 doc_id——这是
  既有的、文档里已经写明的已知限制（原本就覆盖 `:page{n}` 的情况），本
  轮不修复，行为与现状一致。

## 测试计划

- `tests/unit/test_pdf_parser.py`（重写）：mock `subprocess.run`，覆盖：
  - 正常场景：伪造一个临时目录（含 `.md` 文件 + `images/` 子目录若干图
    片），断言 `parse_pdf_to_markdown` 正确返回 markdown 文本和排序后的
    图片路径列表。
  - 子进程返回非零退出码：断言抛出异常。
  - 子进程成功但找不到 `.md` 输出文件：断言抛出异常。
- `tests/unit/test_pipeline_dispatch.py`：
  - 删除 `test_ingest_pdf_multipage_chunks_do_not_collide`（针对的是逐页
    循环重置 `start_index` 的旧 bug，新架构整篇一次性切片，这类 bug 已
    经架构性消除，不再适用）。
  - 新增用例：mock `app.ingest.pipeline.parse_pdf_to_markdown` 返回带标
    题/表格的 markdown 文本 + 2 张伪造图片路径，mock
    `app.ingest.pipeline.get_clip_embedder`，断言：
    - 产出的文本 chunk 带 `heading_path`（证明走的是结构化切片器）。
    - 产生 2 条 `ImageRecord`，`doc_id` 分别为 `{doc_id}:img1`/`:img2`。
    - 存储的正文里图片引用已替换成 `/documents/{doc_id}:img{n}/image`，
      不再包含原始相对路径。
- 不新增真实调用 MinerU（下载权重+GPU 推理）的集成测试——现有
  `tests/integration/test_ingest_and_search.py` 本来就不含 PDF 用例，维
  持现状。
- `tests/unit/test_qdrant_store.py`：新增用例覆盖 `list_all_documents()`
  对 `:img{n}` 命名的父文档挂靠逻辑。
