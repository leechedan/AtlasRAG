# 文档列表 / 分片列表 / 检索命中溯源 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `GET /documents`（文档列表）、`GET /documents/{doc_id}/chunks` + `GET /chunks`（分片列表）三个接口，并让检索响应的每条 hit 在顶层直接带上 `doc_id`/`chunk_index`/`chunk_id`，都基于 spec `docs/superpowers/specs/2026-08-25-document-chunk-listing-design.md`。

**Architecture:** 在 `app/store/qdrant_client.py` 新增聚合/分页/公开 id 计算函数（scroll 现有两个 Qdrant collection，纯读操作，不改变写入路径的核心结构），`app/ingest/pipeline.py` 新增两个 payload 字段的写入，`app/api/schemas.py` 新增响应模型，三个新路由分别加在 `documents.py`（复用已有 `get_store_client` 依赖注入）和新文件 `chunks.py` 里。

**Tech Stack:** FastAPI 0.115.6、pydantic 2.10.4、qdrant-client 1.12.1（Qdrant collection 用 `:memory:` local 模式跑单测）、pytest 8.3.4。

## Global Constraints

- Python 3.12，遵循仓库现有类型注解风格（`int | None` 而非 `Optional[int]`）
- 单测一律用 `get_qdrant_client(location=":memory:")` + `ensure_collections(client)`，不连真实 Qdrant（见 `tests/unit/test_qdrant_store.py` 现有约定）
- API 路由测试一律用 `TestClient(app)` + `app.dependency_overrides[get_store_client]` + `monkeypatch.setattr("app.api.routes.<module>.<name>", ...)`（见 `tests/unit/test_api_search.py`、`tests/unit/test_api_documents.py` 现有约定），每个测试结束前 `app.dependency_overrides.clear()`
- 集成测试放在 `tests/integration/test_ingest_and_search.py`，标 `pytestmark = pytest.mark.slow`，用 `qdrant_test_client` fixture（真实模型，本地默认跳过）
- 所有新代码路径不得改变现有 `ingest_file`/`search_text_hybrid`/`list_chunks_for_document` 等函数的既有调用方行为（向后兼容）

---

### Task 1: qdrant_client.py — origin/created_at 字段 + 公开 compute_chunk_id

**Files:**
- Modify: `app/store/qdrant_client.py:36-118`
- Test: `tests/unit/test_qdrant_store.py`

**Interfaces:**
- Produces: `TextChunkRecord(..., origin: str | None = None, created_at: str | None = None)`，`ImageRecord(..., origin: str | None = None, created_at: str | None = None)`，`compute_chunk_id(doc_id: str, chunk_index: int) -> str`（原 `_point_id` 改名，供 Task 6/8/9 的 API 层直接调用）

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_qdrant_store.py` 里，`test_get_stats_reports_counts` 之前加：

```python
def test_upsert_text_chunk_writes_origin_and_created_at():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc-origin", chunk_index=0, text="text", source_path="/o.txt",
        file_type="txt", page=None, source_type="normal", content_hash="ho",
        origin="upload", created_at="2026-08-25T00:00:00+00:00",
    )
    upsert_text_chunk(client, record, vec)

    chunks = list_chunks_for_document(client, "doc-origin")
    assert chunks[0]["origin"] == "upload"
    assert chunks[0]["created_at"] == "2026-08-25T00:00:00+00:00"


def test_upsert_image_writes_origin_and_created_at():
    client = make_client()
    record = ImageRecord(
        doc_id="img-origin", source_path="/o.png", thumbnail_path="/thumb/o.jpg",
        width=10, height=10, has_ocr_text=False, content_hash="ho2",
        origin="watch", created_at="2026-08-25T00:00:00+00:00",
    )
    upsert_image(client, record, [0.2] * 512)

    payload = find_image_by_doc_id(client, "img-origin")
    assert payload["origin"] == "watch"
    assert payload["created_at"] == "2026-08-25T00:00:00+00:00"


def test_compute_chunk_id_is_deterministic_and_public():
    from app.store.qdrant_client import compute_chunk_id
    assert compute_chunk_id("d1", 0) == compute_chunk_id("d1", 0)
    assert compute_chunk_id("d1", 0) != compute_chunk_id("d1", 1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -k "origin_and_created_at or compute_chunk_id" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'origin'`（`compute_chunk_id` 那条会是 `ImportError`，因为还没导出这个名字，属于同一批预期失败）

- [ ] **Step 3: 实现**

在 `app/store/qdrant_client.py` 里：

```python
@dataclass
class TextChunkRecord:
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    file_type: str
    page: int | None
    source_type: str
    content_hash: str
    origin: str | None = None
    created_at: str | None = None


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
```

把 `_point_id` 改名为 `compute_chunk_id`（去掉下划线，函数体不变），并更新它在 `upsert_text_chunk`/`upsert_image` 里的两处调用：

```python
def compute_chunk_id(doc_id: str, chunk_index: int) -> str:
    # ":" 分隔符是安全的：doc_id 始终来自 make_doc_id 生成的哈希值，不含冒号，
    # 因此 doc_id:chunk_index 不会出现跨字段的歧义拼接。
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()


def upsert_text_chunk(client: QdrantClient, record: TextChunkRecord, vector: DenseSparseVector) -> None:
    point_id = compute_chunk_id(record.doc_id, record.chunk_index)
    sparse_indices = list(vector.sparse.keys())
    sparse_values = list(vector.sparse.values())
    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": vector.dense,
                    "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "doc_id": record.doc_id,
                    "chunk_index": record.chunk_index,
                    "text": record.text,
                    "source_path": record.source_path,
                    "file_type": record.file_type,
                    "page": record.page,
                    "source_type": record.source_type,
                    "content_hash": record.content_hash,
                    "origin": record.origin,
                    "created_at": record.created_at,
                },
            )
        ],
    )


def upsert_image(client: QdrantClient, record: ImageRecord, vector: list[float]) -> None:
    point_id = compute_chunk_id(record.doc_id, 0)
    client.upsert(
        collection_name=IMAGE_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={"clip": vector},
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
                },
            )
        ],
    )
```

在 `tests/unit/test_qdrant_store.py` 顶部 import 里加上 `find_image_by_doc_id`（如果 Task 前置的 bug 修复分支还没加过的话，检查一下 `from app.store.qdrant_client import (...)` 里是否已有它——已有则跳过这步）。

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -v`
Expected: 全部 PASS（含新增 3 条 + 原有全部）

- [ ] **Step 5: Commit**

```bash
git add app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "feat: add origin/created_at fields to Qdrant records, expose compute_chunk_id"
```

---

### Task 2: pipeline.py — 入库时写入 origin/created_at

**Files:**
- Modify: `app/ingest/pipeline.py`
- Test: `tests/unit/test_pipeline_dispatch.py`

**Interfaces:**
- Consumes: `TextChunkRecord(..., origin=..., created_at=...)`、`ImageRecord(..., origin=..., created_at=...)`（Task 1）
- Produces: `ingest_file` 写入的每个 chunk/image record 都带 `origin`（`"upload"` 当 `source_ref` 以 `"upload:"` 开头，否则 `"watch"`）和 `created_at`（ingest 发生时的 UTC ISO 时间戳，同一次 `ingest_file` 调用内所有记录共享同一个值）

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_pipeline_dispatch.py` 末尾加：

```python
@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_upload_source_ref_sets_origin_upload(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=f"upload:2026-08-25T00:00:00+00:00:{f.name}", client=client)

    chunks = list_chunks_for_document(client, result.doc_id)
    assert chunks[0]["origin"] == "upload"
    assert chunks[0]["created_at"] is not None


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_plain_path_source_ref_sets_origin_watch(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    chunks = list_chunks_for_document(client, result.doc_id)
    assert chunks[0]["origin"] == "watch"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_pipeline_dispatch.py -k "origin" -v`
Expected: FAIL — `assert None == "upload"` / `assert None == "watch"`（origin 字段还没被写入，`chunks[0]["origin"]` 当前会是 `KeyError` 或 `None`，取决于 Task 1 是否已在同一分支落地；若 `KeyError` 也算预期失败）

- [ ] **Step 3: 实现**

在 `app/ingest/pipeline.py` 顶部加 import：

```python
from datetime import datetime, timezone
```

修改 `_write_text_chunks`（加 `origin`、`created_at` 形参，传入 `TextChunkRecord`）：

```python
def _write_text_chunks(
    client,
    doc_id: str,
    source_path: str,
    file_type: str,
    text: str,
    page_number: int | None,
    source_type: str,
    content_hash: str,
    chunk_options: ChunkOptions,
    origin: str,
    created_at: str,
    start_index: int = 0,
) -> int:
    chunks = chunk_text(
        text,
        chunk_size=chunk_options.chunk_size,
        chunk_overlap=chunk_options.chunk_overlap,
        separators=chunk_options.separators,
    )
    if not chunks:
        return 0
    vectors = get_text_embedder().embed(chunks)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        record = TextChunkRecord(
            doc_id=doc_id,
            chunk_index=start_index + idx,
            text=chunk,
            source_path=source_path,
            file_type=file_type,
            page=page_number,
            source_type=source_type,
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        )
        upsert_text_chunk(client, record, vector)
    return len(chunks)
```

修改 `_write_csv_chunks`：

```python
def _write_csv_chunks(
    client, doc_id: str, source_path: str, chunks: list[str], content_hash: str, origin: str, created_at: str
) -> int:
    if not chunks:
        return 0
    vectors = get_text_embedder().embed(chunks)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        record = TextChunkRecord(
            doc_id=doc_id,
            chunk_index=idx,
            text=chunk,
            source_path=source_path,
            file_type="csv",
            page=None,
            source_type="normal",
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        )
        upsert_text_chunk(client, record, vector)
    return len(chunks)
```

修改 `_ingest_image_file`：

```python
def _ingest_image_file(
    client, doc_id: str, path: Path, content_hash: str, chunk_options: ChunkOptions, origin: str, created_at: str
) -> int:
    image = Image.open(path).convert("RGB")
    parsed = parse_image(image, doc_id)
    upsert_image(
        client,
        ImageRecord(
            doc_id=doc_id,
            source_path=str(path),
            thumbnail_path=parsed.thumbnail_path,
            width=parsed.width,
            height=parsed.height,
            has_ocr_text=bool(parsed.ocr_text),
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        ),
        parsed.clip_vector,
    )
    written = 1
    if parsed.ocr_text:
        written += _write_text_chunks(
            client,
            doc_id,
            str(path),
            path.suffix.lstrip("."),
            parsed.ocr_text,
            None,
            "ocr_image",
            content_hash,
            chunk_options,
            origin=origin,
            created_at=created_at,
        )
    return written
```

修改 `_ingest_scanned_pdf_page`：

```python
def _ingest_scanned_pdf_page(
    client, doc_id: str, path: Path, page: ParsedPdfPage, content_hash: str, origin: str, created_at: str
) -> None:
    page_doc_id = f"{doc_id}:page{page.page_number}"
    thumbnail_path = _save_thumbnail(page.page_image, page_doc_id)
    clip_vector = get_clip_embedder().embed_image(page.page_image)
    upsert_image(
        client,
        ImageRecord(
            doc_id=page_doc_id,
            source_path=f"{path}#page={page.page_number}",
            thumbnail_path=thumbnail_path,
            width=page.page_image.width,
            height=page.page_image.height,
            has_ocr_text=bool(page.text),
            content_hash=content_hash,
            origin=origin,
            created_at=created_at,
        ),
        clip_vector,
    )
```

修改 `_ingest_pdf_file`：

```python
def _ingest_pdf_file(
    client, doc_id: str, path: Path, content_hash: str, chunk_options: ChunkOptions, origin: str, created_at: str
) -> int:
    pages = parse_pdf(path)
    written = 0
    for page in pages:
        source_type = "ocr_image" if page.is_scanned else "normal"
        written += _write_text_chunks(
            client,
            doc_id,
            str(path),
            "pdf",
            page.text,
            page.page_number,
            source_type,
            content_hash,
            chunk_options,
            origin=origin,
            created_at=created_at,
            start_index=written,
        )
        if page.is_scanned and page.page_image is not None:
            _ingest_scanned_pdf_page(client, doc_id, path, page, content_hash, origin=origin, created_at=created_at)
    return written
```

修改 `ingest_file`：

```python
def ingest_file(
    path: Path, source_ref: str, client=None, chunk_options: ChunkOptions | None = None
) -> IngestResult:
    client = client or get_default_client()
    chunk_options = chunk_options or ChunkOptions()
    doc_id: str | None = None
    origin = "upload" if source_ref.startswith("upload:") else "watch"
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        content_hash = sha256_file(path)
        doc_id = make_doc_id(source_ref, content_hash)

        existing_hash = get_existing_content_hash(client, doc_id)
        if existing_hash == content_hash:
            return IngestResult(doc_id=doc_id, status="skipped_unchanged", chunks_written=0)

        _cleanup_stale_doc_ids(client, doc_id, str(path))

        ext = path.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTS:
            text = parse_text_file(path)
            written = _write_text_chunks(
                client, doc_id, str(path), ext.lstrip("."), text, None, "normal", content_hash, chunk_options,
                origin=origin, created_at=created_at,
            )
        elif ext == ".csv":
            rows, header = parse_csv_file(path)
            chunks = chunk_csv_rows(rows, header, max_chars=chunk_options.chunk_size)
            written = _write_csv_chunks(client, doc_id, str(path), chunks, content_hash, origin=origin, created_at=created_at)
        elif ext in SUPPORTED_IMAGE_EXTS:
            written = _ingest_image_file(client, doc_id, path, content_hash, chunk_options, origin=origin, created_at=created_at)
        elif ext == ".pdf":
            written = _ingest_pdf_file(client, doc_id, path, content_hash, chunk_options, origin=origin, created_at=created_at)
        else:
            raise ValueError(f"unsupported file extension: {ext}")

        return IngestResult(doc_id=doc_id, status="ingested", chunks_written=written)
    except Exception as exc:  # noqa: BLE001 - a single failed file must not abort a batch ingest job
        return IngestResult(doc_id=doc_id or source_ref, status="failed", chunks_written=0, error=str(exc))
```

在 `tests/unit/test_pipeline_dispatch.py` 顶部确认 `make_client`、`list_chunks_for_document`、`ingest_file` 已 import（已有，不用改）。

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_pipeline_dispatch.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/pipeline.py tests/unit/test_pipeline_dispatch.py
git commit -m "feat: derive origin (upload/watch) and stamp created_at during ingest"
```

---

### Task 3: qdrant_client.py — list_all_documents 聚合

**Files:**
- Modify: `app/store/qdrant_client.py`
- Test: `tests/unit/test_qdrant_store.py`

**Interfaces:**
- Consumes: `TEXT_COLLECTION`/`IMAGE_COLLECTION` payload 里的 `doc_id`/`source_path`/`file_type`/`origin`/`chunk_index`/`source_type`/`created_at`/`content_hash`（Task 1）、`thumbnail_path`/`width`/`height`（已有）
- Produces: `list_all_documents(client) -> list[dict]`，每个 dict 的 key 恰好是：`doc_id, source_path, filename, file_type, origin, chunk_count, source_types, created_at, content_hash, related_images`；`related_images` 是 `list[dict]`，每个 dict 恰好是 `doc_id, thumbnail_path, width, height`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_qdrant_store.py` 末尾加：

```python
def test_list_all_documents_aggregates_text_chunks():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for idx in range(2):
        upsert_text_chunk(
            client,
            TextChunkRecord(
                doc_id="doc-agg", chunk_index=idx, text=f"c{idx}", source_path="/agg.txt",
                file_type="txt", page=None, source_type="normal", content_hash="hagg",
                origin="upload", created_at="2026-08-25T00:00:00+00:00",
            ),
            vec,
        )

    docs = list_all_documents(client)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == "doc-agg"
    assert doc["filename"] == "agg.txt"
    assert doc["chunk_count"] == 2
    assert doc["source_types"] == ["normal"]
    assert doc["origin"] == "upload"
    assert doc["related_images"] == []


def test_list_all_documents_attaches_standalone_image_with_same_doc_id():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    upsert_text_chunk(
        client,
        TextChunkRecord(
            doc_id="doc-img-ocr", chunk_index=0, text="ocr text", source_path="/photo.png",
            file_type="png", page=None, source_type="ocr_image", content_hash="himg",
            origin="watch", created_at="2026-08-25T00:00:00+00:00",
        ),
        vec,
    )
    upsert_image(
        client,
        ImageRecord(
            doc_id="doc-img-ocr", source_path="/photo.png", thumbnail_path="/thumb/photo.jpg",
            width=100, height=200, has_ocr_text=True, content_hash="himg",
            origin="watch", created_at="2026-08-25T00:00:00+00:00",
        ),
        [0.2] * 512,
    )

    docs = list_all_documents(client)

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc-img-ocr"
    assert docs[0]["related_images"] == [
        {"doc_id": "doc-img-ocr", "thumbnail_path": "/thumb/photo.jpg", "width": 100, "height": 200}
    ]


def test_list_all_documents_attaches_scanned_pdf_page_image_by_prefix():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    upsert_text_chunk(
        client,
        TextChunkRecord(
            doc_id="doc-pdf", chunk_index=0, text="page text", source_path="/scan.pdf",
            file_type="pdf", page=1, source_type="ocr_image", content_hash="hpdf",
            origin="upload", created_at="2026-08-25T00:00:00+00:00",
        ),
        vec,
    )
    upsert_image(
        client,
        ImageRecord(
            doc_id="doc-pdf:page1", source_path="/scan.pdf#page=1", thumbnail_path="/thumb/doc-pdf:page1.jpg",
            width=800, height=1000, has_ocr_text=True, content_hash="hpdf",
            origin="upload", created_at="2026-08-25T00:00:00+00:00",
        ),
        [0.2] * 512,
    )

    docs = list_all_documents(client)

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc-pdf"
    assert len(docs[0]["related_images"]) == 1
    assert docs[0]["related_images"][0]["doc_id"] == "doc-pdf:page1"


def test_list_all_documents_lists_orphan_image_as_its_own_document():
    client = make_client()
    upsert_image(
        client,
        ImageRecord(
            doc_id="doc-standalone-img", source_path="/pure.jpg", thumbnail_path="/thumb/pure.jpg",
            width=50, height=50, has_ocr_text=False, content_hash="hpure",
            origin="watch", created_at="2026-08-25T00:00:00+00:00",
        ),
        [0.2] * 512,
    )

    docs = list_all_documents(client)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == "doc-standalone-img"
    assert doc["file_type"] == "jpg"
    assert doc["chunk_count"] == 0
    assert doc["source_types"] == []
    assert doc["related_images"] == [
        {"doc_id": "doc-standalone-img", "thumbnail_path": "/thumb/pure.jpg", "width": 50, "height": 50}
    ]


def test_list_all_documents_empty_collections_returns_empty_list():
    client = make_client()
    assert list_all_documents(client) == []
```

在文件顶部 import 列表里加上 `list_all_documents`。

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -k "list_all_documents" -v`
Expected: FAIL — `ImportError: cannot import name 'list_all_documents'`

- [ ] **Step 3: 实现**

在 `app/store/qdrant_client.py` 顶部 import 里加 `from pathlib import Path`。在 `list_chunks_for_document` 函数后面加：

```python
def _scroll_all(client: QdrantClient, collection: str) -> list:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(collection_name=collection, limit=100, offset=offset)
        points.extend(batch)
        if offset is None:
            break
    return points


def list_all_documents(client: QdrantClient) -> list[dict]:
    text_docs: dict[str, dict] = {}
    for p in _scroll_all(client, TEXT_COLLECTION):
        payload = p.payload
        doc_id = payload["doc_id"]
        if doc_id not in text_docs:
            text_docs[doc_id] = {
                "doc_id": doc_id,
                "source_path": payload["source_path"],
                "filename": Path(payload["source_path"]).name,
                "file_type": payload["file_type"],
                "origin": payload.get("origin"),
                "chunk_count": 0,
                "source_types": set(),
                "created_at": payload.get("created_at"),
                "content_hash": payload["content_hash"],
                "related_images": [],
            }
        entry = text_docs[doc_id]
        entry["chunk_count"] += 1
        entry["source_types"].add(payload["source_type"])

    image_only_docs: dict[str, dict] = {}
    for p in _scroll_all(client, IMAGE_COLLECTION):
        payload = p.payload
        img_doc_id = payload["doc_id"]
        image_summary = {
            "doc_id": img_doc_id,
            "thumbnail_path": payload["thumbnail_path"],
            "width": payload["width"],
            "height": payload["height"],
        }

        parent_id = None
        if ":page" in img_doc_id:
            candidate = img_doc_id.split(":page")[0]
            if candidate in text_docs:
                parent_id = candidate
        elif img_doc_id in text_docs:
            parent_id = img_doc_id

        if parent_id is not None:
            text_docs[parent_id]["related_images"].append(image_summary)
        else:
            image_only_docs[img_doc_id] = {
                "doc_id": img_doc_id,
                "source_path": payload["source_path"],
                "filename": Path(payload["source_path"]).name,
                "file_type": Path(payload["source_path"]).suffix.lstrip(".").lower(),
                "origin": payload.get("origin"),
                "chunk_count": 0,
                "source_types": [],
                "created_at": payload.get("created_at"),
                "content_hash": payload["content_hash"],
                "related_images": [image_summary],
            }

    documents = []
    for entry in text_docs.values():
        entry["source_types"] = sorted(entry["source_types"])
        documents.append(entry)
    documents.extend(image_only_docs.values())
    return sorted(documents, key=lambda d: d["doc_id"])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "feat: add list_all_documents aggregation with related-image attachment"
```

---

### Task 4: qdrant_client.py — scroll_chunks 全局游标分页

**Files:**
- Modify: `app/store/qdrant_client.py`
- Test: `tests/unit/test_qdrant_store.py`

**Interfaces:**
- Consumes: `compute_chunk_id`（Task 1）
- Produces: `scroll_chunks(client, cursor: str | None = None, limit: int = 50) -> tuple[list[dict], str | None]`，返回的每个 dict 是 `TEXT_COLLECTION` 的原始 payload 加一个 `chunk_id` key；第二个返回值是 `next_cursor`（`None` 表示到底）

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_qdrant_store.py` 末尾加：

```python
def test_scroll_chunks_paginates_through_all_chunks_without_duplicates():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for idx in range(5):
        upsert_text_chunk(
            client,
            TextChunkRecord(
                doc_id="doc-scroll", chunk_index=idx, text=f"c{idx}", source_path="/s.txt",
                file_type="txt", page=None, source_type="normal", content_hash="hs",
            ),
            vec,
        )

    seen_chunk_ids = set()
    cursor = None
    pages = 0
    while True:
        chunks, cursor = scroll_chunks(client, cursor=cursor, limit=2)
        for c in chunks:
            assert c["chunk_id"]
            seen_chunk_ids.add(c["chunk_id"])
        pages += 1
        if cursor is None:
            break
        assert pages < 10  # safety net against an infinite loop bug

    assert len(seen_chunk_ids) == 5


def test_scroll_chunks_empty_collection_returns_empty_page():
    client = make_client()
    chunks, cursor = scroll_chunks(client)
    assert chunks == []
    assert cursor is None
```

在文件顶部 import 列表里加上 `scroll_chunks`。

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -k "scroll_chunks" -v`
Expected: FAIL — `ImportError: cannot import name 'scroll_chunks'`

- [ ] **Step 3: 实现**

在 `app/store/qdrant_client.py` 里 `list_all_documents` 后面加：

```python
def scroll_chunks(client: QdrantClient, cursor: str | None = None, limit: int = 50) -> tuple[list[dict], str | None]:
    points, next_offset = client.scroll(collection_name=TEXT_COLLECTION, limit=limit, offset=cursor)
    chunks = []
    for p in points:
        payload = dict(p.payload)
        payload["chunk_id"] = compute_chunk_id(payload["doc_id"], payload["chunk_index"])
        chunks.append(payload)
    next_cursor = str(next_offset) if next_offset is not None else None
    return chunks, next_cursor
```

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_qdrant_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "feat: add scroll_chunks cursor pagination across all documents"
```

---

### Task 5: schemas.py — 新响应模型 + SearchHitOut 顶层字段

**Files:**
- Modify: `app/api/schemas.py`

**Interfaces:**
- Produces: `ImageSummaryOut`、`DocumentSummaryOut`、`ChunkOut`、`ChunksPageOut` pydantic 模型；`SearchHitOut` 新增 `doc_id: str`、`chunk_index: int | None`、`chunk_id: str` 字段（`payload: dict` 保留）

- [ ] **Step 1: 实现**（纯数据模型，没有可失败的行为，跳过 TDD 步骤，改完直接靠 Task 6-9 的路由测试间接验证）

把 `app/api/schemas.py` 里的 `SearchHitOut` 替换为：

```python
class SearchHitOut(BaseModel):
    score: float
    doc_id: str
    chunk_index: int | None
    chunk_id: str
    payload: dict
```

在文件末尾加：

```python
class ImageSummaryOut(BaseModel):
    doc_id: str
    thumbnail_path: str
    width: int
    height: int


class DocumentSummaryOut(BaseModel):
    doc_id: str
    source_path: str
    filename: str
    file_type: str
    origin: str | None
    chunk_count: int
    source_types: list[str]
    created_at: str | None
    content_hash: str
    related_images: list[ImageSummaryOut]


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page: int | None
    source_type: str
    source_path: str
    file_type: str


class ChunksPageOut(BaseModel):
    chunks: list[ChunkOut]
    next_cursor: str | None
```

- [ ] **Step 2: 语法检查**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -c "import app.api.schemas"`
Expected: 无报错（Task 6-9 完成前，`SearchHitOut` 的调用方 `search.py` 还没更新，所以先只确认模块本身能 import；`search.py` 的构造调用要等 Task 9 才会同步改）

注意：这一步之后、Task 9 完成之前，`app/api/routes/search.py` 里旧的 `SearchHitOut(score=h.score, payload=h.payload)` 调用会因为缺少必填的 `doc_id`/`chunk_id` 而在运行时报 pydantic `ValidationError`（`chunk_index` 有默认可选但 `doc_id`/`chunk_id` 没有默认值）。这是预期的临时状态——Task 5 和 Task 9 是同一个变更的两半，必须连续做完再提交,不要在这两步之间单独提交或部署。

- [ ] **Step 3: Commit**（与 Task 9 合并一起提交，见 Task 9 Step 5——这一步暂不单独 commit）

---

### Task 6: GET /documents

**Files:**
- Modify: `app/api/routes/documents.py`
- Test: `tests/unit/test_api_documents.py`

**Interfaces:**
- Consumes: `list_all_documents`（Task 3）、`DocumentSummaryOut`（Task 5）
- Produces: `GET /documents` → `200, list[DocumentSummaryOut]`

- [ ] **Step 1: 写失败测试**

`get_store_client` 已经在文件顶部 import 过（`from app.api.deps import get_store_client`），这里不需要再加新 import。

在文件末尾加：

```python
def test_list_documents_returns_aggregated_documents(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.documents.list_all_documents",
        lambda store: [{
            "doc_id": "d1", "source_path": "/a.txt", "filename": "a.txt", "file_type": "txt",
            "origin": "watch", "chunk_count": 2, "source_types": ["normal"], "created_at": None,
            "content_hash": "h1", "related_images": [],
        }],
    )

    client = TestClient(app)
    response = client.get("/documents")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["doc_id"] == "d1"
    assert body[0]["chunk_count"] == 2
    assert body[0]["related_images"] == []


def test_list_documents_empty_when_no_documents(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.list_all_documents", lambda store: [])

    client = TestClient(app)
    response = client.get("/documents")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_documents.py -k "list_documents" -v`
Expected: FAIL — `404 Not Found`（路由还不存在）

- [ ] **Step 3: 实现**

在 `app/api/routes/documents.py` 里，import 部分加：

```python
from app.api.schemas import DeleteResponse, DocumentSummaryOut, UploadResponse
from app.store.qdrant_client import find_image_by_doc_id, list_all_documents
```

（`DeleteResponse`/`UploadResponse` 已存在，只是加 `DocumentSummaryOut`；`find_image_by_doc_id` 已存在，只是加 `list_all_documents` 到同一行。）

在 `get_document_thumbnail` 路由前面加：

```python
@router.get("/documents", response_model=list[DocumentSummaryOut])
def list_documents(store=Depends(get_store_client)):
    return [DocumentSummaryOut(**d) for d in list_all_documents(store)]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_documents.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/documents.py app/api/schemas.py tests/unit/test_api_documents.py
git commit -m "feat: add GET /documents endpoint"
```

---

### Task 7: GET /documents/{doc_id}/chunks

**Files:**
- Modify: `app/api/routes/documents.py`
- Test: `tests/unit/test_api_documents.py`

**Interfaces:**
- Consumes: `list_chunks_for_document`（已有）、`get_existing_content_hash`（已有，用于判断 doc_id 是否存在）、`compute_chunk_id`（Task 1）、`ChunkOut`（Task 5）
- Produces: `GET /documents/{doc_id}/chunks` → `200, list[ChunkOut]`；doc_id 不存在 → `404`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_api_documents.py` 末尾加：

```python
def test_get_document_chunks_returns_ordered_chunks(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.get_existing_content_hash", lambda store, doc_id: "h1")
    monkeypatch.setattr(
        "app.api.routes.documents.list_chunks_for_document",
        lambda store, doc_id: [
            {
                "doc_id": doc_id, "chunk_index": 0, "text": "c0", "page": None,
                "source_type": "normal", "source_path": "/a.txt", "file_type": "txt",
            },
        ],
    )

    client = TestClient(app)
    response = client.get("/documents/d1/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body[0]["chunk_index"] == 0
    assert body[0]["chunk_id"]


def test_get_document_chunks_404_when_document_missing(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr("app.api.routes.documents.get_existing_content_hash", lambda store, doc_id: None)

    client = TestClient(app)
    response = client.get("/documents/unknown/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_documents.py -k "get_document_chunks" -v`
Expected: FAIL — `404 Not Found`（第一条测试本该 200，路由还不存在导致两条都不对）

- [ ] **Step 3: 实现**

在 `app/api/routes/documents.py` import 部分加：

```python
from app.api.schemas import ChunkOut, DeleteResponse, DocumentSummaryOut, UploadResponse
from app.store.qdrant_client import (
    compute_chunk_id,
    find_image_by_doc_id,
    get_existing_content_hash,
    list_all_documents,
    list_chunks_for_document,
)
```

在 `list_documents` 路由后面加：

```python
@router.get("/documents/{doc_id}/chunks", response_model=list[ChunkOut])
def get_document_chunks(doc_id: str, store=Depends(get_store_client)):
    if get_existing_content_hash(store, doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    chunks = list_chunks_for_document(store, doc_id)
    return [
        ChunkOut(
            chunk_id=compute_chunk_id(c["doc_id"], c["chunk_index"]),
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            text=c["text"],
            page=c.get("page"),
            source_type=c["source_type"],
            source_path=c["source_path"],
            file_type=c["file_type"],
        )
        for c in chunks
    ]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_documents.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/documents.py tests/unit/test_api_documents.py
git commit -m "feat: add GET /documents/{doc_id}/chunks endpoint"
```

---

### Task 8: GET /chunks（全局游标分页）

**Files:**
- Create: `app/api/routes/chunks.py`
- Modify: `app/api/main.py`
- Test: `tests/unit/test_api_chunks.py`

**Interfaces:**
- Consumes: `scroll_chunks`（Task 4）、`ChunksPageOut`/`ChunkOut`（Task 5）
- Produces: `GET /chunks?cursor=&limit=` → `200, ChunksPageOut`

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_api_chunks.py`：

```python
from fastapi.testclient import TestClient

from app.api.deps import get_store_client
from app.api.main import app


def test_list_chunks_returns_page(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.chunks.scroll_chunks",
        lambda store, cursor, limit: (
            [
                {
                    "chunk_id": "cid1", "doc_id": "d1", "chunk_index": 0, "text": "t0",
                    "page": None, "source_type": "normal", "source_path": "/a.txt", "file_type": "txt",
                },
            ],
            "next-cursor-value",
        ),
    )

    client = TestClient(app)
    response = client.get("/chunks?limit=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["chunk_id"] == "cid1"
    assert body["next_cursor"] == "next-cursor-value"


def test_list_chunks_last_page_has_null_next_cursor(monkeypatch):
    app.dependency_overrides[get_store_client] = lambda: object()
    monkeypatch.setattr(
        "app.api.routes.chunks.scroll_chunks", lambda store, cursor, limit: ([], None)
    )

    client = TestClient(app)
    response = client.get("/chunks")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"chunks": [], "next_cursor": None}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_chunks.py -v`
Expected: FAIL — `404 Not Found`（`app/api/routes/chunks.py` 还不存在，路由未注册）

- [ ] **Step 3: 实现**

创建 `app/api/routes/chunks.py`：

```python
from fastapi import APIRouter, Depends

from app.api.deps import get_store_client
from app.api.schemas import ChunkOut, ChunksPageOut
from app.store.qdrant_client import scroll_chunks

router = APIRouter()


@router.get("/chunks", response_model=ChunksPageOut)
def list_chunks(cursor: str | None = None, limit: int = 50, store=Depends(get_store_client)):
    chunks, next_cursor = scroll_chunks(store, cursor, limit)
    return ChunksPageOut(
        chunks=[
            ChunkOut(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                page=c.get("page"),
                source_type=c["source_type"],
                source_path=c["source_path"],
                file_type=c["file_type"],
            )
            for c in chunks
        ],
        next_cursor=next_cursor,
    )
```

在 `app/api/main.py` 里加 import 和注册（`from app.api.routes import collections, directories, documents, health, jobs, search` 这一行加上 `chunks`，`app.include_router(...)` 那几行加一行）：

```python
from app.api.routes import chunks, collections, directories, documents, health, jobs, search
```

```python
app.include_router(chunks.router)
```

（加在 `app.include_router(documents.router)` 后面即可，顺序对功能无影响。）

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_chunks.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/chunks.py app/api/main.py tests/unit/test_api_chunks.py
git commit -m "feat: add GET /chunks global cursor-paginated endpoint"
```

---

### Task 9: search.py — 检索响应顶层字段

**Files:**
- Modify: `app/api/routes/search.py`
- Test: `tests/unit/test_api_search.py`

**Interfaces:**
- Consumes: `compute_chunk_id`（Task 1）、更新后的 `SearchHitOut`（Task 5）
- Produces: `search_text`/`search_image` 的每个 hit 都带顶层 `doc_id`/`chunk_index`/`chunk_id`（`payload` 保留不变）

这一步和 Task 5 是同一个变更的收尾——Task 5 改了 `SearchHitOut` 的必填字段但没同步改调用方，所以在 Task 9 完成前 `pytest tests/unit/test_api_search.py` 会因为 `ValidationError` 而失败；这里把它修好并一起验证。

- [ ] **Step 1: 更新测试的 fixture 数据 + 断言**

修改 `tests/unit/test_api_search.py` 里的两个 fake 函数，让它们的 payload 带上 `chunk_index`（贴近真实数据）：

```python
def override_search_text_hybrid(*args, **kwargs):
    return [SearchHit(score=0.9, payload={"doc_id": "d1", "chunk_index": 0, "text": "命中文本"})]


def override_search_images(*args, **kwargs):
    return [SearchHit(score=0.8, payload={"doc_id": "img1", "source_path": "/a.png"})]


def override_list_chunks_for_document(*args, **kwargs):
    return [{"doc_id": "img1", "chunk_index": 0, "text": "关联OCR文本", "source_type": "ocr_image"}]
```

（`override_search_images`/`override_list_chunks_for_document` 内容不变，只是贴在这里方便对照；只有 `override_search_text_hybrid` 真的要改。）

在 `test_search_text_returns_text_hits` 里追加断言：

```python
def test_search_text_returns_text_hits(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/search/text", json={"query": "你好", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["text_hits"][0]["payload"]["doc_id"] == "d1"
    assert body["text_hits"][0]["doc_id"] == "d1"
    assert body["text_hits"][0]["chunk_index"] == 0
    assert body["text_hits"][0]["chunk_id"]
    assert body["image_hits"] == []
    app.dependency_overrides.clear()
```

在 `test_search_image_returns_image_hits` 里追加断言：

```python
def test_search_image_returns_image_hits(monkeypatch):
    client = make_client(monkeypatch)
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/search/image", files={"image": ("q.png", buf, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["image_hits"][0]["payload"]["doc_id"] == "img1"
    assert body["image_hits"][0]["doc_id"] == "img1"
    assert body["image_hits"][0]["chunk_index"] is None
    assert body["image_hits"][0]["chunk_id"]
    app.dependency_overrides.clear()
```

在 `test_search_image_returns_related_text_hits` 里追加断言：

```python
def test_search_image_returns_related_text_hits(monkeypatch):
    client = make_client(monkeypatch)
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/search/image", files={"image": ("q.png", buf, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["related_text_hits"]) == 1
    assert body["related_text_hits"][0]["payload"]["text"] == "关联OCR文本"
    assert body["related_text_hits"][0]["doc_id"] == "img1"
    assert body["related_text_hits"][0]["chunk_index"] == 0
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_search.py -v`
Expected: FAIL — `pydantic_core._pydantic_core.ValidationError: ... doc_id Field required ... chunk_id Field required`（`search.py` 里旧的 `SearchHitOut(score=..., payload=...)` 调用缺 Task 5 新增的必填字段）

- [ ] **Step 3: 实现**

替换 `app/api/routes/search.py` 全文为：

```python
import io

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image

from app.api.deps import get_clip_embedder_dep, get_store_client, get_text_embedder_dep
from app.api.schemas import (
    SearchHitOut,
    SearchImageResponse,
    SearchTextRequest,
    SearchTextResponse,
)
from app.store.qdrant_client import (
    compute_chunk_id,
    list_chunks_for_document,
    search_images,
    search_text_hybrid,
)

router = APIRouter()

# Sentinel score for related-text hits returned by search_image: these come
# from a direct doc_id lookup (list_chunks_for_document), not a similarity
# ranking, so there is no meaningful "score" to report. 1.0 signals "exact
# association" without implying a comparable relevance ranking against the
# image_hits' real similarity scores.
RELATED_TEXT_SENTINEL_SCORE = 1.0


def _make_hit_out(score: float, payload: dict, chunk_index: int | None) -> SearchHitOut:
    doc_id = payload["doc_id"]
    point_chunk_index = chunk_index if chunk_index is not None else 0
    return SearchHitOut(
        score=score,
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_id=compute_chunk_id(doc_id, point_chunk_index),
        payload=payload,
    )


@router.post("/search/text", response_model=SearchTextResponse)
def search_text(
    request: SearchTextRequest,
    store=Depends(get_store_client),
    text_embedder=Depends(get_text_embedder_dep),
    clip_embedder=Depends(get_clip_embedder_dep),
):
    text_hits = []
    image_hits = []

    if "text" in request.modalities:
        query_vector = text_embedder.embed([request.query])[0]
        text_hits = search_text_hybrid(store, query_vector, top_k=request.top_k)

    if "image" in request.modalities:
        clip_query_vector = clip_embedder.embed_text(request.query)
        image_hits = search_images(store, clip_query_vector, top_k=request.top_k)

    return SearchTextResponse(
        text_hits=[_make_hit_out(h.score, h.payload, h.payload.get("chunk_index")) for h in text_hits],
        image_hits=[_make_hit_out(h.score, h.payload, None) for h in image_hits],
    )


@router.post("/search/image", response_model=SearchImageResponse)
async def search_image(
    image: UploadFile = File(...),
    top_k: int = 10,
    store=Depends(get_store_client),
    clip_embedder=Depends(get_clip_embedder_dep),
):
    data = await image.read()
    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    query_vector = clip_embedder.embed_image(pil_image)
    image_hits = search_images(store, query_vector, top_k=top_k)

    # For each returned image hit, look up its associated OCR text chunks
    # (written under the same doc_id for standalone images -- see
    # _ingest_image_file in app/ingest/pipeline.py) and flatten them into a
    # single related_text_hits list. We query per-hit (not just the top hit)
    # so that OCR text associated with any of the returned images is
    # surfaced, not only the single best match. Known limitation: a scanned
    # PDF page's image record uses a composite doc_id
    # (f"{doc_id}:page{N}"), which doesn't match the parent PDF's plain
    # doc_id used for its text_chunks records, so list_chunks_for_document
    # correctly returns [] for those hits rather than finding the page's
    # OCR text (mirrors the same known gap in Task 14's on_deleted handler).
    related_text_hits: list[SearchHitOut] = []
    for hit in image_hits:
        doc_id = hit.payload.get("doc_id")
        if not doc_id:
            continue
        for chunk in list_chunks_for_document(store, doc_id):
            related_text_hits.append(
                _make_hit_out(RELATED_TEXT_SENTINEL_SCORE, chunk, chunk.get("chunk_index"))
            )

    return SearchImageResponse(
        image_hits=[_make_hit_out(h.score, h.payload, None) for h in image_hits],
        related_text_hits=related_text_hits,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit/test_api_search.py -v`
Expected: 全部 PASS

再跑一次全量单测确认没有破坏别的地方：

Run: `docker run --rm -v "$(pwd)":/srv -w /srv retrieval-service-api python3 -m pytest tests/unit -q --ignore=tests/unit/test_embeddings.py --ignore=tests/unit/test_clip_model.py`
Expected: 全部 PASS（这两个 ignore 是因为它们要下载/跑真实模型权重，和本次改动无关，本地一次性容器没有 HF 缓存卷，见本 repo 已知情况）

- [ ] **Step 5: Commit**

```bash
git add app/api/schemas.py app/api/routes/search.py tests/unit/test_api_search.py
git commit -m "feat: promote doc_id/chunk_index/chunk_id to top-level search hit fields"
```

---

### Task 10: 集成测试

**Files:**
- Modify: `tests/integration/test_ingest_and_search.py`

**Interfaces:**
- Consumes: `ingest_file`、`list_all_documents`、`scroll_chunks`、`qdrant_test_client` fixture（都已在前面 Task 完成）

- [ ] **Step 1: 写测试**

在 `tests/integration/test_ingest_and_search.py` 末尾加：

```python
from app.store.qdrant_client import list_all_documents, scroll_chunks


def test_list_documents_after_real_ingest(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    assert result.status == "ingested"

    docs = list_all_documents(qdrant_test_client)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == result.doc_id
    assert doc["chunk_count"] == result.chunks_written
    assert doc["origin"] == "watch"
    assert doc["source_types"] == ["normal"]


def test_scroll_chunks_paginates_across_multiple_real_ingests(qdrant_test_client):
    txt_result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    csv_result = ingest_file(FIXTURES / "sample.csv", source_ref=str(FIXTURES / "sample.csv"), client=qdrant_test_client)
    total_expected = txt_result.chunks_written + csv_result.chunks_written

    seen_chunk_ids = set()
    cursor = None
    while True:
        chunks, cursor = scroll_chunks(qdrant_test_client, cursor=cursor, limit=2)
        seen_chunk_ids.update(c["chunk_id"] for c in chunks)
        if cursor is None:
            break

    assert len(seen_chunk_ids) == total_expected
```

- [ ] **Step 2: 运行测试**（这是标了 `slow` 的集成测试，需要真实模型，默认跳过；本地手动验证一次即可）

Run: `docker run --rm -v "$(pwd)":/srv -w /srv -e HF_HOME=/hf_cache -v "${HOME}/.cache/huggingface:/hf_cache" --gpus all retrieval-service-api python3 -m pytest tests/integration/test_ingest_and_search.py -v -m slow`
Expected: 全部 PASS（如果本机没有 GPU/权重缓存，这一步在 CI/上线前手动跑，参考 `pytest.ini` 里 `slow` marker 的既有约定；本地开发环境没有条件跑的话，跳过此 Step 2，直接进行代码审查确认逻辑正确，并在 PR 描述里注明"集成测试待 CI/上线前验证"）

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ingest_and_search.py
git commit -m "test: add integration coverage for document/chunk listing after real ingest"
```

---

## 完成后自检清单

- [ ] `GET /documents` 返回聚合后的文档列表，含关联图片（独立图片场景 + PDF扫描页场景）
- [ ] `GET /documents/{doc_id}/chunks` 返回排序后的分片，doc_id 不存在时 404
- [ ] `GET /chunks` 支持游标分页，能遍历到所有分片
- [ ] `POST /search/text`、`POST /search/image` 的每个 hit 顶层都有 `doc_id`/`chunk_index`/`chunk_id`
- [ ] 存量数据（无 `origin`/`created_at`）在文档列表里正常显示为 `null`，不报错
- [ ] `app/sdk/langgraph_tools.py` 现有的 `h['payload'].get(...)` 用法不受影响（`payload` 字段保留全量内容）
