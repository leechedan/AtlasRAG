# 多模态检索服务平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 DGX Spark (GB10/ARM64) 上搭建一个提供文字检索、图片检索、文档上传入库、目录批量入库的内网检索服务，含 FastAPI 服务、RQ worker、Qdrant 混合检索、Gradio 管理界面、LangGraph 工具封装。

**Architecture:** FastAPI 承担查询路径（同步、常驻加载 BGE-M3+Chinese-CLIP 做实时向量化），RQ worker 承担入库路径（解析/OCR/切分/向量化后写入 Qdrant，并内置 watchdog 目录监听）。Qdrant 存两个 collection（`text_chunks` 混合检索、`images` 跨模态检索），Redis 承担任务队列与 job 状态。所有服务用 Docker Compose 单机编排。

**Tech Stack:** Python 3.12, FastAPI, RQ + Redis, Qdrant (qdrant-client), PyTorch (cu130 aarch64 wheel), FlagEmbedding (BGE-M3), transformers (Chinese-CLIP), PyMuPDF, pandas, PaddleOCR (CPU), watchdog, Gradio, langchain-text-splitters, Docker Compose.

## Global Constraints

- 部署形态：单机 Docker Compose，仅内网访问，v1 不做鉴权（源自 spec 第1、6节）
- Python 版本：3.12（与宿主机一致）
- PyTorch 安装源：`--index-url https://download.pytorch.org/whl/cu130`（GB10 aarch64 CUDA 13.0 官方 wheel，spec 第6节）
- PaddleOCR v1 使用官方 CPU wheel，不做 GPU 源码编译（spec 第6、10节）
- 文本切分粒度：~500-800 token 等价的字符窗口，默认 chunk_size=1000 字符、chunk_overlap=150 字符（spec 第5节，字符数为实现选择，需在 Task 4 中体现为可配置项）。这个 1000 字符上限对 txt/md/pdf/OCR 文本和 csv 行分组**统一生效**——csv 分组同时按行数(`rows_per_chunk`)和字符数(`max_chars`)双重限制，避免单条记录内容很长时拼出超大 chunk（检索接口按原样返回整个 chunk 文本，没有额外截断，所以入库侧必须保证不产出超大 chunk）
- 切分参数按次入库动态可配：`chunk_size`/`chunk_overlap`/自定义分隔符 `separators` 有全局默认值（`app.config.settings`），但每次上传文件（`/documents/upload`）或指定目录入库（`/directories/index`）时都可以覆盖为该次入库专用的值；目录若开启 `watch`，覆盖值要跟随该目录的监听配置一起持久化，保证后续监听触发的入库沿用同一套切分参数（spec 未预先规定此项，属本次用户追加需求，见 Task 4/12/13/14/17/18 的联动修改）
- Qdrant collection：`text_chunks`（dense 1024维 + sparse，来自 BGE-M3）、`images`（单一 dense 向量，来自 Chinese-CLIP，命名为 `clip`）（spec 第3节）
- `doc_id` 生成规则：目录入库用绝对路径作为来源标识，手动上传用 `upload:<timestamp>:<filename>` 作为来源标识，来源标识+内容hash 拼接后取 hash 得到 doc_id；`content_hash` 单独存 payload 用于增量判断（spec 第3节，已修订版）
- 目录监听：watchdog 实时监听 + 每小时兜底全量扫描（spec 第5节）
- 图片/扫描页 OCR 出的文字要同时写入 `text_chunks`（`source_type=ocr_image`），图片本身独立写入 `images`（spec 第3节）
- 入库任务里单文件失败不能拖垮整批任务，job 状态要记录逐文件成功/失败明细（spec 第7节）

---

## 文件结构总览

```
retrieval-service/
  docker-compose.yml
  Dockerfile
  requirements.txt
  .env.example
  scripts/
    check_gpu_env.py
  app/
    __init__.py
    config.py
    models/
      __init__.py
      embeddings.py         # BGE-M3 dense+sparse 封装
      clip_model.py          # Chinese-CLIP 图/文封装
    store/
      __init__.py
      qdrant_client.py        # collection 管理 + upsert/search/delete/stats
    ingest/
      __init__.py
      hashing.py               # content_hash / doc_id
      chunking.py               # 文本切分
      parsers/
        __init__.py
        text_parser.py           # txt/md
        csv_parser.py
        pdf_parser.py
        image_parser.py           # OCR + CLIP embed + 缩略图
      pipeline.py                  # 按文件类型分发、编排入库
      watcher.py                    # watchdog 目录监听 + 兜底扫描
    jobs/
      __init__.py
      queue.py                       # RQ/Redis 连接
      tasks.py                        # RQ job 函数
      status.py                        # job 状态读写
    api/
      __init__.py
      main.py                           # FastAPI app + 启动时加载模型
      deps.py                            # 依赖注入（store/embedder单例）
      schemas.py                          # pydantic 请求/响应模型
      routes/
        __init__.py
        health.py
        search.py
        documents.py
        directories.py
        jobs.py
        collections.py
    worker.py                                # RQ worker 入口，启动监听线程
    sdk/
      __init__.py
      langgraph_tools.py                        # @tool 封装
  admin_ui/
    app.py
  tests/
    conftest.py
    fixtures/
      sample.txt
      sample.md
      sample.csv
    unit/
      test_hashing.py
      test_chunking.py
      test_pipeline_dispatch.py
      test_jobs.py
      test_watcher_reconcile.py
      test_api_search.py
      test_api_documents.py
      test_sdk_tools.py
    integration/
      test_ingest_and_search.py
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `retrieval-service/requirements.txt`
- Create: `retrieval-service/.env.example`
- Create: `retrieval-service/app/__init__.py`
- Create: `retrieval-service/app/config.py`
- Create: `retrieval-service/.gitignore`

**Interfaces:**
- Produces: `app.config.Settings` 类（pydantic-settings），提供 `settings = Settings()` 单例，字段：`qdrant_host`, `qdrant_port`, `redis_url`, `data_dir`, `chunk_size`, `chunk_overlap`, `text_embed_model`, `clip_model_name`, `ocr_lang`, `watch_reconcile_interval_seconds`

- [ ] **Step 1: 创建 requirements.txt**

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.20
pydantic==2.10.4
pydantic-settings==2.7.0
qdrant-client==1.12.1
redis==5.2.1
rq==2.1.0
fakeredis==2.26.2
transformers==4.47.1
FlagEmbedding==1.3.4
pymupdf==1.25.1
pandas==2.2.3
paddleocr==2.9.1
paddlepaddle==2.6.2
pillow==11.0.0
watchdog==6.0.0
langchain-text-splitters==0.3.4
gradio==5.9.1
httpx==0.28.1
respx==0.22.0
pytest==8.3.4
pytest-asyncio==0.25.0
numpy==1.26.4
```

- [ ] **Step 2: 创建 .env.example**

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379/0
DATA_DIR=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
# JSON 数组字符串，pydantic-settings 按 JSON 解析 list 类型的环境变量
CHUNK_SEPARATORS=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
TEXT_EMBED_MODEL=BAAI/bge-m3
CLIP_MODEL_NAME=OFA-Sys/chinese-clip-vit-base-patch16
OCR_LANG=ch
WATCH_RECONCILE_INTERVAL_SECONDS=3600
```

- [ ] **Step 3: 创建 app/__init__.py**（空文件即可）

```python
```

- [ ] **Step 4: 创建 app/config.py**

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("./data")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_separators: list[str] = ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
    text_embed_model: str = "BAAI/bge-m3"
    clip_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    ocr_lang: str = "ch"
    watch_reconcile_interval_seconds: int = 3600

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: 创建 .gitignore**

```text
__pycache__/
*.pyc
.env
data/
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 6: 验证可导入**

Run: `cd retrieval-service && python -c "from app.config import settings; print(settings.chunk_size)"`
Expected: 输出 `1000`（此时 requirements 还未安装不影响，因为 config.py 只依赖 pydantic-settings，需先 `pip install pydantic-settings` 或整体装 requirements.txt）

Run: `cd retrieval-service && python -m venv .venv && .venv/bin/pip install -r requirements.txt`
Expected: 全部依赖安装成功（PaddleOCR/paddlepaddle 是 CPU 版，aarch64 官方有 wheel，不需要编译）

- [ ] **Step 7: Commit**

```bash
cd retrieval-service
git add requirements.txt .env.example app/__init__.py app/config.py .gitignore
git commit -m "chore: project scaffolding and settings"
```

---

### Task 2: GPU/模型环境验证脚本

**Files:**
- Create: `retrieval-service/scripts/check_gpu_env.py`

**Interfaces:**
- Produces: 无代码接口，产出的是"GPU+模型可用"这一验证结论，后续所有依赖 GPU 的任务都基于这个脚本跑通为前提

- [ ] **Step 1: 编写验证脚本**

```python
"""验证 DGX Spark (GB10) 上 PyTorch CUDA + BGE-M3 + Chinese-CLIP 是否可用。"""
import sys


def check_cuda() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA 不可用，检查 PyTorch 是否装了 cu130 aarch64 wheel"
    print(f"[OK] CUDA available. device={torch.cuda.get_device_name(0)}")


def check_bge_m3() -> None:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    output = model.encode(["你好，世界"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = output["dense_vecs"][0]
    sparse = output["lexical_weights"][0]
    assert len(dense) == 1024, f"期望 dense 向量维度 1024，实际 {len(dense)}"
    assert isinstance(sparse, dict) and len(sparse) > 0
    print(f"[OK] BGE-M3 dense dim={len(dense)}, sparse terms={len(sparse)}")


def check_chinese_clip() -> None:
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    model = ChineseCLIPModel.from_pretrained(model_name).to("cuda").eval()
    processor = ChineseCLIPProcessor.from_pretrained(model_name)

    image = Image.new("RGB", (224, 224), color="white")
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt").to("cuda")
        image_features = model.get_image_features(**image_inputs)
        text_inputs = processor(text=["一张白色的图片"], return_tensors="pt", padding=True).to("cuda")
        text_features = model.get_text_features(**text_inputs)

    print(f"[OK] Chinese-CLIP image dim={image_features.shape[-1]}, text dim={text_features.shape[-1]}")


if __name__ == "__main__":
    try:
        check_cuda()
        check_bge_m3()
        check_chinese_clip()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
    print("[OK] 环境验证全部通过")
```

- [ ] **Step 2: 运行验证**

Run: `cd retrieval-service && .venv/bin/python scripts/check_gpu_env.py`
Expected: 三行 `[OK]` 输出，最后 `[OK] 环境验证全部通过`。**Chinese-CLIP 实际输出维度以这里打印的为准**——如果不是 512，后续 Task 7（Qdrant collection 定义）里 `images` collection 的向量维度要相应调整为脚本实际打印的值。

若 `check_cuda` 失败：确认 `pip show torch` 版本里带 `+cu130`，没有的话重新 `pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall`。

- [ ] **Step 3: 记录实际结果到 README**

Create: `retrieval-service/README.md`

```markdown
# 多模态检索服务

设计文档: `docs/superpowers/specs/2026-08-25-multimodal-retrieval-service-design.md`

## 环境验证结果

运行 `scripts/check_gpu_env.py` 后记录：
- CUDA device: <运行后填入>
- BGE-M3 dense dim: 1024
- Chinese-CLIP image/text dim: <运行后填入，Task 7 的 collection 维度需与此一致>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/check_gpu_env.py README.md
git commit -m "feat: add GPU/model environment validation script"
```

---

### Task 3: content_hash / doc_id 生成

**Files:**
- Create: `retrieval-service/app/ingest/__init__.py`
- Create: `retrieval-service/app/ingest/hashing.py`
- Test: `retrieval-service/tests/unit/test_hashing.py`

**Interfaces:**
- Produces:
  - `def sha256_bytes(data: bytes) -> str`
  - `def sha256_file(path: Path) -> str`
  - `def make_doc_id(source_ref: str, content_hash: str) -> str`
  - `def upload_source_ref(filename: str) -> str`（生成 `upload:<iso timestamp>:<filename>`）

- [ ] **Step 1: 写失败的测试**

```python
# tests/unit/test_hashing.py
from pathlib import Path

from app.ingest.hashing import make_doc_id, sha256_bytes, sha256_file, upload_source_ref


def test_sha256_bytes_is_deterministic():
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")


def test_sha256_file_matches_bytes(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"content")
    assert sha256_file(f) == sha256_bytes(b"content")


def test_make_doc_id_deterministic_and_source_sensitive():
    h = sha256_bytes(b"same content")
    id_a = make_doc_id("/data/a.txt", h)
    id_b = make_doc_id("/data/b.txt", h)
    id_a_again = make_doc_id("/data/a.txt", h)
    assert id_a == id_a_again
    assert id_a != id_b


def test_upload_source_ref_contains_filename():
    ref = upload_source_ref("report.pdf")
    assert ref.startswith("upload:")
    assert ref.endswith(":report.pdf")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_hashing.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: 实现**

```python
# app/ingest/__init__.py
```

```python
# app/ingest/hashing.py
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_doc_id(source_ref: str, content_hash: str) -> str:
    return hashlib.sha256(f"{source_ref}|{content_hash}".encode("utf-8")).hexdigest()


def upload_source_ref(filename: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return f"upload:{ts}:{filename}"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_hashing.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/__init__.py app/ingest/hashing.py tests/unit/test_hashing.py
git commit -m "feat: add content hashing and doc_id generation"
```

---

### Task 4: 文本切分

**Files:**
- Create: `retrieval-service/app/ingest/chunking.py`
- Test: `retrieval-service/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `app.config.settings.chunk_size`, `app.config.settings.chunk_overlap`, `app.config.settings.chunk_separators`（Task 1 Step 4 `config.py` 里定义）
- Produces:
  - `def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None) -> list[str]`（三个参数都可覆盖全局默认值；`separators` 是按优先级尝试的切分标记列表，命中即在该处切开，用于支持"遇到特定字符也切分"，比如按 `"\n## "` 章节标题或自定义分隔符切）
  - `def chunk_csv_rows(rows: list[dict], header: list[str], rows_per_chunk: int = 20, max_chars: int | None = None) -> list[str]`（双重限制：达到 `rows_per_chunk` 行或者再加一行会超过 `max_chars` 字符，先到先切；保证返回的每个 chunk 都不超过 `max_chars`，即使单行内容本身很长也会独立成块）

- [ ] **Step 1: 写失败的测试**

```python
# tests/unit/test_chunking.py
from app.ingest.chunking import chunk_csv_rows, chunk_text


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("短文本", chunk_size=100, chunk_overlap=10)
    assert chunks == ["短文本"]


def test_chunk_text_long_text_multiple_chunks_with_overlap():
    text = "A" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_uses_default_settings_when_not_overridden():
    from app.config import settings

    text = "B" * (settings.chunk_size + 50)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= settings.chunk_size for c in chunks)


def test_chunk_text_splits_on_custom_separator():
    text = "第一段内容" + "---" + "第二段内容" + "---" + "第三段内容"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=0, separators=["---", ""])
    assert len(chunks) == 3
    assert chunks[0] == "第一段内容"
    assert chunks[1] == "第二段内容"
    assert chunks[2] == "第三段内容"


def test_chunk_csv_rows_groups_with_header_context():
    header = ["name", "age"]
    rows = [{"name": f"p{i}", "age": str(i)} for i in range(5)]
    chunks = chunk_csv_rows(rows, header, rows_per_chunk=2, max_chars=1000)
    assert len(chunks) == 3
    assert "name" in chunks[0] and "age" in chunks[0]
    assert "p0" in chunks[0] and "p1" in chunks[0]
    assert "p2" in chunks[1]


def test_chunk_csv_rows_splits_early_when_row_text_is_huge():
    header = ["name", "bio"]
    rows = [
        {"name": "p0", "bio": "x" * 900},
        {"name": "p1", "bio": "y" * 900},
        {"name": "p2", "bio": "short"},
    ]
    chunks = chunk_csv_rows(rows, header, rows_per_chunk=20, max_chars=1000)
    # p0 和 p1 单独一行就接近/超过 max_chars，各自都会触发提前 flush；
    # p2 很短，追加到 p1 所在的 chunk 后仍不超过 max_chars，所以和 p1 合并。
    assert len(chunks) == 2
    assert all(len(c) <= 1000 for c in chunks)
    assert "p0" in chunks[0]
    assert "p1" in chunks[1] and "p2" in chunks[1]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_chunking.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.chunking'`

- [ ] **Step 3: 实现**

```python
# app/ingest/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=separators or settings.chunk_separators,
    )
    return splitter.split_text(text)


def chunk_csv_rows(
    rows: list[dict], header: list[str], rows_per_chunk: int = 20, max_chars: int | None = None
) -> list[str]:
    max_chars = max_chars or settings.chunk_size
    header_line = f"columns: {', '.join(header)}"
    chunks: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            chunks.append(header_line + "\n" + "\n".join(current_lines))

    for row in rows:
        line = ", ".join(f"{k}={row.get(k, '')}" for k in header)
        single_row_len = len(header_line) + 1 + len(line)

        if single_row_len > max_chars:
            # 这一行自己就装不下（哪怕独占一个 chunk 也超限），先冲掉已攒的，
            # 再把这一行本身用 chunk_text 硬切，保证任何情况下产出的 chunk 都不超过 max_chars
            flush()
            current_lines = []
            piece_budget = max(max_chars - len(header_line) - 1, 1)
            for piece in chunk_text(line, chunk_size=piece_budget, chunk_overlap=0, separators=[""]):
                chunks.append(header_line + "\n" + piece)
            continue

        candidate_len = len(header_line) + 1 + sum(len(l) + 1 for l in current_lines) + len(line) + 1

        if current_lines and (len(current_lines) >= rows_per_chunk or candidate_len > max_chars):
            flush()
            current_lines = []

        current_lines.append(line)

    flush()
    return chunks
```

**这个函数最初的版本有个 bug（在本 task 的 code review 时发现）：** 单行内容本身（含表头）就超过 `max_chars` 时，因为 `current_lines` 是空的，触发不了"先 flush 再开新 chunk"的判断，会直接把这个超限的单行原样塞进 chunk，产出一个超过 `max_chars` 的 chunk——违反了"检索接口原样返回 chunk、不做二次截断"的前提。上面的实现加了 `single_row_len > max_chars` 分支，遇到这种超大单行就用 `chunk_text`（复用本文件里的函数）把这一行自己硬切开，保证任何情况下产出的 chunk 都不超过 `max_chars`，内容也不会被截断丢失。

- [ ] **Step 3b: 补一个超大单行的回归测试**

在 `tests/unit/test_chunking.py` 追加：

```python
def test_chunk_csv_rows_splits_single_oversized_row():
    header = ["name", "bio"]
    rows = [{"name": "p0", "bio": "z" * 3000}]

    chunks = chunk_csv_rows(rows, header, rows_per_chunk=20, max_chars=1000)

    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(c.count("z") for c in chunks) == 3000
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_chunking.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/chunking.py tests/unit/test_chunking.py
git commit -m "feat: add text and csv chunking"
```

---

### Task 5: BGE-M3 文本向量封装

**Files:**
- Create: `retrieval-service/app/models/__init__.py`
- Create: `retrieval-service/app/models/embeddings.py`
- Test: `retrieval-service/tests/unit/test_embeddings.py`

**Interfaces:**
- Consumes: `app.config.settings.text_embed_model`（Task 1）
- Produces:
  - `@dataclass class DenseSparseVector: dense: list[float]; sparse: dict[int, float]`
  - `class TextEmbedder: def embed(self, texts: list[str]) -> list[DenseSparseVector]`
  - `def get_text_embedder() -> TextEmbedder`（进程内单例，供 Task 12/16 使用）

- [ ] **Step 1: 写测试（需要真实模型，标记为慢测试）**

```python
# tests/unit/test_embeddings.py
import pytest

from app.models.embeddings import get_text_embedder

pytestmark = pytest.mark.slow


def test_embed_returns_dense_and_sparse():
    embedder = get_text_embedder()
    results = embedder.embed(["检索服务", "retrieval service"])
    assert len(results) == 2
    for r in results:
        assert len(r.dense) == 1024
        assert isinstance(r.sparse, dict)
        assert len(r.sparse) > 0


def test_get_text_embedder_is_singleton():
    assert get_text_embedder() is get_text_embedder()
```

在 `retrieval-service/pytest.ini` 里注册标记：

```ini
[pytest]
markers =
    slow: 需要真实模型/GPU 的测试，本地默认跳过，CI/上线前手动跑
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_embeddings.py -v -m slow`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: 实现**

```python
# app/models/__init__.py
```

```python
# app/models/embeddings.py
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings


@dataclass
class DenseSparseVector:
    dense: list[float]
    sparse: dict[int, float]


class TextEmbedder:
    def __init__(self, model_name: str):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3FlagModel(model_name, use_fp16=True, device=device)

    def embed(self, texts: list[str]) -> list[DenseSparseVector]:
        output = self._model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        results = []
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
            sparse_int_keys = {int(token_id): float(weight) for token_id, weight in sparse.items()}
            results.append(DenseSparseVector(dense=dense.tolist(), sparse=sparse_int_keys))
        return results


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder(settings.text_embed_model)
```

**这个函数最初的版本有个问题（code review 时发现）：** 没有显式传 `device="cuda"`，而是依赖 `BGEM3FlagModel` 内部的 `torch.cuda.is_available()` 自动探测。在这台机器上能跑是因为 GPU 确实在，但换到没装好 CUDA 的环境会不报错、不告警地静默退化到 CPU——而这条查询时向量化路径恰恰是对延迟敏感、必须走 GPU 的（架构设计的前提）。已经改成和 `scripts/check_gpu_env.py`、Task 6 的 Chinese-CLIP 封装一样的显式判断写法：`device = "cuda" if torch.cuda.is_available() else "cpu"`，再把 `device` 传给 `BGEM3FlagModel`。

- [ ] **Step 4: 运行测试确认通过（需要 GPU 环境，即 Task 2 验证通过的机器上跑）**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_embeddings.py -v -m slow`
Expected: 2 passed（首次运行会下载 BGE-M3 权重，需要网络）

- [ ] **Step 5: Commit**

```bash
git add app/models/__init__.py app/models/embeddings.py tests/unit/test_embeddings.py pytest.ini
git commit -m "feat: add BGE-M3 text embedder"
```

---

### Task 6: Chinese-CLIP 图文向量封装

**Files:**
- Create: `retrieval-service/app/models/clip_model.py`
- Test: `retrieval-service/tests/unit/test_clip_model.py`

**Interfaces:**
- Consumes: `app.config.settings.clip_model_name`（Task 1）
- Produces:
  - `class ClipEmbedder: def embed_image(self, image: PIL.Image.Image) -> list[float]`
  - `class ClipEmbedder: def embed_text(self, text: str) -> list[float]`
  - `def get_clip_embedder() -> ClipEmbedder`（进程内单例）
  - 模块级常量 `CLIP_VECTOR_DIM: int`（**Task 2 验证脚本的实际输出维度覆盖此处默认值 512**）

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_clip_model.py
import pytest
from PIL import Image

from app.models.clip_model import CLIP_VECTOR_DIM, get_clip_embedder

pytestmark = pytest.mark.slow


def test_embed_image_and_text_same_dim_and_normalized():
    embedder = get_clip_embedder()
    image = Image.new("RGB", (224, 224), color="red")
    image_vec = embedder.embed_image(image)
    text_vec = embedder.embed_text("一张红色的图片")

    assert len(image_vec) == CLIP_VECTOR_DIM
    assert len(text_vec) == CLIP_VECTOR_DIM

    import math

    norm = math.sqrt(sum(v * v for v in image_vec))
    assert 0.99 < norm < 1.01


def test_get_clip_embedder_is_singleton():
    assert get_clip_embedder() is get_clip_embedder()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_clip_model.py -v -m slow`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.models.clip_model'`

- [ ] **Step 3: 实现**

```python
# app/models/clip_model.py
from functools import lru_cache

import torch
from PIL import Image

from app.config import settings

CLIP_VECTOR_DIM = 512  # 若 Task 2 验证脚本打印的维度不同，改这里


class ClipEmbedder:
    def __init__(self, model_name: str):
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChineseCLIPModel.from_pretrained(model_name).to(self._device).eval()
        self._processor = ChineseCLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image: Image.Image) -> list[float]:
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()


@lru_cache(maxsize=1)
def get_clip_embedder() -> ClipEmbedder:
    return ClipEmbedder(settings.clip_model_name)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_clip_model.py -v -m slow`
Expected: 2 passed。**如果 `CLIP_VECTOR_DIM` 断言失败，把常量改成实际值后重跑。**

- [ ] **Step 5: Commit**

```bash
git add app/models/clip_model.py tests/unit/test_clip_model.py
git commit -m "feat: add Chinese-CLIP image/text embedder"
```

---

### Task 7: Qdrant 存储层

**Files:**
- Create: `retrieval-service/app/store/__init__.py`
- Create: `retrieval-service/app/store/qdrant_client.py`
- Test: `retrieval-service/tests/unit/test_qdrant_store.py`

**Interfaces:**
- Consumes: `DenseSparseVector`（Task 5），`CLIP_VECTOR_DIM`（Task 6）
- Produces:
  - `def get_qdrant_client(location: str | None = None) -> QdrantClient`
  - `def ensure_collections(client) -> None`
  - `@dataclass class TextChunkRecord: doc_id: str; chunk_index: int; text: str; source_path: str; file_type: str; page: int | None; source_type: str; content_hash: str`
  - `@dataclass class ImageRecord: doc_id: str; source_path: str; thumbnail_path: str; width: int; height: int; has_ocr_text: bool; content_hash: str`
  - `def upsert_text_chunk(client, record: TextChunkRecord, vector: DenseSparseVector) -> None`
  - `def upsert_image(client, record: ImageRecord, vector: list[float]) -> None`
  - `@dataclass class SearchHit: score: float; payload: dict`
  - `def search_text_hybrid(client, query: DenseSparseVector, top_k: int = 10) -> list[SearchHit]`
  - `def search_images(client, query_vector: list[float], top_k: int = 10) -> list[SearchHit]`
  - `def delete_document(client, doc_id: str) -> None`
  - `def get_existing_content_hash(client, doc_id: str) -> str | None`
  - `def list_chunks_for_document(client, doc_id: str) -> list[dict]`（scroll `text_chunks` collection 按 `doc_id` 过滤，按 `chunk_index` 升序返回每条的完整 payload；供 Task 24 的"查看已入库文档分片"接口用）
  - `def get_stats(client) -> dict`

- [ ] **Step 1: 写测试（用 qdrant-client 的内存模式，不需要真实 Qdrant 服务）**

```python
# tests/unit/test_qdrant_store.py
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import (
    ImageRecord,
    TextChunkRecord,
    delete_document,
    ensure_collections,
    get_existing_content_hash,
    get_qdrant_client,
    get_stats,
    list_chunks_for_document,
    search_images,
    search_text_hybrid,
    upsert_image,
    upsert_text_chunk,
)


def make_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client


def test_upsert_and_search_text_chunk():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5, 2: 0.3})
    record = TextChunkRecord(
        doc_id="doc1", chunk_index=0, text="你好世界", source_path="/a.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h1",
    )
    upsert_text_chunk(client, record, vec)

    hits = search_text_hybrid(client, vec, top_k=5)
    assert len(hits) == 1
    assert hits[0].payload["doc_id"] == "doc1"
    assert hits[0].payload["text"] == "你好世界"


def test_upsert_and_search_image():
    client = make_client()
    vector = [0.2] * 512
    record = ImageRecord(
        doc_id="img1", source_path="/a.png", thumbnail_path="/thumb/a.jpg",
        width=100, height=100, has_ocr_text=False, content_hash="h2",
    )
    upsert_image(client, record, vector)

    hits = search_images(client, vector, top_k=5)
    assert len(hits) == 1
    assert hits[0].payload["doc_id"] == "img1"


def test_get_existing_content_hash_none_when_missing():
    client = make_client()
    assert get_existing_content_hash(client, "unknown-doc") is None


def test_get_existing_content_hash_returns_hash_after_upsert():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc2", chunk_index=0, text="text", source_path="/b.txt",
        file_type="txt", page=None, source_type="normal", content_hash="hash-b",
    )
    upsert_text_chunk(client, record, vec)
    assert get_existing_content_hash(client, "doc2") == "hash-b"


def test_delete_document_removes_chunks():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc3", chunk_index=0, text="text", source_path="/c.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h3",
    )
    upsert_text_chunk(client, record, vec)
    delete_document(client, "doc3")
    assert get_existing_content_hash(client, "doc3") is None


def test_list_chunks_for_document_returns_ordered_payloads():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for idx in (2, 0, 1):
        record = TextChunkRecord(
            doc_id="doc5", chunk_index=idx, text=f"chunk-{idx}", source_path="/e.txt",
            file_type="txt", page=None, source_type="normal", content_hash="h5",
        )
        upsert_text_chunk(client, record, vec)

    chunks = list_chunks_for_document(client, "doc5")

    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
    assert [c["text"] for c in chunks] == ["chunk-0", "chunk-1", "chunk-2"]


def test_list_chunks_for_document_empty_when_missing():
    client = make_client()
    assert list_chunks_for_document(client, "unknown-doc") == []


def test_get_stats_reports_counts():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    record = TextChunkRecord(
        doc_id="doc4", chunk_index=0, text="text", source_path="/d.txt",
        file_type="txt", page=None, source_type="normal", content_hash="h4",
    )
    upsert_text_chunk(client, record, vec)
    stats = get_stats(client)
    assert stats["text_chunks_count"] == 1
    assert stats["images_count"] == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_qdrant_store.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: 实现**

```python
# app/store/__init__.py
```

```python
# app/store/qdrant_client.py
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings
from app.models.clip_model import CLIP_VECTOR_DIM
from app.models.embeddings import DenseSparseVector

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"
TEXT_DENSE_DIM = 1024


def get_qdrant_client(location: str | None = None) -> QdrantClient:
    if location:
        return QdrantClient(location=location)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collections(client: QdrantClient) -> None:
    if not client.collection_exists(TEXT_COLLECTION):
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=TEXT_DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    if not client.collection_exists(IMAGE_COLLECTION):
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config={"clip": models.VectorParams(size=CLIP_VECTOR_DIM, distance=models.Distance.COSINE)},
        )


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


@dataclass
class ImageRecord:
    doc_id: str
    source_path: str
    thumbnail_path: str
    width: int
    height: int
    has_ocr_text: bool
    content_hash: str


@dataclass
class SearchHit:
    score: float
    payload: dict


def _point_id(doc_id: str, chunk_index: int) -> str:
    import hashlib

    return hashlib.md5(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()


def upsert_text_chunk(client: QdrantClient, record: TextChunkRecord, vector: DenseSparseVector) -> None:
    point_id = _point_id(record.doc_id, record.chunk_index)
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
                },
            )
        ],
    )


def upsert_image(client: QdrantClient, record: ImageRecord, vector: list[float]) -> None:
    point_id = _point_id(record.doc_id, 0)
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
                },
            )
        ],
    )


def search_text_hybrid(client: QdrantClient, query: DenseSparseVector, top_k: int = 10) -> list[SearchHit]:
    result = client.query_points(
        collection_name=TEXT_COLLECTION,
        prefetch=[
            models.Prefetch(query=query.dense, using="dense", limit=top_k * 2),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(query.sparse.keys()), values=list(query.sparse.values())
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    return [SearchHit(score=p.score, payload=p.payload) for p in result.points]


def search_images(client: QdrantClient, query_vector: list[float], top_k: int = 10) -> list[SearchHit]:
    result = client.query_points(
        collection_name=IMAGE_COLLECTION, query=query_vector, using="clip", limit=top_k
    )
    return [SearchHit(score=p.score, payload=p.payload) for p in result.points]


def delete_document(client: QdrantClient, doc_id: str) -> None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    client.delete(collection_name=TEXT_COLLECTION, points_selector=models.FilterSelector(filter=flt))
    client.delete(collection_name=IMAGE_COLLECTION, points_selector=models.FilterSelector(filter=flt))


def get_existing_content_hash(client: QdrantClient, doc_id: str) -> str | None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
        points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1)
        if points:
            return points[0].payload["content_hash"]
    return None


def list_chunks_for_document(client: QdrantClient, doc_id: str) -> list[dict]:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(collection_name=TEXT_COLLECTION, scroll_filter=flt, limit=100, offset=offset)
        payloads.extend(p.payload for p in points)
        if offset is None:
            break
    return sorted(payloads, key=lambda p: p["chunk_index"])


def get_stats(client: QdrantClient) -> dict:
    text_info = client.get_collection(TEXT_COLLECTION)
    image_info = client.get_collection(IMAGE_COLLECTION)
    return {
        "text_chunks_count": text_info.points_count,
        "images_count": image_info.points_count,
    }


@lru_cache(maxsize=1)
def get_default_client() -> QdrantClient:
    client = get_qdrant_client()
    ensure_collections(client)
    return client
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_qdrant_store.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add app/store/__init__.py app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "feat: add Qdrant store layer with hybrid text search and image search"
```

---

### Task 8: txt/md 解析器

**Files:**
- Create: `retrieval-service/app/ingest/parsers/__init__.py`
- Create: `retrieval-service/app/ingest/parsers/text_parser.py`
- Test: `retrieval-service/tests/unit/test_text_parser.py`

**Interfaces:**
- Produces: `def parse_text_file(path: Path) -> str`（读取 txt/md 全文，md 不做特殊渲染，直接当纯文本切分）

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_text_parser.py
from pathlib import Path

from app.ingest.parsers.text_parser import parse_text_file


def test_parse_text_file_reads_utf8(tmp_path: Path):
    f = tmp_path / "note.md"
    f.write_text("# 标题\n正文内容", encoding="utf-8")
    assert parse_text_file(f) == "# 标题\n正文内容"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_text_parser.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.parsers'`

- [ ] **Step 3: 实现**

```python
# app/ingest/parsers/__init__.py
```

```python
# app/ingest/parsers/text_parser.py
from pathlib import Path


def parse_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_text_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/parsers/__init__.py app/ingest/parsers/text_parser.py tests/unit/test_text_parser.py
git commit -m "feat: add txt/md parser"
```

---

### Task 9: csv 解析器

**Files:**
- Create: `retrieval-service/app/ingest/parsers/csv_parser.py`
- Test: `retrieval-service/tests/unit/test_csv_parser.py`

**Interfaces:**
- Produces: `def parse_csv_file(path: Path) -> tuple[list[dict], list[str]]`（返回 (rows, header)）

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_csv_parser.py
from pathlib import Path

from app.ingest.parsers.csv_parser import parse_csv_file


def test_parse_csv_file_returns_rows_and_header(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nalice,30\nbob,25\n", encoding="utf-8")

    rows, header = parse_csv_file(f)

    assert header == ["name", "age"]
    assert rows == [{"name": "alice", "age": "30"}, {"name": "bob", "age": "25"}]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_csv_parser.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.parsers.csv_parser'`

- [ ] **Step 3: 实现**

```python
# app/ingest/parsers/csv_parser.py
from pathlib import Path

import pandas as pd


def parse_csv_file(path: Path) -> tuple[list[dict], list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    header = list(df.columns)
    rows = df.to_dict(orient="records")
    return rows, header
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_csv_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/parsers/csv_parser.py tests/unit/test_csv_parser.py
git commit -m "feat: add csv parser"
```

---

### Task 10: 图片解析器（OCR + CLIP + 缩略图）

**Files:**
- Create: `retrieval-service/app/ingest/parsers/image_parser.py`
- Test: `retrieval-service/tests/unit/test_image_parser.py`

**Interfaces:**
- Consumes: `get_clip_embedder()`（Task 6），`app.config.settings.thumbnails_dir`（Task 1）
- Produces:
  - `@dataclass class ParsedImage: ocr_text: str; clip_vector: list[float]; thumbnail_path: str; width: int; height: int`
  - `def parse_image(image: PIL.Image.Image, doc_id: str) -> ParsedImage`
  - `def run_ocr(image: PIL.Image.Image) -> str`（单独暴露，供 Task 11 PDF 解析器复用）

- [ ] **Step 1: 写测试（mock OCR 引擎和 CLIP，避免每次跑单元测试都要真实模型）**

```python
# tests/unit/test_image_parser.py
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
    from app.config import settings

    monkeypatch.setattr(settings, "thumbnails_dir", tmp_path)
    mock_get_clip.return_value.embed_image.return_value = [0.1] * 512

    result = parse_image(make_text_image(), doc_id="img-doc-1")

    assert result.ocr_text == "hello"
    assert len(result.clip_vector) == 512
    assert result.width == 200
    assert result.height == 80
    assert (tmp_path / "img-doc-1.jpg").exists()
    assert result.thumbnail_path == str(tmp_path / "img-doc-1.jpg")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_image_parser.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.parsers.image_parser'`

- [ ] **Step 3: 实现**

```python
# app/ingest/parsers/image_parser.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_image_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/parsers/image_parser.py tests/unit/test_image_parser.py
git commit -m "feat: add image parser with OCR, CLIP embedding, thumbnail"
```

---

### Task 11: PDF 解析器（文字层 + 扫描页 OCR fallback）

**Files:**
- Create: `retrieval-service/app/ingest/parsers/pdf_parser.py`
- Test: `retrieval-service/tests/unit/test_pdf_parser.py`

**Interfaces:**
- Consumes: `run_ocr`（Task 10）
- Produces:
  - `@dataclass class ParsedPdfPage: page_number: int; text: str; is_scanned: bool; page_image: PIL.Image.Image | None`
  - `def parse_pdf(path: Path, text_threshold: int = 20) -> list[ParsedPdfPage]`

- [ ] **Step 1: 写测试（用 PyMuPDF 现场生成测试 PDF，一页正常文字、一页纯图片触发 OCR 分支）**

```python
# tests/unit/test_pdf_parser.py
from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw

from app.ingest.parsers.pdf_parser import parse_pdf


def make_test_pdf(path):
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "这是可提取的文字层")

    page2 = doc.new_page()
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "scanned text", fill="black")
    img_path = path.parent / "scan.png"
    img.save(img_path)
    rect = fitz.Rect(0, 0, page2.rect.width, page2.rect.height)
    page2.insert_image(rect, filename=str(img_path))

    doc.save(path)
    doc.close()


@patch("app.ingest.parsers.pdf_parser.run_ocr", return_value="scanned text")
def test_parse_pdf_extracts_text_layer_and_ocr_fallback(mock_ocr, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    make_test_pdf(pdf_path)

    pages = parse_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0].is_scanned is False
    assert "可提取的文字层" in pages[0].text

    assert pages[1].is_scanned is True
    assert pages[1].text == "scanned text"
    assert pages[1].page_image is not None
    mock_ocr.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_pdf_parser.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.parsers.pdf_parser'`

- [ ] **Step 3: 实现**

```python
# app/ingest/parsers/pdf_parser.py
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
    doc = fitz.open(path)
    pages = []
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
    doc.close()
    return pages
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_pdf_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/parsers/pdf_parser.py tests/unit/test_pdf_parser.py
git commit -m "feat: add PDF parser with text layer extraction and OCR fallback"
```

---

### Task 12: 入库编排管线（按文件类型分发 + 增量去重）

**Files:**
- Create: `retrieval-service/app/ingest/pipeline.py`
- Test: `retrieval-service/tests/unit/test_pipeline_dispatch.py`

**Interfaces:**
- Consumes: `sha256_file`, `make_doc_id`（Task 3）; `chunk_text`, `chunk_csv_rows`（Task 4）; `get_text_embedder`（Task 5）; `get_clip_embedder`（Task 6）; `get_qdrant_client`, `ensure_collections`, `upsert_text_chunk`, `upsert_image`, `get_existing_content_hash`, `TextChunkRecord`, `ImageRecord`（Task 7）; `parse_text_file`（Task 8）; `parse_csv_file`（Task 9）; `parse_image`（Task 10）; `parse_pdf`（Task 11）
- Produces:
  - `@dataclass class IngestResult: doc_id: str; status: str; chunks_written: int; error: str | None`
  - `@dataclass class ChunkOptions: chunk_size: int | None = None; chunk_overlap: int | None = None; separators: list[str] | None = None`（三个字段任一为 `None` 时在 `chunk_text`/`chunk_csv_rows` 里回落到 `app.config.settings` 的全局默认值；本次入库传了值就只对本次生效，不改全局配置）
  - `def ingest_file(path: Path, source_ref: str, client=None, chunk_options: ChunkOptions | None = None) -> IngestResult`（`status` 取值：`"ingested"` / `"skipped_unchanged"` / `"failed"`）

- [ ] **Step 1: 写测试（mock 所有解析器和模型调用，只验证分发逻辑和去重逻辑）**

```python
# tests/unit/test_pipeline_dispatch.py
from pathlib import Path
from unittest.mock import patch

from app.ingest.pipeline import ingest_file
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import ensure_collections, get_qdrant_client


def make_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_txt_file_writes_chunks(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world " * 100, encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "ingested"
    assert result.chunks_written > 0


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_skips_unchanged_file(mock_get_embedder, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("stable content", encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    first = ingest_file(f, source_ref=str(f), client=client)
    second = ingest_file(f, source_ref=str(f), client=client)

    assert first.status == "ingested"
    assert second.status == "skipped_unchanged"
    assert first.doc_id == second.doc_id


def test_ingest_unsupported_extension_fails_gracefully(tmp_path):
    f = tmp_path / "note.xyz"
    f.write_text("data", encoding="utf-8")

    client = make_client()
    result = ingest_file(f, source_ref=str(f), client=client)

    assert result.status == "failed"
    assert result.error is not None


@patch("app.ingest.pipeline.get_text_embedder")
def test_ingest_honors_custom_chunk_options(mock_get_embedder, tmp_path):
    from app.ingest.pipeline import ChunkOptions

    f = tmp_path / "note.txt"
    f.write_text("word " * 400, encoding="utf-8")
    mock_get_embedder.return_value.embed.side_effect = lambda texts: [
        DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts
    ]

    client = make_client()
    small_chunks = ingest_file(f, source_ref=str(f) + ":a", client=client, chunk_options=ChunkOptions(chunk_size=50, chunk_overlap=0))
    large_chunks = ingest_file(f, source_ref=str(f) + ":b", client=client, chunk_options=ChunkOptions(chunk_size=5000, chunk_overlap=0))

    assert small_chunks.chunks_written > large_chunks.chunks_written
    assert large_chunks.chunks_written == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_pipeline_dispatch.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.pipeline'`

- [ ] **Step 3: 实现**

```python
# app/ingest/pipeline.py
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.ingest.chunking import chunk_csv_rows, chunk_text
from app.ingest.hashing import make_doc_id, sha256_file
from app.ingest.parsers.csv_parser import parse_csv_file
from app.ingest.parsers.image_parser import parse_image
from app.ingest.parsers.pdf_parser import parse_pdf
from app.ingest.parsers.text_parser import parse_text_file
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import (
    ImageRecord,
    TextChunkRecord,
    delete_document,
    get_default_client,
    get_existing_content_hash,
    upsert_image,
    upsert_text_chunk,
)

SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class IngestResult:
    doc_id: str
    status: str
    chunks_written: int
    error: str | None = None


@dataclass
class ChunkOptions:
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    separators: list[str] | None = None


def _write_text_chunks(client, doc_id: str, source_path: str, file_type: str, texts_with_page: list[tuple[str, int | None]], source_type: str, content_hash: str, chunk_options: ChunkOptions) -> int:
    embedder = get_text_embedder()
    written = 0
    for page, page_text in enumerate(texts_with_page):
        text, page_number = page_text
        chunks = chunk_text(
            text,
            chunk_size=chunk_options.chunk_size,
            chunk_overlap=chunk_options.chunk_overlap,
            separators=chunk_options.separators,
        )
        if not chunks:
            continue
        vectors = embedder.embed(chunks)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            record = TextChunkRecord(
                doc_id=doc_id,
                chunk_index=page * 10_000 + idx,
                text=chunk,
                source_path=source_path,
                file_type=file_type,
                page=page_number,
                source_type=source_type,
                content_hash=content_hash,
            )
            upsert_text_chunk(client, record, vector)
            written += 1
    return written


def ingest_file(path: Path, source_ref: str, client=None, chunk_options: ChunkOptions | None = None) -> IngestResult:
    client = client or get_default_client()
    chunk_options = chunk_options or ChunkOptions()
    ext = path.suffix.lower()

    try:
        content_hash = sha256_file(path)
        doc_id = make_doc_id(source_ref, content_hash)

        existing_hash = get_existing_content_hash(client, doc_id)
        if existing_hash == content_hash:
            return IngestResult(doc_id=doc_id, status="skipped_unchanged", chunks_written=0)

        if ext in SUPPORTED_TEXT_EXTS:
            text = parse_text_file(path)
            written = _write_text_chunks(
                client, doc_id, str(path), ext.lstrip("."), [(text, None)], "normal", content_hash, chunk_options
            )
        elif ext == ".csv":
            rows, header = parse_csv_file(path)
            chunks = chunk_csv_rows(rows, header, max_chars=chunk_options.chunk_size)
            written = _write_csv_chunks(client, doc_id, str(path), chunks, content_hash)
        elif ext in SUPPORTED_IMAGE_EXTS:
            image = Image.open(path).convert("RGB")
            parsed = parse_image(image, doc_id)
            upsert_image(
                client,
                ImageRecord(
                    doc_id=doc_id, source_path=str(path), thumbnail_path=parsed.thumbnail_path,
                    width=parsed.width, height=parsed.height, has_ocr_text=bool(parsed.ocr_text),
                    content_hash=content_hash,
                ),
                parsed.clip_vector,
            )
            written = 1
            if parsed.ocr_text:
                written += _write_text_chunks(
                    client, doc_id, str(path), ext.lstrip("."), [(parsed.ocr_text, None)], "ocr_image", content_hash, chunk_options
                )
        elif ext == ".pdf":
            pages = parse_pdf(path)
            written = 0
            for page in pages:
                source_type = "ocr_image" if page.is_scanned else "normal"
                written += _write_text_chunks(
                    client, doc_id, str(path), "pdf", [(page.text, page.page_number)], source_type, content_hash, chunk_options
                )
                if page.is_scanned and page.page_image is not None:
                    from app.models.clip_model import get_clip_embedder

                    page_doc_id = f"{doc_id}:page{page.page_number}"
                    clip_vector = get_clip_embedder().embed_image(page.page_image)
                    from app.ingest.parsers.image_parser import parse_image as _reuse  # noqa: F401
                    thumb = page.page_image.copy()
                    thumb.thumbnail((256, 256))
                    from app.config import settings

                    thumb_path = settings.thumbnails_dir / f"{page_doc_id}.jpg"
                    thumb.convert("RGB").save(thumb_path, "JPEG")
                    upsert_image(
                        client,
                        ImageRecord(
                            doc_id=page_doc_id, source_path=f"{path}#page={page.page_number}",
                            thumbnail_path=str(thumb_path), width=page.page_image.width,
                            height=page.page_image.height, has_ocr_text=True, content_hash=content_hash,
                        ),
                        clip_vector,
                    )
        else:
            raise ValueError(f"unsupported file extension: {ext}")

        return IngestResult(doc_id=doc_id, status="ingested", chunks_written=written)
    except Exception as exc:  # noqa: BLE001
        try:
            fallback_doc_id = make_doc_id(source_ref, sha256_file(path))
        except Exception:  # noqa: BLE001
            fallback_doc_id = source_ref
        return IngestResult(doc_id=fallback_doc_id, status="failed", chunks_written=0, error=str(exc))


def _write_csv_chunks(client, doc_id: str, source_path: str, chunks: list[str], content_hash: str) -> int:
    embedder = get_text_embedder()
    if not chunks:
        return 0
    vectors = embedder.embed(chunks)
    written = 0
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        record = TextChunkRecord(
            doc_id=doc_id, chunk_index=idx, text=chunk, source_path=source_path,
            file_type="csv", page=None, source_type="normal", content_hash=content_hash,
        )
        upsert_text_chunk(client, record, vector)
        written += 1
    return written


def delete_ingested_document(doc_id: str, client=None) -> None:
    client = client or get_default_client()
    delete_document(client, doc_id)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_pipeline_dispatch.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/pipeline.py tests/unit/test_pipeline_dispatch.py
git commit -m "feat: add ingestion pipeline with per-filetype dispatch and dedup"
```

---

### Task 13: Redis/RQ 任务队列与 job 状态

**Files:**
- Create: `retrieval-service/app/jobs/__init__.py`
- Create: `retrieval-service/app/jobs/queue.py`
- Create: `retrieval-service/app/jobs/status.py`
- Create: `retrieval-service/app/jobs/tasks.py`
- Test: `retrieval-service/tests/unit/test_jobs.py`

**Interfaces:**
- Consumes: `ingest_file`, `delete_ingested_document`（Task 12）
- Produces:
  - `def get_redis_connection()`（真实场景连 `settings.redis_url`）
  - `def get_queue(connection=None) -> rq.Queue`
  - `def create_job_status(redis_conn, job_id: str, total_files: int) -> None`
  - `def update_job_file_result(redis_conn, job_id: str, file_path: str, status: str, error: str | None = None) -> None`
  - `def get_job_status(redis_conn, job_id: str) -> dict`（含 `total`, `processed`, `files: {path: {status, error}}`）
  - `def ingest_file_job(file_path: str, source_ref: str, job_id: str, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None) -> None`
  - `def ingest_directory_job(dir_path: str, recursive: bool, watch: bool, job_id: str, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None) -> None`（`chunk_size`/`chunk_overlap`/`separators` 用原始类型而不是 `ChunkOptions` dataclass 传参，因为这几个值要经 RQ 序列化入队、也要从 FastAPI 请求体直接解出来，用基础类型两头都省转换）
  - `def delete_document_job(doc_id: str) -> None`

- [ ] **Step 1: 写测试（用 fakeredis 代替真实 Redis，用 RQ 的 SimpleWorker 同步执行 job）**

```python
# tests/unit/test_jobs.py
from unittest.mock import patch

import fakeredis
from rq import Queue, SimpleWorker

from app.jobs.status import create_job_status, get_job_status, update_job_file_result
from app.jobs.tasks import ingest_directory_job, ingest_file_job


def make_redis():
    return fakeredis.FakeStrictRedis()


def test_job_status_lifecycle():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-1", total_files=2)
    update_job_file_result(redis_conn, "job-1", "/a.txt", "ingested")
    update_job_file_result(redis_conn, "job-1", "/b.txt", "failed", error="bad file")

    status = get_job_status(redis_conn, "job-1")

    assert status["total"] == 2
    assert status["processed"] == 2
    assert status["files"]["/a.txt"]["status"] == "ingested"
    assert status["files"]["/b.txt"]["error"] == "bad file"


@patch("app.jobs.tasks.ingest_file")
def test_ingest_file_job_updates_status(mock_ingest_file, tmp_path):
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d1", status="ingested", chunks_written=3)

    redis_conn = make_redis()
    queue = Queue(connection=redis_conn, is_async=False)
    create_job_status(redis_conn, "job-2", total_files=1)

    queue.enqueue(ingest_file_job, str(tmp_path / "a.txt"), str(tmp_path / "a.txt"), "job-2")

    status = get_job_status(redis_conn, "job-2")
    assert status["files"][str(tmp_path / "a.txt")]["status"] == "ingested"


@patch("app.jobs.tasks.ingest_file")
def test_ingest_directory_job_enumerates_supported_files(mock_ingest_file, tmp_path):
    from app.ingest.pipeline import IngestResult

    mock_ingest_file.return_value = IngestResult(doc_id="d", status="ingested", chunks_written=1)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.unsupported").write_text("y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("z")

    redis_conn = make_redis()
    with patch("app.jobs.tasks.get_redis_connection", return_value=redis_conn):
        ingest_directory_job(str(tmp_path), recursive=True, watch=False, job_id="job-3")

    status = get_job_status(redis_conn, "job-3")
    assert status["total"] == 2
    assert mock_ingest_file.call_count == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_jobs.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.jobs'`

- [ ] **Step 3: 实现**

```python
# app/jobs/__init__.py
```

```python
# app/jobs/queue.py
import redis
from rq import Queue

from app.config import settings


def get_redis_connection():
    return redis.from_url(settings.redis_url)


def get_queue(connection=None) -> Queue:
    return Queue(connection=connection or get_redis_connection())
```

```python
# app/jobs/status.py
import json

JOB_KEY_PREFIX = "job:"


def _key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def create_job_status(redis_conn, job_id: str, total_files: int) -> None:
    redis_conn.set(_key(job_id), json.dumps({"total": total_files, "files": {}}))


def update_job_file_result(redis_conn, job_id: str, file_path: str, status: str, error: str | None = None) -> None:
    raw = redis_conn.get(_key(job_id))
    data = json.loads(raw) if raw else {"total": 0, "files": {}}
    data["files"][file_path] = {"status": status, "error": error}
    redis_conn.set(_key(job_id), json.dumps(data))


def get_job_status(redis_conn, job_id: str) -> dict:
    raw = redis_conn.get(_key(job_id))
    if not raw:
        return {"total": 0, "processed": 0, "files": {}}
    data = json.loads(raw)
    return {"total": data["total"], "processed": len(data["files"]), "files": data["files"]}
```

```python
# app/jobs/tasks.py
from pathlib import Path

from app.ingest.hashing import upload_source_ref
from app.ingest.pipeline import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_TEXT_EXTS,
    ChunkOptions,
    delete_ingested_document,
    ingest_file,
)
from app.jobs.queue import get_redis_connection
from app.jobs.status import create_job_status, update_job_file_result

SUPPORTED_EXTS = SUPPORTED_TEXT_EXTS | SUPPORTED_IMAGE_EXTS | {".csv", ".pdf"}


def ingest_file_job(
    file_path: str,
    source_ref: str,
    job_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    redis_conn = get_redis_connection()
    chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    result = ingest_file(Path(file_path), source_ref, chunk_options=chunk_options)
    update_job_file_result(redis_conn, job_id, file_path, result.status, result.error)


def ingest_directory_job(
    dir_path: str,
    recursive: bool,
    watch: bool,
    job_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    redis_conn = get_redis_connection()
    chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
    directory = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    create_job_status(redis_conn, job_id, total_files=len(files))
    for file_path in files:
        result = ingest_file(file_path, source_ref=str(file_path), chunk_options=chunk_options)
        update_job_file_result(redis_conn, job_id, str(file_path), result.status, result.error)

    if watch:
        from app.ingest.watcher import register_watch

        register_watch(
            redis_conn, dir_path, recursive,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators,
        )


def delete_document_job(doc_id: str) -> None:
    delete_ingested_document(doc_id)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_jobs.py -v`
Expected: 3 passed（Task 14 完成前 `register_watch` 尚未定义，但因为测试用 `watch=False` 不会触发该分支，不影响这里通过）

- [ ] **Step 5: Commit**

```bash
git add app/jobs/ tests/unit/test_jobs.py
git commit -m "feat: add RQ job queue, job status tracking, ingest/delete job functions"
```

---

### Task 14: 目录监听（watchdog + 兜底扫描）

**Files:**
- Create: `retrieval-service/app/ingest/watcher.py`
- Test: `retrieval-service/tests/unit/test_watcher_reconcile.py`

**Interfaces:**
- Consumes: `ingest_file`, `delete_ingested_document`（Task 12）；`get_redis_connection`（Task 13）
- Produces:
  - `def register_watch(redis_conn, dir_path: str, recursive: bool, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None) -> None`（把目录+切分参数一起记录进 Redis set `watched_directories`，保证该目录后续被 watchdog 或兜底扫描触发的入库都沿用同一套切分设置）
  - `def list_watched_directories(redis_conn) -> list[dict]`（每个 dict 含 `path`, `recursive`, `chunk_size`, `chunk_overlap`, `separators`）
  - `def reconcile_directory(dir_path: str, recursive: bool, known_hashes: dict[str, str]) -> dict`（纯函数：比较磁盘文件与已知 hash，返回 `{"to_ingest": [...], "unchanged": [...]}`，不做真正的 IO 之外的副作用）
  - `class DirectoryWatchManager: def start(self) -> None`（用 watchdog Observer 监听 `list_watched_directories` 里的目录，文件变化时调用 `ingest_file`/`delete_ingested_document`；同时起一个后台线程每 `settings.watch_reconcile_interval_seconds` 跑一次 `reconcile_directory` 兜底）

- [ ] **Step 1: 写测试（只测试可判定输入输出的 `reconcile_directory` 纯函数和 `register_watch`/`list_watched_directories`，不测试 watchdog 的 OS 事件本身——那部分留给集成测试用真实文件系统操作验证）**

```python
# tests/unit/test_watcher_reconcile.py
import fakeredis

from app.ingest.hashing import sha256_file
from app.ingest.watcher import list_watched_directories, reconcile_directory, register_watch


def test_register_and_list_watched_directories():
    redis_conn = fakeredis.FakeStrictRedis()
    register_watch(redis_conn, "/data/docs", recursive=True, chunk_size=500, chunk_overlap=50, separators=["\n\n"])
    register_watch(redis_conn, "/data/notes", recursive=False)

    watched = list_watched_directories(redis_conn)

    assert {
        "path": "/data/docs", "recursive": True,
        "chunk_size": 500, "chunk_overlap": 50, "separators": ["\n\n"],
    } in watched
    assert {
        "path": "/data/notes", "recursive": False,
        "chunk_size": None, "chunk_overlap": None, "separators": None,
    } in watched


def test_reconcile_directory_detects_new_and_changed_files(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("original")
    f2 = tmp_path / "b.txt"
    f2.write_text("unchanged")

    known_hashes = {
        str(f2): sha256_file(f2),
        str(f1): "stale-hash-not-matching",
    }

    result = reconcile_directory(str(tmp_path), recursive=False, known_hashes=known_hashes)

    assert str(f1) in result["to_ingest"]
    assert str(f2) in result["unchanged"]


def test_reconcile_directory_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "a.xyz").write_text("data")

    result = reconcile_directory(str(tmp_path), recursive=False, known_hashes={})

    assert result["to_ingest"] == []
    assert result["unchanged"] == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_watcher_reconcile.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.ingest.watcher'`

- [ ] **Step 3: 实现**

```python
# app/ingest/watcher.py
import json
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.config import settings
from app.ingest.hashing import sha256_file
from app.ingest.pipeline import SUPPORTED_IMAGE_EXTS, SUPPORTED_TEXT_EXTS, delete_ingested_document, ingest_file

SUPPORTED_EXTS = SUPPORTED_TEXT_EXTS | SUPPORTED_IMAGE_EXTS | {".csv", ".pdf"}
WATCHED_DIRS_KEY = "watched_directories"


def register_watch(
    redis_conn,
    dir_path: str,
    recursive: bool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    entry = {
        "path": dir_path,
        "recursive": recursive,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    }
    redis_conn.sadd(WATCHED_DIRS_KEY, json.dumps(entry))


def list_watched_directories(redis_conn) -> list[dict]:
    return [json.loads(raw) for raw in redis_conn.smembers(WATCHED_DIRS_KEY)]


def reconcile_directory(dir_path: str, recursive: bool, known_hashes: dict[str, str]) -> dict:
    directory = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    to_ingest, unchanged = [], []

    for path in directory.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        current_hash = sha256_file(path)
        if known_hashes.get(str(path)) == current_hash:
            unchanged.append(str(path))
        else:
            to_ingest.append(str(path))

    return {"to_ingest": to_ingest, "unchanged": unchanged}


class _EventHandler(FileSystemEventHandler):
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None):
        self._debounce: dict[str, float] = {}
        from app.ingest.pipeline import ChunkOptions

        self._chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    def _should_handle(self, path: str) -> bool:
        if Path(path).suffix.lower() not in SUPPORTED_EXTS:
            return False
        now = time.time()
        last = self._debounce.get(path, 0)
        self._debounce[path] = now
        return now - last > 1.0

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            ingest_file(Path(event.src_path), source_ref=event.src_path, chunk_options=self._chunk_options)

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            ingest_file(Path(event.src_path), source_ref=event.src_path, chunk_options=self._chunk_options)

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in SUPPORTED_EXTS:
            from app.ingest.hashing import make_doc_id

            doc_id = make_doc_id(event.src_path, "")
            delete_ingested_document(doc_id)


class DirectoryWatchManager:
    def __init__(self, redis_conn):
        self._redis_conn = redis_conn
        self._observer = Observer()
        self._stop_event = threading.Event()

    def start(self) -> None:
        for entry in list_watched_directories(self._redis_conn):
            handler = _EventHandler(
                chunk_size=entry.get("chunk_size"),
                chunk_overlap=entry.get("chunk_overlap"),
                separators=entry.get("separators"),
            )
            self._observer.schedule(handler, entry["path"], recursive=entry["recursive"])
        self._observer.start()

        thread = threading.Thread(target=self._reconcile_loop, daemon=True)
        thread.start()

    def _reconcile_loop(self) -> None:
        while not self._stop_event.wait(settings.watch_reconcile_interval_seconds):
            for entry in list_watched_directories(self._redis_conn):
                from app.store.qdrant_client import get_default_client, get_existing_content_hash
                from app.ingest.hashing import make_doc_id

                client = get_default_client()
                directory = Path(entry["path"])
                pattern = "**/*" if entry["recursive"] else "*"
                known_hashes = {}
                for path in directory.glob(pattern):
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                        doc_id = make_doc_id(str(path), "")
                        existing = get_existing_content_hash(client, doc_id)
                        if existing:
                            known_hashes[str(path)] = existing
                result = reconcile_directory(entry["path"], entry["recursive"], known_hashes)
                from app.ingest.pipeline import ChunkOptions

                chunk_options = ChunkOptions(
                    chunk_size=entry.get("chunk_size"),
                    chunk_overlap=entry.get("chunk_overlap"),
                    separators=entry.get("separators"),
                )
                for file_path in result["to_ingest"]:
                    ingest_file(Path(file_path), source_ref=file_path, chunk_options=chunk_options)

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_watcher_reconcile.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/ingest/watcher.py tests/unit/test_watcher_reconcile.py
git commit -m "feat: add directory watcher with watchdog events and hourly reconciliation"
```

---

### Task 15: FastAPI 应用骨架 + 健康检查

**Files:**
- Create: `retrieval-service/app/api/__init__.py`
- Create: `retrieval-service/app/api/deps.py`
- Create: `retrieval-service/app/api/main.py`
- Create: `retrieval-service/app/api/routes/__init__.py`
- Create: `retrieval-service/app/api/routes/health.py`
- Test: `retrieval-service/tests/unit/test_api_health.py`

**Interfaces:**
- Consumes: `get_default_client`（Task 7）, `get_text_embedder`（Task 5）, `get_clip_embedder`（Task 6）
- Produces:
  - `app.api.main.app`（FastAPI 实例，供后续所有路由挂载和测试用 `TestClient(app)`）
  - `app.api.deps.get_store_client`, `app.api.deps.get_text_embedder_dep`, `app.api.deps.get_clip_embedder_dep`（FastAPI `Depends` 用的依赖函数，测试时可通过 `app.dependency_overrides` 替换）

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_api_health.py
from fastapi.testclient import TestClient

from app.api.main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_health.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.api'`

- [ ] **Step 3: 实现**

```python
# app/api/__init__.py
```

```python
# app/api/deps.py
from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import get_default_client


def get_store_client():
    return get_default_client()


def get_text_embedder_dep():
    return get_text_embedder()


def get_clip_embedder_dep():
    return get_clip_embedder()
```

```python
# app/api/routes/__init__.py
```

```python
# app/api/routes/health.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}
```

```python
# app/api/main.py
from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_health.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/__init__.py app/api/deps.py app/api/main.py app/api/routes/__init__.py app/api/routes/health.py tests/unit/test_api_health.py
git commit -m "feat: add FastAPI app skeleton with health endpoint and dependency injection"
```

---

### Task 16: 检索接口 `/search/text`、`/search/image`

**Files:**
- Create: `retrieval-service/app/api/schemas.py`
- Create: `retrieval-service/app/api/routes/search.py`
- Modify: `retrieval-service/app/api/main.py`
- Test: `retrieval-service/tests/unit/test_api_search.py`

**Interfaces:**
- Consumes: `get_store_client`, `get_text_embedder_dep`, `get_clip_embedder_dep`（Task 15）; `search_text_hybrid`, `search_images`（Task 7）
- Produces: `POST /search/text`（body: `{query: str, top_k: int = 10, modalities: list[str] = ["text"]}` → `{text_hits: [...], image_hits: [...]}`），`POST /search/image`（multipart 文件上传 `image`，query param `top_k`）→ `{image_hits: [...], related_text_hits: [...]}`

- [ ] **Step 1: 写测试（用 `app.dependency_overrides` 注入假的 embedder/store，不需要真实模型/Qdrant）**

```python
# tests/unit/test_api_search.py
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_clip_embedder_dep, get_store_client, get_text_embedder_dep
from app.api.main import app
from app.models.embeddings import DenseSparseVector
from app.store.qdrant_client import SearchHit


class FakeTextEmbedder:
    def embed(self, texts):
        return [DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5}) for _ in texts]


class FakeClipEmbedder:
    def embed_text(self, text):
        return [0.2] * 512

    def embed_image(self, image):
        return [0.3] * 512


class FakeStoreClient:
    pass


def override_search_text_hybrid(*args, **kwargs):
    return [SearchHit(score=0.9, payload={"doc_id": "d1", "text": "命中文本"})]


def override_search_images(*args, **kwargs):
    return [SearchHit(score=0.8, payload={"doc_id": "img1", "source_path": "/a.png"})]


def make_client(monkeypatch):
    app.dependency_overrides[get_text_embedder_dep] = lambda: FakeTextEmbedder()
    app.dependency_overrides[get_clip_embedder_dep] = lambda: FakeClipEmbedder()
    app.dependency_overrides[get_store_client] = lambda: FakeStoreClient()
    monkeypatch.setattr("app.api.routes.search.search_text_hybrid", override_search_text_hybrid)
    monkeypatch.setattr("app.api.routes.search.search_images", override_search_images)
    return TestClient(app)


def test_search_text_returns_text_hits(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/search/text", json={"query": "你好", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["text_hits"][0]["payload"]["doc_id"] == "d1"
    assert body["image_hits"] == []
    app.dependency_overrides.clear()


def test_search_text_with_image_modality_returns_both(monkeypatch):
    client = make_client(monkeypatch)
    response = client.post("/search/text", json={"query": "你好", "top_k": 5, "modalities": ["text", "image"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["text_hits"]) == 1
    assert len(body["image_hits"]) == 1
    app.dependency_overrides.clear()


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
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_search.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.api.schemas'`

- [ ] **Step 3: 实现**

```python
# app/api/schemas.py
from pydantic import BaseModel


class SearchTextRequest(BaseModel):
    query: str
    top_k: int = 10
    modalities: list[str] = ["text"]


class SearchHitOut(BaseModel):
    score: float
    payload: dict


class SearchTextResponse(BaseModel):
    text_hits: list[SearchHitOut]
    image_hits: list[SearchHitOut]


class SearchImageResponse(BaseModel):
    image_hits: list[SearchHitOut]
    related_text_hits: list[SearchHitOut]
```

```python
# app/api/routes/search.py
from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image
import io

from app.api.deps import get_clip_embedder_dep, get_store_client, get_text_embedder_dep
from app.api.schemas import SearchHitOut, SearchImageResponse, SearchTextRequest, SearchTextResponse
from app.store.qdrant_client import search_images, search_text_hybrid

router = APIRouter()


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
        text_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in text_hits],
        image_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in image_hits],
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

    return SearchImageResponse(
        image_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in image_hits],
        related_text_hits=[],
    )
```

Modify `app/api/main.py`：

```python
# app/api/main.py
from fastapi import FastAPI

from app.api.routes import health, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_search.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/schemas.py app/api/routes/search.py app/api/main.py tests/unit/test_api_search.py
git commit -m "feat: add /search/text and /search/image endpoints"
```

---

### Task 17: 文档上传/删除接口

**Files:**
- Create: `retrieval-service/app/api/routes/documents.py`
- Modify: `retrieval-service/app/api/schemas.py`
- Modify: `retrieval-service/app/api/main.py`
- Test: `retrieval-service/tests/unit/test_api_documents.py`

**Interfaces:**
- Consumes: `get_queue`（Task 13）, `ingest_file_job`, `delete_document_job`（Task 13）, `upload_source_ref`（Task 3）
- Produces: `POST /documents/upload`（multipart `file` + 可选 Form 字段 `chunk_size`、`chunk_overlap`、`separators`，`separators` 用同名字段重复出现表示多个值，不传就都是 `None` 回落到全局默认）→ `{job_id: str}`；`DELETE /documents/{doc_id}` → `{status: "deleted"}`

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_api_documents.py
import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import app


@patch("app.api.routes.documents.get_queue")
def test_upload_document_enqueues_job_and_returns_job_id(mock_get_queue):
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-123")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/documents/upload", files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-job-123"
    mock_queue.enqueue.assert_called_once()
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["chunk_size"] is None
    assert kwargs["separators"] is None


@patch("app.api.routes.documents.get_queue")
def test_upload_document_with_custom_chunk_options(mock_get_queue):
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-job-456")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/documents/upload",
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
        data=[("chunk_size", "500"), ("chunk_overlap", "50"), ("separators", "\n\n"), ("separators", "---")],
    )

    assert response.status_code == 200
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["chunk_size"] == 500
    assert kwargs["chunk_overlap"] == 50
    assert kwargs["separators"] == ["\n\n", "---"]


@patch("app.api.routes.documents.get_queue")
def test_delete_document_enqueues_delete_job(mock_get_queue):
    mock_queue = MagicMock()
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.delete("/documents/some-doc-id")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_queue.enqueue.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_documents.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.api.routes.documents'`

- [ ] **Step 3: 实现**

在 `app/api/schemas.py` 追加：

```python
class UploadResponse(BaseModel):
    job_id: str


class DeleteResponse(BaseModel):
    status: str
```

```python
# app/api/routes/documents.py
from fastapi import APIRouter, File, Form, UploadFile

from app.api.schemas import DeleteResponse, UploadResponse
from app.config import settings
from app.ingest.hashing import upload_source_ref
from app.jobs.queue import get_queue
from app.jobs.tasks import delete_document_job, ingest_file_job

router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
    separators: list[str] | None = Form(None),
):
    source_ref = upload_source_ref(file.filename)
    dest_dir = settings.uploads_dir / source_ref.replace(":", "_").replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    content = await file.read()
    dest_path.write_bytes(content)

    queue = get_queue()
    job = queue.enqueue(
        ingest_file_job,
        str(dest_path),
        source_ref,
        str(dest_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return UploadResponse(job_id=job.id)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    queue = get_queue()
    queue.enqueue(delete_document_job, doc_id)
    return DeleteResponse(status="deleted")
```

Modify `app/api/main.py`：

```python
# app/api/main.py
from fastapi import FastAPI

from app.api.routes import documents, health, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_documents.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/documents.py app/api/schemas.py app/api/main.py tests/unit/test_api_documents.py
git commit -m "feat: add document upload and delete endpoints"
```

---

### Task 18: 目录入库、任务状态、统计接口

**Files:**
- Create: `retrieval-service/app/api/routes/directories.py`
- Create: `retrieval-service/app/api/routes/jobs.py`
- Create: `retrieval-service/app/api/routes/collections.py`
- Modify: `retrieval-service/app/api/schemas.py`
- Modify: `retrieval-service/app/api/main.py`
- Test: `retrieval-service/tests/unit/test_api_directories_jobs_collections.py`

**Interfaces:**
- Consumes: `get_queue`, `ingest_directory_job`（Task 13）; `get_job_status`, `get_redis_connection`（Task 13）; `get_stats`, `get_store_client`（Task 7/15）
- Produces: `POST /directories/index`（body `{path, recursive=True, watch=False, chunk_size=None, chunk_overlap=None, separators=None}`，切分参数不传则整批文件用全局默认；若 `watch=True`，这批切分参数会跟随监听配置持久化，供后续新文件沿用，见 Task 14）→ `{job_id}`；`GET /jobs/{job_id}` → job 状态；`GET /collections/stats` → `{text_chunks_count, images_count}`

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_api_directories_jobs_collections.py
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.deps import get_store_client
from app.api.main import app


@patch("app.api.routes.directories.get_queue")
def test_index_directory_enqueues_job(mock_get_queue):
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="rq-dir-job")
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post("/directories/index", json={"path": "/data/docs", "recursive": True, "watch": True})

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-dir-job"
    args, _ = mock_queue.enqueue.call_args
    assert args[1] == "/data/docs"
    assert args[2] is True
    assert args[3] is True


@patch("app.api.routes.jobs.get_job_status")
@patch("app.api.routes.jobs.get_redis_connection")
def test_get_job_status_returns_progress(mock_get_conn, mock_get_status):
    mock_get_status.return_value = {"total": 2, "processed": 1, "files": {"/a.txt": {"status": "ingested", "error": None}}}

    client = TestClient(app)
    response = client.get("/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["processed"] == 1


def test_get_collections_stats():
    app.dependency_overrides[get_store_client] = lambda: object()
    with patch("app.api.routes.collections.get_stats", return_value={"text_chunks_count": 5, "images_count": 2}):
        client = TestClient(app)
        response = client.get("/collections/stats")

    assert response.status_code == 200
    assert response.json() == {"text_chunks_count": 5, "images_count": 2}
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_directories_jobs_collections.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.api.routes.directories'`

- [ ] **Step 3: 实现**

在 `app/api/schemas.py` 追加：

```python
class IndexDirectoryRequest(BaseModel):
    path: str
    recursive: bool = True
    watch: bool = False
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    separators: list[str] | None = None


class JobStatusResponse(BaseModel):
    total: int
    processed: int
    files: dict


class CollectionsStatsResponse(BaseModel):
    text_chunks_count: int
    images_count: int
```

```python
# app/api/routes/directories.py
from fastapi import APIRouter

from app.api.schemas import IndexDirectoryRequest, UploadResponse
from app.jobs.queue import get_queue
from app.jobs.tasks import ingest_directory_job

router = APIRouter()


@router.post("/directories/index", response_model=UploadResponse)
def index_directory(request: IndexDirectoryRequest):
    queue = get_queue()
    job_id_placeholder = f"dir:{request.path}"
    job = queue.enqueue(
        ingest_directory_job, request.path, request.recursive, request.watch, job_id_placeholder
    )
    return UploadResponse(job_id=job.id)
```

```python
# app/api/routes/jobs.py
from fastapi import APIRouter

from app.api.schemas import JobStatusResponse
from app.jobs.queue import get_redis_connection
from app.jobs.status import get_job_status

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    redis_conn = get_redis_connection()
    status = get_job_status(redis_conn, job_id)
    return JobStatusResponse(**status)
```

```python
# app/api/routes/collections.py
from fastapi import APIRouter, Depends

from app.api.deps import get_store_client
from app.api.schemas import CollectionsStatsResponse
from app.store.qdrant_client import get_stats

router = APIRouter()


@router.get("/collections/stats", response_model=CollectionsStatsResponse)
def collections_stats(store=Depends(get_store_client)):
    stats = get_stats(store)
    return CollectionsStatsResponse(**stats)
```

Modify `app/api/main.py`：

```python
# app/api/main.py
from fastapi import FastAPI

from app.api.routes import collections, directories, documents, health, jobs, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(directories.router)
app.include_router(jobs.router)
app.include_router(collections.router)
```

**注意：** `index_directory` 里用 `job_id_placeholder`（基于路径拼的字符串）作为传给 `ingest_directory_job` 的第4个参数（真正写入 job 状态用的 key），而返回给调用方的 `job.id` 是 RQ 自己生成的 job id——两者不是同一个值。这是本任务遗留的一个已知不一致，下一步（Step 4 之后）需要修正为统一使用 RQ 生成的 `job.id` 作为状态 key，方案是改用 `queue.enqueue` 时先不指定，用 `job.id` 里的值通过 RQ 的 `on_success`/`meta` 机制传给任务函数——但这需要改 `ingest_directory_job` 签名。为避免过度设计，这里改成更直接的方式：`enqueue` 后立刻用返回的 `job.id` 覆盖调用（见下一步）。

- [ ] **Step 3b: 修正 job_id 一致性问题**

```python
# app/api/routes/directories.py
from fastapi import APIRouter

from app.api.schemas import IndexDirectoryRequest, UploadResponse
from app.jobs.queue import get_queue

router = APIRouter()


@router.post("/directories/index", response_model=UploadResponse)
def index_directory(request: IndexDirectoryRequest):
    from app.jobs.tasks import ingest_directory_job

    queue = get_queue()
    job = queue.enqueue(
        ingest_directory_job,
        args=(request.path, request.recursive, request.watch),
        kwargs={
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "separators": request.separators,
        },
        job_id=None,
    )
    job.args = (*job.args, job.id)
    job.save()
    return UploadResponse(job_id=job.id)
```

这里改为：先 enqueue 拿到 RQ 生成的 `job.id`，再把这个 id 补进任务参数并 `job.save()` 持久化，保证 `GET /jobs/{job_id}` 用的 key 和 `POST /directories/index` 返回的 `job_id` 是同一个值；切分参数走 `kwargs` 单独传，不受这个 id 回填逻辑影响。相应地更新测试：

```python
@patch("app.api.routes.directories.get_queue")
def test_index_directory_enqueues_job(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job", args=("/data/docs", True, True))
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = mock_job
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post("/directories/index", json={"path": "/data/docs", "recursive": True, "watch": True})

    assert response.status_code == 200
    assert response.json()["job_id"] == "rq-dir-job"
    mock_job.save.assert_called_once()
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["kwargs"] == {"chunk_size": None, "chunk_overlap": None, "separators": None}


@patch("app.api.routes.directories.get_queue")
def test_index_directory_passes_custom_chunk_options(mock_get_queue):
    mock_job = MagicMock(id="rq-dir-job-2", args=("/data/docs", True, False))
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = mock_job
    mock_get_queue.return_value = mock_queue

    client = TestClient(app)
    response = client.post(
        "/directories/index",
        json={
            "path": "/data/docs", "recursive": True, "watch": False,
            "chunk_size": 800, "chunk_overlap": 100, "separators": ["\n\n", "---"],
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_queue.enqueue.call_args
    assert kwargs["kwargs"] == {"chunk_size": 800, "chunk_overlap": 100, "separators": ["\n\n", "---"]}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_directories_jobs_collections.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/directories.py app/api/routes/jobs.py app/api/routes/collections.py app/api/schemas.py app/api/main.py tests/unit/test_api_directories_jobs_collections.py
git commit -m "feat: add directory indexing, job status, and collection stats endpoints"
```

---

### Task 19: RQ Worker 入口

**Files:**
- Create: `retrieval-service/app/worker.py`

**Interfaces:**
- Consumes: `get_redis_connection`（Task 13）, `DirectoryWatchManager`（Task 14）
- Produces: 可执行入口 `python -m app.worker`，启动 RQ worker 消费默认队列，同时启动目录监听后台线程

- [ ] **Step 1: 实现**

```python
# app/worker.py
import logging

from rq import Worker

from app.ingest.watcher import DirectoryWatchManager
from app.jobs.queue import get_redis_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    redis_conn = get_redis_connection()

    watch_manager = DirectoryWatchManager(redis_conn)
    watch_manager.start()
    logger.info("directory watch manager started")

    worker = Worker(["default"], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 手动验证可启动（需要 Redis 已运行，本地可用 `docker run -p 6379:6379 redis:7`）**

Run: `cd retrieval-service && docker run -d --name test-redis -p 6379:6379 redis:7`
Run: `cd retrieval-service && timeout 5 .venv/bin/python -m app.worker || true`
Expected: 日志打印 `directory watch manager started` 和 RQ worker 启动信息，5秒后被 timeout 正常终止（没有 traceback）

Run: `docker rm -f test-redis`

- [ ] **Step 3: Commit**

```bash
git add app/worker.py
git commit -m "feat: add RQ worker entrypoint with directory watch manager"
```

---

### Task 20: Gradio 管理界面

**Files:**
- Create: `retrieval-service/admin_ui/app.py`

**Interfaces:**
- Consumes: 纯 HTTP 调用 API（`/search/text`, `/search/image`, `/documents/upload`, `/directories/index`, `/jobs/{id}`, `/collections/stats`），通过环境变量 `API_BASE_URL` 配置目标地址

- [ ] **Step 1: 实现**

```python
# admin_ui/app.py
import os

import gradio as gr
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def do_search_text(query: str, top_k: int, search_images_too: bool):
    modalities = ["text", "image"] if search_images_too else ["text"]
    response = httpx.post(
        f"{API_BASE_URL}/search/text", json={"query": query, "top_k": int(top_k), "modalities": modalities}, timeout=30
    )
    response.raise_for_status()
    body = response.json()
    text_lines = [f"[{h['score']:.3f}] {h['payload'].get('text', '')[:200]}" for h in body["text_hits"]]
    image_lines = [f"[{h['score']:.3f}] {h['payload'].get('source_path', '')}" for h in body["image_hits"]]
    return "\n\n".join(text_lines) or "无结果", "\n".join(image_lines) or "无结果"


def do_search_image(image_path: str, top_k: int):
    with open(image_path, "rb") as f:
        response = httpx.post(
            f"{API_BASE_URL}/search/image", files={"image": f}, params={"top_k": int(top_k)}, timeout=30
        )
    response.raise_for_status()
    body = response.json()
    lines = [f"[{h['score']:.3f}] {h['payload'].get('source_path', '')}" for h in body["image_hits"]]
    return "\n".join(lines) or "无结果"


def do_upload(file_path: str, chunk_size: int, chunk_overlap: int, separators_text: str):
    data = {}
    if chunk_size:
        data["chunk_size"] = str(int(chunk_size))
    if chunk_overlap:
        data["chunk_overlap"] = str(int(chunk_overlap))
    separators = [s for s in separators_text.split(",") if s] if separators_text else []

    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{API_BASE_URL}/documents/upload",
            files={"file": f},
            data=[(k, v) for k, v in data.items()] + [("separators", s) for s in separators],
            timeout=60,
        )
    response.raise_for_status()
    return f"已提交，job_id={response.json()['job_id']}"


def do_index_directory(path: str, recursive: bool, watch: bool, chunk_size: int, chunk_overlap: int, separators_text: str):
    separators = [s for s in separators_text.split(",") if s] if separators_text else None
    response = httpx.post(
        f"{API_BASE_URL}/directories/index",
        json={
            "path": path, "recursive": recursive, "watch": watch,
            "chunk_size": int(chunk_size) if chunk_size else None,
            "chunk_overlap": int(chunk_overlap) if chunk_overlap else None,
            "separators": separators,
        },
        timeout=30,
    )
    response.raise_for_status()
    return f"已提交，job_id={response.json()['job_id']}"


def do_check_job(job_id: str):
    response = httpx.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def do_stats():
    response = httpx.get(f"{API_BASE_URL}/collections/stats", timeout=30)
    response.raise_for_status()
    return response.json()


with gr.Blocks(title="检索服务管理界面") as demo:
    gr.Markdown("# 多模态检索服务 — 管理/测试界面")

    with gr.Tab("文字检索"):
        query_input = gr.Textbox(label="查询文字")
        top_k_input = gr.Slider(1, 50, value=10, step=1, label="返回条数")
        search_image_checkbox = gr.Checkbox(label="同时跨模态检索图片")
        search_button = gr.Button("检索")
        text_result = gr.Textbox(label="文字结果", lines=10)
        image_result = gr.Textbox(label="图片结果", lines=5)
        search_button.click(do_search_text, [query_input, top_k_input, search_image_checkbox], [text_result, image_result])

    with gr.Tab("图片检索"):
        query_image = gr.Image(type="filepath", label="上传查询图片")
        image_top_k = gr.Slider(1, 50, value=10, step=1, label="返回条数")
        image_search_button = gr.Button("检索")
        image_search_result = gr.Textbox(label="结果", lines=10)
        image_search_button.click(do_search_image, [query_image, image_top_k], image_search_result)

    with gr.Tab("上传文档"):
        upload_file = gr.File(label="选择文件 (csv/txt/pdf/jpg/png/md)", type="filepath")
        with gr.Accordion("切分参数（不填用全局默认）", open=False):
            upload_chunk_size = gr.Number(label="chunk_size（字符数）", value=None)
            upload_chunk_overlap = gr.Number(label="chunk_overlap（字符数）", value=None)
            upload_separators = gr.Textbox(label="自定义分隔符（逗号分隔，如 \\n\\n,---）")
        upload_button = gr.Button("上传入库")
        upload_result = gr.Textbox(label="结果")
        upload_button.click(
            do_upload,
            [upload_file, upload_chunk_size, upload_chunk_overlap, upload_separators],
            upload_result,
        )

    with gr.Tab("目录入库"):
        dir_path_input = gr.Textbox(label="本地目录路径")
        recursive_checkbox = gr.Checkbox(label="递归子目录", value=True)
        watch_checkbox = gr.Checkbox(label="持续监听")
        with gr.Accordion("切分参数（不填用全局默认，watch=True 时会跟随目录持久化）", open=False):
            dir_chunk_size = gr.Number(label="chunk_size（字符数）", value=None)
            dir_chunk_overlap = gr.Number(label="chunk_overlap（字符数）", value=None)
            dir_separators = gr.Textbox(label="自定义分隔符（逗号分隔，如 \\n\\n,---）")
        dir_index_button = gr.Button("开始入库")
        dir_index_result = gr.Textbox(label="结果")
        dir_index_button.click(
            do_index_directory,
            [dir_path_input, recursive_checkbox, watch_checkbox, dir_chunk_size, dir_chunk_overlap, dir_separators],
            dir_index_result,
        )

    with gr.Tab("任务状态"):
        job_id_input = gr.Textbox(label="Job ID")
        job_check_button = gr.Button("查询")
        job_status_output = gr.JSON(label="状态")
        job_check_button.click(do_check_job, job_id_input, job_status_output)

    with gr.Tab("索引库状态"):
        stats_button = gr.Button("刷新")
        stats_output = gr.JSON(label="统计")
        stats_button.click(do_stats, None, stats_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: 手动验证（需要 API 服务已启动）**

Run: `cd retrieval-service && .venv/bin/uvicorn app.api.main:app --port 8000 &`
Run: `cd retrieval-service && API_BASE_URL=http://localhost:8000 .venv/bin/python admin_ui/app.py`

打开浏览器访问 `http://localhost:7860`，依次验证：
1. "上传文档" tab 上传一个 txt 文件，确认返回 job_id
2. "任务状态" tab 输入该 job_id，确认能看到进度/结果
3. "文字检索" tab 搜索刚上传文件里的关键词，确认能召回
4. "索引库状态" tab 点刷新，确认文档数 > 0

Expected: 四步全部符合预期，浏览器控制台无报错

- [ ] **Step 3: Commit**

```bash
git add admin_ui/app.py
git commit -m "feat: add Gradio admin UI for upload, search, and index status"
```

---

### Task 21: LangGraph 工具封装（SDK）

**Files:**
- Create: `retrieval-service/app/sdk/__init__.py`
- Create: `retrieval-service/app/sdk/langgraph_tools.py`
- Test: `retrieval-service/tests/unit/test_sdk_tools.py`

**Interfaces:**
- Produces: `search_text_tool`、`search_image_tool`（用 `langchain_core.tools.tool` 装饰的函数，供 LangGraph agent 的 `tools=[...]` 列表直接使用），均通过环境变量 `RETRIEVAL_API_BASE_URL` 配置目标地址

- [ ] **Step 1: 写测试（用 respx mock httpx 请求）**

```python
# tests/unit/test_sdk_tools.py
import httpx
import respx

from app.sdk.langgraph_tools import search_image_tool, search_text_tool


@respx.mock
def test_search_text_tool_calls_api_and_returns_hits():
    respx.post("http://localhost:8000/search/text").mock(
        return_value=httpx.Response(
            200, json={"text_hits": [{"score": 0.9, "payload": {"text": "结果A"}}], "image_hits": []}
        )
    )

    result = search_text_tool.invoke({"query": "测试查询", "top_k": 5})

    assert "结果A" in result


@respx.mock
def test_search_image_tool_calls_api_with_image_path(tmp_path):
    from PIL import Image

    img_path = tmp_path / "q.png"
    Image.new("RGB", (10, 10)).save(img_path)

    respx.post("http://localhost:8000/search/image").mock(
        return_value=httpx.Response(
            200, json={"image_hits": [{"score": 0.8, "payload": {"source_path": "/a.png"}}], "related_text_hits": []}
        )
    )

    result = search_image_tool.invoke({"image_path": str(img_path), "top_k": 5})

    assert "/a.png" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_sdk_tools.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.sdk'`（需要先 `pip install langchain-core`，在 requirements.txt 里补上 `langchain-core==0.3.28`）

- [ ] **Step 3: 补依赖并实现**

在 `requirements.txt` 追加一行：`langchain-core==0.3.28`

```python
# app/sdk/__init__.py
```

```python
# app/sdk/langgraph_tools.py
import os

import httpx
from langchain_core.tools import tool

API_BASE_URL = os.environ.get("RETRIEVAL_API_BASE_URL", "http://localhost:8000")


@tool
def search_text_tool(query: str, top_k: int = 5) -> str:
    """在检索服务里做文字检索，返回最相关的文本片段列表（用于回答需要引用本地文档知识的问题）。"""
    response = httpx.post(f"{API_BASE_URL}/search/text", json={"query": query, "top_k": top_k}, timeout=30)
    response.raise_for_status()
    hits = response.json()["text_hits"]
    if not hits:
        return "未找到相关内容"
    return "\n\n".join(f"[score={h['score']:.3f}] {h['payload'].get('text', '')}" for h in hits)


@tool
def search_image_tool(image_path: str, top_k: int = 5) -> str:
    """上传一张本地图片路径，在检索服务里做以图搜图，返回最相似图片的路径列表。"""
    with open(image_path, "rb") as f:
        response = httpx.post(f"{API_BASE_URL}/search/image", files={"image": f}, params={"top_k": top_k}, timeout=30)
    response.raise_for_status()
    hits = response.json()["image_hits"]
    if not hits:
        return "未找到相关图片"
    return "\n".join(f"[score={h['score']:.3f}] {h['payload'].get('source_path', '')}" for h in hits)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pip install -r requirements.txt && .venv/bin/pytest tests/unit/test_sdk_tools.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/sdk/ tests/unit/test_sdk_tools.py requirements.txt
git commit -m "feat: add LangGraph tool wrappers for text/image search"
```

---

### Task 22: Dockerfile + Docker Compose 部署

**Files:**
- Create: `retrieval-service/Dockerfile`
- Create: `retrieval-service/docker-compose.yml`
- Create: `retrieval-service/admin_ui/Dockerfile`

**Interfaces:**
- Consumes: `requirements.txt`（Task 1），`app/api/main.py`（Task 15），`app/worker.py`（Task 19），`admin_ui/app.py`（Task 20）
- Produces: `docker compose up` 拉起 `qdrant`、`redis`、`api`、`worker`、`admin-ui` 五个服务

- [ ] **Step 1: 编写 Dockerfile（api/worker 共用）**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 编写 admin_ui/Dockerfile**

```dockerfile
# admin_ui/Dockerfile
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir gradio==5.9.1 httpx==0.28.1 pillow==11.0.0

COPY admin_ui/app.py ./app.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
```

- [ ] **Step 3: 编写 docker-compose.yml**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

  redis:
    image: redis:7
    volumes:
      - ./data/redis:/data
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn app.api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
      REDIS_URL: redis://redis:6379/0
      DATA_DIR: /srv/data
    volumes:
      - ./data:/srv/data
    depends_on:
      - qdrant
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m app.worker
    environment:
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
      REDIS_URL: redis://redis:6379/0
      DATA_DIR: /srv/data
    volumes:
      - ./data:/srv/data
    depends_on:
      - qdrant
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  admin-ui:
    build:
      context: .
      dockerfile: admin_ui/Dockerfile
    ports:
      - "7860:7860"
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api
    restart: unless-stopped
```

- [ ] **Step 4: 验证构建与启动**

Run: `cd retrieval-service && docker compose build`
Expected: 五个服务镜像全部构建成功（api/worker 镜像构建耗时较长，因为要装 torch cu130 + paddleocr + transformers）

Run: `cd retrieval-service && docker compose up -d && sleep 20 && curl -sf http://localhost:8000/health && curl -sf http://localhost:6333/readyz`
Expected: `/health` 返回 `{"status":"ok"}`，`/readyz` 返回 Qdrant 就绪状态

Run: `docker compose logs worker --tail 30`
Expected: 日志里能看到 "directory watch manager started"，没有模型加载报错

- [ ] **Step 5: Commit**

```bash
git add Dockerfile admin_ui/Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and docker-compose for full stack deployment"
```

---

### Task 23: 端到端集成测试

**Files:**
- Create: `retrieval-service/tests/integration/conftest.py`
- Create: `retrieval-service/tests/integration/fixtures/sample.txt`
- Create: `retrieval-service/tests/integration/fixtures/sample.csv`
- Create: `retrieval-service/tests/integration/test_ingest_and_search.py`

**Interfaces:**
- Consumes: `ingest_file`（Task 12），`search_text_hybrid`（Task 7），`get_qdrant_client`, `ensure_collections`（Task 7）
- 说明：这是唯一需要真实 GPU + 真实 Qdrant 的测试，标记为 `slow`，在 Task 2 验证通过的机器上手动运行，不进日常单测流程

- [ ] **Step 1: 准备 fixture 文件**

```text
# tests/integration/fixtures/sample.txt
DGX Spark 是 NVIDIA 推出的桌面级 AI 超级计算机，搭载 GB10 Grace Blackwell 超级芯片。
本检索服务支持文字检索、图片检索、文档上传入库和目录批量入库。
```

```csv
name,role
张三,工程师
李四,产品经理
```

- [ ] **Step 2: 写集成测试**

```python
# tests/integration/conftest.py
import pytest

from app.store.qdrant_client import ensure_collections, get_qdrant_client


@pytest.fixture
def qdrant_test_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client
```

```python
# tests/integration/test_ingest_and_search.py
from pathlib import Path

import pytest

from app.ingest.pipeline import ingest_file
from app.models.embeddings import get_text_embedder
from app.store.qdrant_client import search_text_hybrid

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_txt_and_search_finds_it(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.txt", source_ref=str(FIXTURES / "sample.txt"), client=qdrant_test_client)
    assert result.status == "ingested"

    query_vector = get_text_embedder().embed(["GB10 是什么芯片"])[0]
    hits = search_text_hybrid(qdrant_test_client, query_vector, top_k=3)

    assert any("Grace Blackwell" in h.payload["text"] for h in hits)


def test_ingest_csv_and_search_finds_row(qdrant_test_client):
    result = ingest_file(FIXTURES / "sample.csv", source_ref=str(FIXTURES / "sample.csv"), client=qdrant_test_client)
    assert result.status == "ingested"

    query_vector = get_text_embedder().embed(["谁是产品经理"])[0]
    hits = search_text_hybrid(qdrant_test_client, query_vector, top_k=3)

    assert any("李四" in h.payload["text"] for h in hits)
```

- [ ] **Step 3: 运行验证（需要 Task 2 验证过的 GPU 环境）**

Run: `cd retrieval-service && .venv/bin/pytest tests/integration/ -v -m slow`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add tests/integration/
git commit -m "test: add end-to-end ingest and search integration test"
```

---

### Task 24: 分片效果查看接口（入库前预览 + 已入库文档查询）

**Files:**
- Create: `retrieval-service/app/api/routes/chunks.py`
- Modify: `retrieval-service/app/api/schemas.py`
- Modify: `retrieval-service/app/api/main.py`
- Test: `retrieval-service/tests/unit/test_api_chunks.py`

**Interfaces:**
- Consumes: `chunk_text`, `chunk_csv_rows`（Task 4）; `parse_text_file`（Task 8）; `parse_csv_file`（Task 9）; `parse_image`（Task 10）; `parse_pdf`（Task 11）; `get_store_client`（Task 15）; `list_chunks_for_document`（Task 7）
- Produces:
  - `POST /chunks/preview`（multipart，`text: str = Form(None)` 和 `file: UploadFile = File(None)` 二选一，另加可选 Form 字段 `chunk_size`/`chunk_overlap`/`separators`）→ `{chunks: list[str], count: int}`，**不写入 Qdrant、不占用 embedding 模型**，纯粹跑分词/切分逻辑，用于入库前调参
  - `GET /documents/{doc_id}/chunks` → `{chunks: [{chunk_index, text, page, source_type, file_type}], count: int}`，查已入库文档实际落到 Qdrant 里的分片

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_api_chunks.py
import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.deps import get_store_client
from app.api.main import app


def test_preview_chunks_from_raw_text():
    client = TestClient(app)
    response = client.post(
        "/chunks/preview",
        data={"text": "A" * 250, "chunk_size": "100", "chunk_overlap": "20", "separators": ""},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["chunks"])
    assert all(len(c) <= 100 for c in body["chunks"])


def test_preview_chunks_from_uploaded_txt_file():
    client = TestClient(app)
    response = client.post(
        "/chunks/preview",
        files={"file": ("note.txt", io.BytesIO("你好，这是一段测试文字。" * 20), "text/plain")},
        data={"chunk_size": "50", "chunk_overlap": "0", "separators": ""},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 1


def test_preview_chunks_requires_exactly_one_of_text_or_file():
    client = TestClient(app)
    response = client.post("/chunks/preview", data={})
    assert response.status_code == 400


def test_get_document_chunks_returns_stored_chunks():
    app.dependency_overrides[get_store_client] = lambda: object()
    with patch(
        "app.api.routes.chunks.list_chunks_for_document",
        return_value=[
            {"chunk_index": 0, "text": "第一块", "page": None, "source_type": "normal", "file_type": "txt"},
            {"chunk_index": 1, "text": "第二块", "page": None, "source_type": "normal", "file_type": "txt"},
        ],
    ):
        client = TestClient(app)
        response = client.get("/documents/doc-abc/chunks")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["chunks"][0]["text"] == "第一块"
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_chunks.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.api.routes.chunks'`

- [ ] **Step 3: 实现**

在 `app/api/schemas.py` 追加：

```python
class ChunkPreviewResponse(BaseModel):
    chunks: list[str]
    count: int


class DocumentChunkOut(BaseModel):
    chunk_index: int
    text: str
    page: int | None
    source_type: str
    file_type: str


class DocumentChunksResponse(BaseModel):
    chunks: list[DocumentChunkOut]
    count: int
```

```python
# app/api/routes/chunks.py
import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_store_client
from app.api.schemas import ChunkPreviewResponse, DocumentChunksResponse
from app.ingest.chunking import chunk_csv_rows, chunk_text
from app.ingest.parsers.csv_parser import parse_csv_file
from app.ingest.parsers.image_parser import parse_image
from app.ingest.parsers.pdf_parser import parse_pdf
from app.ingest.parsers.text_parser import parse_text_file
from app.store.qdrant_client import list_chunks_for_document

router = APIRouter()

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _extract_chunks_from_file(filename: str, content: bytes, chunk_size, chunk_overlap, separators) -> list[str]:
    import tempfile

    ext = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)

        if ext in TEXT_EXTS:
            text = parse_text_file(tmp_path)
            return chunk_text(text, chunk_size, chunk_overlap, separators)
        if ext == ".csv":
            rows, header = parse_csv_file(tmp_path)
            return chunk_csv_rows(rows, header, max_chars=chunk_size)
        if ext == ".pdf":
            pages = parse_pdf(tmp_path)
            chunks: list[str] = []
            for page in pages:
                chunks.extend(chunk_text(page.text, chunk_size, chunk_overlap, separators))
            return chunks
        if ext in IMAGE_EXTS:
            from PIL import Image

            image = Image.open(tmp_path).convert("RGB")
            parsed = parse_image(image, doc_id="preview")
            return chunk_text(parsed.ocr_text, chunk_size, chunk_overlap, separators)

        raise HTTPException(status_code=400, detail=f"unsupported file extension: {ext}")


@router.post("/chunks/preview", response_model=ChunkPreviewResponse)
async def preview_chunks(
    text: str | None = Form(None),
    file: UploadFile | None = File(None),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
    separators: list[str] | None = Form(None),
):
    if (text is None) == (file is None):
        raise HTTPException(status_code=400, detail="provide exactly one of 'text' or 'file'")

    if text is not None:
        chunks = chunk_text(text, chunk_size, chunk_overlap, separators or None)
    else:
        content = await file.read()
        chunks = _extract_chunks_from_file(file.filename, content, chunk_size, chunk_overlap, separators or None)

    return ChunkPreviewResponse(chunks=chunks, count=len(chunks))


@router.get("/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(doc_id: str, store=Depends(get_store_client)):
    payloads = list_chunks_for_document(store, doc_id)
    chunks = [
        {
            "chunk_index": p["chunk_index"],
            "text": p["text"],
            "page": p.get("page"),
            "source_type": p["source_type"],
            "file_type": p["file_type"],
        }
        for p in payloads
    ]
    return DocumentChunksResponse(chunks=chunks, count=len(chunks))
```

Modify `app/api/main.py`：

```python
# app/api/main.py
from fastapi import FastAPI

from app.api.routes import chunks, collections, directories, documents, health, jobs, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(directories.router)
app.include_router(jobs.router)
app.include_router(collections.router)
app.include_router(chunks.router)
```

**注意：** `separators` 是 Form 字段，客户端不传时 FastAPI 会给出空 list `[]` 而不是 `None`（和 `chunk_text` 期望的"不传就是 None 回落默认值"语义不一致），所以路由里统一用 `separators or None` 转一次。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd retrieval-service && .venv/bin/pytest tests/unit/test_api_chunks.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/routes/chunks.py app/api/schemas.py app/api/main.py tests/unit/test_api_chunks.py
git commit -m "feat: add chunk preview and per-document chunk inspection endpoints"
```

---

## Self-Review 记录

**Spec 覆盖检查：**
- 文字检索(spec §4) → Task 16 `/search/text`
- 图片检索(spec §4) → Task 16 `/search/image`
- 文档上传 csv/txt/pdf/jpg/png/md(spec §1、§5) → Task 8/9/10/11 各解析器 + Task 17 上传接口
- 指定本地目录批量入库(spec §1、§5) → Task 13 `ingest_directory_job` + Task 18 `/directories/index`
- 目录持续监听(spec §5) → Task 14
- Qdrant 混合检索数据模型(spec §3) → Task 7
- OCR(spec §6，CPU版) → Task 10/11 `run_ocr`，`use_gpu=False`
- 错误处理逐文件记录(spec §7) → Task 13 job status 结构 + Task 12 `IngestResult.error`
- 测试方案(spec §8) → 每个 Task 都有单元测试，Task 23 补端到端集成测试
- LangGraph 集成(spec §9) → Task 21
- Docker Compose 部署(spec §6) → Task 22
- Admin UI(补充需求) → Task 20
- 切分参数动态可配 + 自定义分隔符切分(用户复审时追加需求) → Task 1(`chunk_separators`全局默认) / Task 4(`chunk_text`接受`separators`覆盖) / Task 12(`ChunkOptions`贯穿`ingest_file`) / Task 13(job函数透传) / Task 14(watch目录持久化切分配置) / Task 17(上传接口Form字段) / Task 18(目录入库接口JSON字段) / Task 20(Admin UI输入框)
- 查看分片效果(用户复审时追加需求) → Task 7(`list_chunks_for_document`) / Task 24(`POST /chunks/preview` 入库前预览 + `GET /documents/{doc_id}/chunks` 已入库查询)

**占位符扫描：** 未发现 TBD/TODO；Task 18 中途出现的 job_id 不一致问题已在同一任务的 Step 3b 里发现并修正，不是遗留占位符。

**追加修正（用户复审时发现）：** 最初 Task 4 的 `chunk_csv_rows` 只按行数分组，没有字符数上限，行内容很长时会产出远超 1000 字符的超大 chunk，而检索接口是原样返回整个 chunk 文本、没有二次截断——已改为按行数和字符数双重限制，任何一个先达到就切分，保证 csv 分片和 txt/pdf/OCR 分片遵守同一个 `max_chars` 上限。

**类型一致性检查：** `DenseSparseVector`（Task 5 定义）在 Task 7/12/16 中字段名 `dense`/`sparse` 全程一致；`IngestResult` 字段名 `doc_id/status/chunks_written/error` 在 Task 12/13/23 引用处一致；`SearchHit.payload` 在 Task 7/16 中一致为 dict；`ChunkOptions`（Task 12 定义，字段 `chunk_size/chunk_overlap/separators`）在 Task 13/14 都是导入同一个类使用，字段名和默认值（均为 `None`）一致；watcher 里 Redis 存的 watch 目录记录字典键名 `chunk_size/chunk_overlap/separators` 与 `register_watch` 参数名、`ChunkOptions` 字段名三处保持一致，没有出现改名不同步的情况。

**范围检查：** 单个 plan 覆盖 24 个 task，规模偏大但都围绕同一个内聚系统（检索服务），没有可独立拆分成不同产品的子系统，符合"一个 spec 对应一个 plan"的判断。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-25-multimodal-retrieval-service.md`. Two execution options:

1. **Subagent-Driven (recommended)** - 我给每个 task 派一个全新 subagent 去实现，task 之间做 review，迭代快
2. **Inline Execution** - 在当前会话里按 task 顺序执行，批量执行+检查点

选哪个？
