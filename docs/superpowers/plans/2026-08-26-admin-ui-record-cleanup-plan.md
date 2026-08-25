# admin-ui 记录清理与文档删除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 admin-ui 加两个运维入口：查看/删除 Redis 里残留的任务状态记录，以及删除已入库文档（后端已有接口，只缺前端）。

**Architecture:** 后端在 `app/jobs/status.py` 加两个纯函数（列出全部任务摘要、删除单个任务记录），在 `app/api/routes/jobs.py` 包一层 REST 路由暴露出去；`app/api/routes/documents.py` 的 `DELETE /documents/{doc_id}` 已存在，不用改。admin-ui 那边加一个新 tab（任务记录）+ 给已有的「文档管理」tab 加一个删除按钮，都是给 gradio 回调函数发 HTTP 请求打后端。

**Tech Stack:** FastAPI, Redis（`redis`/`fakeredis` 测试），Gradio 5.9.1，httpx，respx（测试用的 httpx mock）。

## Global Constraints

- 重新分片功能（给已入库文档换 chunk_size 重新切片）不在本次范围内，spec 里已标记为 future work，本计划不实现。
- `list_all_job_statuses` 用 `redis_conn.scan_iter`（不是 `KEYS`）扫描 `job:*:total`。
- 任务摘要里 `success_count`/`failed_count` 的统计规则：`:files` hash 里每个文件的 `status` 字段，`status == "failed"` 计入 `failed_count`，其余（`"ingested"`/`"skipped_unchanged"`）计入 `success_count`。
- `DELETE /jobs/{job_id}` 返回 204 无 body，`DELETE /documents/{doc_id}`（已有接口）返回 200 + `{"status": "deleted"}`，两者返回格式不同，admin-ui 两个删除回调要分别处理。

---

## Task 1: Redis 任务记录列表 + 删除（纯函数层）

**Files:**
- Modify: `app/jobs/status.py`（在文件末尾追加两个函数，不改动已有函数）
- Test: `tests/unit/test_job_status_cleanup.py`（新建。不追加到已有的 `tests/unit/test_jobs.py`，因为那个文件顶部 `from app.jobs.tasks import ...` 会连带拉入 `app.ingest.pipeline → app.models.clip_model → torch` 整条重依赖链，而这里两个新函数本身零依赖，单独建文件可以在轻量环境里跑，不用装 torch）

**Interfaces:**
- Consumes: `app.jobs.status` 里已有的 `JOB_KEY_PREFIX`、`_total_key`、`_files_key`、`get_job_status`、`create_job_status`、`update_job_file_result`
- Produces：
  - `list_all_job_statuses(redis_conn) -> list[dict]`，每个 dict 形如 `{"job_id": str, "total": int, "processed": int, "success_count": int, "failed_count": int}`，按 `job_id` 字典序排序
  - `delete_job_status(redis_conn, job_id: str) -> None`

- [ ] **Step 1: 写失败的测试**

创建 `tests/unit/test_job_status_cleanup.py`：

```python
import fakeredis

from app.jobs.status import create_job_status, delete_job_status, list_all_job_statuses, update_job_file_result


def make_redis():
    return fakeredis.FakeStrictRedis()


def test_list_all_job_statuses_returns_summary_per_job():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-a", total_files=2)
    update_job_file_result(redis_conn, "job-a", "/a.txt", "ingested")
    update_job_file_result(redis_conn, "job-a", "/b.txt", "failed", error="bad file")
    create_job_status(redis_conn, "job-b", total_files=1)
    update_job_file_result(redis_conn, "job-b", "/c.txt", "ingested")

    summaries = {s["job_id"]: s for s in list_all_job_statuses(redis_conn)}

    assert summaries["job-a"] == {
        "job_id": "job-a",
        "total": 2,
        "processed": 2,
        "success_count": 1,
        "failed_count": 1,
    }
    assert summaries["job-b"] == {
        "job_id": "job-b",
        "total": 1,
        "processed": 1,
        "success_count": 1,
        "failed_count": 0,
    }


def test_list_all_job_statuses_returns_empty_list_when_no_jobs():
    redis_conn = make_redis()
    assert list_all_job_statuses(redis_conn) == []


def test_delete_job_status_removes_both_keys():
    redis_conn = make_redis()
    create_job_status(redis_conn, "job-x", total_files=1)
    update_job_file_result(redis_conn, "job-x", "/a.txt", "ingested")

    delete_job_status(redis_conn, "job-x")

    assert list_all_job_statuses(redis_conn) == []
    assert redis_conn.exists("job:job-x:total") == 0
    assert redis_conn.exists("job:job-x:files") == 0


def test_delete_job_status_is_a_noop_for_unknown_job_id():
    redis_conn = make_redis()
    delete_job_status(redis_conn, "does-not-exist")  # 不应抛异常
```

- [ ] **Step 2: 运行测试确认失败**

```bash
mkdir -p /tmp/retrieval-service-test-scratch
pip install --target /tmp/retrieval-service-test-scratch fakeredis==2.26.2 pytest==8.3.4
cd /home/jack/tools/waibao/langgraph/retrieval-service
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_job_status_cleanup.py -v -p no:cacheprovider
```

预期：`ImportError: cannot import name 'list_all_job_statuses'`（或 `delete_job_status`）导致全部测试 FAIL/ERROR。

- [ ] **Step 3: 写最小实现**

在 `app/jobs/status.py` 文件末尾（`get_job_status` 函数之后）追加：

```python
def list_all_job_statuses(redis_conn) -> list[dict]:
    job_ids = []
    for raw_key in redis_conn.scan_iter(f"{JOB_KEY_PREFIX}*:total"):
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        job_ids.append(key[len(JOB_KEY_PREFIX) : -len(":total")])

    summaries = []
    for job_id in sorted(job_ids):
        status = get_job_status(redis_conn, job_id)
        success_count = sum(1 for f in status["files"].values() if f["status"] != "failed")
        failed_count = sum(1 for f in status["files"].values() if f["status"] == "failed")
        summaries.append(
            {
                "job_id": job_id,
                "total": status["total"],
                "processed": status["processed"],
                "success_count": success_count,
                "failed_count": failed_count,
            }
        )
    return summaries


def delete_job_status(redis_conn, job_id: str) -> None:
    redis_conn.delete(_total_key(job_id), _files_key(job_id))
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_job_status_cleanup.py -v -p no:cacheprovider
```

预期：4 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add app/jobs/status.py tests/unit/test_job_status_cleanup.py
git commit -m "feat: add Redis job-status listing and deletion helpers"
```

---

## Task 2: `GET /jobs` / `DELETE /jobs/{job_id}` 路由

**Files:**
- Modify: `app/api/schemas.py`（追加 `JobSummaryOut`）
- Modify: `app/api/routes/jobs.py`（追加 `list_jobs`、`delete_job` 两个路由）
- Test: `tests/unit/test_api_directories_jobs_collections.py`（追加到已有的 `test_get_job_status_returns_progress` 之后，`test_get_collections_stats` 之前——这个文件已经在测 `/jobs/{job_id}`，风格保持一致）

**Interfaces:**
- Consumes: Task 1 产出的 `list_all_job_statuses(redis_conn) -> list[dict]`、`delete_job_status(redis_conn, job_id: str) -> None`
- Produces：`GET /jobs` → `200` + `list[JobSummaryOut]`（JSON 数组，字段同 Task 1 的 dict）；`DELETE /jobs/{job_id}` → `204` 空 body

> 注意：这个文件顶部 `from app.api.main import app` 会连带导入 `app.api.routes.documents` → `app.jobs.tasks` → `app.ingest.pipeline` → `torch`，所以测试跑不了轻量环境，必须在已经装好完整依赖的 `api` 容器里跑（`docker compose exec`）。这是这个代码库已有测试文件的既有限制，不是本次改动引入的。

- [ ] **Step 1: 写失败的测试**

在 `app/api/schemas.py` 追加（放在 `JobStatusResponse` 之后）：

```python
class JobSummaryOut(BaseModel):
    job_id: str
    total: int
    processed: int
    success_count: int
    failed_count: int
```

在 `tests/unit/test_api_directories_jobs_collections.py` 里，`test_get_job_status_returns_progress` 函数之后、`test_get_collections_stats` 函数之前插入：

```python
@patch("app.api.routes.jobs.list_all_job_statuses")
@patch("app.api.routes.jobs.get_redis_connection")
def test_list_jobs_returns_all_summaries(mock_get_conn, mock_list_statuses):
    mock_list_statuses.return_value = [
        {"job_id": "job-1", "total": 2, "processed": 2, "success_count": 2, "failed_count": 0},
        {"job_id": "job-2", "total": 1, "processed": 1, "success_count": 0, "failed_count": 1},
    ]

    client = TestClient(app)
    response = client.get("/jobs")

    assert response.status_code == 200
    assert response.json() == mock_list_statuses.return_value


@patch("app.api.routes.jobs.delete_job_status")
@patch("app.api.routes.jobs.get_redis_connection")
def test_delete_job_removes_status(mock_get_conn, mock_delete_status):
    client = TestClient(app)
    response = client.delete("/jobs/job-1")

    assert response.status_code == 204
    mock_delete_status.assert_called_once_with(mock_get_conn.return_value, "job-1")
```

- [ ] **Step 2: 运行测试确认失败**

把改动同步进已经在跑的 `api` 容器（源码是 build 时 `COPY` 进去的，不是 live mount，所以要手动同步；这两个新测试函数此时还引用不存在的 `app.api.routes.jobs.list_all_job_statuses`/`delete_job_status`，会在 `@patch` 时报 `AttributeError: <module 'app.api.routes.jobs'> does not have the attribute 'list_all_job_statuses'`）：

```bash
cd /home/jack/tools/waibao/langgraph/retrieval-service
docker compose cp app api:/srv/app
docker compose cp tests api:/srv/tests
docker compose exec api python -m pytest tests/unit/test_api_directories_jobs_collections.py -v
```

预期：`test_list_jobs_returns_all_summaries` 和 `test_delete_job_removes_status` 两个新测试 FAIL（`AttributeError`），其余既有测试仍然 PASS。

- [ ] **Step 3: 写最小实现**

在 `app/api/routes/jobs.py` 文件末尾追加（保留已有的 `get_job` 不变，只加 import 和两个新路由）：

```python
from app.jobs.status import delete_job_status, get_job_status, list_all_job_statuses
```

替换原来的：
```python
from app.jobs.status import get_job_status
```

然后在文件末尾追加：

```python
@router.get("/jobs", response_model=list[JobSummaryOut])
def list_jobs():
    redis_conn = get_redis_connection()
    return [JobSummaryOut(**summary) for summary in list_all_job_statuses(redis_conn)]


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    redis_conn = get_redis_connection()
    delete_job_status(redis_conn, job_id)
```

同时把文件顶部的 schema import 改成：

```python
from app.api.schemas import JobStatusResponse, JobSummaryOut
```

- [ ] **Step 4: 运行测试确认通过**

```bash
docker compose cp app api:/srv/app
docker compose exec api python -m pytest tests/unit/test_api_directories_jobs_collections.py -v
```

预期：全部测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add app/api/schemas.py app/api/routes/jobs.py tests/unit/test_api_directories_jobs_collections.py
git commit -m "feat: add GET /jobs and DELETE /jobs/{job_id} routes"
```

---

## Task 3: admin-ui「任务记录」tab

**Files:**
- Modify: `admin_ui/app.py`（追加 `do_list_jobs`/`do_select_job`/`do_delete_job` 三个函数 + 新 tab）
- Test: `tests/unit/test_admin_ui.py`（新建）

**Interfaces:**
- Consumes: Task 2 产出的 `GET /jobs`、`DELETE /jobs/{job_id}` 两个接口
- Produces：
  - `do_list_jobs() -> list[list]`，每行 `[job_id, total, processed, success_count, failed_count]`
  - `do_select_job(jobs_table, evt) -> str`，返回点击行的 `job_id`
  - `do_delete_job(job_id: str | None) -> tuple[str, list[list]]`，返回 `(提示信息, 刷新后的任务表格行)`
  - 供 Task 4 复用的测试辅助类 `FakeSelectData(row_index)`（`.index = [row_index]`，模拟 gradio 的 `gr.SelectData`）

> admin_ui 容器（`admin_ui/Dockerfile`）目前不含 pytest/respx，也没有把 `tests/` 拷进去，所以这个测试文件在本机跑（同 Task 1，轻量依赖，不需要 torch）。

- [ ] **Step 1: 写失败的测试**

创建 `tests/unit/test_admin_ui.py`：

```python
import httpx
import respx

from admin_ui.app import do_delete_job, do_list_jobs, do_select_job


class FakeSelectData:
    def __init__(self, row_index):
        self.index = [row_index]


@respx.mock
def test_do_list_jobs_returns_rows():
    respx.get("http://localhost:8000/jobs").mock(
        return_value=httpx.Response(
            200,
            json=[{"job_id": "job-1", "total": 2, "processed": 2, "success_count": 2, "failed_count": 0}],
        )
    )

    rows = do_list_jobs()

    assert rows == [["job-1", 2, 2, 2, 0]]


def test_do_select_job_returns_job_id_from_clicked_row():
    jobs_table = [["job-0", 1, 1, 1, 0], ["job-1", 2, 2, 1, 1]]

    job_id = do_select_job(jobs_table, FakeSelectData(1))

    assert job_id == "job-1"


@respx.mock
def test_do_delete_job_calls_api_and_refreshes_list():
    respx.delete("http://localhost:8000/jobs/job-1").mock(return_value=httpx.Response(204))
    respx.get("http://localhost:8000/jobs").mock(return_value=httpx.Response(200, json=[]))

    message, rows = do_delete_job("job-1")

    assert "job-1" in message
    assert rows == []


def test_do_delete_job_without_selection_does_not_call_api():
    message, rows = do_delete_job(None)

    assert "选择" in message
```

- [ ] **Step 2: 运行测试确认失败**

```bash
mkdir -p /tmp/retrieval-service-test-scratch
pip install --target /tmp/retrieval-service-test-scratch gradio==5.9.1 httpx==0.28.1 pillow==11.0.0 respx==0.22.0 pytest==8.3.4
cd /home/jack/tools/waibao/langgraph/retrieval-service
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_admin_ui.py -v -p no:cacheprovider
```

预期：`ImportError: cannot import name 'do_list_jobs' from 'admin_ui.app'`。

- [ ] **Step 3: 写最小实现**

在 `admin_ui/app.py` 里 `do_select_document` 函数之后、`with gr.Blocks(title="检索服务管理界面") as demo:` 之前，插入：

```python
def do_list_jobs():
    response = httpx.get(f"{API_BASE_URL}/jobs", timeout=30)
    response.raise_for_status()
    jobs = response.json()
    return [
        [j["job_id"], j["total"], j["processed"], j["success_count"], j["failed_count"]]
        for j in jobs
    ]


def do_select_job(jobs_table, evt: gr.SelectData):
    row_index = evt.index[0]
    return jobs_table[row_index][0]


def do_delete_job(job_id: str | None):
    if not job_id:
        return "请先在任务记录列表中选择一行", gr.update()
    response = httpx.delete(f"{API_BASE_URL}/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    return f"已删除任务记录 {job_id}", do_list_jobs()
```

在 `with gr.Blocks(...) as demo:` 内部，「文档管理」`with gr.Tab(...)` 块结束之后（仍在 `with gr.Blocks` 缩进内）追加新 tab：

```python
    with gr.Tab("任务记录"):
        refresh_jobs_button = gr.Button("刷新任务记录")
        jobs_table = gr.Dataframe(
            headers=["job_id", "总数", "已处理", "成功数", "失败数"],
            type="array",
            interactive=False,
            label="任务记录（点击一行选中，用下方按钮删除）",
        )
        selected_job_id = gr.State(None)
        delete_job_button = gr.Button("删除所选任务记录")
        delete_job_result = gr.Textbox(label="删除结果")
        refresh_jobs_button.click(do_list_jobs, None, jobs_table)
        jobs_table.select(do_select_job, jobs_table, selected_job_id)
        delete_job_button.click(do_delete_job, selected_job_id, [delete_job_result, jobs_table])
        demo.load(do_list_jobs, None, jobs_table)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_admin_ui.py -v -p no:cacheprovider
```

预期：4 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add admin_ui/app.py tests/unit/test_admin_ui.py
git commit -m "feat: add job-record list/delete tab to admin-ui"
```

---

## Task 4: admin-ui「文档管理」tab 加删除入口

**Files:**
- Modify: `admin_ui/app.py`（改 `do_select_document` 返回值 + 新增 `do_delete_document` + tab 内加按钮）
- Test: `tests/unit/test_admin_ui.py`（追加到 Task 3 建的文件里）

**Interfaces:**
- Consumes: 已有的 `DELETE /documents/{doc_id}` 接口（返回 200 + `{"status": "deleted"}`）；Task 3 产出的 `FakeSelectData`；已有的 `do_list_documents() -> list[list]`
- Produces：
  - `do_select_document(documents_table, evt) -> tuple[list[list], str]`（改动：现在额外返回 `doc_id`，供 `gr.State` 接住）
  - `do_delete_document(doc_id: str | None) -> tuple[str, list[list], list[list]]`，返回 `(提示信息, 刷新后的文档表格行, 清空后的分片表格行)`

- [ ] **Step 1: 写失败的测试**

在 `tests/unit/test_admin_ui.py` 末尾追加：

```python
from admin_ui.app import do_delete_document, do_select_document


@respx.mock
def test_do_select_document_returns_chunks_and_doc_id():
    respx.get("http://localhost:8000/documents/doc-1/chunks").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "chunk_index": 0,
                    "page": None,
                    "source_type": "normal",
                    "chunk_id": "c1",
                    "text": "hello",
                    "doc_id": "doc-1",
                    "source_path": "/a.txt",
                    "file_type": "txt",
                }
            ],
        )
    )
    documents_table = [["doc-1", "a.txt", "txt", "upload", 1, "normal"]]

    rows, doc_id = do_select_document(documents_table, FakeSelectData(0))

    assert rows == [[0, None, "normal", "c1", "hello"]]
    assert doc_id == "doc-1"


@respx.mock
def test_do_delete_document_calls_api_and_refreshes_list():
    respx.delete("http://localhost:8000/documents/doc-1").mock(
        return_value=httpx.Response(200, json={"status": "deleted"})
    )
    respx.get("http://localhost:8000/documents").mock(return_value=httpx.Response(200, json=[]))

    message, table, chunks = do_delete_document("doc-1")

    assert "doc-1" in message
    assert table == []
    assert chunks == []


def test_do_delete_document_without_selection_does_not_call_api():
    message, table, chunks = do_delete_document(None)

    assert "选择" in message
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_admin_ui.py -v -p no:cacheprovider
```

预期：三个新测试 FAIL——`test_do_select_document_returns_chunks_and_doc_id` 因为当前 `do_select_document` 只返回一个值（`rows, doc_id = ...` 解包报错）；`test_do_delete_document_*` 两个因为 `ImportError: cannot import name 'do_delete_document'`。

- [ ] **Step 3: 写最小实现**

把 `admin_ui/app.py` 里的：

```python
def do_select_document(documents_table, evt: gr.SelectData):
    row_index = evt.index[0]
    doc_id = documents_table[row_index][0]
    response = httpx.get(f"{API_BASE_URL}/documents/{doc_id}/chunks", timeout=30)
    response.raise_for_status()
    chunks = response.json()
    return [[c["chunk_index"], c["page"], c["source_type"], c["chunk_id"], c["text"]] for c in chunks]
```

改成：

```python
def do_select_document(documents_table, evt: gr.SelectData):
    row_index = evt.index[0]
    doc_id = documents_table[row_index][0]
    response = httpx.get(f"{API_BASE_URL}/documents/{doc_id}/chunks", timeout=30)
    response.raise_for_status()
    chunks = response.json()
    rows = [[c["chunk_index"], c["page"], c["source_type"], c["chunk_id"], c["text"]] for c in chunks]
    return rows, doc_id


def do_delete_document(doc_id: str | None):
    if not doc_id:
        return "请先在文档列表中选择一行", gr.update(), gr.update()
    response = httpx.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=30)
    response.raise_for_status()
    return f"已删除文档 {doc_id}", do_list_documents(), []
```

把「文档管理」tab 的内容：

```python
    with gr.Tab("文档管理"):
        refresh_docs_button = gr.Button("刷新文档列表")
        documents_table = gr.Dataframe(
            headers=["doc_id", "文件名", "类型", "来源", "分片数", "内容类型"],
            type="array",
            interactive=False,
            label="文档列表（点击一行查看该文档的分片）",
        )
        chunks_table = gr.Dataframe(
            headers=["序号", "页码", "内容类型", "chunk_id", "文本"],
            type="array",
            interactive=False,
            wrap=True,
            label="分片内容",
        )
        refresh_docs_button.click(do_list_documents, None, documents_table)
        documents_table.select(do_select_document, documents_table, chunks_table)
        demo.load(do_list_documents, None, documents_table)
```

改成：

```python
    with gr.Tab("文档管理"):
        refresh_docs_button = gr.Button("刷新文档列表")
        documents_table = gr.Dataframe(
            headers=["doc_id", "文件名", "类型", "来源", "分片数", "内容类型"],
            type="array",
            interactive=False,
            label="文档列表（点击一行查看该文档的分片）",
        )
        selected_doc_id = gr.State(None)
        delete_doc_button = gr.Button("删除所选文档")
        delete_doc_result = gr.Textbox(label="删除结果")
        chunks_table = gr.Dataframe(
            headers=["序号", "页码", "内容类型", "chunk_id", "文本"],
            type="array",
            interactive=False,
            wrap=True,
            label="分片内容",
        )
        refresh_docs_button.click(do_list_documents, None, documents_table)
        documents_table.select(do_select_document, documents_table, [chunks_table, selected_doc_id])
        delete_doc_button.click(do_delete_document, selected_doc_id, [delete_doc_result, documents_table, chunks_table])
        demo.load(do_list_documents, None, documents_table)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/tmp/retrieval-service-test-scratch python3 -m pytest tests/unit/test_admin_ui.py -v -p no:cacheprovider
```

预期：全部测试（Task 3 + Task 4 共 8 个）PASS。

- [ ] **Step 5: 提交**

```bash
git add admin_ui/app.py tests/unit/test_admin_ui.py
git commit -m "feat: add document delete entry point to admin-ui document management tab"
```

---

## 全部完成后：构建并验证

```bash
cd /home/jack/tools/waibao/langgraph/retrieval-service
docker compose build api worker admin-ui
docker compose up -d api worker admin-ui
```

手动验证（浏览器打开 `http://<host>:7860`）：
1. 「任务记录」tab 刷新，能看到当前 Redis 里的任务记录（含之前遗留的那几条）；选中一行点删除，`redis-cli KEYS 'job:*'` 确认对应 key 消失。
2. 「文档管理」tab 刷新，选中一个文档点「删除所选文档」，文档列表和分片列表都应刷新为不含该文档。
