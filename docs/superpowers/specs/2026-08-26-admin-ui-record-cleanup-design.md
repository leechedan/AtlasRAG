# admin-ui 记录清理与文档删除 — 设计

## 背景

`data/uploads` 下的测试文件（如 `note.txt`）被手动从磁盘删除后，两类残留记录仍留在系统里，且 admin-ui 没有任何入口能看到或清理它们：

1. **Redis 任务状态记录**：`app/jobs/status.py` 的 `create_job_status` 每次上传/目录入库都会写入一对 `job:<job_id>:total` / `job:<job_id>:files` key，从不设置 TTL，也没有删除逻辑。截至本次调查，Redis 里已经有 3 组这样的残留 key（`redis-cli KEYS 'job:*'` 可验证）。
2. **Qdrant 文档记录**：后端 `DELETE /documents/{doc_id}` 接口本来就存在（`app/api/routes/documents.py`），但 admin-ui「文档管理」tab 只接了列表和查看分片，从未接入删除按钮。

删除源文件不会级联清理这两类记录——两者在架构上就是独立的（Qdrant 记录一旦写入，与磁盘上的源文件是否还在无关；Redis 任务记录只是一次任务执行的历史快照）。这是预期行为，不是 bug；本设计要解决的是"没有清理入口"的问题。

## 范围

本设计只做两件事：

1. Redis 任务记录的查看 + 删除
2. 文档删除入口接入 admin-ui（后端已就绪，只缺前端）

**暂不做（已与用户确认标记为后续预留）：**

3. 重新分片入口（给已入库文档换 chunk_size/overlap 重新切片）。原因：`doc_id` 是内容哈希寻址的（`make_doc_id(source_ref, content_hash)`），换分片参数重新走上传流程会被"内容未变"逻辑跳过，需要"先删除旧记录、再用新参数重新摄入同一 source_path"的专门链路，且要处理源文件已被删除时的报错（用户已确认此时应报错，而非允许重新上传替代）。这部分逻辑已经设计清楚（见下方"future work"），但不在本次实现范围内。

## 1. Redis 任务记录：查看 + 删除

### 后端

`app/jobs/status.py` 新增两个函数（沿用现有的 `JOB_KEY_PREFIX`/`_total_key`/`_files_key` 命名约定）：

```python
def list_all_job_statuses(redis_conn) -> list[dict]:
    """扫描所有 job:*:total key，逐个拼出 {job_id, total, processed, files} 摘要。
    用 SCAN（而非 KEYS）避免阻塞 Redis；这是运维工具，调用频率低，不需要分页。"""

def delete_job_status(redis_conn, job_id: str) -> None:
    """删除该 job_id 对应的 :total 和 :files 两个 key。"""
```

新增路由（挂在现有 `app/api/routes/jobs.py`）：

- `GET /jobs` → `list[JobSummaryOut]`，每项包含 `job_id`、`total`、`processed`、`success_count`、`failed_count`（`success_count`/`failed_count` 由 `:files` hash 里每个文件的 `status` 字段统计：`status == "failed"` 计入 failed，其余（`"ingested"`/`"skipped_unchanged"`）计入 success）
- `DELETE /jobs/{job_id}` → 204，调用 `delete_job_status`

`app/api/schemas.py` 新增 `JobSummaryOut`。

### admin-ui

新增「任务记录」tab：
- 表格：job_id / total / processed / 成功数 / 失败数
- 用一个 `gr.State` 记录当前选中行的 `job_id`（选中即写入，参照文档管理 tab 的行选择模式）
- 「刷新」按钮 + 「删除所选任务记录」按钮（调用 `DELETE /jobs/{job_id}` 后刷新列表）

## 2. 文档删除入口

### 后端

不变——`DELETE /documents/{doc_id}` 已存在并可用。

### admin-ui

「文档管理」tab 改动：
- 新增一个 `gr.State` 保存当前选中行的 `doc_id`（`do_select_document` 目前只返回 chunks，不保存 doc_id；改为同时把 doc_id 写入这个 State）
- 新增「删除所选文档」按钮：调用 `DELETE /documents/{doc_id}`，成功后清空 chunks 表格并刷新 documents 表格

## 测试计划

- 后端：`app/jobs/status.py` 新函数的单元测试（用真实/fakeredis 或已有测试用的 redis 测试模式，具体方式在实现计划阶段确定）；`GET /jobs`、`DELETE /jobs/{job_id}` 的路由测试（沿用现有 `tests/unit/test_api_*.py` 风格，mock store/redis 依赖）
- admin-ui 的 gradio 回调函数是纯 Python 函数（`do_xxx`），可以像 `do_list_documents` 现有测试那样直接单元测试，mock `httpx` 调用

## Future work（本次不实现，仅记录）

**重新分片入口**：
- 新路由 `POST /documents/{doc_id}/rechunk`，body 为新的 `chunk_size`/`chunk_overlap`/`separators`
- Handler：用 `list_chunks_for_document` 取该文档任一 chunk 的 `source_path` → 若 `Path(source_path).exists()` 为假 → 400 报错"原文件已删除，无法重新分片" → 否则依次入队 `delete_document_job(doc_id)` 与 `ingest_file_job(source_path, 新source_ref, ...)`（复用现有两个 job 函数；单 worker 保证顺序执行，删除必然先于重新摄入完成）
- 返回 `job_id`，前端复用现有「任务状态」轮询
