# 文档列表 / 分片列表 / 检索命中溯源 — 设计文档

日期：2026-08-25
状态：已确认设计，待编写实现计划

## 1. 背景与目标

检索服务（`retrieval-service`）目前已支持文字/图片检索和文档入库，但缺少三项能力：

1. **文档列表接口**：无法查看已入库有哪些文档（只有 `GET /collections/stats` 的总数统计）
2. **分片列表接口**：无法查看某个文档被切成了哪些分片，也无法跨文档浏览所有分片
3. **检索结果溯源**：检索命中的分片虽然在 payload 里带了 `doc_id`/`chunk_index`，但没有作为顶层字段暴露，也没有显式的 `chunk_id`，调用方需要自己拼 payload 字段才能定位"命中的是哪个文档的哪个分片"

本设计新增这三项能力，均为在现有数据模型和 API 上做增量扩展，不改变入库/检索的核心流程。

## 2. 文档列表接口

`GET /documents`（v1 不做分页，内网单机场景文档量可控）

聚合 Qdrant `text_chunks` + `images` 两个 collection，按 `doc_id` 分组，返回：

```
DocumentSummary {
  doc_id: str
  source_path: str
  filename: str            # Path(source_path).name，运行时派生，不单独存储
  file_type: str
  origin: "upload" | "watch" | null   # 见第4节
  chunk_count: int
  source_types: list[str]  # 该文档下出现过的 source_type 去重集合，如 ["normal"] 或 ["normal", "ocr_image"]
  created_at: str | null   # 见第4节
  content_hash: str
  related_images: list[ImageSummary]
}

ImageSummary {
  doc_id: str
  thumbnail_path: str
  width: int
  height: int
}
```

**聚合逻辑：**
1. scroll `text_chunks`，按 `doc_id` 分组，得到每个文本文档的 `chunk_count`、`source_types`、`source_path`、`file_type`、`origin`、`created_at`（取该组内最早一条）、`content_hash`
2. scroll `images`，对每条 image 记录按 `doc_id` 归类：
   - image 的 `doc_id` 与某文本文档的 `doc_id` **完全相同** → 独立图片+OCR 场景（`_ingest_image_file`），挂到该文档的 `related_images`
   - image 的 `doc_id` 形如 `{parent_doc_id}:page{N}` 且 `parent_doc_id` 命中某文本文档 → PDF 扫描页场景（`_ingest_scanned_pdf_page`），挂到该文档的 `related_images`
   - 都不匹配 → 该 image 记录本身作为一条独立的"图片类文档"加入返回列表（`file_type` 从 `source_path` 后缀派生，`chunk_count=0`，`source_types=[]`）

## 3. 分片列表接口

**`GET /documents/{doc_id}/chunks`**
复用现有 `list_chunks_for_document`，直接暴露为 API，返回该文档全部分片（按 `chunk_index` 排序），每条附带 `chunk_id`。`doc_id` 在 `text_chunks`/`images` 两个 collection 里都找不到时返回 404；文档存在但没有文本分片（纯图片文档）返回空列表。

**`GET /chunks?cursor=&limit=`**
跨文档全局浏览。基于 Qdrant `scroll` 原生的 offset 游标做分页（Qdrant 的 scroll offset 本身是不透明的 point id，不是数字页码），响应里返回 `next_cursor`（为空表示到底）。`cursor` 指向的记录不存在（如翻页间隙里记录被删除，或客户端传了一个乱造的值）时，Qdrant 的 scroll 本身就把它当作"从头开始找不到匹配位置"处理，直接返回空列表——不是错误，因此这里也不视为错误，原样返回空 `chunks` + `next_cursor: null`（实测 verified against qdrant-client 1.12.1 的 `:memory:` local 模式：不存在/格式错乱的 offset 都不抛异常，只是返回空结果）。

**`chunk_id`**：复用现有 `_point_id(doc_id, chunk_index)`（= Qdrant point id 的 md5 值），作为分片的唯一标识，贯穿分片列表和检索响应。

## 4. 数据模型改动（ingest pipeline）

`TextChunkRecord` / `ImageRecord` 新增两个字段，写入 Qdrant payload：

- `origin: "upload" | "watch"`：在 `ingest_file` 里根据 `source_ref` 是否以 `"upload:"` 开头判断（见 `app/ingest/hashing.py: upload_source_ref`），逐层传入 `_write_text_chunks` / `_write_csv_chunks` / `_ingest_image_file` / `_ingest_scanned_pdf_page`
- `created_at`：写入时的 UTC 时间戳（`datetime.now(timezone.utc).isoformat()`）

**存量数据兼容：** Qdrant 中已有的入库记录没有这两个字段。不做回填迁移，文档列表接口对这些记录的 `origin`/`created_at` 返回 `null`。

## 5. 检索响应结构调整

`SearchHitOut` 新增顶层字段，同时保留 `payload`（现有 SDK `app/sdk/langgraph_tools.py` 用 `h['payload'].get(...)` 取值，需保持兼容）：

```
SearchHitOut {
  score: float
  doc_id: str
  chunk_index: int | None   # image_hits 里恒为 None，图片不分片
  chunk_id: str
  payload: dict             # 保留全量字段，向后兼容
}
```

`search_text` 的 `text_hits`/`image_hits`、`search_image` 的 `image_hits`/`related_text_hits` 统一走这个结构。

## 6. 错误处理

- `GET /documents/{doc_id}/chunks`：`doc_id` 不存在 → 404；存在但无文本分片 → 空列表
- `GET /chunks`：`cursor` 不存在/格式错乱 → 不报错，返回空 `chunks` + `next_cursor: null`（Qdrant scroll 原生行为，见第3节）
- 任意接口遇到空 collection → 返回空列表，不报错（与现有 `/search` 行为一致）

## 7. 测试方案

- **单元测试：**
  - 文档聚合分组逻辑（多分片文档、纯图片文档、混合 source_type 文档）
  - 图片关联匹配：独立图片场景（doc_id 相同）、PDF 扫描页场景（`:pageN` 前缀）、无匹配场景（独立图片文档）
  - `origin` 推导（upload 前缀 vs 非前缀）
  - `GET /chunks` 游标分页翻页正确性（含最后一页 `next_cursor=null`）
- **集成测试：** 入库一个含扫描页的 PDF + 一个独立图片（带 OCR 文字）+ 一个 txt，验证：
  - `GET /documents` 返回三个文档，图片正确挂载在 PDF 和独立图片文档的 `related_images` 下
  - `GET /documents/{doc_id}/chunks` 返回正确分片
  - `GET /chunks` 分页能遍历到所有分片
  - `POST /search/text` 响应顶层 `doc_id`/`chunk_index`/`chunk_id` 与命中分片一致

## 8. 范围外（本次不做）

- 存量数据 `origin`/`created_at` 回填迁移
- 文档列表分页（后续文档量增长后再加）
- 文档列表按 `file_type`/`origin`/关键词过滤（当前无此需求，YAGNI）
