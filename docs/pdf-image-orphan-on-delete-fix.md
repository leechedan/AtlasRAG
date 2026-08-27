# 修复：删除 PDF 文档未级联删除内嵌图片

日期：2026-08-26

## 问题

删除一份 PDF 文档后，其内嵌图片没有被一起删除，导致这些图片作为"孤儿"点
残留在 Qdrant `images` collection 里。表现为：这些图片在文档/任务列表里被
当成一堆独立的文档记录展示出来（而不是随父 PDF 一起消失）。

## 根因

`pipeline._ingest_pdf_extracted_image`（`app/ingest/pipeline.py`）给 PDF
内嵌图片用的是组合 doc_id：`f"{doc_id}:img{idx}"`，不等于父文档的 doc_id。

而 `delete_document`（`app/store/qdrant_client.py`，被
`DELETE /documents/{doc_id}` → `delete_document_job` →
`delete_ingested_document` 调用）原来只按 `doc_id` 字段做精确匹配删除，
捞不到这些组合 doc_id 的图片记录，它们就被落下了。

之后 `list_all_documents` 扫描 `images` collection 时，对每个图片记录按
`:` 拆出候选父 doc_id，如果父文档还在（`text_docs` 里能查到）就归并显示
为"关联图片"；父文档已被删掉后，这个归并条件不成立，图片就被当成独立的
`image_only_docs` 展示出来——这正是"图片全部展示到任务记录"的表现。

## 修复

`app/store/qdrant_client.py`：`delete_document` 新增调用
`_delete_related_pdf_images(client, doc_id)`：
- 扫描 `images` collection，找出 `doc_id` 字段以 `f"{doc_id}:"` 开头的点
- 从 Qdrant 里删除这些点
- 同时删除它们对应的磁盘文件（`image_path` 原图、`thumbnail_path` 缩略图），
  避免这部分文件残留占用磁盘

副作用（顺带修复了两处已知但尚未处理的遗留问题，两处原本都用注释标注为
"known limitation"）：
- `app/ingest/watcher.py`：`on_deleted` 删除被监控目录里已删除的 PDF 时，
  现在也会级联清理内嵌图片
- `app/ingest/pipeline.py`：`_cleanup_stale_doc_ids`（内容变更重新入库时
  清理旧 doc_id 残留）现在也会级联清理旧 doc_id 下的内嵌图片

## 验证

- 新增测试
  `tests/unit/test_qdrant_store.py::test_delete_document_removes_pdf_extracted_images`：
  构造一个组合 doc_id 的图片记录（含真实临时文件），删除父文档后断言
  Qdrant 记录和磁盘文件都被清理。
- `python3 -m pytest tests/unit -q`：186 passed。
