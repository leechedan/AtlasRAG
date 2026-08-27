# 100 页级 PDF 摄入的性能压力点

日期：2026-08-27

## 背景

评估上传一份 100 页 PDF 走摄入链路（解析 → 切片 → 向量化 → 写入 Qdrant）
是否有性能压力。结论：结构化 Markdown 切片本身不是瓶颈，压力主要来自
MinerU 解析耗时/超时风险、单批次大 embedding 调用、以及 Qdrant 逐点同步
写入这三处。

## 摄入链路

`app/ingest/pipeline.py:ingest_file` → 按扩展名分支，PDF 走
`_ingest_mineru_document`（`app/ingest/pipeline.py:263-278`）：

1. `parse_pdf_to_markdown`（`app/ingest/parsers/pdf_parser.py:56`）—— 整份
   PDF 一次性 HTTP 发给常驻的 `mineru-api` 服务，同步阻塞等待。
2. 抽取图片逐张入库 + CLIP 向量化（`_ingest_mineru_extracted_image`）。
3. 全文 Markdown 交给 `chunk_markdown_structured`
   （`app/ingest/markdown_chunker.py:189`）切片。
4. 所有 chunk 一次性传给 `get_text_embedder().embed()`
   （`app/ingest/pipeline.py:117`）做 BGE-M3 向量化。
5. 每个 chunk/图片逐条调用 `client.upsert()` 写入 Qdrant
   （`app/store/qdrant_client.py:94-149`）。

整条链路在**同一个 RQ worker job 内完全串行**执行，内部没有并发。

## 压力点

### 1. MinerU 解析——最大瓶颈

`app/ingest/parsers/pdf_parser.py:56-69`、`app/config.py:19-21`

- 整份 PDF 一次性提交，同步阻塞，超时 `mineru_timeout_seconds=600`。
- `mineru_backend="pipeline"` + OCR（`ocr_lang="ch"`）对 100 页扫描/图文
  混排 PDF 可能是分钟级耗时；页数多、单页复杂时有撞到 10 分钟超时的
  风险。
- 一旦超时或解析失败，`parse_pdf_to_markdown` 直接抛 `RuntimeError`，
  整份文档摄入失败（`app/ingest/pipeline.py:325-326` 只是捕获并记为
  `failed`），没有部分结果、没有断点续传，需要整体重跑。
- `docker-compose.yml:46-52` 给 mineru 服务预留了 GPU；如果实际部署没有
  GPU 可用，会退化成纯 CPU 跑 layout+OCR，速度显著变慢。

### 2. Markdown 切片——不是瓶颈

`app/ingest/markdown_chunker.py`

纯 Python 对 markdown-it token 流做线性扫描 + 分组打包，100 页文档产出的
Markdown（几百 KB 量级）在这一步是毫秒到秒级，可忽略不计。

### 3. 向量化——第二大压力点

`app/ingest/pipeline.py:117`、`app/models/embeddings.py:21-29`

- `_write_text_chunks` 把整份文档切出的所有 chunk（100 页大概率几百个）
  一次性传给 `TextEmbedder.embed()`，BGE-M3 单次 `encode()` 处理几百条
  文本，同步跑在 worker 进程里，CPU 下会比较慢，且和 MinerU 共享同一张
  GPU 时也可能排队等待。
- PDF 内嵌的每张图片还要额外过一次 CLIP
  （`_ingest_mineru_extracted_image`，`app/ingest/pipeline.py:232-260`），
  且是**逐张串行**处理，没有批量化。

### 4. Qdrant 写入——逐点同步 upsert，没有批量化

`app/store/qdrant_client.py:94-149`

`upsert_text_chunk`/`upsert_image` 每条记录单独发一次 `client.upsert()`，
几百个 chunk 就是几百次网络往返。单次调用本身很快，但累积起来比批量
upsert 慢不少。

### 5. 内存占用

`app/ingest/parsers/pdf_parser.py:44-47`

`ParsedPdfDocument` 把全文 Markdown 和所有抽取图片的 `PIL.Image` 一次性
放进内存；图片多的 100 页 PDF 内存占用会明显上升。`embed()` 对全部
chunk 一次性编码，也会让所有 chunk 文本+向量同时驻留内存。

### 6. 并发

`docker-compose.yml` 的 `worker` 服务是单容器/单进程，多份大文档摄入任务
会在 RQ 队列里排队串行处理，互相等待。

## 优先级建议

关注顺序：MinerU 超时配置/GPU 可用性 > embedding 分批处理 > Qdrant 批量
upsert。当前没有做任何改动，仅记录分析结论供后续决策参考。
