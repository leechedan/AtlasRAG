# Document → Markdown 直接转换接口设计

日期: 2026-08-31

## 背景 / 目的

现有 `POST /documents/upload` 是"入库"流程：保存文件、入 RQ 任务队列、异步分片
（`app.ingest.markdown_chunker`/`app.ingest.chunking`）、embedding、写入 Qdrant，
调用方只能拿到 `job_id`，再轮询 `/jobs/{job_id}`。

新增需求：调用方只想拿到"整份文档转换后的 Markdown 全文"，不需要分片、不需要
写入向量库、也不想等异步任务。即：一个纯转换（无副作用）的同步接口。

## 接口

`POST /documents/convert-to-markdown`

- 请求：`multipart/form-data`，字段 `file`（同 `/documents/upload` 的 `file` 字段），
  无其它参数（没有 chunk_size 等——本接口不分片）。
- 支持的扩展名：与 `app.ingest.pipeline.ingest_file` 完全一致，直接复用其
  `SUPPORTED_TEXT_EXTS` / `SUPPORTED_IMAGE_EXTS` / `SUPPORTED_OFFICE_EXTS` 常量，
  避免两处定义漂移：`.txt` `.md` `.csv` `.jpg/.jpeg/.png` `.pdf` `.docx` `.pptx` `.xlsx`。
  其它扩展名 → `415 Unsupported Media Type`。
- 响应：`200`，body 就是转换后的 Markdown 全文，
  `Content-Type: text/markdown; charset=utf-8`，
  `Content-Disposition: inline; filename="<原文件名词干>.md"`（可预览也可另存），
  `X-Convert-Id: <convert_id>`；若文档含图片再加一个
  `X-Images-Zip-Url: /documents/convert-to-markdown/{convert_id}/images.zip`。
  **有图片时，body 首行还会插入一行 HTML 注释**
  `<!-- images zip: /documents/convert-to-markdown/{convert_id}/images.zip (convert_id: ...) -->`——
  只放响应头容易被忽略（很多调用方只是 `curl -o out.md` 存body、或者只把 body
  贴到别处，根本不会看响应头），HTML 注释在任何 Markdown 渲染器里都不可见，但
  纯文本原文里一眼可见，body 本身就带上了下载地址。
- 同步执行，不入队、不写 Qdrant、不产生检索用的 `doc_id`、不留 job 记录。

`GET /documents/convert-to-markdown/{convert_id}/images.zip`

- 批量下载某次转换抽出的所有图片，打包成一个 zip，zip 内条目路径统一为
  `images/<filename>`——与返回的 Markdown 里图片引用的相对路径完全一致，
  所以把这个 zip 解压到保存 `.md` 文件的同一目录下（解压出的 `images/` 子目录
  正好挨着 `.md`），用任意基于文件系统的 Markdown 阅读器（VS Code 预览、
  Typora、Obsidian……）直接打开就能看到图片。
- 找不到对应 `convert_id`（没转换过图片，或 id 写错）→ `404`。

### 为什么改成"相对路径 + 批量 zip"而不是最初设想的 base64 内嵌

最初设计是把图片转成 base64 data URI 直接塞进 Markdown 正文，好处是完全自包含、
零副作用（不落盘）。但用户后续明确要求"图片引用到转换的 md 里面，普通阅读器可以
打开 md 并看到图片，图片的下载可以使用某个接口获取到批量图片下载"——这意味着
图片需要是真实存在的文件（可被"批量下载"这个动作指代的对象），而不是内嵌在
文本里的 blob。所以改为：Markdown 里用相对路径引用（PDF/office 路径直接复用
MinerU 自己产出的 `images/<filename>` 写法，不用改写；独立图片文件转换时也按
同一约定生成引用），图片本体落盘到 `settings.converted_images_dir/<convert_id>/`，
并新增上面这个批量 zip 下载接口。

## 各文件类型的转换方式（复用现有 parser，不新增分片逻辑）

| 扩展名 | 复用的现有函数 | 转换方式 |
|---|---|---|
| `.txt` `.md` | `app.ingest.parsers.text_parser.parse_text_file` | 原样返回文本 |
| `.csv` | `app.ingest.parsers.csv_parser.parse_csv_file` | 拼成标准 Markdown 表格（新写的小函数，因为现有 `chunk_csv_rows` 产出的是 `key=val` 行格式，不是 Markdown 表格语法；不引入 `tabulate` 依赖） |
| `.jpg/.jpeg/.png` | `app.ingest.parsers.image_parser.run_ocr`（不是 `parse_image` —— 后者还会做 CLIP embedding、落盘缩略图，这些都是入库副作用，转换接口不需要也不应该产生） | `![filename](images/filename)` + OCR 文本（有则附在图片下方）；图片本体作为一张 `ConvertedImage` 一起返回 |
| `.pdf` `.docx` `.pptx` `.xlsx` | `app.ingest.parsers.pdf_parser.parse_pdf_to_markdown`（走 MinerU，已含 `<sub>/<sup>` 标签清理） | 直接用其 `markdown_text`——MinerU 产出的正文本来就用 `images/<filename>` 相对路径引用图片，不需要改写；`images` 列表原样作为 `ConvertedImage` 列表返回 |

`app.ingest.convert.convert_file_to_markdown` 返回 `ConvertedDocument(markdown_text,
images: list[ConvertedImage])`，只负责"解析 + 产出 Markdown 文本 + 收集图片对象"，
不做任何落盘——落盘（写入 `settings.converted_images_dir/<convert_id>/`）是路由层
的职责，因为落盘路径需要一个只有路由才知道的 `convert_id`。

## 实现位置

- 新模块 `app/ingest/convert.py`：`convert_file_to_markdown(path: Path) ->
  ConvertedDocument`，纯函数，不接受 `client`/`chunk_options`（不分片不入库），
  也不落盘。
- `settings.converted_images_dir`（`app/config.py`，新增属性 + 启动时
  `mkdir`）：转换产出的图片落盘位置，与 `extracted_images_dir`（入库文档的图片，
  由 `delete_document` 级联清理）分开，因为转换产出的图片没有 `doc_id`，也没有
  对应的删除入口。
- `app/api/routes/documents.py` 新增两个路由：
  - `POST /documents/convert-to-markdown`：接收上传 → 写临时目录（`tempfile.
    TemporaryDirectory`，请求结束即删，不落 `settings.uploads_dir`——上传的
    源文件本身不该在磁盘上留下痕迹，与"入库"的语义区分开）→ 调
    `convert_file_to_markdown` → 生成 `convert_id`（`uuid.uuid4().hex`）→ 若有
    图片则落盘到 `converted_images_dir/<convert_id>/` → 返回 Markdown 文本
    + 上述响应头。
  - `GET /documents/convert-to-markdown/{convert_id}/images.zip`：把
    `converted_images_dir/<convert_id>/` 下的文件用 `zipfile` 在内存里打包
    （条目名 `images/<filename>`）返回；目录不存在 → `404`。

## 错误处理

- 不支持的扩展名 → `415`，body 说明支持哪些。
- 解析失败：
  - `convert_file_to_markdown` 对不支持扩展名抛 `ValueError` → 路由捕获转 `415`。
  - MinerU 调用失败（`parse_pdf_to_markdown` 抛 `RuntimeError`，网络/MinerU 服务
    不可用/返回空 markdown 等）→ 路由捕获转 `502`，带上原始错误信息。
- 不做 `ingest_file` 那种"吞异常、返回 status=failed"的兜底——这是单次同步请求，
  失败应该让调用方立刻看到 HTTP 错误，而不是拿到一个 200 但内容为空。

## 不做的事（明确排除范围）

- 不分片、不写 Qdrant、不产生检索用的 `doc_id`、不留 job/status 记录、不落盘持久化
  上传的源文件本身（只落盘转换产出的图片，供批量下载）。
- 不支持"已入库文档按 doc_id 反查全文"这条路径——本接口只做"新上传文件 →
  Markdown"的同步转换，与检索/入库流程完全解耦。
- `converted_images_dir` 下按 `convert_id` 落盘的图片没有过期/清理机制（这点与
  仓库里 `extracted_images_dir`/`thumbnails_dir` 现状一致——都没有 TTL，需要时
  再补）。

## 测试

- `app/ingest/convert.py` 的单元测试（`tests/unit/test_document_convert.py`）：
  txt/md 原样返回且 `images == []`、csv 表格拼接正确、image mock `run_ocr`
  验证 `images/<filename>` 相对路径引用 + OCR 文本拼接 + 返回的 `ConvertedImage`、
  pdf/office mock `parse_pdf_to_markdown` 验证 MinerU 的 `images/<filename>`
  占位符原样保留（不改写）且图片列表透传、未支持扩展名抛 `ValueError`。
- 路由层测试（`tests/unit/test_api_documents.py`）：上传 txt 拿到 `200` + 正确
  `Content-Type`/`Content-Disposition`/`X-Convert-Id`/body，且无 `X-Images-Zip-Url`；
  不支持扩展名 `415`；mock 转换函数抛 `RuntimeError` 验证 `502`；上传图片后
  `X-Images-Zip-Url` 存在，`GET` 该 URL 拿到的 zip 里恰好一个 `images/<filename>`
  条目；未知 `convert_id` 请求 zip 下载 `404`；转换请求前后 `settings.
  uploads_dir` 内容不变（快照对比，因为 `uploads_dir` 是只读 property，没法
  monkeypatch）。
