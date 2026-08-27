# 多模态检索服务平台 — 设计文档

日期：2026-08-25
状态：已确认设计，待编写实现计划

## 1. 背景与目标

在 NVIDIA DGX Spark（GB10 Blackwell GPU，ARM64/aarch64，128GB 统一内存，Ubuntu 24.04，单机部署）上搭建一个检索服务平台，提供：

1. 文字检索（关键词 + 语义向量混合检索）
2. 图片检索（以图搜图 / 以文搜图跨模态检索）
3. 文档上传入库（支持 csv / txt / pdf / jpg / png / md，2026-08-26 起 PDF
   解析方案替换为 MinerU 后新增 docx / pptx / xlsx，见第 5 节）
4. 指定本地目录，批量 + 持续监听将目录内容自动入库

服务定位：既作为 LangGraph agent 的检索工具（tool call），也作为独立的内网 HTTP API 服务供其他系统直接调用。仅内网访问，v1 不做鉴权。

## 2. 整体架构

```
                     ┌─────────────────┐
        用户/LangGraph│   Admin UI       │  (Gradio，内网访问)
        Agent  ──────▶│  上传/检索测试/状态│
                     └────────┬─────────┘
                              │ HTTP
                     ┌────────▼─────────┐
                     │   API 服务 (FastAPI)│  ← 加载 BGE-M3 + Chinese-CLIP
                     │  查询时实时向量化    │     (GPU 常驻，走查询路径)
                     └───┬────────┬──────┘
                         │        │ 入库任务入队
                  同步查询│        ▼
                         │   ┌─────────┐
                         │   │  Redis  │ (任务队列 RQ + job 状态 + 目录监听配置)
                         │   └────┬────┘
                         │        │
                         ▼        ▼
                     ┌──────────────────┐        ┌────────────────────┐
                     │  Worker 进程       │  HTTP  │   mineru-api 服务    │
                     │  解析/OCR/切分/向量化│───────▶│  PDF/docx/pptx/xlsx │
                     │  + 目录监听(watchdog)│◀───────│  → 结构化 Markdown  │
                     │  ← BGE-M3+CLIP(GPU) │ MD+图片 │   + 内嵌图片列表     │
                     │  + PaddleOCR(CPU,   │        │   (自己的 GPU 容器)  │
                     │    仅 jpg/png 独立图)│        └────────────────────┘
                     └────────┬──────────┘
                              │ upsert
                     ┌────────▼─────────┐
                     │     Qdrant        │  (向量库，混合检索：dense+sparse，CPU即可)
                     └───────────────────┘
```

**组件划分理由：**
- 查询路径（低延迟同步）与入库路径（耗时批处理）分离成 API 容器和 Worker 容器两个进程，互不阻塞
- API、Worker 各自常驻加载一份模型到 GPU；DGX Spark 121GB 统一内存下两份模型（GB 级别）无压力，v1 不必为省显存再拆独立推理微服务
- Redis + RQ 承担任务队列（比 Celery 轻，单机场景够用）和目录监听配置存储
- 目录监听直接内置在 Worker 里（`watchdog` 实时监听 + 每小时兜底全量扫描防漏），不单独起容器
- Admin UI（Gradio）只通过 HTTP 调用 API，不直连 DB，保证和外部调用者体验一致

全部服务通过 Docker Compose 编排，单机部署。

> **2026-08-26 更新**：PDF（以及新增支持的 docx/pptx/xlsx）解析方案从最初的
> "Worker 进程内 PyMuPDF 抽取文字层 + PaddleOCR 兜底扫描页"（见下方第 5 节
> 原始设计）替换为调用独立的 **`mineru-api` 服务**（`Dockerfile.mineru` 构建，
> 常驻加载 MinerU 模型，自己占一块 GPU）。Worker 通过 HTTP 把文件发给
> `mineru-api`，拿回统一的结构化 Markdown（保留标题层级/列表/表格）+ 内嵌
> 图片列表，不再自己做版面分析/OCR。PaddleOCR(CPU) 仍保留在 Worker 里，但
> 现在只用于独立上传的 jpg/png 图片文字提取，不再参与 PDF 处理。换方案的
> 详细原因见 `docs/QA.md` Q19。

## 3. 数据模型（Qdrant）

两个 collection：

```
text_chunks  (文字检索用)
  vector: dense (BGE-M3, 1024维) + sparse (BGE-M3 词权重)  → 混合检索
  payload: {doc_id, source_path, file_type, chunk_index,
            page, text, source_type(normal|ocr_image),
            content_hash, created_at}

images  (图片检索用)
  vector: clip (Chinese-CLIP，图像/文本共享同一向量空间)
  payload: {doc_id, source_path, thumbnail_path, width, height,
            has_ocr_text, content_hash, created_at}
```

**关键设计点：** 图片/扫描页如果 OCR 出文字，这段文字同时会作为一条记录存入 `text_chunks`（`source_type=ocr_image`），使得图片内容既能靠"文字检索"命中文字内容，图片本身又能靠"图片检索"按视觉相似度或跨模态文字描述命中——两条检索路径互补，覆盖图文检索需求。

`doc_id` = 来源标识 + 内容 hash（sha256）拼接后取 hash：目录入库的文件用绝对路径作为来源标识（同内容出现在两个路径下视为两个文档），手动上传的文件用"upload"+上传时间戳作为来源标识。`content_hash` 单独存一份在 payload 里，用于增量入库时判断文件内容是否变化、是否需要跳过重新入库。

## 4. API 接口

> **2026-08-27 更新**：下表已按 `app/api/routes/` 实际代码同步（原表只覆盖
> v1 设计阶段规划的 8 个接口，实现过程中新增了 chunk 分页浏览、文档级别的
> 图片/缩略图访问、任务列表/删除、目录监听移除等接口，原表未跟进更新过，
> 属于文档滞后于代码的情况）。为避免这张表以后再次和代码脱节，新增了一个
> **自省式** 的 `GET /endpoints`（见下方说明），会随代码自动保持准确，这
> 张手写表格之后仅作架构层面的说明用途，接口明细以 `GET /endpoints` 或
> `GET /docs`（FastAPI 自带的 OpenAPI/Swagger UI）的实时输出为准。

| 接口 | 说明 |
|---|---|
| `POST /search/text` | 文字查询 → text_chunks 混合检索（dense+sparse+全文子串兜底三路 RRF 融合），可选 `modalities: [text, image]` 同时用 CLIP 文本编码器跨模态检索图片 |
| `POST /search/image` | 上传图片查询 → images 集合按 CLIP 视觉相似度检索，返回关联的 OCR/正文文字（如有） |
| `POST /documents/upload` | 上传单文件（csv/txt/pdf/jpg/png/md/docx/pptx/xlsx），异步入库，返回 job_id |
| `GET /documents` | 列出所有已入库文档（含关联图片摘要） |
| `GET /documents/{doc_id}/chunks` | 查某文档的全部 chunk（按 chunk_index 排序） |
| `GET /documents/{doc_id}/thumbnail` | 返回某文档关联图片的缩略图字节（Admin UI 等外部调用方无法直连 api/worker 容器磁盘，必须经此接口） |
| `GET /documents/{doc_id}/image` | 返回 MinerU 抽取的原图字节（同上，PDF 内嵌图片场景） |
| `DELETE /documents/{doc_id}` | 删除某文档及其所有 chunk/向量（含级联清理关联的内嵌图片） |
| `POST /directories/index` | 指定本地目录路径，`recursive`/`watch` 参数控制是否递归、是否持续监听，返回 job_id |
| `DELETE /directories/watch` | 移除一条已注册的目录监听（历史脏数据/坏路径的清理入口） |
| `GET /jobs/{job_id}` | 查单个入库任务进度/结果（含逐文件成功/失败明细） |
| `GET /jobs` | 列出全部任务状态摘要 |
| `DELETE /jobs/{job_id}` | 删除某任务状态记录 |
| `GET /chunks` | 分页浏览全部 chunk（游标分页，供 Admin UI 数据浏览页使用） |
| `GET /collections/stats` | 文档数/向量数，供 Admin UI 状态面板使用 |
| `GET /endpoints` | **2026-08-27 新增**：自省 FastAPI 自身路由表（`request.app.routes` 里的 `APIRoute`），实时返回当前所有业务接口的 method/path/name/summary/description/tags，排除 `/docs`、`/redoc`、`/openapi.json` 等 FastAPI 内部路由。用来替代容易过期的手写接口清单 |
| `GET /health` | 健康检查（`/health` 返回 200 才代表 BGE-M3/Chinese-CLIP 两个模型单例已真正加载完成，见第 7 节） |

另提供一个轻量 Python 客户端封装（`@tool` 装饰的函数），包装 `/search/text`、`/search/image`，直接给 LangGraph agent 当工具用。

## 5. 入库流程

各文件类型解析方式：

| 类型 | 处理方式 |
|---|---|
| txt/md | 直接读取，按结构化 Markdown 切片器分组切分（标题/段落/表格/代码块等块级单元，带 overlap，见 `docs/markdown-chunker-overlap-fix.md`） |
| csv | pandas 解析，按行分组切分，每个 chunk 带表头上下文 |
| pdf / docx / pptx / xlsx | ~~PyMuPDF 提取文字层；某页文字层为空/很少（扫描件）→ 栅格化该页跑 OCR，同时该页图像存入 `images` 集合~~（**2026-08-26 起替换**，详见上方第 2 节更新说明与 `docs/QA.md` Q19）：统一调用 `mineru-api` 服务转成保留标题层级/列表/HTML 表格的结构化 Markdown（不区分文字层 PDF 还是扫描 PDF，也不区分文字来自扫描页还是文字层），正文复用 txt/md 同一套结构化切片器；内嵌图片单独抽取存入 `images` 集合（CLIP 向量化），正文中的图片引用改写为指向 `GET /documents/{doc_id}/image` 的可访问链接。明确不做页码级定位（`page` 字段统一为 `None`）。 |
| jpg/png | PaddleOCR 提取文字（有文字则存入 `text_chunks`）+ Chinese-CLIP 生成图像向量（存入 `images`）+ 生成缩略图供 Admin UI 预览（此路径未变） |

**目录监听：** `watchdog` 实时监听文件创建/修改/删除 → 触发对应入库/删除任务；每小时兜底全量扫描（防止服务重启期间漏事件），靠 `content_hash` 判断文件是否真变化，未变化则跳过（避免重复消耗 GPU）。文件删除/移动时联动删除 Qdrant 中对应向量记录。

## 6. 部署方案

**Docker Compose 服务清单：**

| 服务 | 镜像/说明 | GPU |
|---|---|---|
| `qdrant` | 官方镜像 `qdrant/qdrant`，原生 arm64 支持 | 否，纯 CPU |
| `redis` | `redis:7`，任务队列 + job 状态 | 否 |
| `api` | 自建镜像，FastAPI + Uvicorn，常驻加载 BGE-M3 + Chinese-CLIP | 是 |
| `worker` | 同代码不同启动命令，常驻加载 BGE-M3 + Chinese-CLIP + PaddleOCR(CPU，仅 jpg/png)，内置目录监听线程 | 是 |
| `mineru` | **2026-08-26 新增**，`Dockerfile.mineru` 构建，`mineru-api` 常驻服务，Worker 处理 pdf/docx/pptx/xlsx 时通过 HTTP 调用它转 Markdown | 是 |
| `admin-ui` | Gradio，纯 HTTP 调用 api | 否 |

**GPU/ARM64 环境结论（已用社区实证信息核实）：**

- **PyTorch**：官方直接提供 CUDA 13.0 的 aarch64 wheel，`pip install torch --index-url https://download.pytorch.org/whl/cu130` 即可，无需编译、无需 NGC 容器
- **BGE-M3 / Chinese-CLIP**：均为 BERT 类 encoder-only 架构，不依赖 Flash-Attention 3 专用内核（该内核在 aarch64 上有已知问题，但只影响特定 causal LM 架构），预期可直接在 GB10 上跑通，风险低
- **Qdrant**：官方原生支持 arm64，且不需要 GPU（社区在 DGX Spark 上有实际部署案例验证），零风险
- **PaddleOCR GPU 版**：aarch64 官方无预编译 GPU wheel，需源码编译（社区有 DGX Spark 专属编译教程，约40分钟，需要 `-DEIGEN_DONT_VECTORIZE=1 -U__ARM_NEON` 编译flag解决 Eigen 库问题）。**v1 决定先用官方 CPU wheel**（零构建风险，立即可用；OCR 官方 benchmark 显示 GPU 比 CPU 快约 3-5倍，但 OCR 不是本流程里最吃 GPU 的环节，入库是离线批处理任务，CPU 吞吐通常可接受）。后续如 OCR 成为实际瓶颈，再按已找到的教程源码编译 GPU 版

**实现第一步（风险前置验证）：** 写一个环境验证脚本，确认 `torch.cuda.is_available()` 为 True，并实际跑一次 BGE-M3 和 Chinese-CLIP 的推理（GPU 上），确认无误后再开始写业务代码。

**数据持久化：**
- Qdrant 数据、Redis 数据：Docker volume
- 上传文件：原始文件存于管理目录 `/data/uploads/{doc_id}/original.ext`，供 Admin UI 预览/下载；目录挂载入库的文件不复制，只保留原路径引用
- 图片缩略图：`/data/thumbnails/{doc_id}.jpg`

## 7. 错误处理

- 入库任务中单个文件失败不影响整批任务，job 状态记录每个文件的成功/失败明细及原因
- 模型加载失败 → 容器健康检查不通过，Docker 自动重启，日志记录清晰
- 检索时 collection 为空 → 返回空结果列表，不报错
- 目录监听：文件被删除/移动 → 联动删除 Qdrant 中对应向量记录

## 8. 测试方案

- 单元测试：分片逻辑、content_hash 去重判断、文件类型分发
- 集成测试：起一套 docker-compose 测试栈，用 fixture 文件（pdf/csv/图片）跑完整"入库→检索"链路，验证召回正确性
- 环境验证脚本（见第6节"实现第一步"）作为整个实现工作的起点

## 9. LangGraph 集成

提供轻量 Python SDK 模块，用 `@tool` 装饰器封装 `/search/text`、`/search/image` 两个 HTTP 接口，供 LangGraph agent 直接 import 使用，不重复实现检索逻辑。

## 10. 范围外（v1不做）

- 鉴权/HTTPS（仅内网场景，明确不需要）
- PaddleOCR GPU 加速（先用 CPU，按需再升级）
- 分布式/多机部署（单机 DGX Spark 场景，Qdrant/Milvus 集群模式不在范围内）
