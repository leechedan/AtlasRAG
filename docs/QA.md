# 开发过程问题复盘 QA

面试细节补充素材。按主题整理项目开发中遇到的具体问题：现象、根因、
解决方案，以及每个问题可以延伸讲的技术点。原始记录见对应 commit 和
`docs/` 下的独立修复文档。

---

## 一、检索质量

### Q1. 相邻 chunk 之间为什么会丢失上下文关联？

**现象**：`chunk_overlap`（默认 150 字符）配置项存在，但在生产链路（`.md`
文件 / MinerU 转换出的 PDF）里几乎完全不生效——同一标题小节内因超过
`chunk_size`（1000 字符）被切成多个分片时，相邻分片之间零重叠，检索命中
其中一片看不到下一片的内容，查询落在切分点附近时经常缺半句上下文。

**根因**：结构化 Markdown 切片器（`markdown_chunker.py`）按块级 token（段
落/表格/代码块/列表等）分组，"块是语义原子单位、重叠会重复整段表格没有
意义"这个判断本该只适用于表格/代码块这类不可拆的保护块，实现上却被无差
别套用到了所有分组边界，包括两个普通段落纯粹因长度超限被切开的情况——
而这恰恰是大段连续正文里最常见的场景。

**解决**：新增 `_with_overlap_prefix()`，在同一小节内拼接连续分组时，把
上一分组结尾的 `chunk_overlap` 个字符前缀拼接到当前分组开头；仅当上一分
组末块不是受保护块（表格/代码）时才借用，避免截出半段表格；英文场景下
把 overlap 片段裁到下一个完整单词起点，避免和当前分片首词粘连成伪 token
影响 sparse 检索分词。不同标题小节之间仍保持零重叠——那是真实语义边界。

**延伸**：这是"实现逻辑对需求做了过度泛化"的典型例子——一个只对特殊场景
成立的假设（保护块不该重叠），在写代码时被直接套用到了所有分组，覆盖率
测试也没有针对性覆盖到"两个普通段落之间"这个最大头的场景。详见
`docs/markdown-chunker-overlap-fix.md`。

### Q1b. 结构化切片器的碎片化问题，"只做 section 内合并"为什么不够？

**现象**：`scripts/eval_chunk_quality.py` 跑出的报告显示 93% 的 chunk 不在
长度目标区间内、中位数只有 5 个字符——大量标题独立成段、分隔线、短列表
项各自打包成了一个 chunk。做了一版"把 section 内偏小的 block 并入相邻
block"的修复，新增单测全部通过，但套到真实语料上几乎零效果：合并前后
`chunk_count`/长度中位数完全没变。

**根因**：最初的假设是"一个 section 内有多个小 block"，但对真实文档统计
section 正文尺寸分布后发现，主导的碎片化模式其实是"标题多、每个 section
天生就只有一小段正文"——section 内根本没有相邻小块可合并，仅 section 内
合并在三份实测文档上一次都没触发。

第一版跨 section 修复本身又踩了一个正确性 bug：直接把两个已经定稿的
chunk 字符串用 `"\n\n"` 拼接，破坏了本模块"内容是原文逐字提取"的不变量
——两段内容在原文里中间隔着被跳过的标题行，拼接结果不再是原文里能找到
的连续子串，导致 `app/eval/structure_metrics.locate_chunk_spans` 用来定
位 chunk 位置的 `text.find()` 退化成猜测区间，把完好的表格/代码块误报成
被截断。

**解决**：改成按原文行区间重新切片（`lines[较早节的标题行 :
较晚节的正文结束行]`），保证结果始终是原文的连续子串，被跨过的标题行也
自然留在正文里；`heading_path` 取合并双方路径的**最长公共前缀**（唯一真
正覆盖全部合并内容的标题，不是"更早那个 piece 的路径"）；公共前缀为空
（跨顶层章节）时拒绝合并——那是真实的语义边界。

**延伸**：这题适合讲两件事——(1) 优化方案要先用真实语料验证假设，而不是
直接实现一个"看起来合理"的方案：两轮分析都是先统计真实文档的 section 尺
寸分布，才发现"section 内有小块"这个假设不成立；(2)"内容必须是原文逐字
提取"这类不变量一旦被打破，受影响的往往不只是功能本身，还有下游依赖它的
工具（这里是 eval 脚本自己的 chunk 定位逻辑）——单测要覆盖"改动后原有的
不变量是否还成立"，不能只测新功能。详见
`docs/superpowers/specs/2026-08-27-markdown-chunker-small-group-merge-design.md`。

### Q2. 专有名词/术语检索召回率低，怎么分析和优化的？

**背景**：检索链路是 BGE-M3 同时产出 dense（1024维）+ sparse（类 SPLADE）
两路向量，Qdrant 用两路 Prefetch 各取 top_k*2 后 RRF 融合排序。

**识别出的短板**：
1. 无 reranker，RRF 只是排名层面粗融合，没有 cross-encoder 精确打分；
2. 无同义词/别名扩展，专有名词缩写、全称、译名等变体容易漏召回；
3. sparse 通道依赖 BGE-M3 自带的 XLM-RoBERTa 分词器，专有名词被切成生僻
   子词时词权重区分度打折扣，不如 BM25 + 精确分词稳定；
4. 无精确短语/子串匹配兜底，向量检索对生僻实体容易语义模糊化。

**已实施的快速修复**（优先做投入产出比最高的 #4）：
- Qdrant 给 `text_chunks` 集合的 `text` 字段建立多语言全文索引
  （`TokenizerType.MULTILINGUAL`）；
- `search_text_hybrid` 新增第三路 Prefetch：用 `MatchText` 过滤器限定"原
  文必须逐字包含查询词"，在这个子集里按 dense 语义排序，再和原有 dense/
  sparse 两路一起 RRF 融合——只要专有名词在原文里逐字出现，即使语义排序
  把它排靠后，这一路也能把它捞回来；
- 顺带修了一个数据清洗问题：MinerU 对化学式/脚注等场景会在 `md_content`
  里插入原生 `<sub>`/`<sup>` HTML 标签（如 `H<sub>2</sub>O`），污染
  embedding 和关键词匹配，在解析边界用正则去标签保留内容字符
  （`H<sub>2</sub>O` → `H2O`）。

**尚未做**（reranker、同义词扩展）：评估过投入产出比后判断应该先看 #4
上线效果再决定要不要做，避免过度设计。

**延伸**：这道题适合讲"如何在有限时间内给检索质量问题排优先级"——先枚举
完整问题空间，再挑成本最低、收益最确定的先做，而不是一上来就上 reranker
这种要多一次模型推理、有延迟代价的方案。详见
`docs/proper-noun-retrieval-quality.md`。

---

## 二、后台任务 / Worker 稳定性

### Q3. RQ Worker 为什么导致模型"名义上常驻、实际每次都重新加载"？

**现象**：`app/worker.py` 用 `@lru_cache` 缓存 BGE-M3/Chinese-CLIP/
PaddleOCR 单例，设计意图是模型常驻内存跨任务复用，但实际每个任务都在
重新加载模型，完全没有起到"常驻"的效果。

**根因**：读了实际安装的 `rq==2.1.0` 源码才定位到——`rq.Worker` 默认给
每个任务 fork 一个 work-horse 子进程执行（`execute_job` →
`fork_work_horse`），子进程启动时 `@lru_cache` 是空的，任务跑完子进程就
退出，模型单例从未真正在长驻进程里活过。

**解决**：换成 `rq.SimpleWorker`，`execute_job` 直接在当前进程/线程里跑，
不 fork，`@lru_cache` 单例才能真正跨任务保持热。已知取舍：不再有 fork 隔
离，一次任务里的 OS 级崩溃（如 segfault）理论上可能拖垮整个 worker 进程，
但换来的是避免了 CUDA-after-fork 的隐患，属于可接受的权衡。正常 Python
异常处理不受影响（`perform_job` 逻辑是继承来的，未改动）。

**延伸**：典型的"配置/依赖行为和自己的假设不一致"问题——`@lru_cache` 本
身没问题，问题出在没有验证运行它的进程模型是否真的是"长驻单进程"。定位
方式是直接读第三方库安装版本的源码，而不是猜测或看文档（文档不一定覆盖
这类底层的 fork 语义细节）。

### Q4. 目录批量导入为什么会被"杀掉"？

**现象**：处理耗时超过 3 分钟的目录导入任务会被 SIGALRM 杀掉，留下部分
完成、无法重试的索引状态。

**根因**：RQ 每个入队任务默认 `job_timeout` 是 180 秒，导入接口没有显式
覆盖这个值。

**解决**：单文件导入设置 `job_timeout=3600`，目录批量导入设置
`job_timeout=-1`（无限，向已安装的 rq 源码验证过 `-1` 在心跳 TTL 和
horse-monitor 逻辑里都被特殊处理为"永不超时"）。

### Q5. 目录导入接口的 job_id 存在竞态条件

**现象**：`POST /directories/index` 最初的实现是先用 `job_id=None` 入队
（worker 可能立刻拿到并开始跑），再把生成的 `job.id` 追加进
`job.args` 并 `job.save()` 写回 Redis。如果 worker 恰好在这两步之间把任
务取走，调用 `ingest_directory_job` 时会因为少了 `job_id` 这个位置参数
直接抛 `TypeError`，导致 `GET /jobs/{job_id}` 永远显示一个空/零状态。

**解决**：改为在调用方提前用 `uuid.uuid4()` 生成 `job_id`，同时作为位置
参数传给任务函数、又作为 RQ 的 `job_id=` kwarg 传给 `enqueue()`，从根上
消除"先入队再补丁"这种两步操作之间的时间窗口。

**延伸**：经典的"先做了一个能跑但有竞态窗口的实现"，回归测试用真实的
`fakeredis` + 同步 `Queue(is_async=False)` 让任务真正执行来验证，而不是
只 mock 断言参数传递正确——因为原来的 bug 恰恰就是"参数看起来传对了，
但执行时序不对"。

### Q6. 一个目录路径配置错误为什么能让整个 worker 崩溃循环？

**现象**：`POST /directories/index` 不校验 `path` 是否存在/是目录，一个
拼写错误会产生一个和"成功"没法区分的 0/0 任务状态，且在 `watch=True`
时会被持久化进 Redis。更严重的是：worker 启动时
`DirectoryWatchManager.start()` 会把所有已注册目录一次性 `schedule()`
完再统一 `observer.start()`，而 watchdog 对不存在的路径在 `start()` 时
抛 `FileNotFoundError`，这个异常会打穿整个 observer——包括其他所有合法
的监控目录——未被捕获，直接让 `app.worker.main()` 在进入 RQ 消费循环前
就崩溃退出。配合 docker-compose 的 `restart: unless-stopped`，worker 进
程无限崩溃重启，所有摄取功能（不只是这一个坏目录）全部停摆。

**解决**：
1. `POST /directories/index` 提前用 `Path(path).is_dir()` 校验，非法路
   径直接 400，不再入队也不再持久化进 Redis；
2. 调整 `DirectoryWatchManager.start()` 的启动顺序——先启动空的
   observer，再逐个目录调用 `schedule()`（读 watchdog 源码确认：只有
   observer 已经在跑的情况下，`schedule()` 才会同步启动这个目录自己的
   emitter 线程、只对这一个目录抛异常），每个 `schedule()` 单独包一层
   `try/except OSError`，坏目录只警告跳过，不再拖累其他目录；
3. 新增 `unregister_watch` + `DELETE /directories/watch`，让校验上线前
   已经写进 Redis 的历史脏数据能被清理，不用手动操作 Redis。

**延伸**：这题适合讲"故障半径"——一个局部输入校验缺失，因为共享了同一个
`Observer` 实例和同一个进程的启动路径，被放大成了全局服务不可用，还叠加
了容器自动重启策略把它变成了无限崩溃循环。修复不是只加校验，还包括给已
经产生的脏状态一条清理路径。

### Q7. 编辑一个已入库文件后，为什么旧版本的检索结果一直没消失？

**现象**：`doc_id` 是内容寻址的（`make_doc_id(source_ref, content_hash)`），
监控目录里的文件被编辑后重新入库会生成一个新的 `doc_id`，但旧 `doc_id`
下的 Qdrant 点从未被清理，导致同一 `source_path` 下堆积多个版本、索引无
限增长、搜索结果里混入过期内容。而且旧版本查找函数
`find_doc_id_by_source_path` 用的是 `scroll(limit=1)`，就算想清理也只能
发现一个历史版本。

**解决**：新增 `find_all_doc_ids_by_source_path`（分页 scroll，返回某
`source_path` 下全部 `doc_id` 集合），在 `ingest_file` 跳过"内容未变"检
查之后、写入新内容之前，调用 `_cleanup_stale_doc_ids` 删除除当前
`doc_id` 外的所有历史版本。明确排除范围：扫描版 PDF 按页生成的组合
`doc_id`（`{doc_id}:page{N}`）是另一个已知的、单独记录的局限，本次不处
理。

**延伸**：内容寻址（content-addressed）ID 方案本身没问题，但"寻址变了就
必须清理旧地址下的数据"这一步容易被漏掉——设计和实现之间的隐含契约没有
被显式检查覆盖到。

---

## 三、API / 服务健壮性

### Q8. `/health` 返回 200 就代表服务真的能用了吗？

**现象**：模型（BGE-M3、Chinese-CLIP）最初是懒加载——第一次请求才真正构
造，而 `/health` 无条件返回 `{"status": "ok"}`，跟模型是否已经加载完全
脱节。容器编排层面如果拿 `/health` 做就绪判断，会在模型还没加载完时就把
流量导进来，第一批请求要素模型冷启动的延迟，且和"API 保持模型常驻"的设
计意图矛盾。

**解决**：改用 FastAPI 的 `lifespan`（`@asynccontextmanager`，而不是过
时的 `on_event`）在 `yield` 之前显式调用 `get_text_embedder()` /
`get_clip_embedder()`，验证过 uvicorn 在 lifespan 里代码跑完前不会开始
接受请求、抛异常会产生 `lifespan.startup.failed` 让进程非零退出而不是静
默带病启动。`/health` 改为读 `cache_info().currsize == 1` 判断两个单例
是否已构造，未就绪时返回 503 `{"status": "loading"}`。同时给
docker-compose 的 api/qdrant 加了 `healthcheck`（api 容器没有 curl，用
`python3 -c "urllib..."`；qdrant 官方镜像也没有 curl/python，用 bash
`/dev/tcp` 直接发 HTTP 请求）。

**延伸**：健康检查接口本身也需要"真的检查了什么"，而不是常量。这题也适
合讲怎么在没有 curl/wget 的精简镜像里做健康检查——用镜像里确定存在的解
释器/shell 内建能力代替额外装包。

### Q9. `chunk_text` 为什么会产出超过 `chunk_size` 上限的分片？

**现象**：自定义分隔符走的递归分支，落到 `RecursiveCharacterTextSplitter`
时如果分隔符列表里没有兜底的空字符串 `""`，一旦自定义分隔符在文本里不出
现或用尽，可能产出单个远超 `chunk_size` 的分片。`/search` 直接返回整段
chunk 不做截断，这就违反了"chunk_size 是硬上限"的约束。

**解决**：新增 `_enforce_chunk_size_cap` 后处理兜底，对自定义分隔符分支
和默认分支的输出都过一遍 `RecursiveCharacterTextSplitter(separators=[""])`
（保证能在任意字符边界切分），强制超限的 chunk 被二次拆分。

### Q10. 服务无鉴权，索引接口怎么防止被用来扫全盘？

**现象**：服务定位是内网 v1、无鉴权，`POST /directories/index` 一开始对
`path` 没有任何范围限制，一个 `{"path": "/", "recursive": true}` 请求就
能让 worker 遍历它能看到的整个文件系统。

**解决**：加 `Settings.allowed_index_roots`（默认空，回退到
`data_dir`）和 `Settings.effective_allowed_index_roots`，在已有的
"路径必须存在"校验之后，再校验请求路径是否在允许的根目录之下，不满足
返回 403。

### Q11. 模型权重为什么会被重复下载好几 GB？

**现象**：api 和 worker 两个容器都没设置 `HF_HOME`，权重默认下载到各自
容器可写层里，每次重建/重新创建容器都要在两个服务里各自重新下载一遍，
是此前"冷启动任务超时"问题的根因。

**解决**：把 `HF_HOME` 指向两个服务本就共享的 `./data` bind mount 下的
`/srv/data/hf_cache`，权重只下载一次，容器重建后复用，两个服务共享同一
份缓存。

---

## 四、部署 / Docker

### Q12. 一次 Docker 构建为什么要 29 分钟？

**现象**：`Dockerfile` 和 `admin_ui/Dockerfile` 都用 `context: .`，仓库
里没有 `.dockerignore`，`docker compose build` 每次都把整个仓库（含
6.3GB 的 `.venv`、运行时 `data/`、完整 `.git` 历史）发给 Docker daemon
作为构建上下文。

**解决**：加 `.dockerignore` 排除 `.venv`、`data`、`.git`、
`__pycache__`、`.pytest_cache`、`.ruff_cache`、`tests`、`docs`（排除前先
确认两个 Dockerfile 都没有 `COPY` 这些目录）。实测构建上下文从原来的体
积降到 68.96kB（主服务）和 7.07kB（admin_ui）。

### Q13. 只改一行 Python 代码为什么要重新下载几 GB 的 torch？

**现象**：`COPY requirements.txt` 和 `pip install`（含 torch）共用同一
个 `RUN` 层，`requirements.txt` 里任何改动，哪怕只是加一个纯 Python 的
小依赖，都会让这一整层的缓存失效，触发几 GB 的 torch+cudnn 重新下载。

**解决**：把 torch 的安装拆到单独一层，和其余依赖的安装层隔离，只有真正
涉及 torch 版本变化时才会失效那一层缓存。

---

## 五、数据正确性

### Q14. CSV 里一行超长数据为什么会突破 chunk 大小限制？

**现象**：`chunk_csv_rows()` 只在"多行拼起来超限"时才切分，如果单独一行
（表头+该行内容拼出来的那一行文本）本身就超过 `max_chars`，会被原样当
成一个超限 chunk 直接输出。

**解决**：检测到单行格式化后的文本本身超过 `max_chars` 时，先把已经攒
的 pending chunk 落盘，再用 `chunk_text(separators=[""])` 对这一行做字
符级二次拆分，保证任何情况下产出的 chunk 都不超过上限。

### Q15. 批量导入 PDF 时为什么会出现文件句柄泄漏？

**现象**：`parse_pdf` 里 `fitz.Document` 没有用上下文管理器，如果处理某
一页时抛异常（比如页面损坏、OCR 引擎出错），文件句柄不会被关闭，批量导
入场景下逐渐累积句柄泄漏。

**解决**：改用 `with` 上下文管理器包住 `fitz.Document`，保证无论是否发
生异常都会正确关闭文件句柄。

### Q16. Embedding 模型为什么会静默退化到 CPU 跑？

**现象**：`TextEmbedder` 构造 `BGEM3FlagModel` 时没有显式指定
`device`，某些环境下会静默 fallback 到 CPU，没有任何报错或警告，只是推
理变得异常慢，很容易被误判成"模型/数据本身的问题"而不是设备问题。

**解决**：显式 `device = "cuda" if torch.cuda.is_available() else "cpu"`
并传给模型构造函数，让设备选择结果可见、可预期，不再依赖库内部的隐式
判断逻辑。

### Q17. 删除一份 PDF 文档，为什么它的内嵌图片没有一起消失？

**现象**：删除 PDF 后，内嵌图片作为"孤儿点"残留在 Qdrant `images`
collection 里，在文档/任务列表页被当成一堆独立文档记录展示出来。

**根因**：PDF 内嵌图片使用组合 `doc_id`（`{doc_id}:img{idx}`），与父文
档 `doc_id` 不同；`delete_document` 原来只按 `doc_id` 精确匹配删除，捞
不到这些组合 ID 的图片记录。`list_all_documents` 扫描时会把
`doc_id` 按 `:` 拆出候选父 ID，父文档还在时归并显示为"关联图片"，父文
档被删后归并条件不成立，图片就被当成独立文档展示——这就是"图片全部散
落展示"的表现。

**解决**：`delete_document` 新增 `_delete_related_pdf_images`：扫描
`images` collection，删除所有 `doc_id` 字段以 `{doc_id}:` 开头的点，同
时删除对应的原图和缩略图磁盘文件。顺带修复了两处此前用注释标注为
"known limitation"但一直没处理的遗留问题：`watcher.py` 的
`on_deleted`（监控目录里文件被删）和 `pipeline.py` 的
`_cleanup_stale_doc_ids`（内容变更重新入库清理旧版本）现在也会级联清理
内嵌图片。详见 `docs/pdf-image-orphan-on-delete-fix.md`。

**延伸**：这题适合讲"组合 ID 引入的隐含契约"——一旦子资源的 ID 不再是父
资源 ID 的精确匹配，所有按 ID 做精确操作（删除、统计、归并展示）的地方
都要重新审视是否覆盖了这种前缀关系,而不是只在写入路径想清楚就够了。

### Q18. 跨模态搜索（文字查图片）明明命中了，为什么界面上什么都看不到？

**现象**：Admin UI 的搜索结果里图片命中只打印 `source_path` 文本，即使
查询正确命中了一张图（比如搜"地产"命中一张 OCR 过的房产广告图），界面
也只显示一行路径文本，看不到图片本身。

**根因**：admin-ui 容器和 api/worker 容器之间没有共享文件系统的 volume
挂载，API 也从未把图片字节通过 HTTP 暴露出来——即使检索结果正确，也没有
物理通道能把图片内容传到前端。

**解决**：新增 `GET /documents/{doc_id}/thumbnail`（配合新增的
`find_image_by_doc_id` 查找），Admin UI 改为通过这个接口拉取缩略图字
节，用 Gallery 组件渲染，而不是纯文本路径列表。

---

## 六、架构演进：PDF 解析方案替换

### Q19. 为什么把 PDF 解析从 fitz + PaddleOCR 换成 MinerU？

**背景**：原方案逐页处理——PyMuPDF 抽取每页文本层，抽不到文字的页判定
为扫描页，栅格化整页图像跑 PaddleOCR 兜底，扫描页整页图像本身也存成一
条独立的 `ImageRecord`。

**问题**：这条路径完全丢失文档结构——章节/标题层级、列表、表格全部丢
失，每页文本直接进字符级 `chunk_text()`，PDF 的 chunk 天生拿不到已经给
`.md` 文件做好的 `heading_path`/`has_table`/`has_code_block` 结构化元数
据，PDF 和 Markdown 两类文档的检索质量因此存在系统性差距。

**方案**：引入开源工具 MinerU（自带版面分析 + OCR），把整份 PDF（不论
文字层 PDF 还是扫描 PDF）统一转成保留标题层级/列表/HTML 表格的
Markdown，PDF 正文复用已有的结构化 Markdown 切片器，内嵌图片单独抽取、
CLIP 向量化、正文里的图片引用改写成指向新增图片接口的可访问链接。

**明确不做的范围**（避免范围蔓延）：不做页码级定位（`page` 字段统一为
`None`，需要的话后续可以用 MinerU `content_list.json` 的 `page_idx`
字段补）；不区分文字来自扫描页还是文字层（`source_type` 统一
`"normal"`）；解析失败不做二次兜底或双引擎切换，直接让异常沿用现有的
`ingest_file` 统一 `try/except` 冒泡成 `IngestResult(status="failed")`。

**延伸**：这题适合讲技术选型和"重写 vs 增量修补"的判断——结构化元数据
缺失不是一个局部 bug，而是逐页处理这个架构本身的局限，与其在旧架构上
一点点补丁式加结构识别，不如换一个天生输出结构化 Markdown 的工具，让
PDF 和 Markdown 复用同一套下游切片/检索逻辑。设计文档里显式列出"本轮不
做"的范围，是为了控制这次重写的影响面，把页级溯源这类可以后置的需求
明确挪到后续迭代。
