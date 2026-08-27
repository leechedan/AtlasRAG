# 结构化 Markdown 切片器 — 设计文档

日期：2026-08-26

## 背景与问题

`app/ingest/chunking.py` 的 `chunk_text()` 是纯字符长度的
`RecursiveCharacterTextSplitter`，不解析 Markdown 结构，标题层级在切片阶段
完全丢失，长 chunk 硬切分时可能从任意字符位置切断表格/代码块。

`docs/superpowers/specs/2026-08-25-chunk-quality-eval-design.md` 已经落地了
量化评估工具（`app/eval/`），用于测量这些问题，但明确不产出新的切片器，
把"结构感知的 Markdown 切片器"留给后续工作，并约定用该评估工具做新旧方案
的 A/B 对比。

本设计就是这个后续工作：实现一个基于标题层级 AST 的结构化 Markdown 切片
器，参考样例见 `data/dataset/备考chunks.json`（对应文档
`data/uploads/upload_2026-08-25T12_57_30.283214+00_00_备考总计划TODO.md/
备考总计划TODO.md`）——该文件展示了目标切分粒度和 chunk 元数据结构，但其
`content` 字段是人工精简改写过的摘要，不代表新切片器要复现的输出形式（新
切片器的 `content` 保留原文，不做任何改写）。

## 范围

- **本轮做**：新增独立的结构化 Markdown 切片函数
  `chunk_markdown_structured()`；接入 `app/eval/runner.py` 和
  `scripts/eval_chunk_quality.py`，支持 `--chunker structured` 与现有字符级
  切片做 A/B 对比。
- **本轮不做**：不改 `app/ingest/pipeline.py`、Qdrant 存储 schema
  （`TextChunkRecord`）、API schema（`ChunkOut`）、Admin UI；不接入生产
  摄取链路；不做检索指标（`retrieval_metrics`）的结构化切片对比（需要真实
  Qdrant 数据，留作后续）。
- **文档类型范围**：只处理 Markdown（`.md`）。非 Markdown 内容（无任何
  标题）会退化为整篇一个 chunk，见"边界情况"。

## 架构

```
app/ingest/markdown_chunker.py   # 新增：结构化切片核心逻辑
  StructuredChunk (dataclass)
  chunk_markdown_structured(text, chunk_size=None, chunk_overlap=None) -> list[StructuredChunk]

app/eval/runner.py               # 改：新增 chunker 参数，支持 "structured"
scripts/eval_chunk_quality.py    # 改：新增 --chunker CLI 参数

tests/unit/test_markdown_chunker.py   # 新增
```

复用已有依赖 `markdown-it-py`（`app/eval/structure_metrics.py` 已引入），
用 `MarkdownIt("commonmark").enable("table")` 解析，`token.map` 给出的行号
区间直接切原文——内容是原文逐字提取，不改写、不摘要。

## 核心算法

单次遍历标题 token，不需要预先建树：

1. 解析文档，取出所有 `heading_open` token（H1–H5，不限层级）。
2. 维护一个"标题栈"：遇到新标题时，弹出栈里 level ≥ 新标题 level 的项，
   再压入新标题；栈内标题文本用 `" > "` 拼接得到 `heading_path`。
3. 每个标题的"直属内容" = 从该标题结束到**下一个标题（不论层级）**开始
   之间的原文行区间。直属内容为空/纯空白则跳过，不产出 chunk。
4. 文档第一个标题之前的内容（如果存在，比如没有前置标题的纯文本开头）单
   独处理，`heading_path` 用占位符 `"(文档开头)"`。

不需要为"标题+引言块单独成 chunk 0"或"没有 H3 的大节按 H2 整体切"写任何
特判分支——它们都是同一条规则的自然结果：H1 的直属内容天然排在最前面；
一个 H2 如果没有自己的直属内容（标题后紧跟子标题），就不产出该 H2 自己的
chunk，只有其子标题各自产出；一个 H2 如果既有自己的直属内容、又有 H3 子
节（如原文档"一、总时间轴"），两者分别产出各自的 chunk。

## 直属内容内部的二次拆分

某标题的直属内容如果本身很长、且内部有多个"不可拆元素"（表格/代码块），
在元素之间的空白处二次拆分，绝不从表格/代码块内部切断：

- 把直属内容按块级 token（段落、表格、代码块、列表、引用块等）顺序切成
  若干不可再拆的"块"。
- 流式打包：依次把连续的块装进当前 chunk，一旦加入下一个块会超过
  `chunk_size`（复用 `settings.chunk_size`，默认 1000），就在此处断开、开
  新 chunk。
- 单个块本身超过 `chunk_size`（比如一个巨大表格/代码块）：保持整体不拆，
  允许该 chunk 超限——这是"表格/代码块不可跨 chunk"硬约束的直接推论。
- 例外：单个**非保护块**（普通段落/列表）自己超过 `chunk_size`，对这一个
  块回退调用现有的 `chunk_text()`（字符级切分）——唯一会退化到旧逻辑的
  地方。
- 二次拆分产生多个 chunk 时，`heading_path` 用纯序号后缀区分：第一个
  chunk 不加后缀，第 2 个及以后是 `"<breadcrumb> #2"`、`"<breadcrumb> #3"`
  ……不使用描述性文字（区别于 `备考chunks.json` 里手工加的"年龄要求表"这
  类摘要式后缀）。
- 二次拆分之间、以及不同标题的 chunk 之间原本都不做字符重叠（overlap）。
  **更新（见 `docs/markdown-chunker-overlap-fix.md`）**：这个决定不加区分
  地套用到了所有分组边界，包括两个都是普通段落、纯粹因为长度超限被切开
  的分组——这类边界现在会借用上一个分组结尾的 `chunk_overlap` 个字符做
  上下文衔接。仍然不做重叠的只剩两种情况：上一个分组的最后一个块是受保
  护块（表格/代码，借用会截出半段没有意义），以及不同标题小节之间（新标
  题是真实的语义边界）。

## Chunk 字段

```python
@dataclass
class StructuredChunk:
    chunk_id: int
    heading_path: str
    has_table: bool
    has_code_block: bool
    char_count: int
    token_estimate: int   # round(len(content) / 1.8)，粗略启发式估算
    content: str           # 原文逐字提取，不改写
```

`token_estimate` 用字符数的启发式估算（`/1.8`），不接真实 tokenizer——项目
里唯一的真实 tokenizer 来自 BGE-M3 embedder，为了一个元数据字段加载该模型
成本不划算。

## A/B 评估集成

- `app/eval/runner.py` 的 `run()` 新增 `chunker: Literal["char", "structured"] = "char"` 参数；
  `"structured"` 时调用 `chunk_markdown_structured()`，取每个 chunk 的
  `content` 拼成 `list[str]`，喂给现有的 `structure_metrics`/
  `length_metrics`/`coherence_metrics`。`retrieval_metrics` 需要真实 Qdrant
  数据，本轮不接结构化切片，跳过。
- `scripts/eval_chunk_quality.py` 新增 `--chunker {char,structured}` CLI
  参数，默认 `char`（不改变现有默认行为）。
- 预期效果：结构化切片器产出的 chunk 天然满足"表格/代码块不截断"，跑
  `structure_metrics.detect_structure_violations` 违规率应为 0——写进测试
  作为回归断言点。

## 错误处理 / 边界情况

- 空文本/全空白：返回空列表（与 `chunk_text` 行为一致）。
- 文档没有任何标题：整篇作为 `chunk_id=0`，`heading_path="(文档开头)"`；如
  果超过 `chunk_size`，走块级打包逻辑（此时"块"就是普通段落）。
- 表格/代码块识别依赖 `markdown-it-py` 的 `table`/`fence` token，与
  `app/eval/structure_metrics.py` 用同一套解析逻辑、同样的已知局限（例如
  缩进代码块不会被识别为 `fence`）。
  **更新（2026-08-26 review 修复）**：`markdown_chunker.py` 现在把
  `code_block`（4 空格缩进代码块）token 和 `fence` 同样归类为
  `kind="code"`，因此在本模块里缩进代码块已经和围栏代码块一样被识别、
  被保护（不会被字符级回退逻辑从中间截断，也会置位
  `has_code_block=True`）。上面这条局限仍然只针对
  `app/eval/structure_metrics.py` 自己的违规检测——它内部只看
  `fence`/`table_open` 两种 token（未跟随本次修复变更），和本模块无关，
  超出本次修复范围。

## 测试计划

`tests/unit/test_markdown_chunker.py`：
- 构造小型 md 字符串（H1–H5 混合、H2 既有自己正文又有 H3 子节、表格、代码
  块、无标题开头文本），断言 `heading_path`/`has_table`/`has_code_block`/
  `chunk_id` 正确。
- 断言表格/代码块永远不被拆到两个 chunk（复用
  `structure_metrics.detect_structure_violations` 校验，违规率为 0）。
- 断言直属内容超长时按块打包二次拆分、序号后缀正确、不做 overlap。
- 断言单个非保护块超过 `chunk_size` 时回退 `chunk_text` 字符级切分。
- 用真实的"备考总计划TODO.md"跑一遍，人工核对产出的 chunk 数量/边界和
  `备考chunks.json` 的**切分粒度**大致对齐（不要求 content 逐字相同）。

扩展 `app/eval/` 相关单测覆盖 `chunker="structured"` 路径；
`scripts/eval_chunk_quality.py` 的 `--chunker` 参数解析加断言。

## 已知限制 / 后续扩展

- 不接入生产摄取链路（`pipeline.py`/Qdrant/API/Admin UI）；如果后续要接入，
  需要给 `TextChunkRecord`/`ChunkOut` 加 `heading_path`/`has_table`/
  `has_code_block` 字段，并处理存量数据兼容。
- `retrieval_metrics` 的结构化切片 A/B 对比留作后续（需要真实 Qdrant 摄取
  + 标注集）。
- 只覆盖 `.md`；`.txt`/`.csv`/PDF/图片 OCR 文本仍用旧 `chunk_text`。
- 含表格/代码块的 chunk 在 `content` 里会保留一个末尾换行符（`_group_text`
  对这类组只做 `lstrip()` 而不是 `.strip()`），这是刻意保留的，用来配合
  `app/eval/structure_metrics.detect_structure_violations` 的区域包含性
  判断：该函数用 `line_offsets[region.end_line]`（含末尾换行）作为受保护
  区域的结束偏移，如果 chunk 文本被完整 `.strip()` 掉末尾换行，一个内容
  完整的 chunk 会比这个结束偏移少一个字符，从而被误判为截断
  （在 `data/uploads` 语料上实测会导致 11 个受保护区域中的 3 个出现这种
  假阳性）。这是 chunker 空白处理和这一个 eval 指标实现之间真实存在的耦
  合，不是格式上的随意不一致；如果以后把 `detect_structure_violations`
  的包含性判断改成对空白容忍，就可以去掉这个不对称处理，统一用
  `.strip()`。
