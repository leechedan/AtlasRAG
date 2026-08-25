# 切片质量量化评估工具 — 设计文档

日期：2026-08-25

## 背景与问题

`app/ingest/chunking.py` 当前的切片实现是纯字符长度的
`RecursiveCharacterTextSplitter`（`chunk_text`），配合按行/CSV 行数切分的
`chunk_csv_rows`。这套实现：

- 不解析 Markdown 结构（无 AST），标题层级信息在切片阶段完全丢失。
- 不识别表格、代码块等语义整体，长 chunk 触发 `_enforce_chunk_size_cap`
  的硬切分时（`separators=[""]`）可能从任意字符位置切断表格/代码块。
- 没有任何量化手段判断当前配置（`chunk_size=1000`,
  `chunk_overlap=150`）切出来的 chunk 是否语义完整、边界是否合理。

用户反馈的不满意点集中在四类：表格/代码块被切断、标题与正文分离、语义不
连贯（句子被截断）、下游检索召回效果不理想。本设计只解决"如何量化测量
这些问题"，不改动 `chunking.py` 本身，也不实现新的切片器 —— 为后续优化
提供可重复运行的基线和 A/B 对比工具。

## 范围

- **本轮做**：结构违规检测、长度分布统计、语义连贯性指标、端到端检索
  指标（Recall@K/MRR）的计算函数与 CLI 报告工具。
- **本轮不做**：不改 `app/ingest/chunking.py`；不实现结构感知的新切片
  器；不做 LLM 自动生成 query-answer 标注集（该功能作为可插拔接口留空，
  由 `--qa-file` 接收人工/后续工具产出的标注集）。
- **文档类型范围**：聚焦 Markdown/纯文本（`.md`/`.txt`）。PDF、CSV、图片
  OCR 文本的切片质量评估留作后续扩展（见"已知限制"）。

## 架构

```
app/eval/
  __init__.py
  structure_metrics.py   # 表格/代码块截断、标题上下文丢失检测（基于 markdown AST）
  length_metrics.py      # 长度分布统计
  coherence_metrics.py   # 相邻句子语义连贯性（复用 app/models/embeddings.py）
  retrieval_metrics.py   # Recall@K / MRR 通用计算函数
  report.py              # 汇总各模块指标 -> dict -> JSON + Markdown 报告
  runner.py              # 编排：读取文档 -> chunk_text() -> 跑各项指标 -> 生成报告

scripts/eval_chunk_quality.py   # CLI 入口，薄封装 app.eval.runner

tests/unit/test_eval_structure_metrics.py
tests/unit/test_eval_length_metrics.py
tests/unit/test_eval_coherence_metrics.py
tests/unit/test_eval_retrieval_metrics.py
```

`app/eval/` 是独立、可被其他脚本 import 的库代码（不依赖 FastAPI/Qdrant
运行时也能跑结构/长度指标），`scripts/eval_chunk_quality.py` 只负责参数
解析和调用 `runner.run(...)`。

## 指标设计

### 1. 结构违规率（structure_metrics.py）

新增依赖 `markdown-it-py` + `mdit-py-plugins`（table 插件）做 AST 解析：

- 解析每个源文档，收集"受保护区间"：`fence`（代码块）、`table`（表格）
  的起止行号，转换为原文的字符偏移区间。
- 对 `chunk_text()` 产出的每个 chunk，在原文中近似定位其字符区间：从上
  一个 chunk 结束位置减去 `chunk_overlap` 的余量开始向后 `find()`，作为
  下一个 chunk 的起始搜索位置。这是启发式定位（非严格的按位置回放),
  在报告中会注明这一限制。
- 违规判定：一个受保护区间的起止不完全落在同一个 chunk 的字符区间内
  （即某个 chunk 边界落在区间内部），记一次结构违规。
- 标题上下文丢失率：对每个 chunk，找到其在原文中对应内容最近的前置标题
  行；若该标题行本身不是这个 chunk 内容的一部分（即 chunk 开头不是紧跟
  在某个标题之后、或标题不在 chunk 文本内），计为"丢失标题上下文"。
- 输出：每文档 & 语料库聚合的「表格/代码块截断 chunk 占比」「标题上下文
  丢失 chunk 占比」。

### 2. 长度分布（length_metrics.py）

- 对语料库全部 chunk 计算字符数的 min/max/mean/median/stdev。
- 目标区间默认 `[0.2 × chunk_size, chunk_size]`（可通过 CLI
  `--min-chars`/`--max-chars` 覆盖），统计超出区间的 chunk 占比。

### 3. 语义连贯性（coherence_metrics.py）

- chunk 内部按 `。！？` 切句（复用与 `chunk_separators` 一致的标点集）。
- 用 `app.models.embeddings.get_text_embedder()`（BGE-M3 dense 向量）算
  相邻句子对的余弦相似度，取均值作为该 chunk 的连贯性分数。
- 聚合为 chunk 级、文档级、语料库级的均值/分布。
- 开销大（需加载 BGE-M3 模型跑推理），CLI 提供 `--no-semantic` 跳过。

### 4. 端到端检索指标（retrieval_metrics.py）

- 纯函数：`evaluate_retrieval(queries, retrieve_fn, k_values=[5, 10])`。
  - `queries: list[{"query": str, "expected_chunk_ids": list[str]}]`
  - `retrieve_fn: (query: str, top_k: int) -> list[str]`（返回 chunk_id
    有序列表）
  - 返回每个 k 的 Recall@k，以及整体 MRR。
- 标注集 JSON schema（供人工标注或后续 LLM 生成工具复用）：
  ```json
  [
    {"query": "...", "expected_chunk_ids": ["doc123:0", "doc123:1"]}
  ]
  ```
  `chunk_id` 与 `app.store.qdrant_client.compute_chunk_id(doc_id,
  chunk_index)` 的产出格式一致。
- CLI 通过 `--qa-file` 可选传入标注集；传了才跑这部分，跑的话直接接
  `search_text_hybrid` 做真实检索。不传则报告中注明"未提供标注集，跳过
  端到端检索指标"。
- 本轮不实现标注集自动生成；`retrieve_fn` 之外不预设任何 LLM 调用。

## CLI 设计

```
python -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunk-size 1000 --chunk-overlap 150 \
  --min-chars 200 --max-chars 1000 \
  --no-semantic \
  --qa-file eval/qa_set.json \
  --output report.json
```

- `--input-dir`：递归扫描 `.md`/`.txt` 文件作为语料库。
- `--chunk-size`/`--chunk-overlap`：覆盖默认配置，用于同一批文档跑多组
  参数做消融对比（多次运行、对比多份 `report.json`）。
- `--no-semantic`：跳过语义连贯性指标（默认跑）。
- `--qa-file`：可选，提供则跑检索指标。
- `--output`：写 JSON 报告；同时在控制台打印一份汇总表（Markdown 格式）。

## 数据流

```
input-dir 下的 .md/.txt 文件
  -> 逐个读取原文
  -> chunk_text(text, chunk_size, chunk_overlap)   # 复用现有实现，不改动
  -> structure_metrics（需要原文 + markdown AST）
  -> length_metrics（只需要 chunk 列表）
  -> coherence_metrics（需要 chunk 列表 + embedder，可跳过）
  -> [可选] retrieval_metrics（需要 --qa-file + 真实 Qdrant 检索）
  -> report.py 汇总 -> JSON + 控制台 Markdown 表格
```

## 测试计划

- `structure_metrics`：人造含表格/代码块/多级标题的 Markdown 字符串，
  构造已知会/不会违规的切片方案，断言检测结果。
- `length_metrics`：构造已知长度分布的 chunk 列表，断言统计值。
- `coherence_metrics`：注入 fake embedder（返回固定/可控向量），不加载
  真实 BGE-M3，断言相似度计算和聚合逻辑正确。
- `retrieval_metrics`：用假 `retrieve_fn` 验证 Recall@k/MRR 计算公式。
- 不新增依赖真实 Qdrant/GPU 模型的集成测试；端到端跑法留给人工通过 CLI
  执行。

## 新增依赖

`requirements.txt` 新增：
```
markdown-it-py==3.0.0
mdit-py-plugins==0.4.2
```

## 已知限制 / 后续扩展

- Chunk 在原文中的字符区间定位是启发式的（基于顺序搜索 + overlap 回退），
  不是切片器内部状态的精确回放；极端情况下（如原文中有大量重复子串）
  可能定位错误，报告会注明这一局限。
- 本轮结构/语义指标只覆盖 `.md`/`.txt`；PDF 分页文本、CSV 行式 chunk、
  图片 OCR 文本的质量评估留作后续扩展。
- 检索指标标注集需要人工提供或后续接入 LLM 自动生成（接口已预留，未
  实现生成逻辑）。
- 本设计不产出新的切片器；后续若要做结构感知 Markdown 切片器，可直接用
  本工具做新旧方案的 A/B 对比。
