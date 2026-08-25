# 切片质量量化评估工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `app/ingest/chunking.py` 现有切片实现建立一套可重复运行的质量量化评估工具（结构违规检测、长度分布、语义连贯性、端到端检索指标），产出 JSON + 控制台报告，为后续优化提供基线，不改动切片实现本身。

**Architecture:** `app/eval/` 下的纯函数指标模块（structure/length/coherence/retrieval）+ `runner.py` 编排 + `report.py` 输出，`scripts/eval_chunk_quality.py` 做参数解析并把真实 `chunk_text()`、BGE-M3 embedder、Qdrant 检索接进 runner。

**Tech Stack:** Python 3.12, `markdown-it-py`（新增依赖，AST 解析）, 复用项目现有 `langchain-text-splitters`（`chunk_text`）与 `FlagEmbedding`（`get_text_embedder`）, pytest。

## Global Constraints

- 不修改 `app/ingest/chunking.py` 的切片逻辑（设计范围明确排除）。
- 本轮只覆盖 `.md`/`.txt` 文档；PDF/CSV/图片 OCR 留作后续扩展。
- 所有单测不得依赖真实 GPU 模型（BGE-M3/CLIP）或真实 Qdrant 实例；用假 embedder / 假 `retrieve_fn` 注入。
- 新增依赖仅 `markdown-it-py==3.0.0`（写入 `requirements.txt`）；表格解析用其核心 `table` 规则（`MarkdownIt("commonmark").enable("table")`），不需要额外插件包。
- Recall@K 采用宏平均的标准召回率定义：每个 query 的 `|命中 expected_chunk_ids| / |expected_chunk_ids|`，再对所有 query 取平均（不是"命中即算1"的 hit-rate）。

---

### Task 1: 长度分布指标（length_metrics.py）

**Files:**
- Create: `app/eval/__init__.py`（空文件）
- Create: `app/eval/length_metrics.py`
- Test: `tests/unit/test_eval_length_metrics.py`

**Interfaces:**
- Produces: `compute_length_metrics(chunks: list[str], min_chars: int, max_chars: int) -> LengthReport`，`LengthReport` dataclass 字段：`count, min_chars, max_chars, mean_chars, median_chars, stdev_chars, out_of_range_count, out_of_range_ratio`（均为普通属性，非 property）。

- [x] **Step 1: 创建 `app/eval/__init__.py`（空文件）**

- [x] **Step 2: 写失败测试**

```python
# tests/unit/test_eval_length_metrics.py
from app.eval.length_metrics import compute_length_metrics


def test_compute_length_metrics_basic_stats():
    chunks = ["a" * 100, "b" * 200, "c" * 300]
    report = compute_length_metrics(chunks, min_chars=50, max_chars=250)

    assert report.count == 3
    assert report.min_chars == 100
    assert report.max_chars == 300
    assert report.mean_chars == 200.0
    assert report.median_chars == 200.0
    assert report.out_of_range_count == 1
    assert report.out_of_range_ratio == 1 / 3


def test_compute_length_metrics_empty_chunks_returns_zeroed_report():
    report = compute_length_metrics([], min_chars=50, max_chars=250)

    assert report.count == 0
    assert report.out_of_range_ratio == 0.0


def test_compute_length_metrics_single_chunk_has_zero_stdev():
    report = compute_length_metrics(["x" * 100], min_chars=50, max_chars=250)

    assert report.stdev_chars == 0.0
```

- [x] **Step 3: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_length_metrics.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.length_metrics'`

- [x] **Step 4: 写实现**

```python
# app/eval/length_metrics.py
"""Chunk 长度分布统计：min/max/mean/median/stdev，以及超出目标区间的占比。"""

import statistics
from dataclasses import dataclass


@dataclass
class LengthReport:
    count: int
    min_chars: int
    max_chars: int
    mean_chars: float
    median_chars: float
    stdev_chars: float
    out_of_range_count: int
    out_of_range_ratio: float


def compute_length_metrics(chunks: list[str], min_chars: int, max_chars: int) -> LengthReport:
    if not chunks:
        return LengthReport(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0)

    lengths = [len(c) for c in chunks]
    out_of_range = sum(1 for length in lengths if length < min_chars or length > max_chars)

    return LengthReport(
        count=len(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        mean_chars=statistics.mean(lengths),
        median_chars=statistics.median(lengths),
        stdev_chars=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        out_of_range_count=out_of_range,
        out_of_range_ratio=out_of_range / len(lengths),
    )
```

- [x] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_length_metrics.py -v`
Expected: PASS（3 passed）

- [x] **Step 6: 提交**

```bash
git add app/eval/__init__.py app/eval/length_metrics.py tests/unit/test_eval_length_metrics.py
git commit -m "feat: add chunk length distribution eval metric"
```

---

### Task 2: 结构违规检测（structure_metrics.py）

**Files:**
- Modify: `requirements.txt`（新增 `markdown-it-py==3.0.0`）
- Create: `app/eval/structure_metrics.py`
- Test: `tests/unit/test_eval_structure_metrics.py`

**Interfaces:**
- Consumes: 无（独立模块，只依赖 `markdown-it-py`）
- Produces:
  - `parse_markdown_structure(text: str) -> DocStructure`（`DocStructure` 字段：`line_offsets: list[int]`, `protected_regions: list[ProtectedRegion]`, `heading_owner_line: list[int | None]`；`ProtectedRegion` 字段：`start_line, end_line, kind`）
  - `locate_chunk_spans(original_text: str, chunks: list[str], chunk_overlap: int) -> list[tuple[int, int]]`
  - `detect_structure_violations(text: str, chunks: list[str], chunk_overlap: int) -> StructureReport`（`StructureReport` 字段：`total_chunks, total_protected_regions, truncated_regions, chunks_with_heading_expected, chunks_missing_heading`，外加 `truncated_region_ratio`/`missing_heading_ratio` 两个 property）
  - Task 5（runner.py）会 import 并调用 `detect_structure_violations`。

- [x] **Step 1: 在 `requirements.txt` 新增依赖**

```
markdown-it-py==3.0.0
```
（追加到文件末尾即可，不要求特定顺序）

- [x] **Step 2: 写失败测试**

```python
# tests/unit/test_eval_structure_metrics.py
from app.eval.structure_metrics import (
    detect_structure_violations,
    locate_chunk_spans,
    parse_markdown_structure,
)

SAMPLE_MD = (
    "# Title\n"
    "\n"
    "intro\n"
    "\n"
    "## Section A\n"
    "\n"
    "| a | b |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "```python\n"
    "print(\"hi\")\n"
    "```\n"
    "\n"
    "tail text\n"
)


def test_parse_markdown_structure_finds_table_and_fence_as_protected_regions():
    structure = parse_markdown_structure(SAMPLE_MD)

    kinds = [(r.kind, r.start_line, r.end_line) for r in structure.protected_regions]
    assert ("table", 6, 9) in kinds
    assert ("code", 10, 13) in kinds


def test_parse_markdown_structure_maps_lines_to_owning_heading():
    structure = parse_markdown_structure(SAMPLE_MD)

    assert structure.heading_owner_line[0] == 0  # "# Title" 归自己
    assert structure.heading_owner_line[2] == 0  # "intro" 归 Title
    assert structure.heading_owner_line[6] == 4  # 表格行归 "## Section A"
    assert structure.heading_owner_line[10] == 4  # 代码块归 "## Section A"


def test_locate_chunk_spans_finds_sequential_non_overlapping_chunks():
    text = "AAAAA" "BBBBB" "CCCCC"
    chunks = ["AAAAA", "BBBBB", "CCCCC"]

    spans = locate_chunk_spans(text, chunks, chunk_overlap=0)

    assert spans == [(0, 5), (5, 10), (10, 15)]


def test_locate_chunk_spans_handles_overlap_by_searching_forward_from_backoff():
    text = "0123456789"
    chunks = ["012345", "345678"]  # 第二个 chunk 与第一个重叠 "345"

    spans = locate_chunk_spans(text, chunks, chunk_overlap=3)

    assert spans == [(0, 6), (3, 9)]


def test_detect_structure_violations_flags_table_and_fence_cut_by_hard_split():
    chunks = [SAMPLE_MD[:45], SAMPLE_MD[45:75], SAMPLE_MD[75:]]

    report = detect_structure_violations(SAMPLE_MD, chunks, chunk_overlap=0)

    assert report.total_protected_regions == 2
    assert report.truncated_regions == 2
    assert report.truncated_region_ratio == 1.0


def test_detect_structure_violations_no_violations_when_whole_doc_is_one_chunk():
    report = detect_structure_violations(SAMPLE_MD, [SAMPLE_MD], chunk_overlap=0)

    assert report.truncated_regions == 0
    assert report.truncated_region_ratio == 0.0


def test_detect_structure_violations_flags_missing_heading_context_on_continuation_chunk():
    chunks = [SAMPLE_MD[:45], SAMPLE_MD[45:75], SAMPLE_MD[75:]]

    report = detect_structure_violations(SAMPLE_MD, chunks, chunk_overlap=0)

    assert report.chunks_with_heading_expected == 3
    assert report.chunks_missing_heading == 2
```

- [x] **Step 3: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_structure_metrics.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.structure_metrics'`

- [x] **Step 4: 写实现**

```python
# app/eval/structure_metrics.py
"""Markdown 结构感知的切片质量检测：表格/代码块截断、标题上下文丢失。

结构解析基于 markdown-it-py 的 AST：`table_open`/`fence` token 的 `.map`
覆盖整个块（含围栏/分隔符行）的起止行号，用来判断一个 chunk 边界是否把
表格或代码块从中间切开。chunk 在原文中的字符区间通过 `locate_chunk_spans`
启发式定位（顺序搜索 + overlap 回退），不是切片器内部状态的精确回放，
原文中有大段重复子串时可能定位有误。
"""

from bisect import bisect_right
from dataclasses import dataclass

from markdown_it import MarkdownIt

_MD = MarkdownIt("commonmark").enable("table")


@dataclass
class ProtectedRegion:
    start_line: int
    end_line: int  # 不含
    kind: str  # "code" | "table"


@dataclass
class DocStructure:
    line_offsets: list[int]
    protected_regions: list[ProtectedRegion]
    heading_owner_line: list[int | None]


@dataclass
class StructureReport:
    total_chunks: int
    total_protected_regions: int
    truncated_regions: int
    chunks_with_heading_expected: int
    chunks_missing_heading: int

    @property
    def truncated_region_ratio(self) -> float:
        if not self.total_protected_regions:
            return 0.0
        return self.truncated_regions / self.total_protected_regions

    @property
    def missing_heading_ratio(self) -> float:
        if not self.chunks_with_heading_expected:
            return 0.0
        return self.chunks_missing_heading / self.chunks_with_heading_expected


def _compute_line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def parse_markdown_structure(text: str) -> DocStructure:
    line_offsets = _compute_line_offsets(text)
    n_lines = len(line_offsets) - 1

    protected_regions: list[ProtectedRegion] = []
    heading_owner_line: list[int | None] = [None] * n_lines
    current_heading_line: int | None = None

    for token in _MD.parse(text):
        if token.map is None:
            continue
        start_line, end_line = token.map
        if token.type == "heading_open":
            current_heading_line = start_line
        if token.type in ("fence", "table_open"):
            kind = "code" if token.type == "fence" else "table"
            protected_regions.append(ProtectedRegion(start_line, end_line, kind))
        for line in range(start_line, min(end_line, n_lines)):
            heading_owner_line[line] = current_heading_line

    return DocStructure(line_offsets, protected_regions, heading_owner_line)


def locate_chunk_spans(
    original_text: str, chunks: list[str], chunk_overlap: int
) -> list[tuple[int, int]]:
    """启发式定位每个 chunk 在原文中的字符区间：从上一个 chunk 结束位置
    减去 chunk_overlap 的余量开始向后搜索。原文中有大段重复内容时可能
    定位错误（见模块 docstring）。"""
    spans: list[tuple[int, int]] = []
    search_from = 0
    for chunk in chunks:
        pos = original_text.find(chunk, search_from)
        if pos == -1:
            pos = original_text.find(chunk)
        if pos == -1:
            spans.append((search_from, search_from + len(chunk)))
            search_from += len(chunk)
            continue
        end = pos + len(chunk)
        spans.append((pos, end))
        search_from = max(pos + 1, end - chunk_overlap)
    return spans


def detect_structure_violations(
    text: str, chunks: list[str], chunk_overlap: int
) -> StructureReport:
    structure = parse_markdown_structure(text)
    spans = locate_chunk_spans(text, chunks, chunk_overlap)
    n_lines = len(structure.heading_owner_line)

    truncated_regions = 0
    for region in structure.protected_regions:
        r_start = structure.line_offsets[region.start_line]
        r_end = structure.line_offsets[min(region.end_line, len(structure.line_offsets) - 1)]
        if not any(c_start <= r_start and c_end >= r_end for c_start, c_end in spans):
            truncated_regions += 1

    chunks_with_heading_expected = 0
    chunks_missing_heading = 0
    for c_start, c_end in spans:
        if n_lines == 0:
            continue
        start_line = min(max(bisect_right(structure.line_offsets, c_start) - 1, 0), n_lines - 1)
        end_line = min(
            max(bisect_right(structure.line_offsets, max(c_end - 1, c_start)) - 1, 0),
            n_lines - 1,
        )
        owner = structure.heading_owner_line[start_line]
        if owner is None:
            continue
        chunks_with_heading_expected += 1
        if not (start_line <= owner <= end_line):
            chunks_missing_heading += 1

    return StructureReport(
        total_chunks=len(chunks),
        total_protected_regions=len(structure.protected_regions),
        truncated_regions=truncated_regions,
        chunks_with_heading_expected=chunks_with_heading_expected,
        chunks_missing_heading=chunks_missing_heading,
    )
```

- [x] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_structure_metrics.py -v`
Expected: PASS（7 passed）

- [x] **Step 6: 提交**

```bash
git add requirements.txt app/eval/structure_metrics.py tests/unit/test_eval_structure_metrics.py
git commit -m "feat: add markdown structure violation eval metric"
```

---

### Task 3: 语义连贯性指标（coherence_metrics.py）

**Files:**
- Create: `app/eval/coherence_metrics.py`
- Test: `tests/unit/test_eval_coherence_metrics.py`

**Interfaces:**
- Consumes: 无（`Embedder` 是本模块定义的 `Protocol`，调用方注入任意满足 `embed(texts: list[str]) -> list[<has .dense>]` 的对象；真实场景注入 `app.models.embeddings.TextEmbedder`，其 `embed()` 返回的 `DenseSparseVector` 有 `.dense: list[float]` 字段，天然满足这个协议）
- Produces: `split_sentences(text: str) -> list[str]`，`compute_coherence(chunks: list[str], embedder: Embedder) -> CoherenceReport`（字段：`chunk_count, scored_chunk_count, mean_coherence`）。Task 5（runner.py）会 import 并调用两者。

- [x] **Step 1: 写失败测试**

```python
# tests/unit/test_eval_coherence_metrics.py
from dataclasses import dataclass

from app.eval.coherence_metrics import compute_coherence, split_sentences


@dataclass
class _FakeVector:
    dense: list[float]


class _FakeEmbedder:
    """按句子内容查表返回预设向量，构造已知相似度的场景。"""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]):
        return [_FakeVector(self._vectors[t]) for t in texts]


def test_split_sentences_splits_on_chinese_terminal_punctuation():
    sentences = split_sentences("第一句。第二句！第三句？")

    assert sentences == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_drops_empty_fragments():
    sentences = split_sentences("只有一句没有标点")

    assert sentences == ["只有一句没有标点"]


def test_compute_coherence_identical_vectors_score_one():
    chunks = ["甲。乙。"]
    embedder = _FakeEmbedder({"甲。": [1.0, 0.0], "乙。": [1.0, 0.0]})

    report = compute_coherence(chunks, embedder)

    assert report.scored_chunk_count == 1
    assert report.mean_coherence == 1.0


def test_compute_coherence_orthogonal_vectors_score_zero():
    chunks = ["甲。乙。"]
    embedder = _FakeEmbedder({"甲。": [1.0, 0.0], "乙。": [0.0, 1.0]})

    report = compute_coherence(chunks, embedder)

    assert report.mean_coherence == 0.0


def test_compute_coherence_single_sentence_chunk_not_scored():
    chunks = ["只有一句。"]
    embedder = _FakeEmbedder({"只有一句。": [1.0, 0.0]})

    report = compute_coherence(chunks, embedder)

    assert report.chunk_count == 1
    assert report.scored_chunk_count == 0
    assert report.mean_coherence == 0.0


def test_compute_coherence_empty_chunks_list():
    report = compute_coherence([], _FakeEmbedder({}))

    assert report.chunk_count == 0
    assert report.mean_coherence == 0.0
```

- [x] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_coherence_metrics.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.coherence_metrics'`

- [x] **Step 3: 写实现**

```python
# app/eval/coherence_metrics.py
"""Chunk 内相邻句子的语义连贯性：用注入的 embedder 给每个 chunk 内相邻
句子对算余弦相似度均值。开销集中在 embedder.embed() 调用；单测用假
embedder 注入固定向量，不加载真实 BGE-M3。"""

import re
from dataclasses import dataclass
from typing import Protocol

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list:
        ...


@dataclass
class CoherenceReport:
    chunk_count: int
    scored_chunk_count: int  # 句子数 >= 2 才有连贯性分数
    mean_coherence: float


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text)]
    return [p for p in parts if p]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_coherence(chunks: list[str], embedder: Embedder) -> CoherenceReport:
    sentence_lists = [split_sentences(c) for c in chunks]
    all_sentences = [s for sentences in sentence_lists for s in sentences]
    if not all_sentences:
        return CoherenceReport(len(chunks), 0, 0.0)

    vectors = embedder.embed(all_sentences)
    dense_vectors = [v.dense for v in vectors]

    chunk_scores: list[float] = []
    cursor = 0
    for sentences in sentence_lists:
        n = len(sentences)
        chunk_vectors = dense_vectors[cursor : cursor + n]
        cursor += n
        if n < 2:
            continue
        sims = [
            _cosine_similarity(chunk_vectors[i], chunk_vectors[i + 1]) for i in range(n - 1)
        ]
        chunk_scores.append(sum(sims) / len(sims))

    mean_coherence = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    return CoherenceReport(len(chunks), len(chunk_scores), mean_coherence)
```

- [x] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_coherence_metrics.py -v`
Expected: PASS（6 passed）

- [x] **Step 5: 提交**

```bash
git add app/eval/coherence_metrics.py tests/unit/test_eval_coherence_metrics.py
git commit -m "feat: add chunk semantic coherence eval metric"
```

---

### Task 4: 端到端检索指标（retrieval_metrics.py）

**Files:**
- Create: `app/eval/retrieval_metrics.py`
- Test: `tests/unit/test_eval_retrieval_metrics.py`

**Interfaces:**
- Consumes: 无
- Produces: `RetrievalQuery`（字段：`query: str, expected_chunk_ids: list[str]`）、`evaluate_retrieval(queries: list[RetrievalQuery], retrieve_fn: Callable[[str, int], list[str]], k_values: list[int] | None = None) -> RetrievalReport`（字段：`query_count, recall_at_k: dict[int, float], mrr: float`）、`load_qa_file(path: str) -> list[RetrievalQuery]`。Task 5（runner.py）与 `scripts/eval_chunk_quality.py`（Task 6）都会用到这三者。

- [x] **Step 1: 写失败测试**

```python
# tests/unit/test_eval_retrieval_metrics.py
import json

from app.eval.retrieval_metrics import RetrievalQuery, evaluate_retrieval, load_qa_file


def test_evaluate_retrieval_perfect_match_gives_full_recall_and_mrr_one():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c1"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2", "c3"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[1, 5])

    assert report.query_count == 1
    assert report.recall_at_k[1] == 1.0
    assert report.recall_at_k[5] == 1.0
    assert report.mrr == 1.0


def test_evaluate_retrieval_miss_gives_zero_recall_and_mrr():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["missing"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[5])

    assert report.recall_at_k[5] == 0.0
    assert report.mrr == 0.0


def test_evaluate_retrieval_rank_affects_mrr():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c2"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2", "c3"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[5])

    assert report.mrr == 0.5  # 命中排第2位 -> 1/2


def test_evaluate_retrieval_partial_recall_with_multiple_expected_chunks():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c1", "c2"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c9"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[2])

    assert report.recall_at_k[2] == 0.5  # 命中1/2


def test_evaluate_retrieval_averages_across_multiple_queries():
    queries = [
        RetrievalQuery(query="q1", expected_chunk_ids=["c1"]),
        RetrievalQuery(query="q2", expected_chunk_ids=["missing"]),
    ]

    def retrieve_fn(query, top_k):
        return {"q1": ["c1"], "q2": ["c9"]}[query]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[1])

    assert report.recall_at_k[1] == 0.5
    assert report.mrr == 0.5


def test_evaluate_retrieval_empty_queries_returns_zeroed_report():
    report = evaluate_retrieval([], lambda q, k: [], k_values=[5])

    assert report.query_count == 0
    assert report.recall_at_k == {5: 0.0}
    assert report.mrr == 0.0


def test_load_qa_file_parses_json_into_retrieval_queries(tmp_path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps([{"query": "什么是切片", "expected_chunk_ids": ["doc1:0", "doc1:1"]}]),
        encoding="utf-8",
    )

    queries = load_qa_file(str(qa_path))

    assert queries == [RetrievalQuery(query="什么是切片", expected_chunk_ids=["doc1:0", "doc1:1"])]
```

- [x] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_retrieval_metrics.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.retrieval_metrics'`

- [x] **Step 3: 写实现**

```python
# app/eval/retrieval_metrics.py
"""端到端检索质量：Recall@K（宏平均标准召回率）与 MRR。

标注集 JSON schema：`[{"query": "...", "expected_chunk_ids": ["doc_id:chunk_index 的
md5，即 app.store.qdrant_client.compute_chunk_id 的产出"]}]`。本模块不生成标注集，
只消费它——生成方式（人工标注或未来接入 LLM）留给调用方。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RetrievalQuery:
    query: str
    expected_chunk_ids: list[str]


@dataclass
class RetrievalReport:
    query_count: int
    recall_at_k: dict[int, float]
    mrr: float


def evaluate_retrieval(
    queries: list[RetrievalQuery],
    retrieve_fn: Callable[[str, int], list[str]],
    k_values: list[int] | None = None,
) -> RetrievalReport:
    resolved_k_values = list(k_values) if k_values else [5, 10]

    if not queries:
        return RetrievalReport(0, {k: 0.0 for k in resolved_k_values}, 0.0)

    max_k = max(resolved_k_values)
    recall_sums = {k: 0.0 for k in resolved_k_values}
    reciprocal_ranks: list[float] = []

    for q in queries:
        expected = set(q.expected_chunk_ids)
        retrieved = retrieve_fn(q.query, max_k)

        rank = next((i + 1 for i, cid in enumerate(retrieved) if cid in expected), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in resolved_k_values:
            if not expected:
                continue
            hits = expected & set(retrieved[:k])
            recall_sums[k] += len(hits) / len(expected)

    n = len(queries)
    return RetrievalReport(
        query_count=n,
        recall_at_k={k: recall_sums[k] / n for k in resolved_k_values},
        mrr=sum(reciprocal_ranks) / n,
    )


def load_qa_file(path: str) -> list[RetrievalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RetrievalQuery(query=item["query"], expected_chunk_ids=item["expected_chunk_ids"])
        for item in data
    ]
```

- [x] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_retrieval_metrics.py -v`
Expected: PASS（7 passed）

- [x] **Step 5: 提交**

```bash
git add app/eval/retrieval_metrics.py tests/unit/test_eval_retrieval_metrics.py
git commit -m "feat: add retrieval Recall@K/MRR eval metric"
```

---

### Task 5: 报告输出与编排（report.py + runner.py）

**Files:**
- Create: `app/eval/report.py`
- Create: `app/eval/runner.py`
- Test: `tests/unit/test_eval_report.py`
- Test: `tests/unit/test_eval_runner.py`

**Interfaces:**
- Consumes:
  - `app.eval.structure_metrics.detect_structure_violations`（Task 2）
  - `app.eval.length_metrics.compute_length_metrics`（Task 1）
  - `app.eval.coherence_metrics.compute_coherence`, `Embedder`（Task 3）
  - `app.eval.retrieval_metrics.evaluate_retrieval`, `load_qa_file`（Task 4）
  - `app.ingest.chunking.chunk_text(text, chunk_size=None, chunk_overlap=None, separators=None) -> list[str]`（已存在，见 `app/ingest/chunking.py:34`）
- Produces:
  - `write_report(report: dict, output_path: str) -> None`
  - `format_summary_table(aggregate: dict) -> str`
  - `run(input_dir: str, chunk_size: int, chunk_overlap: int, min_chars: int, max_chars: int, run_semantic: bool = True, embedder=None, qa_file: str | None = None, retrieve_fn=None) -> dict`（返回 `{"documents": [...], "aggregate": {...}}`）。Task 6（CLI）会 import 并调用 `run`、`write_report`、`format_summary_table`。

- [x] **Step 1: 写失败测试（report.py）**

```python
# tests/unit/test_eval_report.py
import json

from app.eval.report import format_summary_table, write_report


def test_write_report_writes_json_file(tmp_path):
    output_path = tmp_path / "report.json"

    write_report({"aggregate": {"chunk_count": 3}}, str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "aggregate": {"chunk_count": 3}
    }


def test_format_summary_table_renders_markdown_table_with_formatted_floats():
    table = format_summary_table({"chunk_count": 3, "truncated_region_ratio": 0.5})

    assert "| chunk_count | 3 |" in table
    assert "| truncated_region_ratio | 0.5000 |" in table
```

- [x] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_report.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.report'`

- [x] **Step 3: 写实现（report.py）**

```python
# app/eval/report.py
"""汇总报告的落盘与控制台摘要格式化。"""

import json
from pathlib import Path


def write_report(report: dict, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def format_summary_table(aggregate: dict) -> str:
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    for key, value in aggregate.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)
```

- [x] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_report.py -v`
Expected: PASS（2 passed）

- [x] **Step 5: 写失败测试（runner.py）**

```python
# tests/unit/test_eval_runner.py
import json
from dataclasses import dataclass

import pytest

from app.eval.runner import run


@dataclass
class _FakeVector:
    dense: list[float]


class _FakeEmbedder:
    def embed(self, texts: list[str]):
        return [_FakeVector([1.0, 0.0]) for _ in texts]


def _write(path, name, content):
    (path / name).write_text(content, encoding="utf-8")


def test_run_aggregates_structure_and_length_metrics_across_documents(tmp_path):
    _write(tmp_path, "a.md", "# A\n\nhello world, this is doc a.\n")
    _write(tmp_path, "b.txt", "plain text without markdown structure.\n")
    _write(tmp_path, "c.png", "not a text document")  # 应被忽略（不支持的扩展名）

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=False,
    )

    assert report["aggregate"]["document_count"] == 2
    assert report["aggregate"]["chunk_count"] == 2
    assert report["aggregate"]["mean_coherence"] is None
    assert report["aggregate"]["retrieval"] is None
    assert len(report["documents"]) == 2


def test_run_computes_semantic_coherence_when_enabled(tmp_path):
    _write(tmp_path, "a.md", "第一句。第二句。\n")

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=True,
        embedder=_FakeEmbedder(),
    )

    assert report["aggregate"]["mean_coherence"] == 1.0


def test_run_requires_embedder_when_semantic_enabled(tmp_path):
    with pytest.raises(ValueError):
        run(
            input_dir=str(tmp_path),
            chunk_size=1000,
            chunk_overlap=0,
            min_chars=0,
            max_chars=1000,
            run_semantic=True,
        )


def test_run_computes_retrieval_metrics_when_qa_file_and_retrieve_fn_given(tmp_path):
    _write(tmp_path, "a.md", "# A\n\nhello.\n")
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps([{"query": "q", "expected_chunk_ids": ["c1"]}]), encoding="utf-8"
    )

    def retrieve_fn(query, top_k):
        return ["c1"]

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=False,
        qa_file=str(qa_path),
        retrieve_fn=retrieve_fn,
    )

    assert report["aggregate"]["retrieval"]["mrr"] == 1.0
```

- [x] **Step 6: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_runner.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.eval.runner'`

- [x] **Step 7: 写实现（runner.py）**

```python
# app/eval/runner.py
"""编排：读取语料库文档 -> 用现有 chunk_text() 切片 -> 跑各项质量指标 ->
汇总成一份 JSON 兼容的报告 dict。不改动 app.ingest.chunking 里的切片实现
本身，只是拿它的输出做测量。"""

from dataclasses import asdict
from pathlib import Path

from app.eval.coherence_metrics import Embedder, compute_coherence
from app.eval.length_metrics import compute_length_metrics
from app.eval.retrieval_metrics import evaluate_retrieval, load_qa_file
from app.eval.structure_metrics import detect_structure_violations
from app.ingest.chunking import chunk_text

SUPPORTED_EXTS = {".md", ".txt"}


def _iter_documents(input_dir: str):
    for path in sorted(Path(input_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def run(
    input_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    max_chars: int,
    run_semantic: bool = True,
    embedder: Embedder | None = None,
    qa_file: str | None = None,
    retrieve_fn=None,
) -> dict:
    if run_semantic and embedder is None:
        raise ValueError("run_semantic=True 时必须提供 embedder")
    if qa_file and retrieve_fn is None:
        raise ValueError("qa_file 不为空时必须提供 retrieve_fn")

    document_reports = []
    all_chunks: list[str] = []
    total_truncated = 0
    total_regions = 0
    total_missing_heading = 0
    total_heading_expected = 0
    coherence_weighted_sum = 0.0
    coherence_weight = 0

    for path in _iter_documents(input_dir):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

        structure_report = detect_structure_violations(text, chunks, chunk_overlap)
        total_truncated += structure_report.truncated_regions
        total_regions += structure_report.total_protected_regions
        total_missing_heading += structure_report.chunks_missing_heading
        total_heading_expected += structure_report.chunks_with_heading_expected

        doc_entry = {
            "path": str(path),
            "chunk_count": len(chunks),
            "structure": asdict(structure_report),
        }

        if run_semantic and chunks:
            coherence_report = compute_coherence(chunks, embedder)
            doc_entry["coherence"] = asdict(coherence_report)
            coherence_weighted_sum += (
                coherence_report.mean_coherence * coherence_report.scored_chunk_count
            )
            coherence_weight += coherence_report.scored_chunk_count

        document_reports.append(doc_entry)

    length_report = compute_length_metrics(all_chunks, min_chars, max_chars)

    aggregate = {
        "document_count": len(document_reports),
        "chunk_count": len(all_chunks),
        "truncated_region_ratio": (total_truncated / total_regions) if total_regions else 0.0,
        "missing_heading_ratio": (
            total_missing_heading / total_heading_expected if total_heading_expected else 0.0
        ),
        **{f"length_{k}": v for k, v in asdict(length_report).items()},
    }

    if run_semantic:
        aggregate["mean_coherence"] = (
            coherence_weighted_sum / coherence_weight if coherence_weight else 0.0
        )
    else:
        aggregate["mean_coherence"] = None

    if qa_file:
        queries = load_qa_file(qa_file)
        retrieval_report = evaluate_retrieval(queries, retrieve_fn)
        aggregate["retrieval"] = asdict(retrieval_report)
    else:
        aggregate["retrieval"] = None

    return {"documents": document_reports, "aggregate": aggregate}
```

- [x] **Step 8: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_runner.py tests/unit/test_eval_report.py -v`
Expected: PASS（6 passed）

- [x] **Step 9: 提交**

```bash
git add app/eval/report.py app/eval/runner.py tests/unit/test_eval_report.py tests/unit/test_eval_runner.py
git commit -m "feat: add chunk quality report writer and eval runner orchestration"
```

---

### Task 6: CLI 入口（scripts/eval_chunk_quality.py）

**Files:**
- Create: `scripts/eval_chunk_quality.py`
- Test: `tests/unit/test_eval_cli.py`

**Interfaces:**
- Consumes: `app.eval.runner.run`（Task 5）, `app.eval.report.write_report` / `format_summary_table`（Task 5）, `app.config.settings`（已存在，`chunk_size`/`chunk_overlap` 默认值）
- Produces: `main(argv: list[str] | None = None) -> int`，供 `python -m scripts.eval_chunk_quality ...` 调用；同时是 CLI 测试的直接调用入口。

- [x] **Step 1: 写失败测试**

用 `monkeypatch` 替换 `app.eval.runner.run` 与 `app.eval.report.write_report`，只验证 CLI 的参数解析/默认值/wiring 逻辑，不触发真实 embedder/Qdrant。

```python
# tests/unit/test_eval_cli.py
import pytest

from scripts.eval_chunk_quality import main


def test_main_passes_parsed_args_to_run_and_writes_report(tmp_path, monkeypatch, capsys):
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return {"aggregate": {"chunk_count": 1}}

    written = {}

    def fake_write_report(report, output_path):
        written["report"] = report
        written["output_path"] = output_path

    monkeypatch.setattr("scripts.eval_chunk_quality.run", fake_run)
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", fake_write_report)

    output_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--input-dir",
            str(tmp_path),
            "--chunk-size",
            "500",
            "--chunk-overlap",
            "50",
            "--no-semantic",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["input_dir"] == str(tmp_path)
    assert captured_kwargs["chunk_size"] == 500
    assert captured_kwargs["chunk_overlap"] == 50
    assert captured_kwargs["run_semantic"] is False
    assert captured_kwargs["embedder"] is None
    assert captured_kwargs["qa_file"] is None
    assert captured_kwargs["retrieve_fn"] is None
    # --no-semantic 时 min/max 未显式指定，按 chunk_size 派生默认区间
    assert captured_kwargs["min_chars"] == 100  # 0.2 * 500
    assert captured_kwargs["max_chars"] == 500
    assert written["output_path"] == str(output_path)
    assert written["report"] == {"aggregate": {"chunk_count": 1}}

    out = capsys.readouterr().out
    assert "chunk_count" in out
    assert str(output_path) in out


def test_main_uses_explicit_min_max_chars_when_given(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)

    main(
        [
            "--input-dir",
            str(tmp_path),
            "--min-chars",
            "10",
            "--max-chars",
            "20",
            "--no-semantic",
        ]
    )

    assert captured_kwargs["min_chars"] == 10
    assert captured_kwargs["max_chars"] == 20


def test_main_requires_retrieve_fn_wiring_when_qa_file_given(monkeypatch, tmp_path):
    """--qa-file 传了就必须给 run() 传一个可调用的 retrieve_fn（不是 None），
    具体是否真的打了 Qdrant 由 fake run 的调用方断言，这里只验证不是 None。"""
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.eval_chunk_quality._build_retrieve_fn", lambda: (lambda q, k: [])
    )

    qa_path = tmp_path / "qa.json"
    qa_path.write_text("[]", encoding="utf-8")

    main(["--input-dir", str(tmp_path), "--no-semantic", "--qa-file", str(qa_path)])

    assert captured_kwargs["qa_file"] == str(qa_path)
    assert captured_kwargs["retrieve_fn"] is not None
```

- [x] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_eval_cli.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'scripts.eval_chunk_quality'`

- [x] **Step 3: 写实现**

```python
# scripts/eval_chunk_quality.py
"""切片质量量化评估 CLI。

用法示例：
    python -m scripts.eval_chunk_quality --input-dir data/uploads --output report.json
    python -m scripts.eval_chunk_quality --input-dir data/uploads --chunk-size 1500 --output report_1500.json

多次跑不同 --chunk-size/--chunk-overlap，对比生成的 report.json，就是最小可行的
参数消融对比。
"""

import argparse
import sys

from app.config import settings
from app.eval.report import format_summary_table, write_report
from app.eval.runner import run


def _build_retrieve_fn():
    from app.models.embeddings import get_text_embedder
    from app.store.qdrant_client import compute_chunk_id, get_default_client, search_text_hybrid

    client = get_default_client()
    embedder = get_text_embedder()

    def retrieve_fn(query: str, top_k: int) -> list[str]:
        vector = embedder.embed([query])[0]
        hits = search_text_hybrid(client, vector, top_k=top_k)
        return [
            compute_chunk_id(h.payload["doc_id"], h.payload.get("chunk_index", 0)) for h in hits
        ]

    return retrieve_fn


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切片质量量化评估工具")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--output", default="report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    min_chars = args.min_chars if args.min_chars is not None else int(args.chunk_size * 0.2)
    max_chars = args.max_chars if args.max_chars is not None else args.chunk_size

    embedder = None
    if not args.no_semantic:
        from app.models.embeddings import get_text_embedder

        embedder = get_text_embedder()

    retrieve_fn = _build_retrieve_fn() if args.qa_file else None

    report = run(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=min_chars,
        max_chars=max_chars,
        run_semantic=not args.no_semantic,
        embedder=embedder,
        qa_file=args.qa_file,
        retrieve_fn=retrieve_fn,
    )

    write_report(report, args.output)
    print(format_summary_table(report["aggregate"]))
    print(f"\n完整报告已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_eval_cli.py -v`
Expected: PASS（3 passed）

- [x] **Step 5: 跑一遍完整 eval 单测套件确认没有互相破坏**

Run: `pytest tests/unit/test_eval_*.py -v`
Expected: 全部 PASS（length 3 + structure 7 + coherence 6 + retrieval 7 + report 2 + runner 4 + cli 3 = 32 passed）

- [x] **Step 6: 提交**

```bash
git add scripts/eval_chunk_quality.py tests/unit/test_eval_cli.py
git commit -m "feat: add eval_chunk_quality CLI entry point"
```

---

## 手动验证（不属于自动化测试，供你在有 GPU/Qdrant 的真实环境跑）

Task 6 完成后，在真实环境（有 BGE-M3/Qdrant 的机器）跑一次全量：

```bash
cd retrieval-service
python -m scripts.eval_chunk_quality --input-dir data/uploads --output /tmp/report_baseline.json
```

看控制台摘要表里的 `truncated_region_ratio`、`missing_heading_ratio`、`mean_coherence`、`length_*` 是否符合预期（结合本文档开头列的四类不满意点核对）。想做参数消融，多跑几次换 `--chunk-size`/`--chunk-overlap`，对比几份 `report.json`。
