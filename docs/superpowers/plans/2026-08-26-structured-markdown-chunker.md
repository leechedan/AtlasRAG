# 结构化 Markdown 切片器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立的、基于标题层级 AST 的结构化 Markdown 切片函数 `chunk_markdown_structured()`，保证表格/代码块不被跨 chunk 截断，chunk 携带 `heading_path` 面包屑元数据，并接入现有切片质量评估工具做新旧方案 A/B 对比。

**Architecture:** 新模块 `app/ingest/markdown_chunker.py` 用 `markdown-it-py` 解析文档，单次遍历标题 token 维护一个"标题栈"得到每个标题的 `heading_path` 和"直属内容"行区间；直属内容按块级 token（段落/表格/代码块/列表/引用块）打包进 `chunk_size` 预算，表格/代码块作为不可拆原子单位，超限的普通段落回退到现有 `chunk_text()`。不改动 `app/ingest/pipeline.py`/Qdrant/API，只接入 `app/eval/runner.py` 和 `scripts/eval_chunk_quality.py` 做对比评估。

**Tech Stack:** Python, `markdown-it-py`（已是项目依赖，`app/eval/structure_metrics.py` 已在用），pytest。

## Global Constraints

- 只处理 Markdown；`content` 字段是原文逐字提取，不做任何改写/摘要（spec 决定）。
- 不改 `app/ingest/pipeline.py`、Qdrant 存储 schema（`TextChunkRecord`）、API schema（`ChunkOut`）、Admin UI（spec 范围声明）。
- `token_estimate = round(len(content) / 1.8)`，启发式估算，不接真实 tokenizer（spec 决定）。
- 二次拆分（同一标题内部按块打包产生多个 chunk）之间、以及不同标题的 chunk 之间都不做字符 overlap；`chunk_overlap` 只在回退到 `chunk_text()` 的场景生效（spec 决定）。
- 二次拆分的 `heading_path` 用纯序号后缀 `" #2"`、`" #3"`……第一个 chunk 不加后缀（spec 决定，用户明确要求序号而非描述性文字）。
- 设计文档：`docs/superpowers/specs/2026-08-26-structured-markdown-chunker-design.md`。

---

### Task 1: `chunk_markdown_structured()` 核心实现

**Files:**
- Create: `app/ingest/markdown_chunker.py`
- Test: `tests/unit/test_markdown_chunker.py`

**Interfaces:**
- Consumes: `app.config.settings.chunk_size`（默认 1000）、`settings.chunk_overlap`（默认 150）；`app.ingest.chunking.chunk_text(text, chunk_size, chunk_overlap)`（既有函数，签名不变）。
- Produces:
  - `NO_HEADING_PLACEHOLDER: str = "(文档开头)"`
  - `@dataclass StructuredChunk`：字段 `chunk_id: int`、`heading_path: str`、`has_table: bool`、`has_code_block: bool`、`char_count: int`、`token_estimate: int`、`content: str`。
  - `chunk_markdown_structured(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[StructuredChunk]`
  - 后续任务只依赖以上三个导出名字。

- [ ] **Step 1: 写测试文件（一次性写完整套用例）**

创建 `tests/unit/test_markdown_chunker.py`：

```python
from app.eval.structure_metrics import detect_structure_violations
from app.ingest.markdown_chunker import NO_HEADING_PLACEHOLDER, chunk_markdown_structured

HIERARCHY_MD = (
    "leading text before any heading\n"
    "\n"
    "# Title\n"
    "\n"
    "intro under title\n"
    "\n"
    "## Section A\n"
    "\n"
    "content of A\n"
    "\n"
    "### Sub A1\n"
    "\n"
    "content of sub a1\n"
    "\n"
    "## Section B\n"
    "\n"
    "| a | b |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "```python\n"
    "print('hi')\n"
    "```\n"
)


def test_leading_text_before_first_heading_becomes_placeholder_chunk():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    assert chunks[0].heading_path == NO_HEADING_PLACEHOLDER
    assert chunks[0].content == "leading text before any heading"


def test_heading_own_content_uses_heading_text_as_path():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    by_path = {c.heading_path: c for c in chunks}
    assert by_path["Title"].content == "intro under title"


def test_breadcrumb_joins_ancestor_headings_with_arrow():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    by_path = {c.heading_path: c for c in chunks}
    assert by_path["Title > Section A"].content == "content of A"
    assert by_path["Title > Section A > Sub A1"].content == "content of sub a1"


def test_sibling_heading_replaces_previous_sibling_in_breadcrumb():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    paths = [c.heading_path for c in chunks]
    assert "Title > Section B" in paths
    assert "Title > Section A > Section B" not in paths


def test_table_and_code_block_flags_set_and_content_preserved_verbatim():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    by_path = {c.heading_path: c for c in chunks}
    section_b = by_path["Title > Section B"]
    assert section_b.has_table is True
    assert section_b.has_code_block is True
    assert "| a | b |" in section_b.content
    assert "```python" in section_b.content


def test_char_count_and_token_estimate_match_content():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    for c in chunks:
        assert c.char_count == len(c.content)
        assert c.token_estimate == round(len(c.content) / 1.8)


def test_chunk_ids_are_sequential_from_zero():
    chunks = chunk_markdown_structured(HIERARCHY_MD)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


NO_OWN_CONTENT_MD = "## Parent\n### Child\n\ncontent under child\n"


def test_heading_with_no_direct_content_produces_no_chunk_for_itself():
    chunks = chunk_markdown_structured(NO_OWN_CONTENT_MD)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Parent > Child"
    assert chunks[0].content == "content under child"


def test_empty_text_returns_empty_list():
    assert chunk_markdown_structured("") == []
    assert chunk_markdown_structured("   \n\n  ") == []


def test_no_heading_document_uses_placeholder_and_keeps_whole_text():
    text = "just a plain paragraph with no headings at all.\n"
    chunks = chunk_markdown_structured(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == NO_HEADING_PLACEHOLDER
    assert chunks[0].content == "just a plain paragraph with no headings at all."


def _build_long_section(paragraph_count: int) -> str:
    paragraphs = [f"paragraph number {i} unique marker" for i in range(paragraph_count)]
    return "## Section\n\n" + "\n\n".join(paragraphs) + "\n"


def test_long_section_splits_into_multiple_chunks_with_numeric_suffix():
    text = _build_long_section(20)
    chunks = chunk_markdown_structured(text, chunk_size=100, chunk_overlap=20)

    section_chunks = [c for c in chunks if c.heading_path.startswith("Section")]
    assert len(section_chunks) > 1
    assert section_chunks[0].heading_path == "Section"
    assert section_chunks[1].heading_path == "Section #2"


def test_long_section_split_has_no_overlap_and_no_lost_content():
    text = _build_long_section(20)
    chunks = chunk_markdown_structured(text, chunk_size=100, chunk_overlap=20)

    for i in range(20):
        marker = f"paragraph number {i} unique marker"
        occurrences = sum(c.content.count(marker) for c in chunks)
        assert occurrences == 1


def test_table_never_split_across_two_chunks_even_in_long_section():
    text = (
        "## Big Section\n"
        "\n"
        + ("word " * 200) + "\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=50, chunk_overlap=0)

    table_text = "| a | b |\n|---|---|\n| 1 | 2 |"
    code_text = "```python\nprint('hi')\n```"
    table_hits = [c for c in chunks if table_text in c.content]
    code_hits = [c for c in chunks if code_text in c.content]
    assert len(table_hits) == 1
    assert len(code_hits) == 1
    assert table_hits[0].has_table is True
    assert code_hits[0].has_code_block is True


def test_oversized_protected_table_stays_whole_even_over_chunk_size():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(100))
    text = "## T\n\n| a | b |\n|---|---|\n" + rows + "\n"
    chunks = chunk_markdown_structured(text, chunk_size=50, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].has_table is True
    assert chunks[0].char_count > 50


def test_single_oversized_plain_paragraph_falls_back_to_char_level_split():
    text = "## Section\n\n" + ("x" * 3000) + "\n"
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(c.char_count <= 1000 for c in chunks)
    assert sum(c.content.count("x") for c in chunks) == 3000


def test_structured_chunks_never_truncate_protected_regions_per_eval_metrics():
    chunks = chunk_markdown_structured(HIERARCHY_MD, chunk_size=1000, chunk_overlap=0)
    report = detect_structure_violations(
        HIERARCHY_MD, [c.content for c in chunks], chunk_overlap=0
    )
    assert report.truncated_regions == 0
```

- [ ] **Step 2: 跑测试确认失败（模块还不存在）**

Run: `pytest tests/unit/test_markdown_chunker.py -v`
Expected: `ModuleNotFoundError: No module named 'app.ingest.markdown_chunker'`

- [ ] **Step 3: 实现核心模块**

创建 `app/ingest/markdown_chunker.py`：

```python
"""结构化 Markdown 切片器：按标题层级 AST 切分，保证表格/代码块不被跨
chunk 截断，每个 chunk 携带标题面包屑路径（heading_path）作为元数据。
内容是原文逐字提取，不做任何改写/摘要。

设计文档：docs/superpowers/specs/2026-08-26-structured-markdown-chunker-design.md
"""

from dataclasses import dataclass

from markdown_it import MarkdownIt

from app.config import settings
from app.ingest.chunking import chunk_text

_MD = MarkdownIt("commonmark").enable("table")

_BLOCK_TOKEN_TYPES = {
    "paragraph_open",
    "table_open",
    "fence",
    "bullet_list_open",
    "ordered_list_open",
    "blockquote_open",
    "hr",
    "html_block",
}

NO_HEADING_PLACEHOLDER = "(文档开头)"


@dataclass
class StructuredChunk:
    chunk_id: int
    heading_path: str
    has_table: bool
    has_code_block: bool
    char_count: int
    token_estimate: int
    content: str


@dataclass
class _Block:
    start_line: int
    end_line: int  # 不含
    kind: str | None  # "table" | "code" | None


def _heading_text(tokens: list, idx: int) -> str:
    if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
        return tokens[idx + 1].content.strip()
    return ""


def _headings(tokens: list) -> list[tuple[int, str, int, int]]:
    """按文档顺序返回 (level, text, start_line, content_start_line)。
    content_start_line 是标题自身那一行结束的位置（下一行开始才是正文）。"""
    result = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.map is not None:
            level = int(token.tag[1:])
            result.append((level, _heading_text(tokens, i), token.map[0], token.map[1]))
    return result


def _top_level_blocks(tokens: list) -> list[_Block]:
    blocks = []
    for token in tokens:
        if token.level != 0 or token.map is None or token.type == "heading_open":
            continue
        if token.type not in _BLOCK_TOKEN_TYPES:
            continue
        if token.type == "table_open":
            kind = "table"
        elif token.type == "fence":
            kind = "code"
        else:
            kind = None
        blocks.append(_Block(token.map[0], token.map[1], kind))
    return blocks


def _build_sections(lines: list[str], tokens: list) -> list[tuple[str, int, int]]:
    """返回 [(heading_path, content_start_line, content_end_line), ...]，
    end 不含。没有标题时返回单个 (NO_HEADING_PLACEHOLDER, 0, len(lines))。"""
    headings = _headings(tokens)
    n_lines = len(lines)

    if not headings:
        return [(NO_HEADING_PLACEHOLDER, 0, n_lines)]

    sections: list[tuple[str, int, int]] = []
    _, _, first_heading_start, _ = headings[0]
    if first_heading_start > 0:
        sections.append((NO_HEADING_PLACEHOLDER, 0, first_heading_start))

    stack: list[tuple[int, str]] = []
    for i, (level, htext, _start, content_start) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, htext))
        heading_path = " > ".join(t for _, t in stack)
        content_end = headings[i + 1][2] if i + 1 < len(headings) else n_lines
        sections.append((heading_path, content_start, content_end))

    return sections


def _blocks_within(blocks: list[_Block], start_line: int, end_line: int) -> list[_Block]:
    return [b for b in blocks if b.start_line >= start_line and b.end_line <= end_line]


def _pack_blocks(blocks: list[_Block], lines: list[str], chunk_size: int) -> list[list[_Block]]:
    """把顺序排列的块打包进尽量接近 chunk_size 的分组，块内部绝不拆开。"""
    groups: list[list[_Block]] = []
    current: list[_Block] = []
    current_len = 0

    for block in blocks:
        block_len = sum(len(lines[i]) for i in range(block.start_line, block.end_line))
        if current and current_len + block_len > chunk_size:
            groups.append(current)
            current = []
            current_len = 0
        current.append(block)
        current_len += block_len

    if current:
        groups.append(current)

    return groups


def _group_text(group: list[_Block], lines: list[str]) -> str:
    return "".join(lines[group[0].start_line : group[-1].end_line]).strip()


def chunk_markdown_structured(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[StructuredChunk]:
    if not text or not text.strip():
        return []

    chunk_size_val = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap_val = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    tokens = _MD.parse(text)
    lines = text.splitlines(keepends=True)
    sections = _build_sections(lines, tokens)
    all_blocks = _top_level_blocks(tokens)

    chunks: list[StructuredChunk] = []

    for heading_path, start_line, end_line in sections:
        section_text = "".join(lines[start_line:end_line]).strip()
        if not section_text:
            continue

        section_blocks = _blocks_within(all_blocks, start_line, end_line)

        pieces: list[tuple[str, bool, bool]] = []
        if not section_blocks:
            pieces.append((section_text, False, False))
        else:
            for group in _pack_blocks(section_blocks, lines, chunk_size_val):
                group_text = _group_text(group, lines)
                has_table = any(b.kind == "table" for b in group)
                has_code = any(b.kind == "code" for b in group)
                oversized_plain = (
                    len(group) == 1
                    and group[0].kind is None
                    and len(group_text) > chunk_size_val
                )
                if oversized_plain:
                    for sub in chunk_text(
                        group_text, chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val
                    ):
                        pieces.append((sub, False, False))
                else:
                    pieces.append((group_text, has_table, has_code))

        for i, (content, has_table, has_code) in enumerate(pieces):
            path = heading_path if i == 0 else f"{heading_path} #{i + 1}"
            chunks.append(
                StructuredChunk(
                    chunk_id=len(chunks),
                    heading_path=path,
                    has_table=has_table,
                    has_code_block=has_code,
                    char_count=len(content),
                    token_estimate=round(len(content) / 1.8),
                    content=content,
                )
            )

    return chunks
```

- [ ] **Step 4: 跑测试确认全部通过**

Run: `pytest tests/unit/test_markdown_chunker.py -v`
Expected: 全部 PASS（17 个用例）

- [ ] **Step 5: Commit**

```bash
git add app/ingest/markdown_chunker.py tests/unit/test_markdown_chunker.py
git commit -m "$(cat <<'EOF'
feat: add structured markdown chunker

Heading-AST-aware chunker that never truncates tables/code blocks
mid-element and tags each chunk with a heading_path breadcrumb. Content
is extracted verbatim from source line ranges, no rewriting.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 用真实文档做集成验证

**Files:**
- Modify: `tests/unit/test_markdown_chunker.py`

**Interfaces:**
- Consumes: `chunk_markdown_structured`（Task 1）、`app.eval.structure_metrics.detect_structure_violations`（既有）。
- Produces: 无新导出，仅新增测试。

- [ ] **Step 1: 追加真实文档测试**

在 `tests/unit/test_markdown_chunker.py` 末尾追加：

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DOC = (
    REPO_ROOT
    / "data"
    / "uploads"
    / "upload_2026-08-25T12_57_30.283214+00_00_备考总计划TODO.md"
    / "备考总计划TODO.md"
)


def test_real_exam_prep_doc_splits_cleanly_with_no_structure_violations():
    text = REAL_DOC.read_text(encoding="utf-8")

    chunks = chunk_markdown_structured(text)

    assert len(chunks) >= 10
    assert any(c.has_table for c in chunks)
    assert any(c.has_code_block for c in chunks)

    report = detect_structure_violations(text, [c.content for c in chunks], chunk_overlap=0)
    assert report.truncated_regions == 0

    total_content_chars = sum(c.char_count for c in chunks)
    assert total_content_chars > 0
    assert total_content_chars <= len(text)
```

- [ ] **Step 2: 跑测试确认通过**

Run: `pytest tests/unit/test_markdown_chunker.py -v -k real_exam_prep_doc`
Expected: PASS

如果失败，先用下面这段临时脚本打印实际切出来的 `heading_path` 列表，人工核对是不是漏切/多切了某个标题（不要盲目改断言凑过）：

```bash
python -c "
from pathlib import Path
from app.ingest.markdown_chunker import chunk_markdown_structured
text = Path('data/uploads/upload_2026-08-25T12_57_30.283214+00_00_备考总计划TODO.md/备考总计划TODO.md').read_text(encoding='utf-8')
for c in chunk_markdown_structured(text):
    print(c.chunk_id, c.heading_path, c.has_table, c.has_code_block, c.char_count)
"
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_markdown_chunker.py
git commit -m "$(cat <<'EOF'
test: verify structured chunker against real exam-prep doc

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 接入 `app/eval/runner.py`

**Files:**
- Modify: `app/eval/runner.py`
- Test: `tests/unit/test_eval_runner.py`

**Interfaces:**
- Consumes: `app.ingest.markdown_chunker.chunk_markdown_structured`（Task 1）。
- Produces: `run(..., chunker: str = "char")` — 新增关键字参数，`"char"`（默认，行为不变）或 `"structured"`；非法值抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_eval_runner.py` 末尾追加：

```python
def test_run_uses_structured_chunker_when_selected(tmp_path):
    _write(
        tmp_path,
        "a.md",
        "## Section\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\ntext after table\n",
    )

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=False,
        chunker="structured",
    )

    assert report["aggregate"]["document_count"] == 1
    assert report["aggregate"]["chunk_count"] == 1
    assert report["aggregate"]["truncated_region_ratio"] == 0.0


def test_run_rejects_unknown_chunker():
    with pytest.raises(ValueError):
        run(
            input_dir=".",
            chunk_size=1000,
            chunk_overlap=0,
            min_chars=0,
            max_chars=1000,
            run_semantic=False,
            chunker="not-a-real-chunker",
        )
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_eval_runner.py -v -k "structured_chunker or rejects_unknown"`
Expected: FAIL（`run() got an unexpected keyword argument 'chunker'`）

- [ ] **Step 3: 修改 `app/eval/runner.py`**

在文件顶部 import 区加一行（紧跟 `from app.ingest.chunking import chunk_text` 之后）：

```python
from app.ingest.markdown_chunker import chunk_markdown_structured
```

把 `run()` 的签名和函数体开头改成：

```python
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
    chunker: str = "char",
) -> dict:
    if chunker not in ("char", "structured"):
        raise ValueError(f"未知 chunker: {chunker!r}，可选 'char' 或 'structured'")
    if run_semantic and embedder is None:
        raise ValueError("run_semantic=True 时必须提供 embedder")
    if qa_file and retrieve_fn is None:
        raise ValueError("qa_file 不为空时必须提供 retrieve_fn")
```

把读取+切片那一行：

```python
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
```

改成：

```python
        text = path.read_text(encoding="utf-8")
        if chunker == "structured":
            chunks = [
                c.content
                for c in chunk_markdown_structured(
                    text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            ]
        else:
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
```

其余逻辑不变。

- [ ] **Step 4: 跑测试确认通过（含既有回归）**

Run: `pytest tests/unit/test_eval_runner.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/eval/runner.py tests/unit/test_eval_runner.py
git commit -m "$(cat <<'EOF'
feat: add chunker selector to eval runner for structured vs char A/B

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 接入 `scripts/eval_chunk_quality.py` CLI

**Files:**
- Modify: `scripts/eval_chunk_quality.py`
- Test: `tests/unit/test_eval_cli.py`

**Interfaces:**
- Consumes: `run(..., chunker=...)`（Task 3）。
- Produces: `--chunker {char,structured}` CLI 参数，默认 `char`。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_eval_cli.py` 末尾追加：

```python
def test_main_passes_chunker_arg_to_run(monkeypatch, tmp_path):
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
            "--no-semantic",
            "--chunker",
            "structured",
        ]
    )

    assert captured_kwargs["chunker"] == "structured"


def test_main_defaults_chunker_to_char(monkeypatch, tmp_path):
    captured_kwargs = {}
    monkeypatch.setattr(
        "scripts.eval_chunk_quality.run",
        lambda **kwargs: captured_kwargs.update(kwargs) or {"aggregate": {}},
    )
    monkeypatch.setattr("scripts.eval_chunk_quality.write_report", lambda *a, **k: None)

    main(["--input-dir", str(tmp_path), "--no-semantic"])

    assert captured_kwargs["chunker"] == "char"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_eval_cli.py -v -k chunker`
Expected: FAIL（`run()` 收到意外的 `chunker` 参数 / argparse 报 unrecognized arguments `--chunker`）

- [ ] **Step 3: 修改 `scripts/eval_chunk_quality.py`**

在 `_parse_args` 里，`parser.add_argument("--no-semantic", ...)` 之后加一行：

```python
    parser.add_argument("--chunker", choices=["char", "structured"], default="char")
```

在 `main()` 里调用 `run(...)` 的关键字参数列表末尾加一行：

```python
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
        chunker=args.chunker,
    )
```

- [ ] **Step 4: 跑测试确认通过（含既有回归）**

Run: `pytest tests/unit/test_eval_cli.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_chunk_quality.py tests/unit/test_eval_cli.py
git commit -m "$(cat <<'EOF'
feat: add --chunker CLI flag to eval_chunk_quality

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 全量回归 + 真实数据 A/B 对比验证

**Files:** 无代码改动，仅验证。

- [ ] **Step 1: 跑全量单测**

Run: `pytest tests/unit -v`
Expected: 全部 PASS，无回归

- [ ] **Step 2: 对 `data/uploads` 跑新旧两种切片器做对比**

Run:
```bash
python -m scripts.eval_chunk_quality --input-dir data/uploads --chunker char --no-semantic --output /tmp/report_char.json
python -m scripts.eval_chunk_quality --input-dir data/uploads --chunker structured --no-semantic --output /tmp/report_structured.json
```

Expected: 两次都成功跑完并打印汇总表；对比两份报告里的 `truncated_region_ratio` —— `structured` 应为 `0.0`（或明显低于 `char`），验证结构化切片器确实解决了表格/代码块被截断的问题。这一步是人工验证，不写断言，把两份报告的 `truncated_region_ratio`/`missing_heading_ratio`/`chunk_count` 对比结果贴给用户看。

- [ ] **Step 3（可选）: 如果对比结果不理想，回到 Task 1 调整 `_pack_blocks`/`chunk_size` 假设，不要在这一步硬改评估脚本掩盖问题**

## 验收标准

- `pytest tests/unit -v` 全绿。
- `chunk_markdown_structured()` 对"备考总计划TODO.md"的切分：有表格的 chunk 和有代码块的 chunk 都至少存在一个；`detect_structure_violations` 报告 0 违规。
- `scripts/eval_chunk_quality.py --chunker structured` 可正常跑完并出报告，`truncated_region_ratio` 为 0 或明显优于 `--chunker char`。
