# Markdown Chunker 跨顶层章节合并修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `chunk_markdown_structured()` 里跨 section 合并逻辑的一个边界判断 bug——当文档是"唯一一个 H1 包住全文、`##` 才是真正的章"这种常见结构时，合并会无限制跨过完全不相关的顶层章节，把两章内容混进同一个 chunk 且面包屑（heading_path）丢失章节标识。

**Architecture:** 新增"章级别自动探测"（`_detect_chapter_depth`）：按 heading_path 的树深度扫描，跳过只有 1 个标题、纯粹包住全文的层级，找到第一个真正出现多个标题（有分支）的深度作为"章级别"。每个 piece 在最初产出时（合并发生前）按这个章级别打上 `chapter_id` 标签；合并时 `chapter_id` 必须显式从两侧传递（优先 earlier，earlier 为 `None` 时retreat取 later），不能在合并之后从退化的 heading_path 重新计算——否则"引言并入第一章"之后 `chapter_id` 会重新算出 `None`，绕开边界检查继续吞并第二章。合并前置检查新增一条：两个 piece 的 `chapter_id` 都不是 `None` 且不相等就拒绝合并。

**Tech Stack:** Python 3.12, pytest, markdown-it-py（现有依赖，不新增）。

## Global Constraints

- 不新增配置项：不做"可配置章级别"这个选项，章级别按文档结构自动探测（对应设计文档"范围之外"一节）。
- `app/config.py` 现有默认值不变：`chunk_size=1000`、`chunk_overlap=150`、`chunk_min_chars=200`。
- 本次改动只在合并前新增一个拒绝条件，不改变切片本身的行为——原有的"原文可连续定位"、`detect_structure_violations` 的 `truncated_regions == 0` 等不变量必须继续成立。
- 设计文档：`docs/superpowers/specs/2026-08-31-markdown-chunker-chapter-boundary-design.md`

---

## Task 1: 章级别探测 + 合并前置检查

**Files:**
- Modify: `app/ingest/markdown_chunker.py`
- Test: `tests/unit/test_markdown_chunker.py`

**Interfaces:**
- Consumes: 现有 `_build_sections(lines, tokens) -> list[tuple[str, int, int, int]]`（每项是 `(heading_path, heading_start_line, content_start_line, content_end_line)`）、现有 `NO_HEADING_PLACEHOLDER` 常量、现有 `_Piece` dataclass、现有 `_merge_two_pieces`/`_can_merge`/`_merge_small_pieces_across_sections`/`chunk_markdown_structured`。
- Produces: 新函数 `_detect_chapter_depth(sections: list[tuple[str, int, int, int]]) -> int | None`；新函数 `_chapter_component(heading_path: str, chapter_depth: int | None) -> str | None`；`_Piece` 新增字段 `chapter_id: str | None = None`（其余任务不依赖这两个新函数，只依赖 `chunk_markdown_structured()` 的最终行为变化）。

- [ ] **Step 1: 写失败的测试（4 个新用例）**

在 `tests/unit/test_markdown_chunker.py` 文件末尾（`test_real_exam_prep_doc_splits_cleanly_with_no_structure_violations` 之后）追加：

```python
def test_cross_top_level_chapter_merge_is_refused_even_with_shared_root_heading():
    # 核心 bug 复现：唯一 H1 包住全文，"二"的子节和"三"这个完全不相关的
    # 顶层章节之间，旧逻辑会因为共享 H1 这个公共前缀而错误地合并。
    text = (
        "# Root\n\n"
        "## 二\n\n"
        "### 学历\n\n" + ("x" * 50) + "\n\n"
        "## 三\n\n" + ("y" * 50) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=200)

    assert [c.heading_path for c in chunks] == ["Root > 二 > 学历", "Root > 三"]


def test_same_chapter_sibling_subsections_merge_but_not_into_next_chapter():
    # 同一章"二"下的两个子节应该照常合并（不能被新约束误伤），但合并结果
    # 不能继续吞并紧邻的"三"这个不同顶层章节。
    text = (
        "# Root\n\n"
        "## 二\n\n"
        "### 学历\n\n" + ("x" * 50) + "\n\n"
        "### 户籍\n\n" + ("y" * 50) + "\n\n"
        "## 三\n\n" + ("z" * 50) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=200)

    assert [c.heading_path for c in chunks] == ["Root > 二", "Root > 三"]


def test_no_single_wrapping_root_still_merges_within_same_top_level_chapter():
    # 没有包住全文的单一 H1（两个并列 H1 各自成章）时，章级别应该直接落在
    # 第一层，而不是继续往下找；每章内部的子节合并不受影响。
    text = (
        "# 第一章\n\n"
        "### 子节甲\n\n" + ("a" * 50) + "\n\n"
        "### 子节乙\n\n" + ("b" * 50) + "\n\n"
        "# 第二章\n\n" + ("c" * 50) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=200)

    assert [c.heading_path for c in chunks] == ["第一章", "第二章"]


def test_front_matter_merged_into_first_chapter_does_not_propagate_into_next_chapter():
    # 防止"章分量从合并后的 heading_path 重新计算"这个漏洞：文档开头的
    # 引言（没有章归属）先合法地并入第一章 A，但这一步之后不能把 chapter_id
    # 错误地重置回 None，导致继续吞并完全不相关的第二章 B。
    text = "# Root\n\nfront matter\n\n## A\n\nAAAAAAAAAA\n\n## B\n\nBBBBBBBBBB\n"
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=1000)

    assert [c.heading_path for c in chunks] == ["Root", "Root > B"]
    assert "front matter" in chunks[0].content
    assert "AAAAAAAAAA" in chunks[0].content
    assert "BBBBBBBBBB" not in chunks[0].content
    assert "BBBBBBBBBB" in chunks[1].content
```

- [ ] **Step 2: 运行测试，确认按预期失败**

Run: `python -m pytest tests/unit/test_markdown_chunker.py -k "chapter" -v`

Expected: 4 个新用例全部 FAIL（旧逻辑会把"二"和"三"、以及 "Root" 前言和 "B" 章错误合并到一起，`heading_path` 列表和断言的不一致）。

- [ ] **Step 3: 实现 `_detect_chapter_depth` 和 `_chapter_component`**

在 `app/ingest/markdown_chunker.py` 里，`_build_sections` 函数（第 102-131 行）之后插入：

```python
def _detect_chapter_depth(sections: list[tuple[str, int, int, int]]) -> int | None:
    """章级别自动探测：只跳过最多一层——包住全文的单一根标题（一般是唯一
    的 H1）。不递归继续往下找分支：如果继续跳，遇到"全文只有一个 `##`
    章、章内有多个 `###` 条款"这种单章多条的文档（比如公司章程），会把
    条款层误判成章级别，导致同一章内的条款互相之间也被当成"跨章"拒绝合
    并——这正是 markdown-chunker-small-group-merge 设计文档要覆盖的目标
    场景，不能破坏。

    用 heading_path 按 " > " 拆分后的分量下标当"树深度"，不用原始 Markdown
    的 #/## 层级数字——heading_path 只由文档里实际出现过的标题拼成，就算
    文档跳级用标题（H1 直接跳到 H3），分量下标依然正确对应嵌套深度。"""
    paths = [
        heading_path.split(" > ")
        for heading_path, *_ in sections
        if heading_path != NO_HEADING_PLACEHOLDER
    ]
    if not paths:
        return None
    root_texts = {p[0] for p in paths}
    if len(root_texts) == 1 and any(len(p) > 1 for p in paths):
        return 2
    return 1


def _chapter_component(heading_path: str, chapter_depth: int | None) -> str | None:
    """取 heading_path 在章级别那一层的分量，作为这个 piece 的章归属标识。
    还没到章这一层（比如文档开头、标题自己的引言段）或者整份文档没有章级
    别时返回 None——None 不代表"属于某个特殊的章"，而是"还没归到任何章"，
    在 `_can_merge` 里和任何章都视为不冲突。"""
    if chapter_depth is None or heading_path == NO_HEADING_PLACEHOLDER:
        return None
    parts = heading_path.split(" > ")
    return parts[chapter_depth - 1] if len(parts) >= chapter_depth else None
```

- [ ] **Step 4: `_Piece` 新增 `chapter_id` 字段**

修改第 50-65 行的 `_Piece` dataclass，在 `content_start_line` 字段之后加一行：

```python
@dataclass
class _Piece:
    heading_path: str
    content: str
    has_table: bool
    has_code: bool
    line_span: tuple[int, int] | None = None
    content_start_line: int | None = None
    # 这个 piece 所属的"章"标识：section 最初产出 piece 时按章级别算一次
    # （见 `_chapter_component`），合并时显式从两侧传递（见
    # `_merge_two_pieces`），不能在合并之后重新从退化的 heading_path 计算
    # ——否则"引言并入第一章"之后会重新算出 None，绕开 `_can_merge` 的边
    # 界检查、继续传递合并进下一章。None 表示"还没归到任何章"（比如文档
    # 开头的引言），不是"属于某个特殊的章"。
    chapter_id: str | None = None
```

- [ ] **Step 5: `_merge_two_pieces` 传递 `chapter_id`**

修改第 215-253 行的 `_merge_two_pieces` 函数，在 `return _Piece(...)` 里加一行：

```python
    return _Piece(
        heading_path=merged_path,
        content=content,
        has_table=has_table,
        has_code=has_code,
        line_span=(start, end),
        content_start_line=start,
        chapter_id=earlier.chapter_id if earlier.chapter_id is not None else later.chapter_id,
    )
```

- [ ] **Step 6: `_can_merge` 新增章边界检查**

修改第 256-261 行的 `_can_merge` 函数：

```python
def _can_merge(earlier: _Piece, later: _Piece, lines: list[str], chunk_size: int) -> bool:
    if earlier.line_span is None or later.line_span is None:
        return False
    if not _common_heading_path(earlier.heading_path, later.heading_path):
        return False
    if (
        earlier.chapter_id is not None
        and later.chapter_id is not None
        and earlier.chapter_id != later.chapter_id
    ):
        return False
    return len(_merge_two_pieces(earlier, later, lines).content) <= chunk_size
```

- [ ] **Step 7: `chunk_markdown_structured` 里算 `chapter_depth` 并赋给每个 piece**

修改第 365-447 行的 `chunk_markdown_structured` 函数。在

```python
    sections = _build_sections(lines, tokens)
    all_blocks = _top_level_blocks(tokens)
```

之后加一行：

```python
    sections = _build_sections(lines, tokens)
    all_blocks = _top_level_blocks(tokens)
    chapter_depth = _detect_chapter_depth(sections)
```

然后在 `for heading_path, heading_line, start_line, end_line in sections:` 循环体内、`section_blocks = _blocks_within(...)` 之后加一行：

```python
        section_blocks = _blocks_within(all_blocks, start_line, end_line)
        section_chapter_id = _chapter_component(heading_path, chapter_depth)
```

最后把循环末尾构造 `_Piece` 的地方（原第 440-443 行）：

```python
        for content, has_table, has_code in pieces:
            all_pieces.append(
                _Piece(heading_path, content, has_table, has_code, span, body_start)
            )
```

改成：

```python
        for content, has_table, has_code in pieces:
            all_pieces.append(
                _Piece(
                    heading_path, content, has_table, has_code, span, body_start,
                    section_chapter_id,
                )
            )
```

- [ ] **Step 8: 运行新用例，确认全部通过**

Run: `python -m pytest tests/unit/test_markdown_chunker.py -k "chapter" -v`

Expected: 4 个用例全部 PASS。

- [ ] **Step 9: 更新两个断言了旧 bug 行为的既有用例**

跑一遍整个测试文件会发现 `test_hierarchy_doc_merges_into_common_ancestor_chunks_by_default`
和 `test_repeated_heading_path_after_merging_gets_numeric_suffix` 这两个既
有用例失败——不是这次改动破坏了它们，是它们的断言本身就在期待"跨顶层
章节合并"这个 bug 的行为（前者期待不同章节的 `Section A`/`Section B` 合
并成一个 chunk；后者期待不同章节的 `C`/`D` 跨章合并撞名来验证 `#n` 编
号）。两个都需要按修复后的正确行为改写。

把 `test_hierarchy_doc_merges_into_common_ancestor_chunks_by_default`
（第 73-85 行）整个替换成：

```python
def test_hierarchy_doc_merges_within_chapter_but_not_across_chapters():
    # 默认配置下 Title 自己的引言、Section A、Sub A1 都远小于 chunk_min_
    # chars，合并成一片，面包屑退到共同祖先 Title；但 Section B 是另一个
    # 顶层章节，不会被并进来——这正是本次修复要保证的边界。开头那段无标
    # 题正文和 Title 没有共同祖先，保持独立。
    chunks = chunk_markdown_structured(HIERARCHY_MD)

    assert [c.heading_path for c in chunks] == [
        NO_HEADING_PLACEHOLDER,
        "Title",
        "Title > Section B",
    ]
    merged = chunks[1]
    assert merged.content in HIERARCHY_MD
    for heading in ("## Section A", "### Sub A1"):
        assert heading in merged.content
    assert "## Section B" not in merged.content
    assert merged.has_table is False
    assert merged.has_code_block is False

    section_b = chunks[2]
    assert section_b.has_table is True
    assert section_b.has_code_block is True
```

把 `test_repeated_heading_path_after_merging_gets_numeric_suffix`（原第
368-381 行）整个替换成：

```python
def test_repeated_heading_path_after_merging_gets_numeric_suffix():
    # 两个"合并孤岛"都在同一章（Chapter）内、被中间偏大的 Big 隔开，各自
    # 独立合并后面包屑都退到同一个章级祖先 "R > Chapter"，产生撞名——沿
    # 用 section 内拆分时就在用的 #n 约定把它们区分开。两个孤岛都没有跨
    # 过 Chapter 这个顶层章节边界，符合本次修复的约束。
    text = (
        "# R\n\n"
        "## Chapter\n\n"
        "### Sub1\n\n" + ("a" * 20) + "\n\n"
        "### Sub2\n\n" + ("b" * 20) + "\n\n"
        "### Big\n\n" + ("c" * 80) + "\n\n"
        "### Sub3\n\n" + ("d" * 20) + "\n\n"
        "### Sub4\n\n" + ("e" * 20) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=100, chunk_overlap=0, min_chars=60)

    assert [c.heading_path for c in chunks] == [
        "R > Chapter",
        "R > Chapter > Big",
        "R > Chapter #2",
    ]
```

- [ ] **Step 10: 跑整个测试文件，确认全部通过**

Run: `python -m pytest tests/unit/test_markdown_chunker.py -v`

Expected: 全部 PASS。

- [ ] **Step 11: Commit**

```bash
git add app/ingest/markdown_chunker.py tests/unit/test_markdown_chunker.py
git commit -m "$(cat <<'EOF'
fix: markdown chunker 跨 section 合并新增章边界检查

单 H1 包住全文的文档结构下，"公共前缀非空即允许合并"形同虚设，
导致合并会跨过完全不相关的顶层章节、面包屑丢失章节标识。新增
章级别自动探测 + chapter_id 显式传递（不从合并后的 heading_path
重新计算，避免绕开检查）两道约束堵住这个漏洞。

设计文档：docs/superpowers/specs/2026-08-31-markdown-chunker-chapter-boundary-design.md
EOF
)"
```

---

## Task 2: 真实文档回归测试

**Files:**
- Create: `tests/unit/fixtures/广东省考公务员.md`
- Test: `tests/unit/test_markdown_chunker.py`

**Interfaces:**
- Consumes: Task 1 完成后的 `chunk_markdown_structured()`（行为已经修复，本任务不改动实现代码）。
- Produces: 无新公开接口，只新增一个 fixture 文件和一个测试函数，供后续回归检测使用。

**说明**：这个任务不是"先写失败测试、再实现让它通过"的标准 TDD 流程——Task 1 已经把通用修复实现并提交了，这里只是用真实语料（brainstorming 阶段发现问题的那份文档）补一个端到端回归断言，防止以后有人改动合并逻辑时又引入同类 bug。所以下面第 2 步预期直接 PASS，不是先 FAIL。

- [ ] **Step 1: 复制真实文档到测试 fixture 目录**

```bash
mkdir -p tests/unit/fixtures
cp "data/uploads/upload_2026-08-28T12_52_53.367843+00_00_广东省考公务员.md/广东省考公务员.md" "tests/unit/fixtures/广东省考公务员.md"
```

- [ ] **Step 2: 加 `import re` 和新测试**

在 `tests/unit/test_markdown_chunker.py` 顶部的 import 区（第 1-4 行）加一行：

```python
import re
from pathlib import Path

from app.eval.structure_metrics import detect_structure_violations
from app.ingest.markdown_chunker import NO_HEADING_PLACEHOLDER, chunk_markdown_structured
```

在文件末尾追加：

```python
REAL_GDSK_DOC = Path(__file__).resolve().parent / "fixtures" / "广东省考公务员.md"


def test_real_gdsk_doc_never_mixes_two_top_level_chapters_in_one_chunk():
    # 回归这次 brainstorming 发现的具体 bug：修复前，这份文档会产出把
    # "## 一、公告发布规律"整节内容混进标题自己那个 chunk、以及把
    # "### 学历与户籍"（属于"二"）和"## 三、报名规则"（完全独立的一章）
    # 混进同一个 chunk 这两种跨章合并。用"任意一个 chunk 的正文里最多只
    # 能出现一条 `## ` 顶层章节标题行"这条不变量兜底，不依赖具体章节数。
    text = REAL_GDSK_DOC.read_text(encoding="utf-8")
    chapter_heading_lines = re.findall(r"^## .+$", text, re.MULTILINE)
    assert len(chapter_heading_lines) >= 2  # 前提检查：这份文档确实有多章

    chunks = chunk_markdown_structured(text)
    assert len(chunks) > 0

    for chunk in chunks:
        present = [line for line in chapter_heading_lines if line in chunk.content]
        assert len(present) <= 1, (chunk.heading_path, present)
```

- [ ] **Step 3: 运行新测试，确认通过**

Run: `python -m pytest tests/unit/test_markdown_chunker.py -k "gdsk" -v`

Expected: PASS。（如果在 Task 1 提交之前跑这个断言会 FAIL——旧版本会产出至少两个同时包含两条不同 `## ` 标题行的 chunk，可以用 `git stash` 临时验证，不是必须步骤。）

- [ ] **Step 4: 跑整个测试文件，确认全部通过**

Run: `python -m pytest tests/unit/test_markdown_chunker.py -v`

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/unit/fixtures/广东省考公务员.md tests/unit/test_markdown_chunker.py
git commit -m "$(cat <<'EOF'
test: 用真实文档回归跨顶层章节合并 bug

补一个端到端断言：任意 chunk 正文里最多只能出现一条顶层章节
标题行，防止以后改动合并逻辑重新引入同类 bug。

EOF
)"
```
