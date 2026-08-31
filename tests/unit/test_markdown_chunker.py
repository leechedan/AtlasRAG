import re
from pathlib import Path

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


# HIERARCHY_MD 每一节都远小于默认 chunk_min_chars，默认配置下会被跨
# section 合并掉。下面这几个用例针对的是标题层级/面包屑本身，用
# min_chars=0 关掉合并来隔离被测行为；默认配置下的合并结果另见
# test_hierarchy_doc_merges_into_common_ancestor_chunks_by_default。
def test_heading_own_content_uses_heading_text_as_path():
    chunks = chunk_markdown_structured(HIERARCHY_MD, min_chars=0)
    by_path = {c.heading_path: c for c in chunks}
    assert by_path["Title"].content == "intro under title"


def test_breadcrumb_joins_ancestor_headings_with_arrow():
    chunks = chunk_markdown_structured(HIERARCHY_MD, min_chars=0)
    by_path = {c.heading_path: c for c in chunks}
    assert by_path["Title > Section A"].content == "content of A"
    assert by_path["Title > Section A > Sub A1"].content == "content of sub a1"


def test_sibling_heading_replaces_previous_sibling_in_breadcrumb():
    chunks = chunk_markdown_structured(HIERARCHY_MD, min_chars=0)
    paths = [c.heading_path for c in chunks]
    assert "Title > Section B" in paths
    assert "Title > Section A > Section B" not in paths


def test_table_and_code_block_flags_set_and_content_preserved_verbatim():
    chunks = chunk_markdown_structured(HIERARCHY_MD, min_chars=0)
    by_path = {c.heading_path: c for c in chunks}
    section_b = by_path["Title > Section B"]
    assert section_b.has_table is True
    assert section_b.has_code_block is True
    assert "| a | b |" in section_b.content
    assert "```python" in section_b.content


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


def test_long_section_split_has_overlap_and_no_lost_content():
    text = _build_long_section(20)
    chunk_overlap = 20
    chunks = chunk_markdown_structured(text, chunk_size=100, chunk_overlap=chunk_overlap)

    # 每个完整段落只应该完整出现一次——overlap 只借用上一个分片结尾的
    # 一小段字符做上下文衔接，不是把整段内容复制一遍。
    for i in range(20):
        marker = f"paragraph number {i} unique marker"
        occurrences = sum(c.content.count(marker) for c in chunks)
        assert occurrences == 1

    # 但相邻分片之间应该共享一小段上下文（overlap 实际生效），而不是在
    # chunk_size 硬切处完全断裂、互相没有关联性。
    section_chunks = [c for c in chunks if c.heading_path.startswith("Section")]
    assert len(section_chunks) > 1
    for prev, nxt in zip(section_chunks, section_chunks[1:]):
        prev_tail = prev.content[-10:].strip()
        assert prev_tail
        assert prev_tail in nxt.content[: chunk_overlap + len(prev_tail)]


def test_overlap_skipped_when_previous_chunk_ends_in_protected_block():
    # 上一个分片的最后一个块是表格：借用它结尾的字符做 overlap 等于截出
    # 半段表格，所以这条边界应该保持干净、不加 overlap。
    text = (
        "## Sec\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        + ("next paragraph word " * 10) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=40, chunk_overlap=15)

    table_chunk = next(c for c in chunks if c.has_table)
    following = chunks[chunks.index(table_chunk) + 1]
    assert not following.content.startswith("|")
    assert "|" not in following.content


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


def test_indented_code_block_content_is_preserved_not_dropped():
    text = (
        "## Section\n"
        "\n"
        "first paragraph\n"
        "\n"
        "    indented code line one\n"
        "    indented code line two\n"
    )
    chunks = chunk_markdown_structured(text)
    combined = "\n".join(c.content for c in chunks)
    assert "indented code line one" in combined
    assert "indented code line two" in combined
    assert any(c.has_code_block for c in chunks)


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


def test_small_group_merges_into_neighbor_within_same_section():
    text = (
        "## Sec\n\n"
        "---\n\n"
        + ("word " * 30) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=50)

    section_chunks = [c for c in chunks if c.heading_path.startswith("Sec")]
    assert len(section_chunks) == 1
    assert "---" in section_chunks[0].content
    assert "word" in section_chunks[0].content


def test_cascading_merge_of_several_small_groups():
    # 三个用 hr 分隔的短列表项，单独打包都会 < min_chars，级联合并成更少
    # 的 chunk，但每个仍然不超过 chunk_size。
    text = "## Sec\n\n" + "\n\n---\n\n".join(f"- item {i}" for i in range(6)) + "\n"
    chunks = chunk_markdown_structured(text, chunk_size=60, chunk_overlap=0, min_chars=40)

    section_chunks = [c for c in chunks if c.heading_path.startswith("Sec")]
    assert len(section_chunks) < 6
    assert all(c.char_count <= 60 for c in section_chunks)
    for i in range(6):
        assert sum(c.content.count(f"item {i}") for c in section_chunks) == 1


def test_merge_refuses_to_exceed_chunk_size():
    text = "## Sec\n\n---\n\n" + ("word " * 40) + "\n"
    chunks = chunk_markdown_structured(text, chunk_size=30, chunk_overlap=0, min_chars=1000)

    section_chunks = [c for c in chunks if c.heading_path.startswith("Sec")]
    assert len(section_chunks) > 1
    assert all(c.char_count <= 30 for c in section_chunks)


def test_lone_small_piece_in_section_stays_unmerged_across_section_boundary():
    text = "## A\n\ntiny\n\n## B\n\nalso tiny\n"
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=500)

    paths = [c.heading_path for c in chunks]
    assert paths == ["A", "B"]
    assert chunks[0].content == "tiny"
    assert chunks[1].content == "also tiny"


# 真实语料的主导碎片化模式：标题多、每节天生只有一小段正文，section 内
# 没有相邻小块可合并（实测见 docs 里的小片合并设计文档）。跨 section 合并
# 针对的就是这种文档。
CHARTER_MD = (
    "# 公司章程\n"
    "\n"
    "## 第一章 总则\n"
    "\n"
    "### 第一条\n"
    "\n"
    "公司名称为示例科技有限公司。\n"
    "\n"
    "### 第二条\n"
    "\n"
    "公司住所为广东省广州市天河区。\n"
    "\n"
    "### 第三条\n"
    "\n"
    "公司经营范围以登记机关核准的内容为准。\n"
)


def test_adjacent_small_sections_merge_into_verbatim_slice_of_source():
    # 合并结果必须仍然是原文的一段连续切片——本模块承诺"内容是原文逐字
    # 提取"，下游 app/eval 的 locate_chunk_spans 也靠 text.find(chunk) 定位
    # chunk 才能判断表格/代码块有没有被切断。
    chunks = chunk_markdown_structured(CHARTER_MD, chunk_size=1000, chunk_overlap=0, min_chars=200)

    assert len(chunks) == 1
    assert chunks[0].content in CHARTER_MD


def test_cross_section_merge_keeps_every_merged_heading_in_content():
    # 合并跨越了多个标题，被跨过的标题行必须留在正文里，否则这些标题文本
    # 在 heading_path（取公共祖先）和 content 里都不存在，信息直接丢失。
    chunks = chunk_markdown_structured(CHARTER_MD, chunk_size=1000, chunk_overlap=0, min_chars=200)

    content = chunks[0].content
    assert "### 第一条" in content
    assert "### 第二条" in content
    assert "### 第三条" in content


def test_cross_section_merge_uses_common_ancestor_as_heading_path():
    # 不能取"更早那个 piece"的 heading_path——那条路径只覆盖合并内容的
    # 前一部分，对后面几节是错的面包屑。取最长公共前缀才是真正包含全部
    # 合并内容的标题。
    chunks = chunk_markdown_structured(CHARTER_MD, chunk_size=1000, chunk_overlap=0, min_chars=200)

    assert chunks[0].heading_path == "公司章程 > 第一章 总则"


def test_cross_section_merge_refused_when_sections_share_no_ancestor_heading():
    # 两个顶层章节之间没有共同祖先标题，合并出来的 chunk 找不到一个能覆盖
    # 全部内容的 heading_path，而且跨顶层章节本身就是语义边界，不该合并。
    text = "# 第一章\n\n很短的正文\n\n# 第二章\n\n也很短的正文\n"
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=500)

    assert [c.heading_path for c in chunks] == ["第一章", "第二章"]


def test_cross_section_merge_refused_when_result_exceeds_chunk_size():
    text = (
        "# Doc\n\n"
        "## A\n\n" + ("word " * 20) + "\n\n"
        "## B\n\n" + ("word " * 20) + "\n"
    )
    chunks = chunk_markdown_structured(text, chunk_size=120, chunk_overlap=0, min_chars=1000)

    assert len(chunks) > 1
    assert all(c.char_count <= 120 for c in chunks)


def test_split_section_pieces_do_not_participate_in_cross_section_merge():
    # 被 chunk_size 拆开的 section，它的每个 piece 都不对应一段完整的原文
    # 行区间（还可能被注入过 overlap 前缀），没法按行区间重新切片，所以
    # 不参与跨 section 合并。
    text = "# Doc\n\n## Long\n\n" + ("x" * 3000) + "\n\n## Tiny\n\ntiny body\n"
    chunks = chunk_markdown_structured(text, chunk_size=1000, chunk_overlap=0, min_chars=200)

    tiny = [c for c in chunks if c.content == "tiny body"]
    assert len(tiny) == 1
    assert tiny[0].heading_path == "Doc > Tiny"


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


def test_every_chunk_of_real_doc_is_a_verbatim_substring_of_the_source():
    text = REAL_DOC.read_text(encoding="utf-8")

    chunks = chunk_markdown_structured(text, chunk_overlap=0)

    not_found = [c.heading_path for c in chunks if c.content not in text]
    assert not_found == []


def test_structured_chunks_never_truncate_protected_regions_per_eval_metrics():
    chunks = chunk_markdown_structured(HIERARCHY_MD, chunk_size=1000, chunk_overlap=0)
    report = detect_structure_violations(
        HIERARCHY_MD, [c.content for c in chunks], chunk_overlap=0
    )
    assert report.truncated_regions == 0


REAL_DOC = Path(__file__).resolve().parent / "fixtures" / "备考总计划TODO.md"


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
