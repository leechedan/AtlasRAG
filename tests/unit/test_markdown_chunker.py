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
