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
