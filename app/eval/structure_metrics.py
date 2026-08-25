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
