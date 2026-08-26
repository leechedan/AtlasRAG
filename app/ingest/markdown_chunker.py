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
    "code_block",
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
        elif token.type in ("fence", "code_block"):
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
    """拼接一组块对应的原文行。

    是否用 `.strip()` 还是只 `lstrip()` 取决于组里有没有受保护块（表格/
    代码），这不是随意的格式选择，而是和
    `app/eval/structure_metrics.detect_structure_violations` 的实现耦合：
    该函数计算受保护区域结束偏移量时用的是 `line_offsets[region.end_line]`，
    这个偏移量包含了区域末尾的换行符。如果这里对含受保护块的 chunk 做完
    整的 `.strip()`（会去掉末尾换行），chunk 文本就会比 region 的结束偏移
    少一个字符，即使内容完整也会被 `detect_structure_violations` 误判为
    截断（"truncated"）。所以含受保护块的组只做 `lstrip()`，保留末尾换行，
    让 chunk 文本能完整覆盖到 region 的结束偏移。
    不含受保护块的组没有这个下游比较，用普通 `.strip()` 去掉首尾空白即可。
    如果以后把 `detect_structure_violations` 的包含性判断改成对空白容忍，
    这里就可以恢复成统一的 `.strip()`。
    """
    content = "".join(lines[group[0].start_line : group[-1].end_line])
    has_protected = any(b.kind is not None for b in group)
    if has_protected:
        return content.lstrip()
    else:
        return content.strip()


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
