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


@dataclass
class _Piece:
    heading_path: str
    content: str
    has_table: bool
    has_code: bool
    # 该 piece 在原文里对应的行区间 [标题行, 正文结束行)，只有"整节恰好
    # 产出这一个 piece"时才填。为 None 表示这个 piece 是 section 被
    # chunk_size 拆开（或注入过 overlap 前缀）的产物，不对应一段完整的原
    # 文行区间，因此不能参与跨 section 合并。见
    # `_merge_small_pieces_across_sections`。
    line_span: tuple[int, int] | None = None
    # 正文起始行（跳过标题行本身）。合并时如果这一节的路径正好就是合并
    # 结果的 heading_path，就从这里而不是标题行开始切，免得同一个标题既
    # 出现在 heading_path 里又重复出现在正文开头。
    content_start_line: int | None = None


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


def _build_sections(lines: list[str], tokens: list) -> list[tuple[str, int, int, int]]:
    """返回 [(heading_path, heading_start_line, content_start_line,
    content_end_line), ...]，end 不含。没有标题时返回单个
    (NO_HEADING_PLACEHOLDER, 0, 0, len(lines))。

    heading_start_line 是标题自身所在的行，正常切片时用不到（标题文本走
    heading_path 元数据，不进正文）；只有跨 section 合并按原文行区间重新
    切片时需要它，好让被合并进来的各节标题留在正文里，
    见 `_merge_small_pieces_across_sections`。"""
    headings = _headings(tokens)
    n_lines = len(lines)

    if not headings:
        return [(NO_HEADING_PLACEHOLDER, 0, 0, n_lines)]

    sections: list[tuple[str, int, int, int]] = []
    _, _, first_heading_start, _ = headings[0]
    if first_heading_start > 0:
        sections.append((NO_HEADING_PLACEHOLDER, 0, 0, first_heading_start))

    stack: list[tuple[int, str]] = []
    for i, (level, htext, start, content_start) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, htext))
        heading_path = " > ".join(t for _, t in stack)
        content_end = headings[i + 1][2] if i + 1 < len(headings) else n_lines
        sections.append((heading_path, start, content_start, content_end))

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


def _group_len(group: list[_Block], lines: list[str]) -> int:
    return sum(len(lines[i]) for i in range(group[0].start_line, group[-1].end_line))


def _merge_small_groups(
    groups: list[list[_Block]], lines: list[str], chunk_size: int, min_chars: int
) -> list[list[_Block]]:
    """把 section 内偏小的 block-group 并入相邻 group，减少过短分片。

    只在 group（block 列表）这一层合并，发生在 `_group_text`/
    `_with_overlap_prefix` 之前：如果改成在已经注入过 overlap 前缀的最终
    文本上做字符串拼接合并，会把上一个 group 借给下一个 group 的 overlap
    片段重复拼进去，产生不易察觉的文本重复。在这一层合并，overlap 注入
    照常在合并之后的最终 group 序列上只发生一次。

    优先并入下一个 group，行不通（没有下一个，或者合并后会超过
    chunk_size）就尝试并入上一个；两个方向都不行就保留原样——不跨
    section、不无视 chunk_size 上限地生造超大 chunk。"""
    result = list(groups)
    i = 0
    while i < len(result):
        if len(result) == 1 or _group_len(result[i], lines) >= min_chars:
            i += 1
            continue

        if i + 1 < len(result) and (
            _group_len(result[i], lines) + _group_len(result[i + 1], lines) <= chunk_size
        ):
            result[i] = result[i] + result[i + 1]
            del result[i + 1]
            continue

        if i > 0 and (
            _group_len(result[i - 1], lines) + _group_len(result[i], lines) <= chunk_size
        ):
            result[i - 1] = result[i - 1] + result[i]
            del result[i]
            i -= 1
            continue

        i += 1

    return result


def _common_heading_path(a: str, b: str) -> str:
    """两条面包屑路径的最长公共前缀，也就是同时包含两者内容的最深标题。
    没有公共祖先时返回空串。"""
    common: list[str] = []
    for x, y in zip(a.split(" > "), b.split(" > ")):
        if x != y:
            break
        common.append(x)
    return " > ".join(common)


def _merge_two_pieces(earlier: _Piece, later: _Piece, lines: list[str]) -> _Piece:
    """按原文行区间把两个相邻 piece 重新切成一片。

    关键是**重新切片**而不是把两段已定稿的 content 字符串拼起来：两个
    section 的正文之间隔着标题行（可能还隔着若干个没有正文、因此没产出
    piece 的空标题节），字符串拼接的结果在原文里根本找不到连续匹配，会
    破坏本模块"内容是原文逐字提取"的承诺，也会让
    `app/eval/structure_metrics.locate_chunk_spans` 的 `text.find(chunk)`
    定位失败、退化成猜测区间，进而把完好的表格/代码块误报成被截断。
    取 `lines[较早节的标题行 : 较晚节的正文结束行]` 就仍然是原文的一段连
    续切片，中间被跨过的标题行也自然留在正文里。"""
    merged_path = _common_heading_path(earlier.heading_path, later.heading_path)
    # 合并结果的面包屑就是较早这一节自己的路径时（父节的正文并入了子节），
    # 它的标题已经由 heading_path 表达了，正文就从标题行之后开始，不重复。
    # 其余情况从标题行开始切，让这一节的标题作为原文的一部分留在正文里
    # ——否则它既不在 heading_path（那是更浅的公共祖先）里，也不在正文里，
    # 就彻底丢了。
    if merged_path == earlier.heading_path and earlier.content_start_line is not None:
        start = earlier.content_start_line
    else:
        start = earlier.line_span[0]
    end = later.line_span[1]
    has_table = earlier.has_table or later.has_table
    has_code = earlier.has_code or later.has_code
    content = "".join(lines[start:end])
    # 和 `_group_text` 同样的理由：含受保护块时只 lstrip，保留末尾换行，
    # 免得被 `detect_structure_violations` 误判为截断。
    content = content.lstrip() if (has_table or has_code) else content.strip()
    return _Piece(
        heading_path=merged_path,
        content=content,
        has_table=has_table,
        has_code=has_code,
        line_span=(start, end),
        # 合并结果的起点已经定了，级联再合并时不能再往回挪：两个分支都取
        # 到同一个 start，而不是继承较早那一节的正文起点（那会把它的标题
        # 行又切掉）。
        content_start_line=start,
    )


def _can_merge(earlier: _Piece, later: _Piece, lines: list[str], chunk_size: int) -> bool:
    if earlier.line_span is None or later.line_span is None:
        return False
    if not _common_heading_path(earlier.heading_path, later.heading_path):
        return False
    return len(_merge_two_pieces(earlier, later, lines).content) <= chunk_size


def _merge_small_pieces_across_sections(
    pieces: list[_Piece], lines: list[str], chunk_size: int, min_chars: int
) -> list[_Piece]:
    """章节内合并（`_merge_small_groups`）只能消除"一个 section 内有多个
    小块"的碎片化，消除不了真实语料里占主导的模式——标题多、每个 section
    天生就只有一小段正文，section 内根本没有相邻小块可合并。这一轮在整个
    文档的最终 piece 序列上再扫一遍，把小 piece 并入相邻 section。

    三条约束，缺一不可：

    1. 两边都必须是"整节恰好产出一个 piece"（`line_span` 不为 None）。被
       chunk_size 拆开的 piece 不对应完整的原文行区间，没法重新切片。
    2. 合并后的 `heading_path` 取两条路径的最长公共前缀——那是唯一真正
       包含全部合并内容的标题。取"更早那个 piece 的路径"是错的：它只覆
       盖合并内容的前一部分，对后面几节就是指错地方的面包屑。
    3. 公共前缀为空（两节分属不同顶层章节）时**拒绝合并**：既找不到能覆
       盖全部内容的 heading_path，跨顶层章节本身也是语义边界。

    被跨过的各节标题不会丢失——它们作为原文行留在合并后的正文里（见
    `_merge_two_pieces`）。

    不会重复注入 overlap：overlap 只在同一个 section 内部注入过一次
    （`prev_group`/`prev_text` 每个 section 开始时都重置），而这里参与合
    并的 piece 按约束 1 都是没被注入过 overlap 的整节 piece。"""
    result = list(pieces)
    i = 0
    while i < len(result):
        if len(result) == 1 or len(result[i].content) >= min_chars:
            i += 1
            continue

        if i + 1 < len(result) and _can_merge(result[i], result[i + 1], lines, chunk_size):
            result[i] = _merge_two_pieces(result[i], result[i + 1], lines)
            del result[i + 1]
            continue

        if i > 0 and _can_merge(result[i - 1], result[i], lines, chunk_size):
            result[i - 1] = _merge_two_pieces(result[i - 1], result[i], lines)
            del result[i]
            i -= 1
            continue

        i += 1

    return result


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


def _with_overlap_prefix(
    text_piece: str, prev_group: list[_Block], prev_text: str | None, chunk_overlap: int
) -> str:
    """把上一个块打包组结尾的 chunk_overlap 个字符前缀到当前分片，弥补
    "同一节内容因超过 chunk_size 被硬切"产生的分片边界断裂——两个相邻
    分片原本共享的上下文完全丢失，检索命中其中一个时看不到另一个的内容，
    互相之间就没有关联性了。

    只在上一组的最后一个块不是受保护块（表格/代码）时借用：受保护块整体
    不可拆，借用它结尾的字符等于截出半段表格/代码，没有意义，也违反"块
    不可跨 chunk 拆开"的约束——遇到这种情况就不加 overlap，保持原有的
    干净边界。
    """
    if chunk_overlap <= 0 or not prev_text or prev_group[-1].kind is not None:
        return text_piece
    overlap = prev_text[-chunk_overlap:]
    # 字符级硬切可能从英文单词中间开始，尝试丢掉切到的第一个空白字符之前
    # 的残缺片段，让 overlap 从一个完整词开始，避免和当前分片的首词粘连成
    # 一个不存在的新 token（影响 sparse/关键词检索的分词）。中文没有空格
    # 分词，找不到空白边界时直接用原始切片——从句子中间开始对中文阅读
    # 体验的影响远小于英文单词被粘连成乱码 token。
    boundary = next((i for i, ch in enumerate(overlap) if ch.isspace()), None)
    if boundary is not None:
        overlap = overlap[boundary + 1 :]
    overlap = overlap.strip()
    if not overlap or text_piece.startswith(overlap):
        return text_piece
    return overlap + "\n\n" + text_piece


def chunk_markdown_structured(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chars: int | None = None,
) -> list[StructuredChunk]:
    if not text or not text.strip():
        return []

    chunk_size_val = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap_val = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    min_chars_val = min_chars if min_chars is not None else settings.chunk_min_chars

    tokens = _MD.parse(text)
    lines = text.splitlines(keepends=True)
    sections = _build_sections(lines, tokens)
    all_blocks = _top_level_blocks(tokens)

    all_pieces: list[_Piece] = []

    for heading_path, heading_line, start_line, end_line in sections:
        section_text = "".join(lines[start_line:end_line]).strip()
        if not section_text:
            continue

        section_blocks = _blocks_within(all_blocks, start_line, end_line)

        pieces: list[tuple[str, bool, bool]] = []
        if not section_blocks:
            pieces.append((section_text, False, False))
        else:
            # prev_group/prev_text track the immediately preceding packed
            # group so a same-section, size-driven cut (the common case: a
            # long run of plain paragraphs split only because it exceeds
            # chunk_size_val) can carry chunk_overlap_val characters of
            # trailing context into the next piece -- see "为什么块打包
            # 之间也要 overlap" in the markdown-chunker-overlap design doc.
            prev_group: list[_Block] | None = None
            prev_text: str | None = None
            packed_groups = _pack_blocks(section_blocks, lines, chunk_size_val)
            packed_groups = _merge_small_groups(packed_groups, lines, chunk_size_val, min_chars_val)
            for group in packed_groups:
                group_text = _group_text(group, lines)
                has_table = any(b.kind == "table" for b in group)
                has_code = any(b.kind == "code" for b in group)
                oversized_plain = (
                    len(group) == 1
                    and group[0].kind is None
                    and len(group_text) > chunk_size_val
                )
                if oversized_plain:
                    sub_chunks = chunk_text(
                        group_text, chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val
                    )
                    if sub_chunks and prev_group is not None:
                        sub_chunks[0] = _with_overlap_prefix(
                            sub_chunks[0], prev_group, prev_text, chunk_overlap_val
                        )
                    for sub in sub_chunks:
                        pieces.append((sub, False, False))
                    prev_text = sub_chunks[-1] if sub_chunks else prev_text
                else:
                    if prev_group is not None:
                        group_text = _with_overlap_prefix(
                            group_text, prev_group, prev_text, chunk_overlap_val
                        )
                    pieces.append((group_text, has_table, has_code))
                    prev_text = group_text
                prev_group = group

        # 只有整节恰好产出一个 piece 时，这个 piece 才对应一段完整的原文
        # 行区间（也才没被注入过 overlap 前缀），能参与按行区间重新切片的
        # 跨 section 合并。区间从标题行开始，这样合并时各节标题会留在正文里。
        span = (heading_line, end_line) if len(pieces) == 1 else None
        body_start = start_line if len(pieces) == 1 else None
        for content, has_table, has_code in pieces:
            all_pieces.append(
                _Piece(heading_path, content, has_table, has_code, span, body_start)
            )

    all_pieces = _merge_small_pieces_across_sections(
        all_pieces, lines, chunk_size_val, min_chars_val
    )

    # 重名的面包屑统一在这里编号，一条规则覆盖两种重名来源：同一节被
    # chunk_size 拆成多片，以及多个合并结果的面包屑都退到了同一个公共祖
    # 先。对前者产生的编号和分节时逐片编号的老写法完全一致（同一节的多片
    # 在列表里相邻且同名）。
    seen: dict[str, int] = {}

    chunks: list[StructuredChunk] = []
    for piece in all_pieces:
        seen[piece.heading_path] = seen.get(piece.heading_path, 0) + 1
        occurrence = seen[piece.heading_path]
        path = piece.heading_path if occurrence == 1 else f"{piece.heading_path} #{occurrence}"
        chunks.append(
            StructuredChunk(
                chunk_id=len(chunks),
                heading_path=path,
                has_table=piece.has_table,
                has_code_block=piece.has_code,
                char_count=len(piece.content),
                token_estimate=round(len(piece.content) / 1.8),
                content=piece.content,
            )
        )

    return chunks
