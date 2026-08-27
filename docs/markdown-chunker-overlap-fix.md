# 修复：结构化 Markdown 切片器丢失 chunk_overlap，导致相邻分片没有关联性

日期：2026-08-26

## 问题

`app/config.py` 里 `chunk_overlap` 默认值是 150，从 API/`ChunkOptions` 一路
透传到 `chunk_markdown_structured()`，但实际生产链路（`.md` 文件、经 MinerU
转换的 PDF——两者都走 `_write_text_chunks` 里 `file_type in ("md", "pdf")`
分支）里，这个参数在绝大多数情况下**完全不生效**：同一个标题小节内，如果
正文因为超过 `chunk_size`（默认 1000 字符）被切成多个分片，相邻分片之间
是零重叠的硬切——上一个分片结尾和下一个分片开头之间不共享任何字符。

后果：检索命中其中一个分片时，看不到紧邻分片的内容，两个本来语义连续的
分片之间失去了关联性；如果用户查询恰好落在切分点附近，召回的单个分片经
常缺半句话的上下文。

## 根因

`app/ingest/markdown_chunker.py` 的 `_pack_blocks`/`chunk_markdown_structured`
把一个标题小节的正文按块级 token（段落/表格/代码块/列表/引用块等）顺序
打包进不超过 `chunk_size` 的分组，分组之间原本就没有重叠逻辑
（`chunk_overlap` 只在单个块本身超过 `chunk_size`、回退到字符级
`chunk_text()` 时才生效——见
`docs/superpowers/specs/2026-08-26-structured-markdown-chunker-design.md`
"直属内容内部的二次拆分"一节）。

当时不做重叠的理由是"块是语义原子单位，重叠会重复整段表格/代码块，没有
意义"——这个理由本身只适用于**表格/代码块**这类不可拆的保护块，但实现上
被无差别套用到了**所有**分组边界，包括两个都是普通段落、纯粹因为长度超限
被切开的分组。这部分场景占绝大多数（大段连续正文），也正是这次要修的
场景。

## 修复

`app/ingest/markdown_chunker.py` 新增 `_with_overlap_prefix()`，在
`chunk_markdown_structured()` 组装同一小节内连续分组的过程中调用：

- 把上一个分组结尾的 `chunk_overlap` 个字符，前缀拼接到当前分组开头。
- **仅当上一个分组的最后一个块不是受保护块（表格/代码）时**才借用——如果
  是，保持原有的干净边界不加 overlap，避免截出半段表格/代码，不违反"块
  不可跨 chunk 拆开"的约束。
- 英文场景下会把 overlap 片段裁到下一个完整单词的起点，避免和当前分片的
  首词粘连成一个不存在的新 token（影响 sparse/关键词检索的分词）；中文没
  有空格分词边界，找不到就直接用原始字符切片。
- 单个超长段落回退到字符级 `chunk_text()` 时（`chunk_text` 内部本来就有
  overlap），只需要把这个回退序列的第一个子分片也接上前一个分组的
  overlap，衔接住"结构化分组 → 字符级回退"这条路径上的边界。
- 不同标题小节之间的分片仍然保持零重叠（未改动）：新标题是真实的语义边界，
  不是因为长度被迫切开的，不需要借用上下文。

## 已知取舍

- `app/eval/structure_metrics.locate_chunk_spans` 用 `str.find()` 在原文里
  启发式定位每个 chunk 的字符区间，本身就有"原文有大段重复子串时可能定位
  有误"的已知局限（见该模块 docstring）。这次修复让相邻分片确实产生了同
  一段原文的字面重复，属于该定位工具的容忍范围内，不影响生产环境实际写入
  Qdrant 的分片内容——只可能在极少数边界情况下让评估脚本（`scripts/
  eval_chunk_quality.py`）的定位结果不够精确，不影响检索本身。

## 验证

- `tests/unit/test_markdown_chunker.py`：
  - `test_long_section_split_has_overlap_and_no_lost_content`（原
    `test_long_section_split_has_no_overlap_and_no_lost_content` 改名+改写）：
    断言相邻分片之间确实共享一小段上下文，且每个完整段落仍然只完整出现
    一次（不是整段重复）。
  - 新增 `test_overlap_skipped_when_previous_chunk_ends_in_protected_block`：
    断言紧跟在表格分片后面的下一个分片不会借到任何表格片段。
- `python3 -m pytest tests/unit -q`：187 passed。
