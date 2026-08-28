# 结构化 Markdown 分片：小块合并设计

## 背景

`scripts/eval_chunk_quality.py --chunker structured` 跑出的 report.json
显示 `length_out_of_range_ratio = 0.9333`：93% 的 chunk 都不在
`[chunk_size*0.2, chunk_size]` 目标区间内，`length_median_chars = 5` 远低于
`length_mean_chars = 63`，说明大量 chunk 是极短的碎片（标题独立成段、`---`
分隔线、短列表项等各自打包成一个 group）。`chunk_markdown_structured()` 按
标题 section 分组、section 内再用 `_pack_blocks()` 把块打包到接近
`chunk_size`，但从不检查打包出的某个 group 是否偏小，也从不跨 group 合并。

`chunk_markdown_structured()` 不只是 eval 脚本用的路径——`app/ingest/
pipeline.py` 里所有 `.md` 文件、以及 MinerU 转出 markdown 的 PDF/docx/pptx/
xlsx（`_MINERU_MARKDOWN_FILE_TYPES`）都走这同一个函数，所以这是生产摄取链
路的真实问题，不只是评估指标好看不好看。

## 范围

只改 `app/ingest/markdown_chunker.py` 的 `chunk_markdown_structured()`。不
改 `app/ingest/chunking.py` 的 `chunk_text()`（无结构字符切片器，用于
`.txt` 等文档，没有 heading/section 概念）。

合并只在**同一个 heading section 内部**进行，不跨标题边界合并——保持
`heading_path` 语义准确、不需要处理多标题拼接的展示问题。代价是：如果整
个 section 本身只有一个天然很短的 piece（比如只有一行正文的小节），这次
改动不会消除它——这是明确接受的局限，不是遗漏。

## 合并发生的层次

在 `_pack_blocks()` 产出的 `list[list[_Block]]`（block-group）这一层合并，
**在**用 `_group_text()` 拼出最终文本、用 `_with_overlap_prefix()` 注入
overlap 前缀**之前**。

这个顺序不是随意的：如果在最终文本（已经带 overlap 前缀）上做字符串拼接
合并，会把上一个 group 借给下一个 group 的 overlap 片段重复拼进去，产生
不易察觉的文本重复。在 block-group（结构层）合并就没有这个问题——overlap
注入照常在合并之后的最终 group 序列上进行一次，不会重复。

## 算法

新增 `_merge_small_groups(groups, lines, chunk_size, min_chars) -> list[list[_Block]]`，
在 `_pack_blocks()` 之后、原有的逐 group 生成文本循环之前调用：

- 从左到右扫描 `groups`。某个 group 的字符数（复用 `_pack_blocks` 里同样
  的按行长度求和方式）< `min_chars` 时：
  1. 优先尝试并入**下一个** group（block 列表直接拼接，`_group_text` 天
     然支持任意 block 序列，只要求组内 block 是原文顺序）；如果合并后总
     长度会超过 `chunk_size` 就放弃这个方向。
  2. 下一个方向不行（超限，或没有下一个 group 了）就尝试并入**上一个**
     group，同样受 `chunk_size` 上限约束。
  3. 两个方向都不行（都会超限，或者前后都没有邻居）——放弃合并，保留这
     个小 group 原样。
- 合并是级联的：小 group 并入邻居后如果合并结果仍然 < `min_chars`，继续
  尝试和下一个邻居合并，直到达标或者两个方向都合并不动为止。

合并结果直接替换原来的 `groups`，后续 `_group_text`/`oversized_plain`/
`_with_overlap_prefix` 逻辑不变——它们本来就是按 group 处理，对合并后的
group（多个 block 而不是一个 block）同样适用。

## 配置

`app/config.py` 新增字段：

```python
chunk_min_chars: int = 200
```

对应 `chunk_size=1000` 默认值的 20%，和 `scripts/eval_chunk_quality.py`
里 `--min-chars` 未显式传参时的默认公式 `chunk_size * 0.2` 口径一致，但作
为独立可调字段（不做动态派生，和 `chunk_size`/`chunk_overlap` 现有写法保
持一致）。

`chunk_markdown_structured()` 新增参数 `min_chars: int | None = None`，不
传时读 `settings.chunk_min_chars`（和现有 `chunk_size`/`chunk_overlap` 参
数的处理方式一致）。

## 已知不覆盖的残留情况

`oversized_plain` 分支（单个超长段落被 `chunk_text()` 硬切出的子分片）如
果切出一个很小的尾片，这次不处理——`chunk_text()` 内部用的
`RecursiveCharacterTextSplitter` 自己也做 overlap，在这一层再做合并会有
同样的"重复 overlap 文本"风险，且出现频率远低于 block-group 层面的碎片
化，不在本次范围内。

## 测试计划

`tests/unit/test_markdown_chunker.py` 新增：
- 一个 section 内有一个很短的 group（如独立的 `---` 分隔线或一行短列表）
  紧邻一个正常大小的 group：断言合并后只剩一个 chunk，内容按原文顺序拼
  接完整。
- 级联合并：连续三个以上都偏小的 group，断言最终合并成尽量接近
  `chunk_size` 又不超限的更少数量的 chunk。
- 合并会导致超过 `chunk_size` 时拒绝合并，断言小 group 保持独立（验证上
  限约束生效）。
- 整个 section 只有一个天然偏小的 piece：断言原样保留（不会跟其他
  section 合并，验证 section 边界不被打破）。
- 合并不影响受保护块（表格/代码块）本身完整性：复用
  `detect_structure_violations` 断言违规率仍为 0。
- 现有用例里因为合并导致 chunk 数量/`heading_path` `#n` 序号变化的，同步
  更新预期值。

## 待办

无——这是本次改动的完整范围。
