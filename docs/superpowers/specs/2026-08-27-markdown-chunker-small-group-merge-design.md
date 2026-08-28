# 结构化 Markdown 分片：小片合并设计

## 背景

`scripts/eval_chunk_quality.py --chunker structured` 跑出的 report.json
显示 `length_out_of_range_ratio = 0.9333`：93% 的 chunk 都不在
`[chunk_size*0.2, chunk_size]` 目标区间内，`length_median_chars = 5` 远低于
`length_mean_chars = 63`，说明大量 chunk 是极短的碎片。
`chunk_markdown_structured()` 按标题 section 分组、section 内再用
`_pack_blocks()` 把块打包到接近 `chunk_size`，但从不检查打包出的某个 group
是否偏小，也从不跨 group 合并。

`chunk_markdown_structured()` 不只是 eval 脚本用的路径——`app/ingest/
pipeline.py` 里所有 `.md` 文件、以及 MinerU 转出 markdown 的 PDF/docx/pptx/
xlsx（`_MINERU_MARKDOWN_FILE_TYPES`）都走这同一个函数，所以这是生产摄取链
路的真实问题，不只是评估指标好看不好看。

## 实测：碎片化的主导模式是"小章节多"，不是"一个 section 内小块多"

对三份文档统计 section 正文尺寸（`chunk_size=1000`，小 section 阈值 200）：

| 文档 | section 数 | 正文中位数 | 小 section | 其中 section 内只有 ≤1 个 block | 长 section(>chunk_size) |
|---|---|---|---|---|---|
| 广东省考公务员.md | 14 | 196 | 57.1% | 57.1% | 0% |
| 备考总计划TODO.md | 16 | 281 | 25.0% | 6.2% | 6.2% |
| K8S部署文档.docx→md ⚠️ | 31 | 45 | 93.5% | 77.4% | 3.2% |

⚠️ K8S 那行只能看方向：它是用 python-docx 按 Heading 样式近似转的，没有
MinerU 的代码块围栏，shell 注释行（`# X86环境：...`）被误当成标题，section
数被高估。

结论：**长 section 几乎不存在，`chunk_size` 上限在真实语料里基本不触发**；
碎片全部来自"标题多、每节天生只有一小段正文"。因此只在 section 内部合并是
不够的——实测 A（不合并）和 B（仅 section 内合并）在这三份文档上输出**逐项
完全相同**，一次都没触发。必须允许跨 section 合并。

## 两层合并

### 第一层：section 内合并（`_merge_small_groups`）

在 `_pack_blocks()` 产出的 `list[list[_Block]]`（block-group）这一层合并，
**在**用 `_group_text()` 拼出最终文本、用 `_with_overlap_prefix()` 注入
overlap 前缀**之前**。

这个顺序不是随意的：如果在最终文本（已经带 overlap 前缀）上做字符串拼接
合并，会把上一个 group 借给下一个 group 的 overlap 片段重复拼进去，产生
不易察觉的文本重复。在 block-group（结构层）合并就没有这个问题——overlap
注入照常在合并之后的最终 group 序列上进行一次，不会重复。

算法：从左到右扫描，group 字符数 < `min_chars` 时优先并入**下一个** group，
超 `chunk_size` 就改试**上一个**，两个方向都不行就保留原样。合并是级联的。

这一层在实测的三份文档上零触发（见上表：小 section 里绝大多数 section 内
只有 ≤1 个 block，根本没有相邻小块可合并）。保留它是因为它覆盖另一种真实
存在的形态：少数大章节、每章节内很多小块（例如"第一章"是标题、"第一条…
第十条"是它下面的并列段落的章程/制度类文档）。

### 第二层：跨 section 合并（`_merge_small_pieces_across_sections`）

在整个文档的最终 piece 序列上再扫一遍，把小 piece 并入相邻 section。

**必须按原文行区间重新切片，而不是把两段已定稿的 content 字符串拼接。**
两个 section 的正文之间隔着标题行（可能还隔着若干没有正文、因此没产出
piece 的空标题节），字符串拼接的结果在原文里根本找不到连续匹配：

- 破坏本模块"内容是原文逐字提取"的承诺；
- 让 `app/eval/structure_metrics.locate_chunk_spans` 的 `text.find(chunk)`
  定位失败、退化成猜测区间，把完好的表格/代码块误报成被截断
  （实测 `truncated_regions` 由 0 变成 1~2）。

取 `lines[较早节的标题行 : 较晚节的正文结束行]` 就仍然是原文的一段连续切片，
被跨过的标题行也自然留在正文里。

三条约束，缺一不可：

1. **两边都必须是"整节恰好产出一个 piece"**（`_Piece.line_span` 不为
   None）。被 `chunk_size` 拆开的 piece 不对应完整的原文行区间，也可能被
   注入过 overlap 前缀，没法重新切片。
2. **合并后的 `heading_path` 取两条路径的最长公共前缀**——那是唯一真正
   包含全部合并内容的标题。取"更早那个 piece 的路径"是错的：它只覆盖合并
   内容的前一部分，对后面几节就是指错地方的面包屑。
3. **公共前缀为空（两节分属不同顶层章节）时拒绝合并**：既找不到能覆盖全部
   内容的 `heading_path`，跨顶层章节本身也是语义边界。

被跨过的各节标题不会丢失——它们作为原文行留在合并后的正文里。唯一的例外
是合并结果的面包屑正好等于较早那一节自己的路径时（父节的正文并入了子节），
这一节的标题已经由 `heading_path` 表达，正文就从标题行之后开始，不重复。

### 面包屑编号

`#n` 编号在最后对最终 chunk 列表统一做，一条规则覆盖两种重名来源：同一节
被 `chunk_size` 拆成多片，以及多个合并结果的面包屑都退到了同一个公共祖先。
对前者产生的编号和"分节时逐片编号"的老写法完全一致（同一节的多片在列表里
相邻且同名）。

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
传时读 `settings.chunk_min_chars`；传 `0` 可以完全关掉两层合并（单测里用来
隔离标题层级/面包屑本身的行为）。

## 效果（chunk_size=1000, chunk_overlap=0, min_chars=200）

| 文档 | | chunk 数 | 中位数 | 出界比例 | truncated_regions | 原文可定位 |
|---|---|---|---|---|---|---|
| 广东省考公务员.md | 合并前 | 14 | 196 | 50% | 0/3 | 14/14 |
| | 合并后 | 10 | 259 | **0%** | 0/3 | 10/10 |
| 备考总计划TODO.md | 合并前 | 17 | 263 | 29% | 0/8 | 17/17 |
| | 合并后 | 15 | 323 | **13%** | 0/8 | 15/15 |

"原文可定位"= chunk 内容是原文的连续子串。两项不变量（`truncated_regions`
为 0、可定位 100%）在合并前后都保持。

## 已知不覆盖的残留情况

- `oversized_plain` 分支（单个超长段落被 `chunk_text()` 硬切出的子分片）
  如果切出一个很小的尾片，不处理——`chunk_text()` 内部用的
  `RecursiveCharacterTextSplitter` 自己也做 overlap，在这一层再做合并会有
  "重复 overlap 文本"的风险，且实测长 section 占比 0%~6.2%，出现频率很低。
- 没有单一根标题、通篇是并列顶层章节的文档，章节之间因为约束 3 不会合并。
  这是有意的：跨顶层章节是语义边界。章节**内部**的小节照常合并。

## 待办

- 语料里没有任何公司章程/制度类文档，第一层合并（section 内）对这类文档
  是否真的生效未经实测。拿到真实样本后应补一次 `_merge_small_groups` 的
  触发验证。

## 测试

`tests/unit/test_markdown_chunker.py`：

- section 内：小 group 并入邻居、级联合并、超 `chunk_size` 时拒绝合并。
- 跨 section：合并结果是原文连续切片、被跨过的各节标题留在正文里、
  `heading_path` 取公共祖先、无公共祖先时拒绝合并、超 `chunk_size` 时拒绝
  合并、被拆分过的 section 不参与合并。
- 重名面包屑按 `#n` 编号。
- 真实文档：每个 chunk 都是原文的连续子串；`detect_structure_violations`
  的 `truncated_regions` 为 0。
- 测标题层级/面包屑本身的用例统一传 `min_chars=0` 关掉合并来隔离被测行为，
  默认配置下的合并结果另有专门用例覆盖。
