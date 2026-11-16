# 分片质量与评估

本文说明 AtlasRAG 如何量化和改进文本分片质量，适用于摄入链路的字符分片器和结构化 Markdown 分片器。

## 质量目标

高质量分片应保留回答查询所需的局部上下文，长度适合向量化与重排，保留文档结构，并避免不必要的重叠重复。

对于 Markdown，标题提供章节上下文；表格和围栏代码块应尽量完整；重叠应弥补人为边界处的上下文，但不应重复受保护的表格或代码块内容。

## 已实现指标

使用 `scripts/eval_chunk_quality.py` 递归评估 `.md`、`.txt` 文件，并输出 JSON 报告。

```bash
# 只运行静态结构和长度检查，不加载 embedding 模型。
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --no-semantic \
  --output /tmp/chunk-quality.json

# 加入句子级语义连贯度，需要已配置的 embedding 模型。
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --output /tmp/chunk-quality-semantic.json
```

| 指标 | 定义 | 解读 |
| --- | --- | --- |
| `length_min_chars`、`length_max_chars`、`length_mean_chars`、`length_median_chars`、`length_stdev_chars` | 分片字符长度分布 | 发现过短、过长或波动过大的分片。 |
| `length_out_of_range_ratio` | 超出 `min_chars..max_chars` 目标区间的分片比例 | 通常越低越好；刻意保留的超长结构需单独判断。 |
| `truncated_region_ratio` | 未被单个分片完整包含的 Markdown 表格、代码块比例 | 结构化 Markdown 分片的目标为零。 |
| `mean_coherence` | 可评分分片内相邻句子的平均余弦相似度 | 仅作诊断；高相似度也可能意味着内容重复。 |
| `recall_at_k`、`mrr` | 基于标注问题的检索召回率和排名质量 | 选择分片配置的首要依据。 |

默认情况下，`min_chars` 是 `chunk_size` 的 20%，`max_chars` 等于 `chunk_size`。服务默认 `CHUNK_SIZE=1000`、`CHUNK_OVERLAP=150`，实验可覆盖这些参数。

## 检索评估

传入 `--qa-file` 后可计算 `Recall@5`、`Recall@10` 和 MRR。标注文件是如下 JSON 数组：

```json
[
  {
    "query": "合同续签通知如何处理？",
    "expected_chunk_ids": ["<chunk-id>"]
  }
]
```

`expected_chunk_ids` 使用由 `doc_id` 和 `chunk_index` 计算出的 Qdrant chunk ID。

```bash
python3 -m scripts.eval_chunk_quality \
  --input-dir data/uploads \
  --chunker structured \
  --qa-file qa-gold.json \
  --output /tmp/chunk-quality-retrieval.json
```

**重要：** 当前 CLI 查询的是已经写入 Qdrant 的集合。仅修改 `--chunk-size`、`--chunk-overlap` 或 `--chunker` **不会**重建索引。有效的检索 A/B 测试必须为每套候选配置将相同语料重新摄入独立索引，并查询匹配的索引。重新切分后 ID 会变化，因此长期金标除当前 chunk ID 外，还应记录稳定的源位置或答案原文片段。

## 推荐评估流程

1. 固化有代表性的版本化语料，并准备至少 50–100 条真实用户问题；记录答案原文位置，而不仅是当前 chunk ID。
2. 选择候选配置，例如字符分片与结构化分片，以及多组大小/重叠参数。
3. 将相同语料重新摄入彼此隔离的索引，每个配置对应一个索引。
4. 先通过静态门槛：原文完全覆盖、表格/代码不截断、短分片比例可接受、重叠重复受到控制。
5. 在同一金标问题集上计算 `Recall@5`、`Recall@10`、MRR，并将延迟、索引大小和 token 成本作为运行约束比较。
6. 在采用某配置前，人工复核未命中的问题和切分边界案例。

## 当前限制与后续指标

- **标题上下文：** 当前评估器检查正文是否含标题；结构化分片将标题存为 `heading_path` 元数据，因此 `missing_heading_ratio` 可能产生假阳性。在将其作为发布门槛前，应直接评估 `heading_path`。
- **区间定位：** 结构检查通过顺序文本搜索定位 chunk；重复文本或注入的 overlap 会造成歧义，切片器输出精确源偏移量会更可靠。
- **字符与 token：** 当前长度按字符计量；生产限制还应使用实际 embedding/reranker 的 tokenizer。
- **语义连贯性：** 当前拆句主要面向中文句末标点，连贯度也不是检索有效性的通用指标。

建议补充原文覆盖率、重叠/重复率、断句/断段率、标题元数据完整率和 token 长度分位数。这些属于诊断或安全指标；稳定金标集上的检索效果仍是最终决策指标。

## 相关实现

- CLI：`scripts/eval_chunk_quality.py`
- 评估编排：`app/eval/runner.py`
- 结构、长度、连贯度和检索指标：`app/eval/`
- 结构化 Markdown 分片器：`app/ingest/markdown_chunker.py`
