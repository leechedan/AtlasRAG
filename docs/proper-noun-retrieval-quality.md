# 专有名词检索质量评估与改进

日期：2026-08-26

## 背景

评估当前向量检索链路对专有名词/术语的检索质量，并对发现的短板做快速修复。

## 检索链路现状

`app/api/routes/search.py` → `search_text_hybrid`（`app/store/qdrant_client.py`）：

- Embedding 模型：`BAAI/bge-m3`（`app/models/embeddings.py`），一次编码同时产出
  **dense（1024维）** 和 **sparse（学习型词权重，类似 SPLADE）** 两路向量。
- Qdrant 用两路 `Prefetch` 各取 `top_k*2`，再用 **RRF（Reciprocal Rank Fusion）** 融合排序。
- Chunk 大小 1000 字符（`app/config.py`），按 markdown 结构切分（`app/ingest/markdown_chunker.py`）。

## 识别出的四个短板

1. **无 reranker**：RRF 融合是排名层面的粗融合，没有 cross-encoder 对
   查询词与 chunk 文本做精确重合度打分。
2. **无同义词/别名扩展**：专有名词的缩写、全称、译名等变体写法容易漏召回。
3. **sparse 通道依赖 BGE-M3 自带的 XLM-RoBERTa 分词器**：专有名词被切成生僻
   子词时，词权重区分度会打折扣，不如传统 BM25 + 精确分词稳定。
4. **无精确短语/子串匹配兜底**：向量检索对生僻实体容易语义模糊化，且缺少
   任何"原文逐字命中就必须召回"的保底机制。

## 已实施的快速修复

### 1. MinerU 输出的 `<sub>`/`<sup>` 标签清洗

`app/ingest/parsers/pdf_parser.py`：MinerU 对化学式、脚注等场景会在
`md_content` 里插入原生 HTML 标签（如 `H<sub>2</sub>O`）。这些标签本身不是
正文语义的一部分，留在 chunk 里会污染 embedding 和 sparse 关键词匹配（包括
下面新增的全文匹配兜底）。在 `parse_pdf_to_markdown` 解析边界处用正则去除
标签本身、保留标签内的字符：`H<sub>2</sub>O` → `H2O`。

### 2. 短板 #4：精确短语匹配兜底

- `ensure_collections`（`app/store/qdrant_client.py`）为 `text_chunks` 集合
  的 `text` payload 字段建立多语言全文索引（`TextIndexParams` +
  `TokenizerType.MULTILINGUAL`）。
- `search_text_hybrid` 新增第三路 `Prefetch`：用 `MatchText` 过滤器限定
  "原文必须逐字包含查询词"，在这个子集里按 dense 语义排序，再和原有的
  dense/sparse 两路一起做 RRF 融合。
- `search_text_hybrid` 新增可选参数 `query_text`，由
  `app/api/routes/search.py` 的 `/search/text` 路由把原始查询字符串透传进来
  （之前只传向量，拿不到原文用于过滤）。
- 效果：只要专有名词在文档原文里逐字出现，即使 dense 语义排序把它排得靠后、
  sparse 权重因子词切分被稀释，这一路也能把它捞回来参与最终排序，不再完全
  依赖向量相似度。

## 尚未实施（后续可选）

- **#1 reranker**：可复用现有 `FlagEmbedding` 依赖加载
  `BAAI/bge-reranker-v2-m3`，对 RRF 融合后的 top 20~30 候选做二次打分。
  会增加一次模型前向推理的延迟，建议先看 #4 上线后的效果再决定是否需要。
- **#2 同义词/别名扩展**：没有通用的快速方案；应急做法是维护一个领域别名
  表，查询前做字符串扩展成多个 query 变体一起检索。
- **#3 sparse 通道子词切分限制**：和 #4 做的事情有重叠（都是为了捞精确词
  面匹配），#4 上线后大概率没必要单独处理。

## 验证

- `python3 -m pytest tests/unit -q`：185 passed。
- 新增测试：`tests/unit/test_pdf_parser.py::test_parse_pdf_to_markdown_strips_sub_sup_tags`。
- 现有测试 `tests/unit/test_qdrant_store.py` 在内存版 Qdrant 上验证了
  `ensure_collections` 新增的全文索引创建不会报错、`search_text_hybrid`
  行为不受影响（`query_text` 为可选参数，默认 `None`，不传入等价于原行为）。
