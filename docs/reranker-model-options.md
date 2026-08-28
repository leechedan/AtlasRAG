# Rerank 模型轻量化选型

日期：2026-08-27

## 背景

`docs/concurrency-benchmark.md` 的压测结论摘要第 3 点把"换更小的 rerank
模型"列为 `reranker_candidate_k` 之外，继续提升 RPS 的下一个方向：这个
服务是纯 GPU 算力瓶颈（单请求就能把 GPU 打到 88%+ 利用率），rerank 这一
步占了单请求耗时的 90%+，而 dtype（fp16/bf16）在这块 GPU 上不影响吞吐
（同一份文档结果五）。模型本身的计算量（参数量/层数/hidden size）是除
了候选数量之外唯一还没验证过的杠杆。

本文档只做选型分析，**不包含实测数据**——两个候选都还没有跑过
`app/eval` 的检索质量评估，也没有跑过 `scripts/bench_search_concurrency.py`
的吞吐对比，替换前必须先做这两项验证。

## 当前模型

`BAAI/bge-reranker-v2-m3`（`app/config.py:19` 默认值）：XLM-RoBERTa-large
架构，24 层，约 568M 参数，多语言（含中文），是 BAAI 专门为质量做的
升级版本。

## 候选一：`BAAI/bge-reranker-base`

- XLM-RoBERTa-base 架构，12 层，约 278M 参数——约为当前模型的一半。
- 同厂商同系列，加载方式与当前完全一致（`app/models/reranker.py` 里的
  `FlagReranker(model_name, ...)` 不用改代码，只改
  `RERANKER_MODEL_NAME`），兼容性风险最低。
- 代价：这是 BAAI 更早期的模型，中英文 reranking 质量比 v2-m3 有实测
  差距（v2-m3 就是针对 base 版本的质量问题做的升级），排序精度会打
  折扣，需要用 `app/eval` 确认降幅是否可接受。

## 候选二：`Alibaba-NLP/gte-multilingual-reranker-base`

- 2024 年发布，约 306M 参数，支持 70+ 语言。
- 官方 benchmark（C-MTEB reranking 等）显示与 `bge-reranker-v2-m3` 质量
  相当甚至部分指标更优，同时参数量小了近一半——如果这个结论在我们自己
  的数据上也成立，是"质量不掉、速度更快"的更优选项。
- 风险：不是 BAAI 生态模型，能否被 `FlagReranker` 直接加载还没验证过，
  可能需要 `trust_remote_code=True`，或者绕开 `FlagReranker` 直接用
  transformers 的 `AutoModelForSequenceClassification` 加载（意味着
  `app/models/reranker.py` 需要改代码，不是纯配置切换）。**替换前第一步
  必须先验证加载兼容性**，再谈质量/速度对比。

## 替换前必须做的验证（未完成）

1. 加载兼容性：两个模型能否用现有 `Reranker` 类直接加载（尤其是候选二）。
2. 质量：`app/eval` 的检索质量评估跑一遍，对比当前 `bge-reranker-v2-m3`
   基线，确认召回/排序没有明显掉分。
3. 速度：`scripts/bench_search_concurrency.py` 跑一遍并发梯度扫描，对比
   `docs/concurrency-benchmark.md` 结果四（`reranker_candidate_k=10` 时
   RPS ~6.5）的基线，量化实际 RPS 提升幅度。

在以上验证完成前，`.env.example` 里只把这两个模型列为"可尝试的备选
项"（注释形式），默认值不变。
