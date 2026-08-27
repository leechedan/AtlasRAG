# 两阶段 Rerank 设计

日期：2026-08-27

## 背景

`docs/retrieval-relevance-filtering.md` 讨论过检索结果相关性问题，核心结
论是 RRF 融合分数是排名的函数、不是相似度，既不能跨查询标定绝对阈值，
排序精度本身也有限（见方案 A：引入 cross-encoder reranker 对 RRF top-N
候选做二次打分）。该文档写作时 A/B 两个方案均未实施。

本设计落地方案 A 的排序改进部分（暂不做硬阈值过滤，见"范围"一节），并
针对高并发场景采用轻量级 Stage1（复用现有 RRF 分数，零额外模型成本）+
重量级 Stage2（cross-encoder 精排，用信号量限制并发）的两阶段架构，避
免让 cross-encoder 参与全量候选排序或无限制并发推理拖垮服务。

## 范围

- 只作用于 `/search/text` 的文本链路（`search_text_hybrid`）。
- 图片链路（CLIP `search_images`）和 `/search/image` 的
  `related_text_hits`（本身没有相关性排序语义，是 doc_id 直查）不受影响
  ——cross-encoder 是文本对文本打分，不适用于图片检索。
- 不做硬阈值过滤/拒绝回答：只替换排序和分数语义，`docs/retrieval-
  relevance-filtering.md` 里"完全不相关时接口该返回什么"仍待后续按需求
  排期（本设计让分数变得可比，为以后做阈值判断打基础，但本次不做）。

## 架构与数据流

```
查询 → embed → search_text_hybrid(pool_size=候选池大小N)  ← Stage 1：现有 RRF，零改动成本
                        ↓ 候选池 N 条 (payload.text)
              apply_rerank(query, hits, top_k)             ← Stage 2：cross-encoder 精排
                        ↓ 信号量限制并发 GPU 推理
                最终 top_k 条（score = reranker 分数，可跨查询比较）
```

- **Stage 1（轻量，复用现有能力）**：`search_text_hybrid` 的职责从"直接
  产出 top_k"改为"产出更大的候选池"（`pool_size`，默认 30，对齐
  `docs/retrieval-relevance-filtering.md` 里"20~30"的建议）。RRF 融合逻
  辑本身不变，只是把截断点从 `top_k` 挪到 `pool_size`。这一步零额外模型
  推理成本，天然适合高并发。
- **Stage 2（重量，新增）**：对候选池里每条的 `payload["text"]` 与
  query 用 `BAAI/bge-reranker-v2-m3` 做 cross-encoder 打分，重新排序后
  截断为 `top_k`，替换掉不可比的 RRF 分数。这一步有推理成本，用并发信
  号量限流而不是跨请求 batching，实现复杂度和现有模型加载方式保持一致。

## 分层设计（新增/改动文件）

1. **`app/models/reranker.py`（新增）**——模型封装层，与
   `app/models/embeddings.py`、`app/models/clip_model.py` 同构：
   - `Reranker` 类：加载 `FlagReranker(model_name, use_fp16=True,
     device=...)`；`.score(query: str, texts: list[str]) -> list[float]`，
     内部用 `threading.Semaphore(settings.reranker_max_concurrency)` 包住
     实际推理调用（FastAPI 的同步 `def` 路由本身跑在线程池里，并发请求
     是真实线程，用 `threading.Semaphore` 而非 asyncio 版本）；调用
     `compute_score(pairs, normalize=True)` 做 sigmoid 归一化，让分数落
     在 0~1、可跨查询比较。
   - `get_reranker()`：`@lru_cache(maxsize=1)` 单例，与现有两个模型加载
     方式一致，只在 API 进程常驻（reranker 只在检索路径用，不参与摄取
     写入，不需要像 BGE-M3/CLIP 那样在 worker 进程里也加载一份）。

2. **`app/retrieval/rerank.py`（新增包）**——编排/业务逻辑层，与
   `app/ingest/pipeline.py` 的定位类似（不是模型，也不是存储，是"拿到候
   选后做什么"的逻辑）：
   - `apply_rerank(query: str, hits: list[SearchHit], reranker: Reranker,
     top_k: int) -> list[SearchHit]`：取 `payload["text"]`、调用
     `reranker.score`、按分数降序排序、截断 `top_k`，返回替换了 `score`
     字段的新 `SearchHit` 列表。
   - 纯函数、接收已加载好的 `reranker` 对象（依赖注入进来，而不是内部调
     用 `get_reranker()`），可脱离 FastAPI 单独单测。

3. **`app/store/qdrant_client.py::search_text_hybrid`**——新增可选参数
   `pool_size: int | None = None`（默认等于 `top_k`，向后兼容），控制
   RRF 最终 `limit` 和各路 prefetch 的 `limit`（`pool_size * 2`），只改
   候选池大小，不改融合逻辑本身。

4. **`app/api/deps.py`**——新增 `get_reranker_dep()`，与
   `get_text_embedder_dep`/`get_clip_embedder_dep` 同构，方便路由测试里
   override。

5. **`app/api/routes/search.py::search_text`**——编排调整：
   ```python
   if "text" in request.modalities:
       query_vector = text_embedder.embed([request.query])[0]
       pool_size = settings.reranker_candidate_k if settings.reranker_enabled else request.top_k
       text_hits = search_text_hybrid(store, query_vector, query_text=request.query,
                                       top_k=request.top_k, pool_size=pool_size)
       if settings.reranker_enabled and text_hits:
           text_hits = apply_rerank(request.query, text_hits, reranker, top_k=request.top_k)
   ```

6. **`app/config.py`**——新增配置项：
   - `reranker_enabled: bool = True`（总开关，出问题可一键关回单阶段
     RRF，呼应 `docs/retrieval-relevance-filtering.md` 里"是否要做尚未
     决定"的谨慎态度）
   - `reranker_model_name: str = "BAAI/bge-reranker-v2-m3"`
   - `reranker_candidate_k: int = 30`（Stage1 候选池大小）
   - `reranker_max_concurrency: int = 2`（Stage2 信号量并发上限）

## 并发处理细节

- 不做跨请求 micro-batching；每个请求的候选池（≤30 条 `(query, text)`
  pair）在自己的线程内一次性喂给 `reranker.compute_score`（FlagEmbedding
  内部已对这批 pair 做批处理）。
- `threading.Semaphore(reranker_max_concurrency)` 挡在实际推理调用外
  层：超过并发上限的请求会在信号量上排队等待，而不是同时挤爆 GPU 显存
  或互相拖慢推理。
- 分数语义变化：响应里 `score` 字段从"结构性不可比的 RRF 分数"变成
  "0~1、可跨查询比较的 cross-encoder sigmoid 分数"，是有意的行为变化。

## 测试

- `tests/unit/test_api_search.py`：沿用现有 `monkeypatch` +
  `dependency_overrides` 模式，新增用例覆盖：reranker 打分后顺序变化、
  `top_k` 截断正确、`reranker_enabled=False` 时跳过 Stage2（走原有单阶
  段路径）。
- `app/retrieval/rerank.py` 的 `apply_rerank` 单独写单测（纯函数，传入
  假的 `SearchHit` 列表和一个打桩的 `score()` 返回值，断言排序/截断逻
  辑），不依赖真实模型。
- `app/models/reranker.py` 参照 `tests/unit/test_embeddings.py` /
  `test_clip_model.py` 的既有约定，加一个 `pytest.mark.slow`（本地默认
  跳过，CI/上线前手动跑）的真实模型测试，验证 `get_reranker()` 单例、以
  及一条明显相关和一条明显不相关的 `(query, text)` 对分数确实拉开差
  距，不需要额外新增基建。

## 文档更新

实施完成后需要回写 `docs/retrieval-relevance-filtering.md` 的"尚未实
施"结尾，说明方案 A 的排序部分已落地（本设计文档），阈值过滤仍未决
定/未做。
