# API 多 worker 进程改造设计

日期：2026-08-29

## 背景与问题

`docs/concurrency-benchmark.md` 的结果六/结果七发现：关闭 rerank
（`RERANKER_ENABLED=false`）后，`/search/text` 的吞吐**随并发反向劣化**
——c=1 时 54.77 RPS，c=64 时只剩 26.77 RPS，同时 GPU 和 CPU 利用率都低
到个位数。也就是说服务既没有用满算力，也没有获得任何并发扩展能力。

分层隔离压测（结果七）定位到真实原因**不是**锁：

| 层 | c=1 | c=32 | 表现 |
|---|---|---|---|
| `embed()`（持 `threading.Lock`） | 86.4 | 78.2 | 平的，不劣化 |
| `search_text_hybrid()` | 452 | 693 | 正常扩展 |
| HTTP/ASGI 栈 | 1458 | 970 | 远高于被测值 |
| 完整 handler + JSON 序列化 | 64.9 | 34.2 | **反向劣化** |

没有任何单独一层是瓶颈，劣化产生于组合本身。根因是 **GIL convoy**：服务
跑在**单个 uvicorn 进程**里（`docker-compose.yml:65` 无 `--workers`），
而宿主机有 20 个核；压测全程 CPU 稳定在 4.5~6%，即 `6% × 20 ≈ 1.2` 个核
——恰好一个核跑满，这是 GIL 饱和的指纹。持锁线程进入 `encode()` 后
torch 释放 GIL 等 GPU，其余线程趁机抢 GIL 做响应序列化（每个响应
**16.8KB**，10 个 chunk 全文，纯 Python 工作）；GPU 算完后持锁线程要
拿回 GIL 才能继续，得排队等 `sys.setswitchinterval` 默认的 5ms。**临界区
被拖长，串行上限本身随并发下降。**

因此本设计要解决的是"一个 Python 进程只用得上一个核"，而不是那两把锁。

## 目标与非目标

**目标**：让 `/search/text` 在关闭 rerank 时具备真实的并发扩展能力，
把吞吐从 ~26 RPS 提升到 200 RPS 量级，且不引入新的正确性风险。

**非目标**：
- 不提升 rerank 打开时的吞吐。那条路的约束是 GPU 算力
  （`docs/concurrency-benchmark.md` 结论 1、3），加并发无效，得靠降
  `candidate_k` / 换小模型 / 截断候选文本。
- 不做 embed 微批合并调度。见下方"已否决的方案"。
- 不改动 `worker`（RQ ingest）服务，它的进程模型独立且不受影响。

## 方案：uvicorn 多 worker 进程

把 `docker-compose.yml` 的 api 启动命令从

```
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

改为

```
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS:-4}
```

并在 api 服务的 `environment` 里加 `API_WORKERS: ${API_WORKERS:-4}`，
使其可按部署环境覆盖。

**应用代码零改动。** `app/models/embeddings.py` 和
`app/models/reranker.py` 里的两把 `threading.Lock()` 原样保留——它们在
各自进程内继续保证 FlagEmbedding / HF tokenizer 的线程安全，而进程隔离
让它们不再构成全局瓶颈。

### 为什么安全：两项已验证的前提

**1. uvicorn 用 spawn 而非 fork，不存在 CUDA-after-fork 风险。**
读装在容器里的 uvicorn 0.32.1 源码确认，`uvicorn/_subprocess.py:17` 是
`spawn = multiprocessing.get_context("spawn")`，`get_subprocess()` 返回
`spawn.Process(...)`。每个 worker 是全新解释器，独立 import 应用、独立
执行 `app/api/main.py` 的 lifespan 加载模型。`app/worker.py:21-24` 记录
的 "CUDA-after-fork hazard" 在这里不适用。

**2. 没有任何跨进程共享状态。** 全部长生命周期对象都是
`@lru_cache(maxsize=1)` 的进程内单例——`get_text_embedder`
（`app/models/embeddings.py:38`）、`get_clip_embedder`
（`app/models/clip_model.py:34`）、`get_reranker`
（`app/models/reranker.py:49`）、`get_default_client`
（`app/store/qdrant_client.py:387`）。没有模块级可变全局状态，也没有
任何"单进程"假设。Qdrant / Redis 连接本就是每进程一份。

**3. `/health` 天然适配多 worker。** `app/api/routes/health.py:27-31` 用
`cache_info().currsize == 1` 判断本进程的模型是否已就绪，未就绪返回
503。这个语义按进程独立成立，多 worker 下每个 worker 各自如实汇报。

### 预期收益

| 场景 | 现状 | 4 workers（预期） | 4 workers（实测） |
|---|---|---|---|
| rerank 关闭 | 26 RPS @ c=64 | 200~260 RPS | **111.5 RPS**（预测偏高） |
| rerank 打开 | 6.5 RPS | ~8~9 RPS | **7.2~7.4 RPS**（基本吻合） |

预测依据：单进程完整 handler 65 RPS × 4；rerank 打开时 GPU 已 67~85%，
只能榨出剩余余量。**rerank 关闭那一格的预测偏高约 2 倍**，原因见文末
"实施结果"。

### 代价与风险

1. **内存**：1.8GB × 4 ≈ 7.2GB（rerank 打开约 10GB），相对 121.6GB
   可忽略。
2. **冷启动变慢（风险很小）**：每个 worker 独立加载 BGE-M3 +
   Chinese-CLIP (+ reranker)，4 份并发加载会争抢磁盘和 GPU 初始化。但
   实测权重已缓存时单进程只要 BGE-M3 1.8s + CLIP 3.7s ≈ 5.5s，而
   `docker-compose.yml:121` 的 `healthcheck.start_period` 已经是 **120s**，
   余量充足，**预计无需调整**。仍需在验证步骤中确认实际冷启动时间。
3. **启动期 healthcheck 间歇 503**：端口由 4 个 worker 共享，健康检查
   可能命中尚未加载完的 worker。语义上是正确的（确实没就绪），且落在
   120s 的 `start_period` 窗口内不会被判定为不健康。验证时确认一次即可。
4. **rerank 打开时是"吞吐换延迟"**：4 个进程同时抢 GPU，单请求 p50 会
   比现在差，而吞吐只涨一点。**如果最终决定生产开启 rerank，应重新
   评估 worker 数是否要降回 1~2。** 这也是验证计划要把两种情况都扫一遍
   的原因。

## 已否决的方案

**embed 微批合并（micro-batching）**：把并发的单 query embed 攒成一批
跑。实测 batch=16 时每条 11.57ms → 1.16ms，能把单进程 embed 上限从
86 RPS 抬到 800+ RPS。否决理由有二：(a) 它**修不掉 GIL convoy**——
16.8KB 响应的序列化仍是 GIL-bound，单进程实际收益会被吃掉大半；
(b) 需要一整套排队/超时窗口/异常传播逻辑，复杂度远高于一行配置。
且多 worker 落地后各进程并发度下降，攒批更难攒满，收益进一步缩水。

**削减每请求 GIL 工作量**（换 `ORJSONResponse`、裁剪响应里的 chunk
全文）：便宜但治标，预计只能把 65 RPS 抬到 ~90 RPS。可作为多 worker
之后的后续优化，不在本次范围。

## 验证计划

1. 改动前记录基线（已完成，见 `docs/concurrency-benchmark.md` 结果六）。
2. 改为 `--workers 4`，`docker compose up -d api`，等全部 worker 就绪。
3. **rerank 关闭**下跑
   `python3 scripts/bench_search_concurrency.py --levels 1,2,4,8,16,32,64
   --requests-per-level 40`，确认吞吐随并发上升且不再反向劣化，目标
   200+ RPS。
4. 把 `RERANKER_ENABLED` 改为 `true` 重启，同样扫一遍，记录吞吐涨幅和
   p50 延迟的劣化幅度，作为"生产要不要开 rerank / 开了之后 worker 数
   定多少"的决策依据。
5. 确认冷启动期间 compose 不会因 healthcheck 失败而重启 api，并记录
   4 worker 实际冷启动耗时（判断 `start_period: 120s` 是否仍然够用）。
6. 两次结果作为「结果八」补进 `docs/concurrency-benchmark.md`，并更新
   结论摘要。

## 实施结果（2026-08-29，已完成）

改动已落地：`docker-compose.yml` 的 api `command` 加
`--workers ${API_WORKERS:-4}`，`environment` 增加 `API_WORKERS`，并把
`RERANKER_ENABLED` 一并改为 `${RERANKER_ENABLED:-false}` 便于压测切换。
**应用代码零改动**，两把 `Lock` 原样保留。完整数据见
`docs/concurrency-benchmark.md` 结果八。

**达成**：
- rerank 关闭峰值 **26.8 → 111.5 RPS（约 4.2 倍）**，GPU 利用率从个位数
  涨到 90%——GIL 瓶颈确已消除。
- 两种配置共 2940 个请求**零错误**，`RuntimeError: Already borrowed`
  没有回归，验证了 spec 里"spawn 而非 fork + 进程内单例"的安全性论证。
- 冷启动 14s，远小于 `start_period: 120s`，风险 2/3 未发生，无需调整。
- worker 数扫描（2/4/8）确认 4 是拐点：8 个 worker 吞吐不再增长
  （109.5 vs 111.5）却要 2.6 倍内存（14.5GB vs 5.5GB）。

**未达成 / 判断失误**：rerank 关闭下预测 200~260 RPS，实测只有 111.5。
预测是拿单进程 65 RPS 线性外推 ×4，隐含假设"GPU 还有余量"——这个假设
错了。单进程时 GPU 平均只有个位数，看起来很闲，但那是**因为 GIL 让请求
到不了 GPU**，不代表 GPU 有 4 倍余量。实际上单条 query 走的是 `batch=1`
的 embed 调用，GPU 效率极低（batch=1 每条 11.57ms，batch=16 每条
1.16ms），4 个进程各自发 `batch=1` kernel 很快就把 GPU **占满**，而
占满 ≠ 算得多。

教训：**从"某资源利用率低"推断"该资源有余量"是不成立的**，当存在上游
串行瓶颈时利用率低恰恰是瓶颈的症状。正确做法是先解除上游瓶颈再测该
资源的真实上限——这也正是结果七/结果八两步走验证出来的。

**后续**：GPU 现在饱和在低效的 `batch=1` 调用上，因此本文档"已否决的
方案"里的 **embed 微批合并需要重新评估**——当时否决理由是"收益会被
GIL convoy 吃掉"，而该前提已不复存在。若最终决定生产开启 rerank，则
应重新评估把 `API_WORKERS` 降回 1~2 以换回尾延迟（rerank 打开时
4 worker 只换来 +12% 吞吐，却让 c=64 的 p95 从 7342ms 涨到 12134ms）。
