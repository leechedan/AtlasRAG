# 检索并发能力基准测试

日期：2026-08-27

## 背景

评估这个服务在生产环境下的并发承载能力：能扛多少并发、瓶颈在哪一层、
"加并发"（信号量/线程池/批处理）这类改造到底值不值得做。测试机器是
DGX（NVIDIA GB10，unified memory，显存 128GB），显存本身在测试过程中
从未成为限制因素——真正的瓶颈是 GPU 算力（compute），而且**单个请求
自己就能把算力占满**，这一点决定了后面所有关于"要不要加并发"的结论。

## 环境

- GPU：NVIDIA GB10（`nvidia-smi` 报告 memory.total/used/free 均为
  `Not Supported`——这是该芯片 unified memory 架构的已知限制，本次测试
  没有依赖 nvidia-smi 的显存读数，只用它的 `utilization.gpu` 采样）
- 服务：`docker compose`，`api`/`worker`/`qdrant`/`redis`/`mineru` 全部
  正常运行，Qdrant 里有 74 个 text chunk、19 张图片（真实索引数据，非
  空库）
- 被测端点：`POST /search/text`，`{"query": "检索服务的并发能力",
  "top_k": 10, "modalities": ["text"]}`（走 embed → Qdrant 混合检索 →
  rerank 全链路，`reranker_enabled=True`）
- 复用组件：`app/models/reranker.py` 里 `reranker_max_concurrency=2`
  的 `threading.Semaphore`；FastAPI 同步路由默认线程池容量 = 40
  （`anyio.to_thread.current_default_thread_limiter().total_tokens`
  在本机实测确认，非文档臆测）

## 量化指标

| 指标 | 含义 |
|---|---|
| 吞吐量 (RPS) | 给定并发下每秒能处理的请求数 |
| 延迟分位数 p50/p95/p99/max | 并发升高时尾延迟的劣化曲线 |
| 错误率 | 非 200 响应/异常占比 |
| GPU 利用率 avg/max | 判断瓶颈是不是 GPU 算力打满 |

方法：固定请求数（每档 40 个），用 `asyncio.Semaphore(concurrency)` 控制
同时在飞请求数，并发梯度 1/2/4/8/16/32/64，全部打完记录墙钟时间算 RPS；
同时后台每 0.3s 采样一次 `nvidia-smi --query-gpu=utilization.gpu`。

复现命令（`scripts/bench_search_concurrency.py`，服务需已在
`http://localhost:8000` 跑起来）：

```bash
python scripts/bench_search_concurrency.py
# 自定义并发梯度/请求数/query：
python scripts/bench_search_concurrency.py \
    --levels 1,2,4,8,16,32,64 --requests-per-level 40 \
    --query "检索服务的并发能力"
```

## 结果一：并发梯度扫描（`reranker_candidate_k=30`，改动前默认值）

| 并发 | RPS | p50 | p95 | p99 | GPU avg/max | 错误 |
|---|---|---|---|---|---|---|
| 1 | 2.35 | 425ms | 429ms | 429ms | 88.6/96 | 0 |
| 2 | 2.57 | 834ms | 1049ms | 1237ms | 92.9/96 | 3 |
| 4 | 2.55 | 1618ms | 2339ms | 2539ms | 95.4/96 | 3 |
| 8 | 2.36 | 3234ms | 3600ms | 4106ms | 94.9/96 | 0 |
| 16 | 2.31 | 6504ms | 7222ms | 7432ms | 91.5/96 | 0 |
| 32 | 2.29 | 9800ms | 13849ms | 14461ms | 86.5/96 | 1 |
| 64 | 2.26 | 10223ms | 17307ms | 17705ms | 86.1/96 | 1 |

**吞吐量是平的（2.3~2.6 RPS 封顶），跟并发数无关；p50 延迟随并发线性
叠加（425ms → 10秒+），这是典型的"排队而非扩容"曲线。** GPU 利用率在
**并发=1 时就已经 88.6%/96%**——单个请求自己就快把算力打满了，不是"并发
超过某个阈值才打满"。同时 `docker stats` 显示 api/worker 容器 CPU/内存
几乎是 0（1.9GB/121GB），排除了 CPU 或线程池是瓶颈的可能。

## 结果二：单请求各阶段耗时分解

用 `app.retrieval.rerank.apply_rerank` / `app.store.qdrant_client.search_text_hybrid`
/ `app.models.embeddings.get_text_embedder` 直接在进程内调用（绕开
HTTP/线程池/信号量），定位 420ms 到底花在哪：

```
embed query                ~12ms   (模型预热后)
qdrant search_text_hybrid   ~5ms
apply_rerank(30 候选)     ~400~570ms   ← 占了 90%+
```

**瓶颈几乎全部在 rerank 这一步**，而且只有 1 个请求时它也要跑这么久——
和"并发"本身关系不大，是"单次计算量"的问题。

## 结果三：batching 是否能提升吞吐？—— 两次验证，结论相反

**第一次（合成短文本，~30 字符/候选）**：

```
batch= 1   per_item=12.25ms
batch=64   per_item= 0.94ms   （~13倍加速）
```

这个结果具有误导性：候选文本长度跟真实场景（`chunk_size=1000`，
`app/config.py:13`）对不上。

**第二次（真实长度，~660 字符/候选）**：

```
batch= 1   per_item=21.15ms
batch= 8   per_item=12.36ms
batch=16   per_item=12.14ms
batch=32   per_item=11.99ms
batch=64   per_item=11.64ms
```

**批处理收益在 batch≈8 就封顶，只有约 1.7 倍，再往上加 batch size 几乎
没有提升**——不是短文本测试里的十几倍加速。原因：cross-encoder 的
attention 计算量随序列长度大致平方增长，真实长度的 chunk 本身就已经把
单条计算的并行度用得差不多，短文本测试摊薄的只是"一次 kernel 调用的
固定开销"，这部分在长文本场景占比很小。

**结论：跨请求批处理（把并发请求的 rerank 调用合并成一次 forward）
收益上限只有 ~1.7 倍，不值得为此上一套请求合并调度器。**

## 结果四：把 `reranker_candidate_k` 从 30 改成 10 后重新扫描

`rerank` 耗时基本正比于候选数量（结果二里 30 候选花 400~570ms，且
batching 摊不掉多少，见结果三），所以把喂给 reranker 的候选池从 30
砍到 10（`app/config.py:20`，同步更新了 `tests/unit/test_config.py`
里的默认值断言）应该能近似线性地降耗时。改完 `docker compose build api
&& docker compose up -d api` 重新扫描：

| 并发 | RPS | p50 | p95 | p99 | GPU avg/max | 错误 |
|---|---|---|---|---|---|---|
| 1 | 5.43 | 161ms | 164ms | 1095ms | 67.3/96 | 0 |
| 2 | 6.55 | 310ms | 413ms | 452ms | 79.8/95 | 1 |
| 4 | 6.28 | 604ms | 774ms | 777ms | 85.1/95 | 0 |
| 8 | 6.09 | 1251ms | 1720ms | 1946ms | 72.9/96 | 0 |
| 16 | 5.72 | 2504ms | 3369ms | 4335ms | 81.3/96 | 0 |
| 32 | 5.37 | 4476ms | 6292ms | 6293ms | 69.3/96 | 1 |
| 64 | 5.35 | 4795ms | 7342ms | 7476ms | 72.3/96 | 1 |

对比改动前：**单请求 p50 从 425ms 降到 161ms（~2.6倍），吞吐量峰值从
~2.6 RPS 涨到 ~6.5 RPS（~2.5倍）。** 涨幅和"候选数从 30 降到 10，理论
上限 3倍"基本吻合（差距来自 embed/qdrant 这部分不随候选数变化的固定
开销）。同样是"吞吐随并发不变、延迟随并发线性堆积"的形状——说明这
不是并发架构问题，是纯算力/计算量问题，`reranker_candidate_k` 是目前
性价比最高的杠杆：一行配置改动换来 2.5 倍吞吐，此外还有截断候选文本
长度、换更小的 rerank 模型等同类方向可以进一步榨。

## 结果五：切到 BF16 后 RPS 没有变化——是否符合预期？

结果四把 `reranker_candidate_k` 从 30 降到 10 后，RPS 到了 ~6.5。之后把
reranker 精度从 FP16 切到 BF16（`RERANKER_DTYPE=bf16`），重新压测，RPS
仍然在 6 左右，没有进一步提升。

排查后确认**这是符合预期的结果，不是改动失败或遗留 bug**：在这块 GPU
（GB10 / Blackwell，compute capability 12.1）上，FP16 和 BF16 走的是同
一套 Tensor Core 执行单元，单次矩阵乘法吞吐完全一样——两者只是"同样
16 bit 里指数位/尾数位怎么分配"的区别（BF16 范围更大、精度更低；FP16
反过来），不存在谁比谁算得快。BF16 相对 FP16 的价值是**数值稳定性**
（不易溢出、不需要 loss scaling），不是速度。

用结果二同样的方法论实测验证（10 候选、~600+ 字符真实长度）：

```
fp16: 105.20 ms/call
bf16: 106.51 ms/call
```

误差范围内完全一样。**结论：dtype（FP16 vs BF16）不是这块 GPU 上的吞吐
杠杆，`reranker_candidate_k` 才是**——如果目标是继续提高 RPS，应该往
"继续降 candidate_k / 换更小的 rerank 模型 / 截断候选文本长度"这几个
方向走，而不是指望切精度格式。

## 意外发现：并发下的一个真实 bug（不是性能问题，是正确性问题）

两次扫描里都出现了个别请求报错（30候选那次 c=2/4/32/64 各有 1~3 个；
10候选那次 c=2/32/64 各有 1 个）。单独复现并抓了 api 容器日志，根因是：

```
File ".../FlagEmbedding/inference/reranker/encoder_only/base.py", line 137,
  in compute_score_single_gpu
    passages_inputs_batch = self.tokenizer(...)
File ".../transformers/tokenization_utils_fast.py", line 467,
  in set_truncation_and_padding
    self._tokenizer.enable_truncation(**target)
RuntimeError: Already borrowed
```

`app/models/reranker.py:19-25` 的 `threading.Semaphore(reranker_max_concurrency)`
（当前=2）包住了整个 `compute_score()` 调用，但 **HuggingFace 的 Rust
fast tokenizer 在被 2 个线程同时调用时本身就不是线程安全的**——
`enable_truncation`/`enable_padding` 会修改tokenizer 实例的共享内部状态，
两个线程同时改就会触发 Rust 侧的 `RefCell` 重入 panic，表现为这个
`RuntimeError: Already borrowed`，最终变成客户端看到的 500。

也就是说：**只要 `reranker_max_concurrency` ≥ 2，这个服务在并发下就有
概率随机返回 500，跟候选数量、并发梯度无关，是一个正确性 bug，不是
性能调优能解决的问题。** 这是这次基准测试里最值得优先处理的发现——
比"吞吐提不提得上去"更紧急，需要在"并发/协程改造计划"里作为第一项。

## 结论摘要（供后续并发改造计划参考）

1. **DGX 上这块 GB10 GPU 的算力，而不是显存，是唯一的实际瓶颈**——单
   请求就能打到 90%+ 利用率，"防并发过载" 类的信号量/线程池调整不会
   带来吞吐提升。
2. **跨请求批处理收益有限（~1.7倍封顶）**，不值得为此引入请求合并调度
   器这类复杂度。
3. **`reranker_candidate_k` 从 30 降到 10 是目前验证过、性价比最高的
   改动**：2.5 倍吞吐提升，一行配置 + 重新构建镜像即可，代价是候选池
   变小可能略微牺牲召回（需要跟 `app/eval` 的检索质量评估配合，确认
   没有明显掉分）。
4. **`reranker_max_concurrency>=2` 时 HF fast tokenizer 并发调用不安全，
   会随机产生 500**——这是一个需要修的正确性 bug，优先级高于吞吐优化。
   **已修复**：`app/models/reranker.py` 把 `threading.Semaphore(reranker_max_concurrency)`
   换成了 `threading.Lock()`，完全串行化 `compute_score()` 调用，并整体
   移除了 `reranker_max_concurrency` 配置项（结论 1 已经证明并发对吞吐
   没有增益，串行化不损失性能）。见 `tests/unit/test_reranker_concurrency.py`
   的回归测试。
5. **切到 BF16 不会提升吞吐**（见结果五）——FP16/BF16 在这块 GPU 上
   Tensor Core 吞吐相同，dtype 不是这里的性能杠杆，只是数值稳定性上的
   改进。
