# Two-Stage Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-stage rerank pipeline to `/search/text` — Stage 1 widens the existing RRF candidate pool at zero extra model cost, Stage 2 reorders that pool with a `bge-reranker-v2-m3` cross-encoder guarded by a concurrency semaphore — improving text search relevance without a single heavy model call per request blocking under high concurrency.

**Architecture:** `app/models/reranker.py` adds a `Reranker` model wrapper (same singleton pattern as `embeddings.py`/`clip_model.py`) with a `threading.Semaphore` around inference. `app/store/qdrant_client.py::search_text_hybrid` gains an optional `pool_size` param so Stage 1 (RRF, unchanged logic) can return more than `top_k` candidates. `app/retrieval/rerank.py` is a new pure-function orchestration layer (`apply_rerank`) that scores the pool and truncates to `top_k`. `app/api/routes/search.py` wires the two stages together behind a `settings.reranker_enabled` flag.

**Tech Stack:** FastAPI, Qdrant (`qdrant-client`), `FlagEmbedding.FlagReranker` (`BAAI/bge-reranker-v2-m3`), pytest.

## Global Constraints

- Only `/search/text`'s text hits are affected. `search_images` (CLIP) and `/search/image`'s `related_text_hits` are out of scope — cross-encoder reranking is text-only.
- No hard score threshold / filtering in this plan — only reordering and score replacement. See `docs/retrieval-relevance-filtering.md` for the still-undecided filtering question.
- `pool_size` on `search_text_hybrid` must default to `None` (→ behaves exactly as `top_k`, i.e. today's behavior) so the two existing callers (`scripts/eval_chunk_quality.py`, and `search_text_hybrid` calls in tests) keep working unchanged.
- New model load (`FlagReranker`) follows the existing convention in `app/models/embeddings.py` / `app/models/clip_model.py`: a class wrapping the model, plus an `@lru_cache(maxsize=1)` module-level getter function.
- Real-model tests are marked `pytest.mark.slow` (see `pytest.ini`) and run explicitly with `-m slow`; they are not part of the default `pytest` run.
- Design reference: `docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md`.

---

### Task 1: Reranker config settings

**Files:**
- Modify: `app/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `settings.reranker_enabled: bool`, `settings.reranker_model_name: str`, `settings.reranker_candidate_k: int`, `settings.reranker_max_concurrency: int` — consumed by Task 2 (`reranker_model_name`, `reranker_max_concurrency`) and Task 5 (`reranker_enabled`, `reranker_candidate_k`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_config.py`:

```python
def test_reranker_settings_have_expected_defaults():
    assert settings.reranker_enabled is True
    assert settings.reranker_model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_candidate_k == 30
    assert settings.reranker_max_concurrency == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_reranker_settings_have_expected_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'reranker_enabled'`

- [ ] **Step 3: Write minimal implementation**

In `app/config.py`, add the four fields to the `Settings` class, right after the existing `clip_model_name` field:

```python
    clip_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    reranker_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidate_k: int = 30
    reranker_max_concurrency: int = 2
```

(Keep every other existing field in `Settings` unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/unit/test_config.py
git commit -m "feat: add reranker config settings"
```

---

### Task 2: Reranker model wrapper

**Files:**
- Create: `app/models/reranker.py`
- Test: `tests/unit/test_reranker.py`

**Interfaces:**
- Consumes: `settings.reranker_model_name`, `settings.reranker_max_concurrency` (Task 1).
- Produces: `class Reranker` with `.score(query: str, texts: list[str]) -> list[float]`; `get_reranker() -> Reranker` (`@lru_cache(maxsize=1)` singleton) — consumed by Task 4 (`Reranker` type hint) and Task 5 (`get_reranker`, via a dependency wrapper).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_reranker.py`:

```python
import pytest

from app.models.reranker import get_reranker

pytestmark = pytest.mark.slow


def test_score_ranks_relevant_text_higher():
    reranker = get_reranker()
    scores = reranker.score(
        "苹果的产地在哪里？",
        ["苹果主要产自中国、美国等国家。", "今天天气晴朗，适合出门跑步。"],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_get_reranker_is_singleton():
    assert get_reranker() is get_reranker()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reranker.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.reranker'`

- [ ] **Step 3: Write minimal implementation**

Create `app/models/reranker.py`:

```python
"""Cross-encoder reranker wrapper for Stage 2 of the two-stage rerank
pipeline. See docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md.
"""

import threading
from functools import lru_cache

from app.config import settings


class Reranker:
    def __init__(self, model_name: str, max_concurrency: int):
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=True, normalize=True)
        self._semaphore = threading.Semaphore(max_concurrency)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        with self._semaphore:
            scores = self._model.compute_score(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model_name, settings.reranker_max_concurrency)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_reranker.py -v -m slow`
Expected: PASS (downloads `BAAI/bge-reranker-v2-m3` on first run if not already cached under `HF_HOME`)

- [ ] **Step 5: Commit**

```bash
git add app/models/reranker.py tests/unit/test_reranker.py
git commit -m "feat: add cross-encoder reranker model wrapper"
```

---

### Task 3: Widen the RRF candidate pool with `pool_size`

**Files:**
- Modify: `app/store/qdrant_client.py:152-185` (`search_text_hybrid`)
- Test: `tests/unit/test_qdrant_store.py`

**Interfaces:**
- Produces: `search_text_hybrid(client, query, query_text=None, top_k=10, pool_size=None) -> list[SearchHit]` — `pool_size` consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_qdrant_store.py`:

```python
def test_search_text_hybrid_pool_size_widens_candidate_limit():
    client = make_client()
    vec = DenseSparseVector(dense=[0.1] * 1024, sparse={1: 0.5})
    for i in range(5):
        record = TextChunkRecord(
            doc_id=f"doc{i}", chunk_index=0, text=f"内容 {i}", source_path=f"/{i}.txt",
            file_type="txt", page=None, source_type="normal", content_hash=f"h{i}",
        )
        upsert_text_chunk(client, record, vec)

    hits_default = search_text_hybrid(client, vec, top_k=2)
    assert len(hits_default) == 2

    hits_pooled = search_text_hybrid(client, vec, top_k=2, pool_size=5)
    assert len(hits_pooled) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_qdrant_store.py::test_search_text_hybrid_pool_size_widens_candidate_limit -v`
Expected: FAIL with `TypeError: search_text_hybrid() got an unexpected keyword argument 'pool_size'`

- [ ] **Step 3: Write minimal implementation**

Replace `search_text_hybrid` in `app/store/qdrant_client.py` (lines 152-185) with:

```python
def search_text_hybrid(
    client: QdrantClient,
    query: DenseSparseVector,
    query_text: str | None = None,
    top_k: int = 10,
    pool_size: int | None = None,
) -> list[SearchHit]:
    limit = pool_size if pool_size is not None else top_k
    prefetch = [
        models.Prefetch(query=query.dense, using="dense", limit=limit * 2),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(query.sparse.keys()), values=list(query.sparse.values())
            ),
            using="sparse",
            limit=limit * 2,
        ),
    ]
    # 第三路：精确子串匹配兜底。专有名词/术语的向量语义有时会被稀释掉，
    # 但只要原文里逐字出现该词，全文索引的 MatchText 就能把它捞回来参与
    # RRF 融合，而不是完全依赖 dense/sparse 的语义相似度排序。
    if query_text and query_text.strip():
        prefetch.append(
            models.Prefetch(
                query=query.dense,
                using="dense",
                filter=models.Filter(
                    must=[models.FieldCondition(key="text", match=models.MatchText(text=query_text))]
                ),
                limit=limit * 2,
            )
        )
    result = client.query_points(
        collection_name=TEXT_COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return [SearchHit(score=p.score, payload=p.payload) for p in result.points]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_qdrant_store.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add app/store/qdrant_client.py tests/unit/test_qdrant_store.py
git commit -m "feat: add pool_size param to search_text_hybrid for rerank candidate pool"
```

---

### Task 4: `apply_rerank` orchestration function

**Files:**
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/rerank.py`
- Test: `tests/unit/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: `SearchHit` (from `app.store.qdrant_client`, has `.score: float`, `.payload: dict`); any object with `.score(query: str, texts: list[str]) -> list[float]` (matches `Reranker` from Task 2, but the function only depends on this method shape).
- Produces: `apply_rerank(query: str, hits: list[SearchHit], reranker, top_k: int) -> list[SearchHit]` — consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Create `app/retrieval/__init__.py` (empty file).

Create `tests/unit/test_retrieval_rerank.py`:

```python
from app.retrieval.rerank import apply_rerank
from app.store.qdrant_client import SearchHit


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, query, texts):
        return self._scores


def test_apply_rerank_reorders_by_score_and_truncates():
    hits = [
        SearchHit(score=0.1, payload={"doc_id": "a", "text": "a"}),
        SearchHit(score=0.9, payload={"doc_id": "b", "text": "b"}),
        SearchHit(score=0.5, payload={"doc_id": "c", "text": "c"}),
    ]
    reranker = FakeReranker([0.2, 0.8, 0.6])

    result = apply_rerank("query", hits, reranker, top_k=2)

    assert [h.payload["doc_id"] for h in result] == ["b", "c"]
    assert [h.score for h in result] == [0.8, 0.6]


def test_apply_rerank_empty_hits_returns_empty():
    reranker = FakeReranker([])
    assert apply_rerank("query", [], reranker, top_k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_retrieval_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.rerank'`

- [ ] **Step 3: Write minimal implementation**

Create `app/retrieval/rerank.py`:

```python
"""Stage 2 of the two-stage rerank pipeline: reorders a Stage 1 candidate
pool by cross-encoder relevance and truncates to top_k. See
docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md.
"""

from app.store.qdrant_client import SearchHit


def apply_rerank(query: str, hits: list[SearchHit], reranker, top_k: int) -> list[SearchHit]:
    if not hits:
        return hits
    texts = [hit.payload.get("text", "") for hit in hits]
    scores = reranker.score(query, texts)
    reranked = sorted(
        (SearchHit(score=score, payload=hit.payload) for hit, score in zip(hits, scores)),
        key=lambda hit: hit.score,
        reverse=True,
    )
    return reranked[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_retrieval_rerank.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/__init__.py app/retrieval/rerank.py tests/unit/test_retrieval_rerank.py
git commit -m "feat: add apply_rerank Stage 2 orchestration function"
```

---

### Task 5: Wire the two stages into `/search/text`

**Files:**
- Modify: `app/api/deps.py`
- Modify: `app/api/routes/search.py`
- Test: `tests/unit/test_api_search.py`

**Interfaces:**
- Consumes: `get_reranker` (Task 2), `search_text_hybrid(..., pool_size=...)` (Task 3), `apply_rerank(...)` (Task 4), `settings.reranker_enabled` / `settings.reranker_candidate_k` (Task 1).
- Produces: `get_reranker_dep()` in `app/api/deps.py` — a FastAPI dependency, same shape as `get_text_embedder_dep`/`get_clip_embedder_dep`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_api_search.py`, add these imports at the top (alongside the existing ones):

```python
import pytest

from app.api.deps import get_reranker_dep
from app.config import settings
from app.store.qdrant_client import SearchHit
```

Replace the existing `make_client` function with a version that also overrides the reranker dependency:

```python
class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, query, texts):
        return self._scores


def make_client(monkeypatch, reranker_scores=None):
    app.dependency_overrides[get_text_embedder_dep] = lambda: FakeTextEmbedder()
    app.dependency_overrides[get_clip_embedder_dep] = lambda: FakeClipEmbedder()
    app.dependency_overrides[get_store_client] = lambda: FakeStoreClient()
    app.dependency_overrides[get_reranker_dep] = lambda: FakeReranker(reranker_scores or [0.9])
    monkeypatch.setattr("app.api.routes.search.search_text_hybrid", override_search_text_hybrid)
    monkeypatch.setattr("app.api.routes.search.search_images", override_search_images)
    monkeypatch.setattr(
        "app.api.routes.search.list_chunks_for_document", override_list_chunks_for_document
    )
    return TestClient(app)
```

(This keeps every existing call site of `make_client(monkeypatch)` working unchanged, since `reranker_scores` defaults to `None` → `[0.9]`, matching the single hit `override_search_text_hybrid` returns.)

Add two new test functions at the end of the file:

```python
def test_search_text_applies_rerank_and_reorders(monkeypatch):
    def two_hits(*args, **kwargs):
        return [
            SearchHit(score=0.5, payload={"doc_id": "low", "chunk_index": 0, "text": "低相关"}),
            SearchHit(score=0.4, payload={"doc_id": "high", "chunk_index": 0, "text": "高相关"}),
        ]

    client = make_client(monkeypatch, reranker_scores=[0.1, 0.9])
    monkeypatch.setattr("app.api.routes.search.search_text_hybrid", two_hits)

    response = client.post("/search/text", json={"query": "你好", "top_k": 5})
    body = response.json()

    assert [h["doc_id"] for h in body["text_hits"]] == ["high", "low"]
    assert body["text_hits"][0]["score"] == pytest.approx(0.9)
    app.dependency_overrides.clear()


def test_search_text_skips_rerank_when_disabled(monkeypatch):
    def two_hits(*args, **kwargs):
        return [
            SearchHit(score=0.5, payload={"doc_id": "low", "chunk_index": 0, "text": "低相关"}),
            SearchHit(score=0.4, payload={"doc_id": "high", "chunk_index": 0, "text": "高相关"}),
        ]

    monkeypatch.setattr(settings, "reranker_enabled", False)
    client = make_client(monkeypatch, reranker_scores=[0.1, 0.9])
    monkeypatch.setattr("app.api.routes.search.search_text_hybrid", two_hits)

    response = client.post("/search/text", json={"query": "你好", "top_k": 5})
    body = response.json()

    assert [h["doc_id"] for h in body["text_hits"]] == ["low", "high"]
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_api_search.py -v`
Expected: FAIL — `test_search_text_applies_rerank_and_reorders` and `test_search_text_skips_rerank_when_disabled` fail (route doesn't rerank yet, order stays `["low", "high"]` for the first test); collection may also error on `ImportError: cannot import name 'get_reranker_dep'` until Step 3 lands.

- [ ] **Step 3: Write minimal implementation**

In `app/api/deps.py`, add:

```python
from app.models.reranker import get_reranker


def get_reranker_dep():
    return get_reranker()
```

(add the import alongside the existing `get_clip_embedder`/`get_text_embedder` imports, and the function alongside `get_clip_embedder_dep`/`get_text_embedder_dep`)

In `app/api/routes/search.py`, update the imports:

```python
from app.api.deps import get_clip_embedder_dep, get_reranker_dep, get_store_client, get_text_embedder_dep
from app.config import settings
from app.retrieval.rerank import apply_rerank
```

(add `get_reranker_dep` to the existing `app.api.deps` import, and the two new imports, alongside the existing imports at the top of the file)

Replace the `search_text` function with:

```python
@router.post("/search/text", response_model=SearchTextResponse)
def search_text(
    request: SearchTextRequest,
    store=Depends(get_store_client),
    text_embedder=Depends(get_text_embedder_dep),
    clip_embedder=Depends(get_clip_embedder_dep),
    reranker=Depends(get_reranker_dep),
):
    text_hits = []
    image_hits = []

    if "text" in request.modalities:
        query_vector = text_embedder.embed([request.query])[0]
        pool_size = settings.reranker_candidate_k if settings.reranker_enabled else request.top_k
        text_hits = search_text_hybrid(
            store, query_vector, query_text=request.query, top_k=request.top_k, pool_size=pool_size
        )
        if settings.reranker_enabled and text_hits:
            text_hits = apply_rerank(request.query, text_hits, reranker, top_k=request.top_k)

    if "image" in request.modalities:
        clip_query_vector = clip_embedder.embed_text(request.query)
        image_hits = search_images(store, clip_query_vector, top_k=request.top_k)

    return SearchTextResponse(
        text_hits=[_make_hit_out(h.score, h.payload, h.payload.get("chunk_index")) for h in text_hits],
        image_hits=[_make_hit_out(h.score, h.payload, None) for h in image_hits],
    )
```

(Leave `search_image` and everything else in the file unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_api_search.py -v`
Expected: PASS (all tests in the file, including the two new ones)

- [ ] **Step 5: Run the full unit test suite**

Run: `pytest tests/unit -v`
Expected: PASS (slow-marked tests from Task 2 are skipped by default unless `-m slow` is passed — this is fine, they're covered separately)

- [ ] **Step 6: Commit**

```bash
git add app/api/deps.py app/api/routes/search.py tests/unit/test_api_search.py
git commit -m "feat: wire two-stage rerank into /search/text"
```

---

### Task 6: Update `docs/retrieval-relevance-filtering.md` implementation status

**Files:**
- Modify: `docs/retrieval-relevance-filtering.md`

**Interfaces:**
- None (docs-only change).

- [ ] **Step 1: Replace the closing "尚未实施" section**

In `docs/retrieval-relevance-filtering.md`, replace the final section:

```markdown
## 尚未实施

本文档仅记录讨论结论，A/B 方案均未实施，是否要做、以及"完全不相关时接
口具体该返回什么（空列表 vs 带信号继续返回）"尚未决定，留待后续按需求
优先级排期。
```

with:

```markdown
## 实施状态

方案 A 的排序部分（RRF 粗筛 top-N 候选池 + cross-encoder 精排，两阶段
架构）已按 `docs/superpowers/specs/2026-08-27-two-stage-rerank-design.md`
实施：`/search/text` 默认走两阶段 rerank（`reranker_enabled=False` 可关
闭回退到单阶段 RRF）。方案 B（透传命中路径信号）以及"完全不相关时接口
具体该返回什么（空列表 vs 带信号继续返回）"仍未实施/未决定，留待后续按
需求优先级排期。
```

- [ ] **Step 2: Commit**

```bash
git add docs/retrieval-relevance-filtering.md
git commit -m "docs: mark two-stage rerank ranking improvements as implemented"
```
