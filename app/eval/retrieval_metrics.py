"""端到端检索质量：Recall@K（宏平均标准召回率）与 MRR。

标注集 JSON schema：`[{"query": "...", "expected_chunk_ids": ["doc_id:chunk_index 的
md5，即 app.store.qdrant_client.compute_chunk_id 的产出"]}]`。本模块不生成标注集，
只消费它——生成方式（人工标注或未来接入 LLM）留给调用方。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RetrievalQuery:
    query: str
    expected_chunk_ids: list[str]


@dataclass
class RetrievalReport:
    query_count: int
    recall_at_k: dict[int, float]
    mrr: float


def evaluate_retrieval(
    queries: list[RetrievalQuery],
    retrieve_fn: Callable[[str, int], list[str]],
    k_values: list[int] | None = None,
) -> RetrievalReport:
    resolved_k_values = list(k_values) if k_values else [5, 10]

    if not queries:
        return RetrievalReport(0, {k: 0.0 for k in resolved_k_values}, 0.0)

    max_k = max(resolved_k_values)
    recall_sums = {k: 0.0 for k in resolved_k_values}
    reciprocal_ranks: list[float] = []

    for q in queries:
        expected = set(q.expected_chunk_ids)
        retrieved = retrieve_fn(q.query, max_k)

        rank = next((i + 1 for i, cid in enumerate(retrieved) if cid in expected), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in resolved_k_values:
            if not expected:
                continue
            hits = expected & set(retrieved[:k])
            recall_sums[k] += len(hits) / len(expected)

    n = len(queries)
    return RetrievalReport(
        query_count=n,
        recall_at_k={k: recall_sums[k] / n for k in resolved_k_values},
        mrr=sum(reciprocal_ranks) / n,
    )


def load_qa_file(path: str) -> list[RetrievalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RetrievalQuery(query=item["query"], expected_chunk_ids=item["expected_chunk_ids"])
        for item in data
    ]
