"""Chunk 内相邻句子的语义连贯性：用注入的 embedder 给每个 chunk 内相邻
句子对算余弦相似度均值。开销集中在 embedder.embed() 调用；单测用假
embedder 注入固定向量，不加载真实 BGE-M3。"""

import re
from dataclasses import dataclass
from typing import Protocol

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list:
        ...


@dataclass
class CoherenceReport:
    chunk_count: int
    scored_chunk_count: int  # 句子数 >= 2 才有连贯性分数
    mean_coherence: float


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text)]
    return [p for p in parts if p]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_coherence(chunks: list[str], embedder: Embedder) -> CoherenceReport:
    sentence_lists = [split_sentences(c) for c in chunks]
    all_sentences = [s for sentences in sentence_lists for s in sentences]
    if not all_sentences:
        return CoherenceReport(len(chunks), 0, 0.0)

    vectors = embedder.embed(all_sentences)
    dense_vectors = [v.dense for v in vectors]

    chunk_scores: list[float] = []
    cursor = 0
    for sentences in sentence_lists:
        n = len(sentences)
        chunk_vectors = dense_vectors[cursor : cursor + n]
        cursor += n
        if n < 2:
            continue
        sims = [
            _cosine_similarity(chunk_vectors[i], chunk_vectors[i + 1]) for i in range(n - 1)
        ]
        chunk_scores.append(sum(sims) / len(sims))

    mean_coherence = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    return CoherenceReport(len(chunks), len(chunk_scores), mean_coherence)
