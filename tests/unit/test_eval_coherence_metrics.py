from dataclasses import dataclass

from app.eval.coherence_metrics import compute_coherence, split_sentences


@dataclass
class _FakeVector:
    dense: list[float]


class _FakeEmbedder:
    """按句子内容查表返回预设向量，构造已知相似度的场景。"""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]):
        return [_FakeVector(self._vectors[t]) for t in texts]


def test_split_sentences_splits_on_chinese_terminal_punctuation():
    sentences = split_sentences("第一句。第二句！第三句？")

    assert sentences == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_drops_empty_fragments():
    sentences = split_sentences("只有一句没有标点")

    assert sentences == ["只有一句没有标点"]


def test_compute_coherence_identical_vectors_score_one():
    chunks = ["甲。乙。"]
    embedder = _FakeEmbedder({"甲。": [1.0, 0.0], "乙。": [1.0, 0.0]})

    report = compute_coherence(chunks, embedder)

    assert report.scored_chunk_count == 1
    assert report.mean_coherence == 1.0


def test_compute_coherence_orthogonal_vectors_score_zero():
    chunks = ["甲。乙。"]
    embedder = _FakeEmbedder({"甲。": [1.0, 0.0], "乙。": [0.0, 1.0]})

    report = compute_coherence(chunks, embedder)

    assert report.mean_coherence == 0.0


def test_compute_coherence_single_sentence_chunk_not_scored():
    chunks = ["只有一句。"]
    embedder = _FakeEmbedder({"只有一句。": [1.0, 0.0]})

    report = compute_coherence(chunks, embedder)

    assert report.chunk_count == 1
    assert report.scored_chunk_count == 0
    assert report.mean_coherence == 0.0


def test_compute_coherence_empty_chunks_list():
    report = compute_coherence([], _FakeEmbedder({}))

    assert report.chunk_count == 0
    assert report.mean_coherence == 0.0
