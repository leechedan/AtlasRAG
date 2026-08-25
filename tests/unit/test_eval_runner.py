import json
from dataclasses import dataclass

import pytest

from app.eval.runner import run


@dataclass
class _FakeVector:
    dense: list[float]


class _FakeEmbedder:
    def embed(self, texts: list[str]):
        return [_FakeVector([1.0, 0.0]) for _ in texts]


def _write(path, name, content):
    (path / name).write_text(content, encoding="utf-8")


def test_run_aggregates_structure_and_length_metrics_across_documents(tmp_path):
    _write(tmp_path, "a.md", "# A\n\nhello world, this is doc a.\n")
    _write(tmp_path, "b.txt", "plain text without markdown structure.\n")
    _write(tmp_path, "c.png", "not a text document")  # 应被忽略（不支持的扩展名）

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=False,
    )

    assert report["aggregate"]["document_count"] == 2
    assert report["aggregate"]["chunk_count"] == 2
    assert report["aggregate"]["mean_coherence"] is None
    assert report["aggregate"]["retrieval"] is None
    assert len(report["documents"]) == 2


def test_run_computes_semantic_coherence_when_enabled(tmp_path):
    _write(tmp_path, "a.md", "第一句。第二句。\n")

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=True,
        embedder=_FakeEmbedder(),
    )

    assert report["aggregate"]["mean_coherence"] == 1.0


def test_run_requires_embedder_when_semantic_enabled(tmp_path):
    with pytest.raises(ValueError):
        run(
            input_dir=str(tmp_path),
            chunk_size=1000,
            chunk_overlap=0,
            min_chars=0,
            max_chars=1000,
            run_semantic=True,
        )


def test_run_computes_retrieval_metrics_when_qa_file_and_retrieve_fn_given(tmp_path):
    _write(tmp_path, "a.md", "# A\n\nhello.\n")
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps([{"query": "q", "expected_chunk_ids": ["c1"]}]), encoding="utf-8"
    )

    def retrieve_fn(query, top_k):
        return ["c1"]

    report = run(
        input_dir=str(tmp_path),
        chunk_size=1000,
        chunk_overlap=0,
        min_chars=0,
        max_chars=1000,
        run_semantic=False,
        qa_file=str(qa_path),
        retrieve_fn=retrieve_fn,
    )

    assert report["aggregate"]["retrieval"]["mrr"] == 1.0
