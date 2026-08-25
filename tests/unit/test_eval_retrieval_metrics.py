import json

from app.eval.retrieval_metrics import RetrievalQuery, evaluate_retrieval, load_qa_file


def test_evaluate_retrieval_perfect_match_gives_full_recall_and_mrr_one():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c1"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2", "c3"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[1, 5])

    assert report.query_count == 1
    assert report.recall_at_k[1] == 1.0
    assert report.recall_at_k[5] == 1.0
    assert report.mrr == 1.0


def test_evaluate_retrieval_miss_gives_zero_recall_and_mrr():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["missing"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[5])

    assert report.recall_at_k[5] == 0.0
    assert report.mrr == 0.0


def test_evaluate_retrieval_rank_affects_mrr():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c2"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c2", "c3"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[5])

    assert report.mrr == 0.5  # 命中排第2位 -> 1/2


def test_evaluate_retrieval_partial_recall_with_multiple_expected_chunks():
    queries = [RetrievalQuery(query="q1", expected_chunk_ids=["c1", "c2"])]

    def retrieve_fn(query, top_k):
        return ["c1", "c9"]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[2])

    assert report.recall_at_k[2] == 0.5  # 命中1/2


def test_evaluate_retrieval_averages_across_multiple_queries():
    queries = [
        RetrievalQuery(query="q1", expected_chunk_ids=["c1"]),
        RetrievalQuery(query="q2", expected_chunk_ids=["missing"]),
    ]

    def retrieve_fn(query, top_k):
        return {"q1": ["c1"], "q2": ["c9"]}[query]

    report = evaluate_retrieval(queries, retrieve_fn, k_values=[1])

    assert report.recall_at_k[1] == 0.5
    assert report.mrr == 0.5


def test_evaluate_retrieval_empty_queries_returns_zeroed_report():
    report = evaluate_retrieval([], lambda q, k: [], k_values=[5])

    assert report.query_count == 0
    assert report.recall_at_k == {5: 0.0}
    assert report.mrr == 0.0


def test_load_qa_file_parses_json_into_retrieval_queries(tmp_path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps([{"query": "什么是切片", "expected_chunk_ids": ["doc1:0", "doc1:1"]}]),
        encoding="utf-8",
    )

    queries = load_qa_file(str(qa_path))

    assert queries == [RetrievalQuery(query="什么是切片", expected_chunk_ids=["doc1:0", "doc1:1"])]
