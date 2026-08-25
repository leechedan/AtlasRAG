from app.eval.length_metrics import compute_length_metrics


def test_compute_length_metrics_basic_stats():
    chunks = ["a" * 100, "b" * 200, "c" * 300]
    report = compute_length_metrics(chunks, min_chars=50, max_chars=250)

    assert report.count == 3
    assert report.min_chars == 100
    assert report.max_chars == 300
    assert report.mean_chars == 200.0
    assert report.median_chars == 200.0
    assert report.out_of_range_count == 1
    assert report.out_of_range_ratio == 1 / 3


def test_compute_length_metrics_empty_chunks_returns_zeroed_report():
    report = compute_length_metrics([], min_chars=50, max_chars=250)

    assert report.count == 0
    assert report.out_of_range_ratio == 0.0


def test_compute_length_metrics_single_chunk_has_zero_stdev():
    report = compute_length_metrics(["x" * 100], min_chars=50, max_chars=250)

    assert report.stdev_chars == 0.0
