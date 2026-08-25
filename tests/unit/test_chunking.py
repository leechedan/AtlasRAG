from app.ingest.chunking import chunk_csv_rows, chunk_text


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("短文本", chunk_size=100, chunk_overlap=10)
    assert chunks == ["短文本"]


def test_chunk_text_long_text_multiple_chunks_with_overlap():
    text = "A" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_uses_default_settings_when_not_overridden():
    from app.config import settings

    text = "B" * (settings.chunk_size + 50)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= settings.chunk_size for c in chunks)


def test_chunk_text_splits_on_custom_separator():
    text = "第一段内容" + "---" + "第二段内容" + "---" + "第三段内容"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=0, separators=["---", ""])
    assert len(chunks) == 3
    assert chunks[0] == "第一段内容"
    assert chunks[1] == "第二段内容"
    assert chunks[2] == "第三段内容"


def test_chunk_csv_rows_groups_with_header_context():
    header = ["name", "age"]
    rows = [{"name": f"p{i}", "age": str(i)} for i in range(5)]
    chunks = chunk_csv_rows(rows, header, rows_per_chunk=2, max_chars=1000)
    assert len(chunks) == 3
    assert "name" in chunks[0] and "age" in chunks[0]
    assert "p0" in chunks[0] and "p1" in chunks[0]
    assert "p2" in chunks[1]


def test_chunk_csv_rows_splits_early_when_row_text_is_huge():
    header = ["name", "bio"]
    rows = [
        {"name": "p0", "bio": "x" * 900},
        {"name": "p1", "bio": "y" * 900},
        {"name": "p2", "bio": "short"},
    ]
    chunks = chunk_csv_rows(rows, header, rows_per_chunk=20, max_chars=1000)
    # p0 和 p1 单独一行就接近/超过 max_chars，各自都会触发提前 flush；
    # p2 很短，追加到 p1 所在的 chunk 后仍不超过 max_chars，所以和 p1 合并。
    assert len(chunks) == 2
    assert all(len(c) <= 1000 for c in chunks)
    assert "p0" in chunks[0]
    assert "p1" in chunks[1] and "p2" in chunks[1]


def test_chunk_csv_rows_splits_single_oversized_row():
    header = ["name", "bio"]
    rows = [{"name": "p0", "bio": "z" * 3000}]

    chunks = chunk_csv_rows(rows, header, rows_per_chunk=20, max_chars=1000)

    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(c.count("z") for c in chunks) == 3000
