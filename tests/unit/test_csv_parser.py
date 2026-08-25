from pathlib import Path

from app.ingest.parsers.csv_parser import parse_csv_file


def test_parse_csv_file_returns_rows_and_header(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nalice,30\nbob,25\n", encoding="utf-8")

    rows, header = parse_csv_file(f)

    assert header == ["name", "age"]
    assert rows == [{"name": "alice", "age": "30"}, {"name": "bob", "age": "25"}]
