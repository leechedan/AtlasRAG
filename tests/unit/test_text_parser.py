from pathlib import Path

from app.ingest.parsers.text_parser import parse_text_file


def test_parse_text_file_reads_utf8(tmp_path: Path):
    f = tmp_path / "note.md"
    f.write_text("# 标题\n正文内容", encoding="utf-8")
    assert parse_text_file(f) == "# 标题\n正文内容"
