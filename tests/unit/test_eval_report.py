import json

from app.eval.report import format_summary_table, write_report


def test_write_report_writes_json_file(tmp_path):
    output_path = tmp_path / "report.json"

    write_report({"aggregate": {"chunk_count": 3}}, str(output_path))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "aggregate": {"chunk_count": 3}
    }


def test_format_summary_table_renders_markdown_table_with_formatted_floats():
    table = format_summary_table({"chunk_count": 3, "truncated_region_ratio": 0.5})

    assert "| chunk_count | 3 |" in table
    assert "| truncated_region_ratio | 0.5000 |" in table
