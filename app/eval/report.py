"""汇总报告的落盘与控制台摘要格式化。"""

import json
from pathlib import Path


def write_report(report: dict, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def format_summary_table(aggregate: dict) -> str:
    lines = ["| 指标 | 值 |", "| --- | --- |"]
    for key, value in aggregate.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)
