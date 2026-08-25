"""Chunk 长度分布统计：min/max/mean/median/stdev，以及超出目标区间的占比。"""

import statistics
from dataclasses import dataclass


@dataclass
class LengthReport:
    count: int
    min_chars: int
    max_chars: int
    mean_chars: float
    median_chars: float
    stdev_chars: float
    out_of_range_count: int
    out_of_range_ratio: float


def compute_length_metrics(chunks: list[str], min_chars: int, max_chars: int) -> LengthReport:
    if not chunks:
        return LengthReport(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0)

    lengths = [len(c) for c in chunks]
    out_of_range = sum(1 for length in lengths if length < min_chars or length > max_chars)

    return LengthReport(
        count=len(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        mean_chars=statistics.mean(lengths),
        median_chars=statistics.median(lengths),
        stdev_chars=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        out_of_range_count=out_of_range,
        out_of_range_ratio=out_of_range / len(lengths),
    )
