from pathlib import Path

import pandas as pd


def parse_csv_file(path: Path) -> tuple[list[dict], list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    header = list(df.columns)
    rows = df.to_dict(orient="records")
    return rows, header
