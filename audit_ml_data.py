"""Inventory local CSVs in bounded-memory chunks; do not modify source data."""
import json
from pathlib import Path

import pandas as pd

from forex_ml import ROOT


def audit(data_dir=ROOT / "data"):
    rows = []
    for path in sorted(Path(data_dir).rglob("*.csv")):
        if "ml" in path.relative_to(data_dir).parts:
            continue
        row = {"file": str(path.relative_to(data_dir)), "bytes": path.stat().st_size,
               "rows": 0, "missing_cells": 0, "duplicate_timestamps": 0}
        earliest, latest, previous = None, None, None
        for chunk in pd.read_csv(path, chunksize=100000):
            row["columns"] = list(chunk.columns)
            row["rows"] += len(chunk)
            row["missing_cells"] += int(chunk.isna().sum().sum())
            col = "timestamp" if "timestamp" in chunk else "date" if "date" in chunk else None
            if col:
                t = pd.to_datetime(chunk[col], unit="ms" if pd.api.types.is_numeric_dtype(chunk[col]) else None,
                                   utc=True, errors="coerce")
                row["invalid_dates"] = row.get("invalid_dates", 0) + int(t.isna().sum())
                if t.notna().any():
                    earliest = t.min() if earliest is None else min(earliest, t.min())
                    latest = t.max() if latest is None else max(latest, t.max())
                if col == "timestamp":
                    row["duplicate_timestamps"] += int(t.duplicated().sum()) + int(previous is not None and t.iloc[0] == previous)
                    row["unordered"] = row.get("unordered", False) or not t.is_monotonic_increasing or bool(previous is not None and t.iloc[0] < previous)
                    previous = t.iloc[-1]
        row.update(start=str(earliest), end=str(latest))
        rows.append(row)
    out = Path(data_dir) / "ml"
    out.mkdir(exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Audited {len(rows)} CSV files, {sum(r['rows'] for r in rows):,} rows")
    return rows


if __name__ == "__main__":
    audit()
