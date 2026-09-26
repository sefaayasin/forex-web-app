"""Build complete hourly bars from the 15-minute archive.

The native Dukascopy hourly download (data/historical_1h) silently skipped whole
trading weeks for many pairs — e.g. GBPCHF is missing ~47k hours in 120-hour
blocks — while the 15-minute archive is complete. Where both exist, the hourly
bars rebuilt here match the native ones to within one tick for ~99.8% of hours.

The native files are left untouched (earlier research records hash them).
Reads data/historical_15m/<PAIR>.csv, writes data/historical_1h_from_15m/<PAIR>.csv
in the same format (UTC bar-open timestamp in ms, open/high/low/close/volume).

Run: python rebuild_hourly_from_15m.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "historical_15m"
TARGET = ROOT / "data" / "historical_1h_from_15m"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def rebuild(frame: pd.DataFrame) -> pd.DataFrame:
    """UTC-aligned hourly OHLCV from 15-minute rows; hours without any 15-minute bar are dropped."""
    data = frame.set_index(pd.to_datetime(frame["timestamp"], unit="ms", utc=True)).drop(columns="timestamp")
    hourly = data.resample("1h", label="left", closed="left").agg(AGG).dropna(subset=["open", "high", "low", "close"])
    hourly.insert(0, "timestamp", hourly.index.as_unit("ms").asi8)
    return hourly.reset_index(drop=True)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for path in sorted(SOURCE.glob("*.csv")):
        hourly = rebuild(pd.read_csv(path))
        hourly.to_csv(TARGET / path.name, index=False)
        print(f"{path.stem}: {len(hourly)} hourly bars", flush=True)


if __name__ == "__main__":
    main()
