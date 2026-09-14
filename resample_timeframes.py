"""Build weekly (1w) bars from the natively-downloaded daily (1d) data.

Dukascopy does not publish a native weekly candle feed (no provider does —
weekly bars are always built from daily ones), so this is the one timeframe
that has to be derived rather than downloaded directly.

Reads data/historical_1d/<PAIR>.csv and writes data/historical_1w/<PAIR>.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DAILY_DIR = BASE_DIR / "data" / "historical_1d"
WEEKLY_DIR = BASE_DIR / "data" / "historical_1w"

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def build_weekly(csv_path: Path) -> None:
    pair = csv_path.stem
    out_path = WEEKLY_DIR / f"{pair}.csv"
    if out_path.exists():
        return

    df = pd.read_csv(csv_path)
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df = df.set_index(ts_col).sort_index()

    weekly = df.resample("W-FRI", label="left", closed="left").agg(AGG)
    weekly = weekly.dropna(subset=["open"])
    weekly.index.name = ts_col
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path)
    print(f"  {pair}: {len(weekly)} satır")


def main() -> None:
    csv_files = sorted(DAILY_DIR.glob("*.csv"))
    if not csv_files:
        print("data/historical_1d içinde CSV bulunamadı, önce günlük veri indirilmeli.")
        return
    for csv_path in csv_files:
        build_weekly(csv_path)
    print("\nHaftalık veri üretimi tamamlandı.")


if __name__ == "__main__":
    main()
