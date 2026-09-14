"""Download free daily forex history (2008-present) for training data.

Uses yfinance (Yahoo Finance) daily OHLC data, which is the only free
source that realistically covers all major/minor FX pairs back to 2008.
Intraday (minute-level) history is NOT available for free that far back
from any provider without a paid tick-data subscription (Yahoo limits
intraday lookback to a few months/years depending on interval).

Output: one CSV per pair under data/historical/<PAIR>.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from forex_config import SYMBOL_LIST

OUT_DIR = Path(__file__).resolve().parent / "data" / "historical"
START_DATE = "2008-01-01"


def download_symbol(symbol: str) -> tuple[bool, str]:
    try:
        df = yf.download(
            symbol,
            start=START_DATE,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"HATA: {exc}"

    if df is None or df.empty:
        return False, "veri yok"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    out_path = OUT_DIR / f"{symbol.replace('=X', '')}.csv"
    df.to_csv(out_path)
    return True, f"{len(df)} satır, {df.index.min().date()} -> {df.index.max().date()}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for symbol in SYMBOL_LIST:
        ok, msg = download_symbol(symbol)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {symbol}: {msg}")
        results.append((symbol, ok, msg))
        time.sleep(0.5)

    failed = [s for s, ok, _ in results if not ok]
    print("\n--- Özet ---")
    print(f"Toplam: {len(results)}, Başarılı: {len(results) - len(failed)}, Başarısız: {len(failed)}")
    if failed:
        print("Başarısız semboller:", ", ".join(failed))


if __name__ == "__main__":
    main()
