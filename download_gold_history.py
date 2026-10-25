"""Download gold (XAUUSD) minute history for the out-of-sample test in research_arda_ideas.py.

Bars: HistData.com free M1 BID files (one zip per past year, one per month for the
current year); their clock is EST without daylight saving, i.e. UTC-5 all year.
Spread: Dukascopy BID and ASK minute candles for the 15th of each month (or the next
weekday), because HistData has no ask side. Dukascopy's free datafeed throttles to
about one file a minute, so only this small sample comes from it. Downloads are
cached in research/arda_ideas/cache (git-ignored); reruns resume.

Writes data/historical_gold/XAUUSD_1m.csv and data/historical_gold/XAUUSD_spread_1m.csv.
Run: python -X utf8 download_gold_history.py [--spread]
"""
from __future__ import annotations

import argparse
import io
import lzma
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "research/arda_ideas/cache"
OUT = ROOT / "data/historical_gold"
HISTDATA = "https://www.histdata.com"
DUKASCOPY = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
HEADERS = {"User-Agent": "Mozilla/5.0"}
YEARS = range(2020, 2026)
CURRENT_YEAR_MONTHS = [(2026, m) for m in range(1, 11)]


def histdata_zip(year: int, month: int | None = None) -> bytes:
    name = f"histdata_{year}{'' if month is None else f'{month:02d}'}.zip"
    cached = CACHE / name
    if cached.exists():
        return cached.read_bytes()
    page_url = f"{HISTDATA}/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd/{year}"
    page_url += "" if month is None else f"/{month}"
    page = urllib.request.urlopen(urllib.request.Request(page_url, headers=HEADERS), timeout=120).read().decode()
    token = re.search(r'name="tk" id="tk" value="([^"]+)"', page).group(1)
    form = {"tk": token, "date": str(year), "datemonth": str(year) if month is None else f"{year}{month:02d}",
        "platform": "ASCII", "timeframe": "M1", "fxpair": "XAUUSD"}
    request = urllib.request.Request(f"{HISTDATA}/get.php", data=urllib.parse.urlencode(form).encode(),
        headers={**HEADERS, "Referer": page_url})
    data = urllib.request.urlopen(request, timeout=300).read()
    zipfile.ZipFile(io.BytesIO(data)).testzip()
    cached.write_bytes(data)
    return data


def histdata_frame(data: bytes) -> pd.DataFrame:
    archive = zipfile.ZipFile(io.BytesIO(data))
    name = next(n for n in archive.namelist() if n.endswith(".csv"))
    d = pd.read_csv(archive.open(name), sep=";", header=None, names=["time", "open", "high", "low", "close", "volume"])
    d.index = pd.to_datetime(d.pop("time"), format="%Y%m%d %H%M%S").dt.tz_localize("UTC") + pd.Timedelta(hours=5)
    return d


def dukascopy(path: str) -> bytes:
    cached = CACHE / path.replace("/", "_")
    if cached.exists():
        return cached.read_bytes()
    for attempt in range(8):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(f"{DUKASCOPY}/{path}", headers=HEADERS), timeout=180).read()
            data = lzma.decompress(raw) if raw else b""
            cached.write_bytes(data)
            return data
        except Exception:  # noqa: BLE001 - the datafeed throttles; retry with backoff
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(path)


def dukascopy_close(day: date, side: str) -> pd.Series:
    data = dukascopy(f"{day.year}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5")
    if not data:
        return pd.Series(dtype=float)
    rows = np.frombuffer(data, dtype=">u4").reshape(-1, 6)
    volume = np.frombuffer(data, dtype=">f4").reshape(-1, 6)[:, 5]
    index = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rows[:, 0].astype(np.int64), unit="s")
    return pd.Series(rows[:, 2] * 0.001, index=index)[volume > 0]


def spread_days() -> list[date]:
    days = []
    for year in range(2020, 2025):
        for month in range(1, 13):
            d = date(year, month, 15)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            days.append(d)
    return days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spread", action="store_true", help="only the Dukascopy spread sample")
    parser.add_argument("--cached", action="store_true", help="with --spread: use only days already downloaded")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.spread:
        parts = []
        for d in spread_days():
            names = [f"{d.year}_{d.month - 1:02d}_{d.day:02d}_{side}_candles_min_1.bi5" for side in ("BID", "ASK")]
            if args.cached and not all((CACHE / n).exists() for n in names):
                continue
            bid, ask = dukascopy_close(d, "BID"), dukascopy_close(d, "ASK")
            joined = pd.concat([bid.rename("bid"), ask.rename("ask")], axis=1, join="inner")
            parts.append((joined.ask - joined.bid).rename("spread"))
            print(f"spread {d}: {len(joined)} minutes", flush=True)
        spread = pd.concat(parts).sort_index()
        spread.index = spread.index.asi8 // 1_000_000
        spread.rename_axis("timestamp").to_csv(OUT / "XAUUSD_spread_1m.csv")
        return
    frames = []
    for year in YEARS:
        frames.append(histdata_frame(histdata_zip(year)))
        print(f"{year}: {len(frames[-1])} bars", flush=True)
    for year, month in CURRENT_YEAR_MONTHS:
        try:
            frames.append(histdata_frame(histdata_zip(year, month)))
            print(f"{year}-{month:02d}: {len(frames[-1])} bars", flush=True)
        except Exception as exc:  # noqa: BLE001 - the newest month may not be published yet
            print(f"{year}-{month:02d}: not available ({exc})", flush=True)
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated()]
    bars.index = bars.index.asi8 // 1_000_000
    bars.rename_axis("timestamp").to_csv(OUT / "XAUUSD_1m.csv")
    print(f"bars {len(bars)}")


if __name__ == "__main__":
    main()
