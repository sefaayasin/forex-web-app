"""Measure real bid/ask spreads from Dukascopy by pair, hour of day, and around US releases.

Dukascopy is an ECN bank, so these are raw interbank-style spreads before any
commission: a realistic floor for a good account, not a retail broker's quote.
Samples 10 recent weekdays (two of each weekday) of 1-minute BID and ASK candles
per pair, plus the NFP/FOMC release days in the same window (minute candles for
USD pairs and tick files for the release hour of three majors).

Writes data/ml/spread_profile.json (read by the app) and research/spreads/results.json.
Downloads are cached in research/spreads/cache (git-ignored); rerunning resumes.

Run: python -X utf8 measure_spreads.py
"""
from __future__ import annotations

import json
import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import symbol_pair

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "spreads"
CACHE = OUTPUT / "cache"
PROFILE_PATH = ROOT / "data" / "ml" / "spread_profile.json"
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")
SAMPLE_DAYS = [date(2026, 6, 15), date(2026, 6, 23), date(2026, 7, 1), date(2026, 7, 9), date(2026, 7, 17),
               date(2026, 7, 20), date(2026, 8, 4), date(2026, 8, 12), date(2026, 8, 20), date(2026, 8, 28)]
# (UTC release time, label) in the sample window; NFP 08:30 ET, FOMC 14:00 ET (EDT = UTC-4).
RELEASES = [(pd.Timestamp("2026-07-02 12:30", tz="UTC"), "NFP"), (pd.Timestamp("2026-07-29 18:00", tz="UTC"), "FOMC"),
            (pd.Timestamp("2026-08-07 12:30", tz="UTC"), "NFP"), (pd.Timestamp("2026-09-04 12:30", tz="UTC"), "NFP")]
TICK_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
WORKERS = 6
RETRIES = 6


def points_per_pip(pair: str) -> float:
    """Dukascopy stores prices as integers in points: 1e-5 (1e-3 for JPY); a pip is 10 points."""
    return 10.0


def fetch(path: str) -> bytes:
    """Decompressed datafeed file, from the local cache when available."""
    cached = CACHE / path.replace("/", "_")
    if cached.exists():
        return cached.read_bytes()
    last_error = None
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(f"{BASE_URL}/{path}", headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(request, timeout=120).read()
            data = lzma.decompress(raw) if raw else b""
            cached.write_bytes(data)
            return data
        except Exception as exc:  # noqa: BLE001 - the datafeed throttles; retry with backoff
            last_error = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{path}: {last_error}")


def day_path(pair: str, day: date, side: str) -> str:
    return f"{pair}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def tick_path(pair: str, when: pd.Timestamp) -> str:
    return f"{pair}/{when.year}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"


def minute_candles(data: bytes, day: date) -> pd.DataFrame:
    rows = np.frombuffer(data, dtype=">u4").reshape(-1, 6) if data else np.empty((0, 6), dtype=">u4")
    volume = np.frombuffer(data, dtype=">f4").reshape(-1, 6)[:, 5].astype(np.float32) if data else np.empty(0)
    index = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(rows[:, 0].astype(np.int64), unit="s")
    return pd.DataFrame({"close": rows[:, 2].astype(np.int64), "volume": volume}, index=index)


def minute_spreads(pair: str, day: date) -> pd.Series:
    """Spread in pips at each minute close, only for minutes with ticks on both sides."""
    bid = minute_candles(fetch(day_path(pair, day, "BID")), day)
    ask = minute_candles(fetch(day_path(pair, day, "ASK")), day)
    joined = bid.join(ask, lsuffix="_bid", rsuffix="_ask", how="inner")
    active = joined[(joined.volume_bid > 0) & (joined.volume_ask > 0)]
    return (active.close_ask - active.close_bid) / points_per_pip(pair)


def available_minute_spreads(pair: str, days: list[date]) -> pd.Series:
    """Minute spreads over the days whose BID and ASK files were downloaded."""
    parts = [minute_spreads(pair, d) for d in days
             if all((CACHE / day_path(pair, d, s).replace("/", "_")).exists() for s in ("BID", "ASK"))]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def tick_spreads(pair: str, when: pd.Timestamp) -> pd.Series:
    if not (CACHE / tick_path(pair, when).replace("/", "_")).exists():
        return pd.Series(dtype=float)
    data = fetch(tick_path(pair, when))
    if not data:
        return pd.Series(dtype=float)
    ints = np.frombuffer(data, dtype=">u4").reshape(-1, 5)
    hour = when.floor("h")
    index = hour + pd.to_timedelta(ints[:, 0].astype(np.int64), unit="ms")
    return pd.Series((ints[:, 1].astype(np.int64) - ints[:, 2].astype(np.int64)) / points_per_pip(pair), index=index)


def prefetch(paths: list[str]) -> list[str]:
    failures = []

    def one(path: str) -> None:
        try:
            fetch(path)
        except RuntimeError as exc:
            failures.append(str(exc))

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(one, paths), start=1):
            if i % 50 == 0:
                print(f"{i}/{len(paths)} files", flush=True)
    return failures


def session_median(spreads: pd.Series, hours: range) -> float:
    part = spreads[spreads.index.hour.isin(list(hours))]
    return float(part.median()) if len(part) else float("nan")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    usd_pairs = [p for p in PAIRS if "USD" in symbol_pair(p)]
    release_days = sorted({when.date() for when, _ in RELEASES})
    paths = [day_path(p, d, s) for p in PAIRS for d in SAMPLE_DAYS for s in ("BID", "ASK")]
    paths += [day_path(p, d, s) for p in usd_pairs for d in release_days for s in ("BID", "ASK")]
    paths += [tick_path(p, when) for p in TICK_PAIRS for when, _ in RELEASES]
    failures = prefetch(paths)
    print(f"downloaded {len(paths) - len(failures)}/{len(paths)} files", flush=True)

    profile, summary = {}, {}
    for pair in PAIRS:
        spreads = available_minute_spreads(pair, SAMPLE_DAYS)
        # Keyed by New York hour: sessions and the 5 pm ET rollover follow local time, so the
        # summer sample stays aligned in winter when the UTC hours shift.
        by_hour = spreads.groupby(spreads.index.tz_convert("America/New_York").hour)
        profile[pair] = {
            "median_by_ny_hour": [round(float(by_hour.median().get(h, np.nan)), 2) for h in range(24)],
            "p90_by_ny_hour": [round(float(by_hour.quantile(.9).get(h, np.nan)), 2) for h in range(24)],
            "minutes": int(len(spreads)),
        }
        summary[pair] = {
            "median_all": round(float(spreads.median()), 2),
            "median_london_ny_12_16": round(session_median(spreads, range(12, 16)), 2),
            "median_london_7_12": round(session_median(spreads, range(7, 12)), 2),
            "median_asia_0_6": round(session_median(spreads, range(0, 6)), 2),
            "median_rollover_21_22": round(session_median(spreads, range(21, 23)), 2),
            "p90_all": round(float(spreads.quantile(.9)), 2),
        }

    news = []
    for when, label in RELEASES:
        for pair in usd_pairs:
            spreads = available_minute_spreads(pair, [when.date()])
            same_hour = available_minute_spreads(pair, SAMPLE_DAYS)
            if spreads.empty or same_hour.empty:
                continue
            normal = float(same_hour[same_hour.index.hour == when.hour].median())
            window = spreads[(spreads.index >= when) & (spreads.index < when + pd.Timedelta(minutes=5))]
            news.append({"release": str(when), "event": label, "pair": pair, "normal_same_hour_median": round(normal, 2),
                         "minute_close_max_first_5min": round(float(window.max()), 2) if len(window) else None})
        for pair in TICK_PAIRS:
            ticks = tick_spreads(pair, when)
            before = ticks[(ticks.index >= when - pd.Timedelta(minutes=5)) & (ticks.index < when)]
            after = ticks[(ticks.index >= when) & (ticks.index < when + pd.Timedelta(seconds=60))]
            normal = float(before.median()) if len(before) else float("nan")
            later = ticks[ticks.index >= when]
            back = later[later <= 2 * normal]
            news.append({"release": str(when), "event": label, "pair": pair, "source": "ticks",
                         "median_5min_before": round(normal, 2),
                         "max_first_60s": round(float(after.max()), 2) if len(after) else None,
                         "median_first_60s": round(float(after.median()), 2) if len(after) else None,
                         "seconds_until_back_within_2x": round(float((back.index[0] - when).total_seconds()), 1) if len(back) else None})

    as_of = str(pd.Timestamp.now(tz="UTC").floor("min"))
    PROFILE_PATH.write_text(json.dumps({
        "as_of": as_of, "source": "Dukascopy 1-minute BID/ASK closes, raw ECN spread without commission",
        "hour_basis": "America/New_York local hour",
        "sample_days": [str(d) for d in SAMPLE_DAYS], "pairs": profile}, indent=2), encoding="utf-8")
    (OUTPUT / "results.json").write_text(json.dumps({"as_of": as_of, "failed_downloads": failures,
                                                    "summary": summary, "news": news}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).T.sort_values("median_all").to_string())
    print(pd.DataFrame(news).to_string())


if __name__ == "__main__":
    main()
