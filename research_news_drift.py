"""Post-release drift after high-impact macro surprises, $30 net target / $100 stop.

Andersen, Bollerslev, Diebold & Vega (2003) show surprises move FX within minutes;
this asks whether, five minutes after the release, the move still has room to run
toward a small target before a wider stop. Direction comes from ForexFactory's own
better/worse flag (actualBetterWorse), so no polarity is chosen from results.
Surprise size uses only earlier releases of the same series. Grid fixed in
protocol.json before any outcome was computed. Periods: development 2010-2017,
validation 2018-2022, audit 2023+. No broker orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from research_bracket_core import bracket_trades, cost_pips, day_test, load_pair, summarize
from research_calendar_sources import parse_value

OUT = ROOT / "research/news_drift"
RAW_DIRS = [OUT / "raw_forexfactory", ROOT / "research/calendar_sources/raw/forexfactory"]
PERIODS = {"development": ("2010-01-01", "2018-01-01"), "validation": ("2018-01-01", "2023-01-01"),
    "audit": ("2023-01-01", "2027-01-01")}
# Each event currency trades against USD; USD events trade the four most liquid majors.
PAIRS = {"EUR": ["EURUSD"], "GBP": ["GBPUSD"], "JPY": ["USDJPY"], "AUD": ["AUDUSD"], "NZD": ["NZDUSD"],
    "CAD": ["USDCAD"], "CHF": ["USDCHF"], "USD": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]}
DELAY_MIN = (5, 15)
MIN_SURPRISE = (1., 2.)
STOPS_DAILY_ATR = (0.25, 0.5)
HOLD_HOURS = 4
NEWS_SPREAD_MULTIPLIER = 1.5


def load_events():
    rows = []
    for folder in RAW_DIRS:
        for path in sorted(folder.glob("*.json")):
            for day in json.loads(path.read_text(encoding="utf-8")):
                for e in day["events"]:
                    if e.get("impactName") != "high" or e.get("currency") not in PAIRS:
                        continue
                    rows.append({"id": e["id"], "series": e["ebaseId"], "name": e["name"], "currency": e["currency"],
                        "time": pd.Timestamp(e["dateline"], unit="s", tz="UTC"),
                        "actual": parse_value(e.get("actual")), "forecast": parse_value(e.get("forecast")),
                        "better_worse": int(e.get("actualBetterWorse") or 0)})
    d = pd.DataFrame(rows).drop_duplicates("id").dropna(subset=["actual", "forecast"]).sort_values("time")
    d["surprise"] = d.actual - d.forecast
    scale = d.groupby("series", sort=False).surprise.transform(lambda s: s.abs().shift(1).rolling(12, min_periods=4).median())
    d["z"] = d.surprise.abs() / scale.replace(0, np.nan)
    d["vote"] = np.select([d.better_worse == 1, d.better_worse == 2], [1, -1], 0)
    return d[d.z.notna() & (d.vote != 0)]


def pair_events(events, symbol, min_z):
    rows = []
    for currency, pairs in PAIRS.items():
        if symbol not in pairs:
            continue
        sign = 1 if symbol.startswith(currency) else -1
        part = events[(events.currency == currency) & (events.z >= min_z)]
        rows.append(part.assign(pair_vote=part.vote * sign))
    joined = pd.concat(rows)
    # Same-minute releases are netted; conflicting votes give no trade.
    votes = joined.groupby("time").pair_vote.sum()
    return np.sign(votes[votes != 0]).astype(int)


def run_pair(symbol, events, data_dir=ROOT / "data"):
    d = load_hourly(Path(data_dir) / f"historical_1d/{symbol}.csv", quarantine=True)
    d = d[d.index.dayofweek < 5]
    atr = compute_atr(d)
    atr.index = d.index + pd.Timedelta(days=1)  # value known after the daily bar closes
    _, fills = load_pair(symbol, Path(data_dir), "1d")
    rows, frames = [], []
    for min_z in MIN_SURPRISE:
        votes = pair_events(events, symbol, min_z)
        for delay in DELAY_MIN:
            times = pd.DatetimeIndex((votes.index + pd.Timedelta(minutes=delay)).ceil("5min"))
            a = atr.reindex(times, method="ffill").to_numpy(float)
            costs = cost_pips(symbol, times) * NEWS_SPREAD_MULTIPLIER
            for k in STOPS_DAILY_ATR:
                trades = bracket_trades(symbol, fills, times, votes.to_numpy(int), a * k,
                    times + pd.Timedelta(hours=HOLD_HOURS), costs)
                key = f"z{min_z:g}_d{delay}_{k:g}atr"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"symbol": symbol, "min_z": min_z, "delay_min": delay, "stop_atr": k,
                        "period": period, **summarize(part)})
    folder = OUT / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat([f for f in frames if len(f)], ignore_index=True).to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "events": "ForexFactory high impact, numeric actual+forecast, direction = actualBetterWorse",
        "surprise_z": "abs(actual-forecast) / median abs surprise of the previous 12 releases (min 4)",
        "min_z": MIN_SURPRISE, "entry_delay_minutes": DELAY_MIN, "stops_daily_atr": STOPS_DAILY_ATR,
        "hold_hours": HOLD_HOURS, "cost": f"measured spread x{NEWS_SPREAD_MULTIPLIER} + 0.7 pip; stress 2x",
        "pairs": PAIRS, "periods": PERIODS,
        "gates": {"development": ">=100 trades and stress mean > 0", "validation": ">=50 trades and mean > 0"},
        "script_sha256": digest(Path(__file__))})
    events = load_events()
    events.to_csv(OUT / "events.csv.gz", index=False, compression="gzip")
    print(f"events: {len(events)} from {events.time.min()} to {events.time.max()}")
    symbols = sorted({p for pairs in PAIRS.values() for p in pairs})
    rows = [r for symbol in symbols for r in run_pair(symbol, events)]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.groupby(["min_z", "delay_min", "stop_atr", "period"])[
        ["trades", "mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"]].mean().round(2).to_string())
    wide = s.pivot_table(index=["symbol", "min_z", "delay_min", "stop_atr"], columns="period",
        values=["trades", "mean_usd", "stress_mean_usd"])
    dev = (wide[("trades", "development")] >= 100) & (wide[("stress_mean_usd", "development")] > 0)
    val = dev & (wide[("trades", "validation")] >= 50) & (wide[("mean_usd", "validation")] > 0)
    print(f"development picks: {int(dev.sum())}; validation survivors: {int(val.sum())}")
    print(wide[val].round(2).to_string())
    for symbol, min_z, delay, stop in wide[val].index:
        t = pd.read_csv(OUT / symbol / "trades.csv.gz", parse_dates=["entry_time"])
        t = t[(t.config == f"z{min_z:g}_d{delay}_{stop:g}atr") & (t.entry_time >= PERIODS["audit"][0])]
        print(symbol, min_z, delay, stop, summarize(t), day_test(t))


if __name__ == "__main__":
    main()
