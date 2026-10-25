"""CFTC speculative positioning (COT) as a weekly signal for a $30 net / $100 stop bracket.

Tuesday positions are published Friday 15:30 New York; entries wait for Monday
00:00 UTC so neither the report nor the weekend open spread is used early.
Hypotheses fixed in protocol.json: extreme speculator positioning reverses
(Tornell & Yuan 2012), speculators follow trends (follow), and recent changes in
positioning continue (flow4). Periods: development 2009-2017, validation
2018-2022, audit 2023+. No broker orders.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from research_bracket_core import bracket_trades, cost_pips, day_test, load_pair, summarize
from research_night_bracket import PERIODS

OUT = ROOT / "research/cot_bracket"
COT = ROOT / "data/news/cot/cot_currency_positioning.csv"
PAIRS = {"EURUSD": ("EUR", 1), "GBPUSD": ("GBP", 1), "AUDUSD": ("AUD", 1),
    "USDJPY": ("JPY", -1), "USDCHF": ("CHF", -1), "USDCAD": ("CAD", -1)}
STOPS_ATR = (2., 4.)
HOLD_DAYS = (7, 14)


def cot_signals(currency):
    d = pd.read_csv(COT, parse_dates=["date"])
    d = d[d.currency == currency].sort_values("date").set_index("date")
    share = d.noncommercial_net / d.open_interest
    z = (share - share.rolling(156, min_periods=104).mean()) / share.rolling(156, min_periods=104).std()
    # Tuesday as-of date -> Friday release -> next Monday 00:00 UTC.
    available = (d.index + pd.Timedelta(days=6)).tz_localize("UTC")
    return pd.DataFrame({"contrarian": np.select([z > 1.5, z < -1.5], [-1, 1], 0),
        "follow": np.sign(share).fillna(0).astype(int).to_numpy(),
        "flow4": np.sign(share.diff(4)).fillna(0).astype(int).to_numpy()},
        index=available)


def run_pair(symbol, data_dir=ROOT / "data", out=OUT):
    currency, orientation = PAIRS[symbol]
    d = load_hourly(Path(data_dir) / f"historical_1d/{symbol}.csv", quarantine=True)
    d = d[d.index.dayofweek < 5]
    atr = compute_atr(d)
    atr.index = d.index + pd.Timedelta(days=1)  # known after the bar closes
    _, fills = load_pair(symbol, Path(data_dir), "1d")
    signals = cot_signals(currency)
    rows, frames = [], []
    for name in signals.columns:
        # One candidate per day of the following week keeps re-entry after an exit.
        daily = signals[name].reindex(pd.date_range(signals.index[0], signals.index[-1] + pd.Timedelta(days=4),
            freq="D", tz="UTC"), method="ffill").fillna(0).astype(int)
        daily = daily[daily.index.dayofweek < 4] * orientation
        daily = daily[daily != 0]
        times = pd.DatetimeIndex(daily.index)
        a = atr.reindex(times, method="ffill").to_numpy(float)
        costs = cost_pips(symbol, times)
        for k in STOPS_ATR:
            for hold in HOLD_DAYS:
                trades = bracket_trades(symbol, fills, times, daily.to_numpy(int), a * k,
                    times + pd.Timedelta(days=hold), costs)
                key = f"{name}_{k:g}atr_{hold}d"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"symbol": symbol, "rule": name, "stop_atr": k, "hold_days": hold,
                        "period": period, **summarize(part)})
    folder = Path(out) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat([f for f in frames if len(f)], ignore_index=True).to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "signals": {"contrarian": "3y z-score of speculative net/OI beyond +-1.5, faded",
            "follow": "sign of speculative net", "flow4": "sign of 4-week change in net/OI"},
        "entry": "Mon-Thu 00:00 UTC after the Friday release; daily ATR stops", "stops_daily_atr": STOPS_ATR,
        "hold_calendar_days": HOLD_DAYS, "swap": "not charged (stated limitation)", "periods": PERIODS,
        "script_sha256": digest(Path(__file__))})
    rows = [r for symbol in PAIRS for r in run_pair(symbol)]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.groupby(["rule", "period"])[["trades", "mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"]].mean().round(2).to_string())
    wide = s.pivot_table(index=["symbol", "rule", "stop_atr", "hold_days"], columns="period",
        values=["trades", "mean_usd", "stress_mean_usd"])
    dev = (wide[("trades", "development")] >= 100) & (wide[("stress_mean_usd", "development")] > 0)
    val = dev & (wide[("trades", "validation")] >= 50) & (wide[("mean_usd", "validation")] > 0)
    print(f"development picks: {int(dev.sum())}; validation survivors: {int(val.sum())}")
    print(wide[val].round(2).to_string())


if __name__ == "__main__":
    main()
