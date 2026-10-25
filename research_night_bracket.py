"""Asian-session (New York evening) mean reversion with a $30 net target / $100 stop.

Rationale: intraday FX returns reverse most at ~15-minute horizons and the
post-New-York session is thin and range bound. Earlier project tests only used
07-17 UTC. Pre-registered grid (protocol.json) is fixed before any result.
Periods: development 2009-2017, validation 2018-2022, audit 2023+.
No broker orders; retrospective research only.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_indicators import add_indicators, compute_rsi
from research_bracket_core import bracket_trades, cost_pips, day_test, load_pair, summarize

OUT = ROOT / "research/night_bracket"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURCHF",
    "EURJPY", "EURAUD", "EURCAD", "EURNZD", "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "CADCHF", "CADJPY", "CHFJPY", "NZDCAD", "NZDCHF", "NZDJPY"]
PERIODS = {"development": ("2009-01-01", "2018-01-01"), "validation": ("2018-01-01", "2023-01-01"),
    "audit": ("2023-01-01", "2027-01-01")}
ENTRY_NY_HOURS = (19, 20, 21, 22, 23)
EXIT_NY_HOUR = 2
STOPS_ATR = (2., 3., 5.)


def signals(bars):
    ind = add_indicators(bars)
    close = bars.Close
    rsi3 = compute_rsi(close, 3)
    low16, high16 = bars.Low.rolling(16).min(), bars.High.rolling(16).max()
    position = (close - low16) / (high16 - low16).replace(0, np.nan)
    mean48, std48 = close.rolling(48).mean(), close.rolling(48).std()
    z = (close - mean48) / std48.replace(0, np.nan)

    def state(long, short):
        return np.select([long, short], [1, -1], default=0).astype(int)

    rules = {"bb_fade": state(close < ind.BBLow, close > ind.BBUp),
        "rsi3_fade": state(rsi3 < 10, rsi3 > 90),
        "range16_fade": state(position < .1, position > .9),
        "z48_fade": state(z < -2, z > 2)}
    return rules, ind.ATR14.to_numpy(float)


def deadlines_for(decisions):
    ny = decisions.tz_convert("America/New_York")
    day = ny.normalize() + pd.Timedelta(days=1)
    local = pd.DatetimeIndex([d.replace(hour=EXIT_NY_HOUR) for d in day.tz_localize(None)])
    return local.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def run_pair(symbol, data_dir=ROOT / "data", out=OUT):
    bars, fills = load_pair(symbol, Path(data_dir))
    rules, atr = signals(bars)
    decisions = bars.index + pd.Timedelta(minutes=15)
    ny_hour = decisions.tz_convert("America/New_York").hour
    window = np.isin(ny_hour, ENTRY_NY_HOURS)
    window &= np.arange(len(bars)) >= 200
    bad15 = np.zeros(len(bars), bool)
    bad15[bars.attrs.get("bad_positions", [])] = True
    window &= ~pd.Series(bad15).rolling(201, min_periods=1).max().astype(bool).to_numpy()
    deadlines = deadlines_for(decisions)
    costs = cost_pips(symbol, decisions)
    rows, frames = [], {}
    for name, state in rules.items():
        fresh = (state != 0) & (state != np.r_[0, state[:-1]]) & window
        idx = np.flatnonzero(fresh)
        for k in STOPS_ATR:
            trades = bracket_trades(symbol, fills, decisions[idx], state[idx], atr[idx] * k,
                deadlines[idx], costs[idx])
            key = f"{name}_{k:g}atr"
            frames[key] = trades
            for period, (start, end) in PERIODS.items():
                part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                rows.append({"symbol": symbol, "rule": name, "stop_atr": k, "period": period, **summarize(part)})
    folder = Path(out) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat([f.assign(config=k) for k, f in frames.items() if len(f)], ignore_index=True).to_csv(
        folder / "trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: done", flush=True)
    return rows


def evaluate(summary, out=OUT):
    """Pre-registered gates; development decides, validation confirms, audit reports."""
    wide = summary.pivot_table(index=["symbol", "rule", "stop_atr"], columns="period",
        values=["trades", "mean_usd", "stress_mean_usd"])
    picks = []
    for (symbol, rule, stop), row in wide.iterrows():
        dev_ok = row[("trades", "development")] >= 150 and row[("stress_mean_usd", "development")] > 0
        val_ok = row[("trades", "validation")] >= 75 and row[("mean_usd", "validation")] > 0
        if dev_ok:
            picks.append({"symbol": symbol, "rule": rule, "stop_atr": stop, "validation_pass": bool(val_ok),
                **{f"{m}_{p}": row[(m, p)] for m in ("trades", "mean_usd", "stress_mean_usd") for p in PERIODS}})
    picks = pd.DataFrame(picks)
    survivors = picks[picks.validation_pass] if len(picks) else picks
    audit = []
    for _, s in survivors.iterrows():
        trades = pd.read_csv(Path(out) / s.symbol / "trades.csv.gz", parse_dates=["entry_time", "exit_time"])
        t = trades[(trades.config == f"{s.rule}_{s.stop_atr:g}atr") & (trades.entry_time >= PERIODS["audit"][0])]
        test = day_test(t)
        audit.append({**s.to_dict(), **{f"audit_{k}": v for k, v in summarize(t).items()},
            "audit_ci_low": test["ci_low"], "audit_p": test["p_value"],
            "audit_years_positive": int((t.groupby(t.entry_time.dt.year).pnl_usd.mean() > 0).sum()) if len(t) else 0})
    return picks, pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "entry_ny_hours": ENTRY_NY_HOURS, "time_exit_ny_hour": EXIT_NY_HOUR, "stops_atr15m": STOPS_ATR,
        "rules": ["bb_fade", "rsi3_fade", "range16_fade", "z48_fade"], "periods": PERIODS,
        "costs": "measured Dukascopy median spread at the decision New York hour + 0.7 pip; stress 2x",
        "gates": {"development": ">=150 trades and stress mean > 0",
            "validation": ">=75 trades and mean > 0", "audit": "report only; daily block bootstrap"},
        "script_sha256": digest(Path(__file__))})
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(run_pair, args.symbols):
            rows.extend(part)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    picks, audit = evaluate(summary)
    picks.to_csv(OUT / "development_picks.csv", index=False)
    audit.to_csv(OUT / "audit.csv", index=False)
    print(summary.groupby(["rule", "stop_atr", "period"])[["mean_usd", "stress_mean_usd", "target_rate"]].mean().round(2).to_string())
    print(f"development picks: {len(picks)}; validation survivors: {len(audit)}")
    if len(audit):
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
