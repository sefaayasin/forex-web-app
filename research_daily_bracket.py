"""Daily-signal entries with a $30 net target / $100 planned stop.

Earlier project tests used 15M stops where spread + commission was 15-25% of
the risk. Here stops are 1-2 daily ATRs, so the same cost is ~1-3% of risk and
a small directional tendency could survive. Literature motivation: time-series
momentum (Moskowitz, Ooi & Pedersen 2012), carry (Lustig et al. 2011), short-term
reversal. Overnight swap is charged from 3-month rate differentials minus a 1%/yr
retail markup, with triple Wednesday rollover. Grid fixed in protocol.json.
Periods: development 2009-2017, validation 2018-2022, audit 2023+.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_config import get_pip_size
from forex_indicators import compute_atr, compute_rsi
from forex_ml_tournament import load_hourly
from research_bracket_core import bracket_trades, cost_pips, day_test, load_pair, summarize
from research_night_bracket import PAIRS, PERIODS

OUT = ROOT / "research/daily_bracket"
RATES = ROOT / "research/fx_factors/rates.csv"
STOPS_ATR = (1., 2.)
HOLD_DAYS = (7, 14)  # calendar days: about 5 and 10 trading days
SWAP_MARKUP = 0.01


def daily_rules(d, base_minus_quote):
    close = d.Close
    rsi2 = compute_rsi(close, 2)
    ibs = (close - d.Low) / (d.High - d.Low).replace(0, np.nan)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()

    def state(long, short):
        return np.select([long, short], [1, -1], default=0).astype(int)

    def fresh(s):
        return np.where(s != np.r_[0, s[:-1]], s, 0)

    rules = {
        "tsmom20": np.sign(close.pct_change(20).fillna(0)).astype(int).to_numpy(),
        "tsmom60": np.sign(close.pct_change(60).fillna(0)).astype(int).to_numpy(),
        "tsmom250": np.sign(close.pct_change(250).fillna(0)).astype(int).to_numpy(),
        "sma50_200": state(sma50 > sma200, sma50 < sma200),
        "donchian20": fresh(state(close > d.High.shift(1).rolling(20).max(), close < d.Low.shift(1).rolling(20).min())),
        "rsi2_fade": state(rsi2 < 10, rsi2 > 90),
        "ibs_fade": state(ibs < .15, ibs > .85),
        "rev5": -np.sign(close.pct_change(5).fillna(0)).astype(int).to_numpy(),
        "carry": np.sign(base_minus_quote).astype(int),
    }
    rules["carry_tsmom60"] = np.where(rules["carry"] == rules["tsmom60"], rules["carry"], 0)
    return rules


def rate_differential(symbol, index):
    rates = pd.read_csv(RATES, index_col="month", parse_dates=True).ffill()
    index = pd.DatetimeIndex(index).tz_localize(None)
    month = index.to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)  # previous month: known in advance
    months = month.unique()
    lagged = rates.reindex(rates.index.union(months)).ffill().reindex(months)
    base, quote = symbol[:3], symbol[3:]
    return (lagged[base] - lagged[quote]).reindex(month).to_numpy(float)


def rollovers(entry, exit_):
    """New York 17:00 rollovers crossed; Wednesday counts three (spot weekend value dates)."""
    days = pd.date_range(entry.tz_convert("America/New_York").normalize(),
        exit_.tz_convert("America/New_York").normalize() + pd.Timedelta(days=1), freq="D")
    roll = (days + pd.Timedelta(hours=17)).tz_convert("UTC") if days.tz else days
    crossed = roll[(roll > entry) & (roll <= exit_)]
    return int(sum(3 if r.tz_convert("America/New_York").dayofweek == 2 else 1 for r in crossed if r.tz_convert("America/New_York").dayofweek < 5))


def run_pair(symbol, data_dir=ROOT / "data", out=OUT):
    d = load_hourly(Path(data_dir) / f"historical_1d/{symbol}.csv", quarantine=True)
    d = d[d.index.dayofweek < 5]  # Sunday stubs are a few hours of Monday's session
    _, fills = load_pair(symbol, Path(data_dir), "1d")
    pip = get_pip_size(symbol)
    diff = rate_differential(symbol, d.index)
    rules = daily_rules(d, diff)
    atr = compute_atr(d).to_numpy(float)
    decisions = d.index + pd.Timedelta(days=1)
    usable = (d.index.dayofweek < 4) & (np.arange(len(d)) >= 260) & np.isfinite(diff)
    costs = cost_pips(symbol, decisions)
    rows, frames = [], []
    for name, state in rules.items():
        idx = np.flatnonzero((state != 0) & usable)
        for k in STOPS_ATR:
            for hold in HOLD_DAYS:
                trades = bracket_trades(symbol, fills, decisions[idx], state[idx], atr[idx] * k,
                    decisions[idx] + pd.Timedelta(days=hold), costs[idx])
                if len(trades):
                    pos = decisions.get_indexer(trades.entry_time)
                    nights = np.array([rollovers(a, b) for a, b in zip(trades.entry_time, trades.exit_time)])
                    price = fills["o"][fills["index"].get_indexer(trades.entry_time)]
                    carry = trades.side * diff[pos] / 100 - SWAP_MARKUP
                    swap_pips = carry * price / 365 * nights / pip
                    risk = trades.stop_pips + trades.cost_pips
                    trades["swap_pips"] = swap_pips
                    trades["pnl_usd"] += swap_pips / risk * 100
                    trades["pnl_stress_usd"] += (swap_pips - np.abs(price * SWAP_MARKUP / 365 * nights / pip)) / risk * 100
                    trades["net_r"] = trades.pnl_usd / 100
                key = f"{name}_{k:g}atr_{hold}d"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"symbol": symbol, "rule": name, "stop_atr": k, "hold_days": hold,
                        "period": period, **summarize(part)})
    folder = Path(out) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat([f for f in frames if len(f)], ignore_index=True).to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: done", flush=True)
    return rows


def evaluate(summary, out=OUT):
    wide = summary.pivot_table(index=["symbol", "rule", "stop_atr", "hold_days"], columns="period",
        values=["trades", "mean_usd", "stress_mean_usd"])
    picks = []
    for (symbol, rule, stop, hold), row in wide.iterrows():
        if row[("trades", "development")] >= 100 and row[("stress_mean_usd", "development")] > 0:
            picks.append({"symbol": symbol, "rule": rule, "stop_atr": stop, "hold_days": hold,
                "validation_pass": bool(row[("trades", "validation")] >= 50 and row[("mean_usd", "validation")] > 0),
                **{f"{m}_{p}": row[(m, p)] for m in ("trades", "mean_usd", "stress_mean_usd") for p in PERIODS}})
    picks = pd.DataFrame(picks)
    audit = []
    for _, s in (picks[picks.validation_pass] if len(picks) else picks).iterrows():
        trades = pd.read_csv(Path(out) / s.symbol / "trades.csv.gz", parse_dates=["entry_time", "exit_time"])
        t = trades[(trades.config == f"{s.rule}_{s.stop_atr:g}atr_{s.hold_days:g}d") & (trades.entry_time >= PERIODS["audit"][0])]
        test = day_test(t)
        audit.append({**s.to_dict(), **{f"audit_{k}": v for k, v in summarize(t).items()},
            "audit_ci_low": test["ci_low"], "audit_p": test["p_value"]})
    return picks, pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "decision": "close of Mon-Thu Dukascopy UTC daily bar; entry next 5M open",
        "rules": ["tsmom20", "tsmom60", "tsmom250", "sma50_200", "donchian20", "rsi2_fade", "ibs_fade",
            "rev5", "carry", "carry_tsmom60"], "stops_daily_atr": STOPS_ATR, "hold_calendar_days": HOLD_DAYS,
        "swap": "previous-month 3M rate differential minus 1%/yr markup; stress also subtracts the markup again",
        "periods": PERIODS, "gates": {"development": ">=100 trades and stress mean > 0",
            "validation": ">=50 trades and mean > 0", "audit": "report only"},
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
    print(summary.groupby(["rule", "period"])[["mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"]].mean().round(2).to_string())
    print(f"development picks: {len(picks)}; validation survivors: {len(audit)}")
    if len(audit):
        print(audit.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
