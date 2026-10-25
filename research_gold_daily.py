"""Gold (COMEX GC=F daily from Yahoo) with the $30 net target / $100 stop bracket.

Trend persistence is better documented in commodities than in currencies
(Moskowitz, Ooi & Pedersen 2012), and a daily stop makes cost a tiny share of
risk. Rules: the untested indicator zoo plus the daily trend/reversal set.
Development 2001-2012 (bull), validation 2013-2018 (bear and flat), audit 2019+.
Only daily bars exist, so when stop and target fall in the same day the stop is
assumed first. Grid fixed in protocol.json. No broker orders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from forex_adaptive import ROOT, digest, write_json
from forex_indicators import compute_atr
from research_bracket_core import bracket_trades, day_test, summarize
from research_daily_bracket import daily_rules
from research_indicator_zoo import zoo_rules

OUT = ROOT / "research/gold_daily"
PERIODS = {"development": ("2001-01-01", "2013-01-01"), "validation": ("2013-01-01", "2019-01-01"),
    "audit": ("2019-01-01", "2027-01-01")}
STOPS_ATR = (1., 2.)
HOLD_DAYS = (7, 14)
COST_USD = 0.40  # per ounce round trip: typical retail XAU/USD spread + commission
FINANCING = 0.02  # per year, charged on both sides (futures roll gaps already favor longs)
PIP = 0.01


def load_gold():
    d = yf.download("GC=F", start="2000-01-01", interval="1d", progress=False, auto_adjust=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    d = d[(d.High >= d.Low) & (d.Open > 0)]
    d.index = pd.DatetimeIndex(d.index).tz_localize("UTC")
    d["High"] = d[["Open", "High", "Close"]].max(axis=1)
    d["Low"] = d[["Open", "Low", "Close"]].min(axis=1)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "instrument": "GC=F daily (Yahoo)",
        "target_net_usd": 30, "planned_stop_usd": 100, "stops_daily_atr": STOPS_ATR, "hold_calendar_days": HOLD_DAYS,
        "cost_usd_per_oz": COST_USD, "financing_per_year_both_sides": FINANCING, "periods": PERIODS,
        "gates": {"development": ">=60 trades and stress mean > 0", "validation": ">=30 trades and mean > 0"},
        "script_sha256": digest(Path(__file__))})
    bars = load_gold()
    bars.to_csv(OUT / "gc_daily.csv")
    zoo, atr = zoo_rules(bars, "4h")
    trend = {k: v for k, v in daily_rules(bars, np.zeros(len(bars))).items() if not k.startswith("carry")}
    rules = {**{f"zoo_{k}": v for k, v in zoo.items()}, **trend}
    # Decision at the daily close, entry at the next day's open: the next bar's timestamp.
    decisions = pd.DatetimeIndex(np.r_[bars.index[1:].asi8, bars.index[-1].value + 86_400_000_000_000]).tz_localize("UTC")
    fills = {"o": bars.Open.to_numpy(float), "h": bars.High.to_numpy(float), "l": bars.Low.to_numpy(float),
        "c": bars.Close.to_numpy(float), "t": bars.index.asi8, "bad": np.zeros(len(bars) + 1, dtype=np.int64),
        "index": bars.index}
    costs = np.full(len(bars), COST_USD / PIP)
    usable = np.arange(len(bars)) >= 260
    rows, frames = [], []
    for name, st in rules.items():
        fresh = st if name in trend and name not in ("donchian20",) else np.where(st != np.r_[0, st[:-1]], st, 0)
        idx = np.flatnonzero((fresh != 0) & usable)[:-1] if len(fresh) else []
        for k in STOPS_ATR:
            for hold in HOLD_DAYS:
                trades = bracket_trades("XAUUSD", fills, decisions[idx], fresh[idx], atr[idx] * k,
                    decisions[idx] + pd.Timedelta(days=hold), costs[idx])
                if len(trades):
                    nights = (trades.exit_time.dt.normalize() - trades.entry_time.dt.normalize()).dt.days.to_numpy()
                    price = fills["o"][bars.index.get_indexer(trades.entry_time)]
                    swap = FINANCING * price / 365 * nights / PIP
                    risk = trades.stop_pips + trades.cost_pips
                    trades["pnl_usd"] -= swap / risk * 100
                    trades["pnl_stress_usd"] -= 2 * swap / risk * 100
                key = f"{name}_{k:g}atr_{hold}d"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"rule": name, "stop_atr": k, "hold_days": hold, "period": period, **summarize(part)})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.concat([f for f in frames if len(f)], ignore_index=True).to_csv(OUT / "trades.csv.gz", index=False, compression="gzip")
    wide = s.pivot_table(index=["rule", "stop_atr", "hold_days"], columns="period", values=["trades", "mean_usd", "stress_mean_usd"])
    dev = (wide[("trades", "development")] >= 60) & (wide[("stress_mean_usd", "development")] > 0)
    val = dev & (wide[("trades", "validation")] >= 30) & (wide[("mean_usd", "validation")] > 0)
    print(wide[("mean_usd",)].sort_values("development", ascending=False).head(20).round(2).to_string())
    print(f"configs {len(wide)}; development passes {int(dev.sum())}; validation passes {int(val.sum())}")
    print(wide[val].round(2).to_string())
    t_all = pd.concat(frames)
    for rule, stop, hold in wide[val].index:
        t = t_all[(t_all.config == f"{rule}_{stop:g}atr_{hold}d") & (t_all.entry_time >= PERIODS["audit"][0])]
        print(rule, stop, hold, summarize(t), day_test(t))


if __name__ == "__main__":
    main()
