"""Weekend gap fade with a $30 net target / $100 planned stop.

The Sunday open often differs from Friday's close; a popular retail rule fades
the gap back toward Friday's close. The first Sunday hour carries the 17:00 New
York rollover spread (5-40x normal), so entries wait until a fixed later New York
hour and only if the gap is still open then. Grid fixed in protocol.json.
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
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from research_bracket_core import bracket_trades, cost_pips, load_pair, summarize
from research_night_bracket import PAIRS, PERIODS

OUT = ROOT / "research/weekend_gap"
ENTRY_NY_HOURS = (18, 20)
MIN_GAP_ATR = (0.1, 0.25)
STOPS_ATR = (0.5, 1.)
HOLD_HOURS = 24


def weekend_events(fills, daily_atr, entry_ny_hour):
    index = fills["index"]
    gaps = np.flatnonzero(np.diff(index.asi8) > pd.Timedelta(hours=36).value) + 1
    rows = []
    for g in gaps:
        friday_close, sunday_open = fills["c"][g - 1], fills["o"][g]
        ny_open = index[g].tz_convert("America/New_York")
        entry_local = ny_open.normalize() + pd.Timedelta(hours=entry_ny_hour)
        if entry_local <= ny_open:
            continue
        entry = entry_local.tz_convert("UTC")
        j = index.searchsorted(entry)
        if j >= len(index) or index[j] != entry:
            continue
        side = -np.sign(sunday_open - friday_close)
        # Still open at the decision: price has not traded back through Friday's close.
        seen_low, seen_high = fills["l"][g:j].min(), fills["h"][g:j].max()
        still_open = seen_low > friday_close if side < 0 else seen_high < friday_close
        atr = daily_atr.asof(index[g - 1])
        rows.append({"entry_time": entry, "side": int(side), "gap": abs(sunday_open - friday_close),
            "atr": atr, "still_open": bool(still_open and side != 0)})
    return pd.DataFrame(rows)


def run_pair(symbol, data_dir=ROOT / "data", out=OUT):
    d = load_hourly(Path(data_dir) / f"historical_1d/{symbol}.csv", quarantine=True)
    d = d[d.index.dayofweek < 5]
    # ATR known at Friday's close: shift so the value at a bar is from completed bars only.
    daily_atr = compute_atr(d).shift(1)
    daily_atr.index = d.index + pd.Timedelta(days=1)
    _, fills = load_pair(symbol, Path(data_dir), "1d")
    rows, frames = [], []
    for hour in ENTRY_NY_HOURS:
        events = weekend_events(fills, daily_atr, hour)
        events = events[events.still_open & np.isfinite(events.atr)]
        times = pd.DatetimeIndex(events.entry_time)
        costs = cost_pips(symbol, times)
        for min_gap in MIN_GAP_ATR:
            sel = (events.gap >= min_gap * events.atr).to_numpy()
            for k in STOPS_ATR:
                trades = bracket_trades(symbol, fills, times[sel], events.side.to_numpy()[sel],
                    (events.atr * k).to_numpy()[sel], times[sel] + pd.Timedelta(hours=HOLD_HOURS), costs[sel])
                key = f"ny{hour}_gap{min_gap:g}_{k:g}atr"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"symbol": symbol, "entry_ny_hour": hour, "min_gap_atr": min_gap, "stop_atr": k,
                        "period": period, **summarize(part)})
    folder = Path(out) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    frames = [f for f in frames if len(f)]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: done", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "entry_ny_hours_sunday": ENTRY_NY_HOURS, "min_gap_daily_atr": MIN_GAP_ATR, "stops_daily_atr": STOPS_ATR,
        "hold_hours": HOLD_HOURS, "periods": PERIODS,
        "gates": {"development": ">=100 trades and stress mean > 0", "validation": ">=50 trades and mean > 0"},
        "script_sha256": digest(Path(__file__))})
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(run_pair, args.symbols):
            rows.extend(part)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.groupby(["entry_ny_hour", "min_gap_atr", "stop_atr", "period"])[
        ["trades", "mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"]].mean().round(2).to_string())
    wide = s.pivot_table(index=["symbol", "entry_ny_hour", "min_gap_atr", "stop_atr"], columns="period",
        values=["trades", "mean_usd", "stress_mean_usd"])
    dev = (wide[("trades", "development")] >= 100) & (wide[("stress_mean_usd", "development")] > 0)
    val = dev & (wide[("trades", "validation")] >= 50) & (wide[("mean_usd", "validation")] > 0)
    print(f"development picks: {int(dev.sum())}; validation survivors: {int(val.sum())}")
    print(wide[val].round(2).to_string())


if __name__ == "__main__":
    main()
