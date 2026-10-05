"""Measured outcomes of the Özet "İşlem Planı" for the live app (descriptive, not a hypothesis test).

Each full hour from 2023-01-01 (the ML direction model's out-of-sample period) the plan enters at the 5M open
in the ML model's direction (probability_up > 0.5 -> LONG), with the app's technical stop of 1.5 x ATR14 on
closed 15M bars and a target of `ratio` x stop. The trade ends at whichever is touched first within the next
288 5M bars (one trading day); a bar that touches both counts as a stop (conservative); otherwise it closes at
the last bar. The measured round-trip cost (spread at the entry's New York hour + 0.7 pip) is subtracted.
The same is measured for the opposite side so the app can show how much the ML side adds over a coin flip.

    python -X utf8 research_trade_plan.py [--pairs EURUSD ...]   # writes data/ml/trade_plan_stats.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_indicators import add_indicators
from research_consensus import PAIRS, ml_probabilities, read_archive

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "ml" / "trade_plan_stats.json"
START, END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-09-10", tz="UTC")
RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
STOP_ATR = 1.5  # build_intraday_opportunity: stop_pips = ATR14 * 1.5 / pip
HORIZON_BARS = 288
WORKERS = 7


def first_touch(hit: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or a value past the window when there is none."""
    return np.where(hit.any(axis=1), hit.argmax(axis=1), hit.shape[1] + 1)


def simulate(entry, highs, lows, last_close, side, stop, target) -> tuple[np.ndarray, np.ndarray]:
    """Gross pips and outcome code (1 target, -1 stop, 0 timeout) per trade."""
    favourable = np.where(side[:, None] > 0, highs - entry[:, None], entry[:, None] - lows)
    adverse = np.where(side[:, None] > 0, entry[:, None] - lows, highs - entry[:, None])
    t_hit, s_hit = first_touch(favourable >= target[:, None]), first_touch(adverse >= stop[:, None])
    outcome = np.where(s_hit <= t_hit, np.where(s_hit <= HORIZON_BARS, -1, 0), 1)
    gross = np.where(outcome == 1, target, np.where(outcome == -1, -stop, side * (last_close - entry)))
    return gross, outcome


def pair_stats(pair: str) -> tuple[str, dict]:
    pip = get_pip_size(pair)
    hourly = read_archive("historical_1h_from_15m", pair, pd.Timestamp("2000-01-01", tz="UTC"))
    probs = ml_probabilities(pair, hourly)
    bars15 = read_archive("historical_15m", pair, START - pd.Timedelta(days=30))
    atr = add_indicators(bars15[["Open", "High", "Low", "Close"]])["ATR14"]
    bars5 = read_archive("historical_5m", pair, START)

    times = bars5.index[(bars5.index >= START) & (bars5.index < END) & (bars5.index.minute == 0)]
    prob = probs.reindex(times - pd.Timedelta(hours=1)).to_numpy()  # last closed hourly bar
    atr_at = atr.reindex(times - pd.Timedelta(minutes=15)).to_numpy()  # last closed 15M bar
    keep = np.isfinite(prob) & (prob != 0.5) & np.isfinite(atr_at) & (atr_at > 0)
    times, prob, atr_at = times[keep], prob[keep], atr_at[keep]

    entry_pos = bars5.index.get_indexer(times)
    keep = entry_pos + HORIZON_BARS <= len(bars5)
    times, prob, atr_at, entry_pos = times[keep], prob[keep], atr_at[keep], entry_pos[keep]
    highs = sliding_window_view(bars5["High"].to_numpy(), HORIZON_BARS)[entry_pos] / pip
    lows = sliding_window_view(bars5["Low"].to_numpy(), HORIZON_BARS)[entry_pos] / pip
    entry = bars5["Open"].to_numpy()[entry_pos] / pip
    last_close = bars5["Close"].to_numpy()[entry_pos + HORIZON_BARS - 1] / pip
    stop = STOP_ATR * atr_at / pip
    profile = load_spread_profile()
    hours = times.tz_convert("America/New_York").hour
    cost = np.array([measured_spread_pips(profile, pair, h) or 1.5 for h in hours]) + ECN_COMMISSION_PIPS
    ml_side = np.where(prob > 0.5, 1, -1)

    stats = {"n": int(len(times)), "median_stop_pips": float(np.median(stop)), "median_cost_pips": float(np.median(cost)),
             "ratios": {}}
    for ratio in RATIOS:
        row = {}
        for name, side in (("ml", ml_side), ("opposite", -ml_side)):
            gross, outcome = simulate(entry, highs, lows, last_close, side, stop, ratio * stop)
            net = gross - cost
            row[name] = {"target_rate": float((outcome == 1).mean()), "stop_rate": float((outcome == -1).mean()),
                         "timeout_rate": float((outcome == 0).mean()), "net_r": float((net / stop).mean()),
                         "net_pips": float(net.mean()), "win_rate": float((net > 0).mean())}
        stats["ratios"][str(ratio)] = row
    return pair, stats


def main(pairs: list[str]) -> None:
    with Pool(min(WORKERS, len(pairs))) as pool:
        results = dict(pool.map(pair_stats, pairs))
    pooled = {}
    total = sum(r["n"] for r in results.values())
    for ratio in map(str, RATIOS):
        pooled[ratio] = {name: {key: sum(r["ratios"][ratio][name][key] * r["n"] for r in results.values()) / total
                                for key in results[pairs[0]]["ratios"][ratio][name]} for name in ("ml", "opposite")}
    payload = {"as_of": "2026-10-05", "period": [str(START.date()), str(END.date())], "stop_atr_15m": STOP_ATR,
               "horizon": "288 5M bars (one trading day)", "ratios": RATIOS,
               "note": "Descriptive. ML side = sign(probability_up - 0.5); cost = measured spread at NY hour + 0.7 pip.",
               "pooled": {"n": total, "ratios": pooled}, "pairs": results}
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for ratio in map(str, RATIOS):
        ml, op = pooled[ratio]["ml"], pooled[ratio]["opposite"]
        print(f"ratio {ratio:>4}: target first {ml['target_rate']:.3f} (opp {op['target_rate']:.3f})  stop {ml['stop_rate']:.3f}  "
              f"net {ml['net_r']:+.3f} R / {ml['net_pips']:+.2f} pip (opp {op['net_r']:+.3f} R)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    main(parser.parse_args().pairs)
