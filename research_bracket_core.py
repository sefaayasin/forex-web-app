"""Shared fast bracket simulator for the $30 net target / $100 planned stop research.

One position per pair; entry at the next 5M open after a closed decision bar;
stop wins when both levels are touched in the same 5M bar; gaps fill at the
open. Targets and stops are NET of the one-off spread + commission, matching
research_cash_target.py. No broker orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_decision_core import stationary_bootstrap_mean_test
from forex_ml_tournament import load_hourly

STOP, TARGET, TIME, STOP_GAP, TARGET_GAP = 0, 1, 2, 3, 4


@njit(cache=True)
def _run(o, h, l, c, t, bad, entries, sides, stops, targets, deadlines):
    n = len(entries)
    exit_price = np.full(n, np.nan)
    exit_pos = np.full(n, -1)
    reason = np.full(n, -1)
    free_from = -1
    for k in range(n):
        j = entries[k]
        if j <= free_from:
            continue
        side, price = sides[k], o[j]
        sp, tp = price - side * stops[k], price + side * targets[k]
        i = j
        done = False
        while i < len(o) and t[i] < deadlines[k]:
            if i > j:
                if side * (o[i] - sp) <= 0:
                    exit_price[k], exit_pos[k], reason[k] = o[i], i, STOP_GAP
                    done = True
                    break
                if side * (o[i] - tp) >= 0:
                    exit_price[k], exit_pos[k], reason[k] = o[i], i, TARGET_GAP
                    done = True
                    break
            if (l[i] <= sp) if side == 1 else (h[i] >= sp):
                exit_price[k], exit_pos[k], reason[k] = sp, i, STOP
                done = True
                break
            if (h[i] >= tp) if side == 1 else (l[i] <= tp):
                exit_price[k], exit_pos[k], reason[k] = tp, i, TARGET
                done = True
                break
            i += 1
        if not done:
            if i == j or i >= len(o):
                continue  # no complete path inside the data
            exit_price[k], exit_pos[k], reason[k] = c[i - 1], i - 1, TIME
        if bad[exit_pos[k] + 1] != bad[j]:
            exit_pos[k], reason[k] = -1, -1  # quarantined price bar inside the path
            exit_price[k] = np.nan
            continue
        free_from = exit_pos[k]
    return exit_price, exit_pos, reason


def load_pair(symbol, data_dir, timeframe="15m"):
    bars = load_hourly(data_dir / f"historical_{timeframe}/{symbol}.csv", quarantine=True)
    fills = load_hourly(data_dir / "historical_5m" / f"{symbol}.csv", quarantine=True)
    bars.attrs["symbol"] = fills.attrs["symbol"] = symbol
    bad = np.zeros(len(fills), dtype=np.int64)
    bad[fills.attrs.get("bad_positions", [])] = 1
    return bars, {"o": fills.Open.to_numpy(float), "h": fills.High.to_numpy(float),
        "l": fills.Low.to_numpy(float), "c": fills.Close.to_numpy(float),
        "t": fills.index.asi8, "bad": np.r_[0, np.cumsum(bad)].astype(np.int64), "index": fills.index}


def cost_pips(symbol, decision_times, profile=None):
    profile = load_spread_profile() if profile is None else profile
    hours = pd.DatetimeIndex(decision_times).tz_convert("America/New_York").hour
    table = np.array([(measured_spread_pips(profile, symbol, hr) or np.nan) for hr in range(24)])
    return table[hours] + ECN_COMMISSION_PIPS


def bracket_trades(symbol, fills, decision_times, sides, stop_dist, deadlines, costs,
        target_ratio=0.3, loss_usd=100.):
    """Simulate candidate entries in time order; returns one row per executed trade."""
    pip = get_pip_size(symbol)
    decision_times = pd.DatetimeIndex(decision_times)
    j = fills["index"].searchsorted(decision_times)
    ok = (j < len(fills["t"])) & np.isfinite(stop_dist) & (stop_dist > 0) & np.isfinite(costs)
    ok[ok] &= fills["t"][j[ok]] == decision_times.asi8[ok]  # entry bar must exist at decision time
    idx = np.flatnonzero(ok)
    if not len(idx):
        return pd.DataFrame()
    stop_pips = stop_dist[idx] / pip
    cost = costs[idx]
    target_pips = cost + target_ratio * (stop_pips + cost)
    price, pos, reason = _run(fills["o"], fills["h"], fills["l"], fills["c"], fills["t"], fills["bad"],
        j[idx].astype(np.int64), sides[idx].astype(np.int64), stop_dist[idx].astype(float),
        (target_pips * pip).astype(float), pd.DatetimeIndex(deadlines)[idx].asi8)
    done = pos >= 0
    side = sides[idx][done]
    entry = fills["o"][j[idx][done]]
    gross = side * (price[done] - entry) / pip
    risk = stop_pips[done] + cost[done]
    out = pd.DataFrame({"entry_time": decision_times[idx][done], "side": side,
        "exit_time": fills["index"][pos[done]] + pd.Timedelta(minutes=5),
        "reason": np.array(["stop", "target", "time", "stop_gap", "target_gap"])[reason[done]],
        "stop_pips": stop_pips[done], "target_pips": target_pips[done], "cost_pips": cost[done],
        "gross_pips": gross})
    out["net_r"] = (gross - cost[done]) / risk
    out["pnl_usd"] = out.net_r * loss_usd
    out["pnl_stress_usd"] = (gross - 2 * cost[done]) / risk * loss_usd
    return out


def summarize(trades):
    if trades is None or trades.empty:
        return {"trades": 0, "mean_usd": None, "stress_mean_usd": None, "target_rate": None, "win_rate": None}
    return {"trades": int(len(trades)), "mean_usd": float(trades.pnl_usd.mean()),
        "stress_mean_usd": float(trades.pnl_stress_usd.mean()),
        "gross_mean_r": float((trades.gross_pips / (trades.stop_pips + trades.cost_pips)).mean()),
        "target_rate": float(trades.reason.isin(["target", "target_gap"]).mean()),
        "win_rate": float(trades.pnl_usd.gt(0).mean()),
        "worst_usd": float(trades.pnl_usd.min()),
        "median_cost_share": float((trades.cost_pips / (trades.stop_pips + trades.cost_pips)).median())}


def day_test(trades, column="pnl_usd", seed=73):
    if trades is None or len(trades) < 10:
        return {"ci_low": None, "ci_high": None, "p_value": None}
    daily = trades.groupby(trades.entry_time.dt.floor("D"))[column].sum().sort_index()
    test = stationary_bootstrap_mean_test(daily.to_list(), simulations=2000, seed=seed)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in test.items()}
