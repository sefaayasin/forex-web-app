"""Bounded 15M entry search for $30 net target / $100 planned stop.

Rules are selected on 2016-2022 only and audited on 2023+. Existing inspected
history is retrospective, never a new blind/forward test. No broker orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import MAJORS, ROOT, digest, write_json
from forex_analysis import bias_score_frame
from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_decision_core import stationary_bootstrap_mean_test
from forex_indicators import add_indicators, compute_rsi, wilder_average
from forex_ml_tournament import load_hourly

OUT = ROOT / "research/cash_target"
FIVE, FIFTEEN = pd.Timedelta(minutes=5), pd.Timedelta(minutes=15)


def rule_states(bars):
    ind = add_indicators(bars)
    close, atr = bars.Close, ind.ATR14
    up, down = bars.High.diff(), -bars.Low.diff()
    plus = wilder_average(up.where((up > down) & (up > 0), 0), 14)
    minus = wilder_average(down.where((down > up) & (down > 0), 0), 14)
    adx = wilder_average((plus - minus).abs() / (plus + minus).replace(0, np.nan), 14) * 100
    rsi2 = compute_rsi(close, 2)
    stochastic = (close - bars.Low.rolling(14).min()) / (
        bars.High.rolling(14).max() - bars.Low.rolling(14).min()).replace(0, np.nan)
    slope = ind.EMA50.diff(8)
    long_trend, short_trend = (ind.EMA50 > ind.EMA200) & (slope > 0), (ind.EMA50 < ind.EMA200) & (slope < 0)
    score = bias_score_frame(ind).score

    def state(long, short):
        return np.select([long, short], [1, -1], default=0).astype(int)

    rules = {
        "current_score": state(score >= 65, score <= -65),
        "score_contrarian": state(score <= -65, score >= 65),
        "ema_pullback": state(long_trend & (close > ind.EMA20) & (close.shift(1) <= ind.EMA20.shift(1)),
            short_trend & (close < ind.EMA20) & (close.shift(1) >= ind.EMA20.shift(1))),
        "rsi2_with_trend": state((close > ind.EMA200) & (rsi2 < 5), (close < ind.EMA200) & (rsi2 > 95)),
        "rsi2_fade": state(rsi2 < 5, rsi2 > 95),
        "rsi14_reentry": state((ind.RSI14 > 30) & (ind.RSI14.shift(1) <= 30),
            (ind.RSI14 < 70) & (ind.RSI14.shift(1) >= 70)),
        "bollinger_reentry": state((close > ind.BBLow) & (close.shift(1) <= ind.BBLow.shift(1)),
            (close < ind.BBUp) & (close.shift(1) >= ind.BBUp.shift(1))),
        "bollinger_range": state((adx < 20) & (close < ind.BBLow) & (ind.RSI14 < 30),
            (adx < 20) & (close > ind.BBUp) & (ind.RSI14 > 70)),
        "stochastic_reentry": state((stochastic > .2) & (stochastic.shift(1) <= .2),
            (stochastic < .8) & (stochastic.shift(1) >= .8)),
        "donchian_break": state(close > bars.High.shift(1).rolling(20).max(),
            close < bars.Low.shift(1).rolling(20).min()),
        "ema20_50_cross": state((ind.EMA20 > ind.EMA50) & (ind.EMA20.shift(1) <= ind.EMA50.shift(1)),
            (ind.EMA20 < ind.EMA50) & (ind.EMA20.shift(1) >= ind.EMA50.shift(1))),
    }
    rules["donchian_contrarian"] = -rules["donchian_break"]
    lower, upper = ind.EMA20 - atr * 1.5, ind.EMA20 + atr * 1.5
    rules["keltner_reentry"] = state((close > lower) & (close.shift(1) <= lower.shift(1)),
        (close < upper) & (close.shift(1) >= upper.shift(1)))
    return rules, atr


def simulate_path(values, entry, side, stop, target, bars):
    price = values[entry, 0]
    stop_price, target_price = price - side * stop, price + side * target
    for i in range(entry, entry + bars):
        opening, high, low, close = values[i]
        if i > entry and side * (opening - stop_price) <= 0:
            return opening, i, "stop_gap"
        if i > entry and side * (opening - target_price) >= 0:
            return target_price, i, "target_gap"
        if (low <= stop_price if side == 1 else high >= stop_price):
            return stop_price, i, "stop"
        if (high >= target_price if side == 1 else low <= target_price):
            return target_price, i, "target"
    return close, i, "time"


def prepare_paths(bars, fills):
    bad = np.zeros(len(fills), dtype=int)
    bad[fills.attrs.get("bad_positions", [])] = 1
    bad15 = pd.Series(False, index=bars.index)
    bad15.iloc[bars.attrs.get("bad_positions", [])] = True
    return {"values": fills[["Open", "High", "Low", "Close"]].to_numpy(),
        "bad_sum": np.r_[0, np.cumsum(bad)],
        "history_bad": bad15.rolling(501, min_periods=1).max().astype(bool).to_numpy()}


def simulate(bars, fills, state, atr, costs, start, end, stop_atr, hours, target_usd=30., loss_usd=100., entries=None, prepared=None):
    """Fixed hypothetical $100 risk budget; report account solvency separately."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    pip = get_pip_size(bars.attrs["symbol"])
    decisions = bars.index + FIFTEEN
    if entries is None:
        entries = np.flatnonzero((state != 0) & (state != np.r_[0, state[:-1]]))
    times = fills.index
    prepared = prepare_paths(bars, fills) if prepared is None else prepared
    values, bad_sum, history_bad = (prepared[k] for k in ("values", "bad_sum", "history_bad"))
    atr_values = atr.to_numpy()
    count = hours * 12
    free_from, rows = None, []
    for i in entries:
        when = decisions[i]
        if when < start or when + pd.Timedelta(hours=hours) >= end or i < 500 or history_bad[i]:
            continue
        # One fixed, declared liquid-session window, no per-period tuning.
        if not 7 <= when.hour < 17 or (free_from is not None and when < free_from):
            continue
        if not np.isfinite(atr_values[i]) or atr_values[i] <= 0 or not np.isfinite(costs[i]):
            continue
        j = times.searchsorted(when)
        if (j + count > len(times) or times[j] != when or
                times[j+count-1] != when + pd.Timedelta(hours=hours) - FIVE or
                bad_sum[j+count] != bad_sum[j]):
            continue
        side, cost = int(state[i]), float(costs[i])
        distance = float(atr_values[i] * stop_atr)
        stop_pips = distance / pip
        # Both budgets are NET of the assumed one-off spread/commission.
        target_pips = cost + target_usd / loss_usd * (stop_pips + cost)
        price, exit_pos, reason = simulate_path(values, j, side, distance, target_pips * pip, count)
        gross = side * (price - values[j, 0]) / pip
        net_r = (gross - cost) / (stop_pips + cost)
        rows.append({"entry_time": when, "exit_time": times[exit_pos] + FIVE,
            "label_end": when + pd.Timedelta(hours=hours), "side": side,
            "reason": reason, "stop_pips": stop_pips, "target_pips": target_pips,
            "cost_pips": cost, "gross_pips": gross, "net_r": net_r,
            "pnl_usd": net_r * loss_usd,
            "pnl_stress_usd": (gross - 2 * cost) / (stop_pips + cost) * loss_usd})
        free_from = times[exit_pos] + FIVE
    return pd.DataFrame(rows)


def describe(trades, initial_balance=1000., loss_usd=100.):
    if trades.empty:
        return {"trades": 0, "target_rate": None, "win_rate": None, "mean_usd": None,
            "stress_mean_usd": None, "fixed_risk_drawdown_usd": 0.}
    equity = np.r_[0., trades.pnl_usd.cumsum().to_numpy()]
    balance, executed = initial_balance, 0
    for pnl in trades.pnl_usd:
        if balance < loss_usd:
            break
        balance += float(pnl)
        executed += 1
    return {"trades": len(trades), "target_rate": float(trades.reason.isin(["target", "target_gap"]).mean()),
        "win_rate": float(trades.pnl_usd.gt(0).mean()), "mean_usd": float(trades.pnl_usd.mean()),
        "stress_mean_usd": float(trades.pnl_stress_usd.mean()),
        "fixed_risk_drawdown_usd": float((np.maximum.accumulate(equity) - equity).max()),
        "worst_loss_usd": float(trades.pnl_usd.min()),
        "fixed_budget_account_balance": balance, "affordable_trades": executed,
        "account_exhausted": executed < len(trades)}


def day_test(trades, column):
    if trades.empty:
        return {"ci_low": None, "ci_high": None, "p_value": None}
    daily = trades.groupby(trades.entry_time.dt.floor("D"))[column].mean().sort_index()
    test = stationary_bootstrap_mean_test(daily.to_list(), simulations=2000, seed=73)
    return {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in test.items()}


def run_pair(symbol, output=OUT, target_usd=30., loss_usd=100.):
    if target_usd <= 0 or loss_usd <= 0 or not np.isfinite([target_usd, loss_usd]).all():
        raise ValueError("Budgets must be finite and positive")
    paths = [ROOT / f"data/historical_{tf}/{symbol}.csv" for tf in ("15m", "5m")]
    bars, fills = [load_hourly(p, quarantine=True) for p in paths]
    bars.attrs["symbol"] = symbol
    rules, atr = rule_states(bars)
    prepared = prepare_paths(bars, fills)
    profile = load_spread_profile()
    hours = (bars.index + FIFTEEN).tz_convert("America/New_York").hour
    # Explicit scenario: today's measured fee profile applied to past prices.
    costs = np.array([(measured_spread_pips(profile, symbol, h) or 1.5) + ECN_COMMISSION_PIPS for h in hours])
    rows, frames = [], {}
    development = (("2016-01-01", "2019-01-01"), ("2019-01-01", "2023-01-01"))
    for name, state in rules.items():
        for stop_atr in (1.5, 3., 6.):
            for hold in (1, 4):
                summaries, samples = [], []
                for start, end in development:
                    trade = simulate(bars, fills, state, atr, costs, start, end, stop_atr, hold, target_usd, loss_usd, prepared=prepared)
                    summaries.append(describe(trade))
                    samples.append(trade)
                combined = pd.concat(samples, ignore_index=True)
                key = f"{name}_{stop_atr:g}_{hold}h"
                row = {"key": key, "rule": name, "stop_atr": stop_atr, "hours": hold,
                    "development": describe(combined), "subperiods": summaries}
                rows.append(row)
                frames[key] = combined
        print(f"{symbol}: {name} development tested", flush=True)
    eligible = [r for r in rows if all(p["trades"] >= 100 for p in r["subperiods"])]
    if not eligible:
        raise ValueError(f"No configuration with sufficient development trades: {symbol}")
    # Stability and stress fees select the candidate, never the audit results.
    selected = max(eligible, key=lambda r: min(p["stress_mean_usd"] for p in r["subperiods"]))
    state = rules[selected["rule"]]
    end = str((fills.index[-1] + FIVE).date() + pd.Timedelta(days=1))[:10]
    test = simulate(bars, fills, state, atr, costs, "2023-01-01", end,
        selected["stop_atr"], selected["hours"], target_usd, loss_usd, prepared=prepared)
    summary = describe(test, loss_usd=loss_usd)
    base_test, stress_test = day_test(test, "pnl_usd"), day_test(test, "pnl_stress_usd")
    checks = {"development_both_positive": all(p["mean_usd"] > 0 and p["stress_mean_usd"] > 0 for p in selected["subperiods"]),
        "at_least_300_audit_trades": len(test) >= 300,
        "audit_positive_with_ci": base_test["ci_low"] is not None and base_test["ci_low"] > 0,
        "positive_stress_with_ci": stress_test["ci_low"] is not None and stress_test["ci_low"] > 0,
        "seven_pair_adjusted_p": base_test["p_value"] is not None and base_test["p_value"] * 7 <= .05}
    year_rows = {str(y): describe(group, loss_usd=loss_usd) for y, group in test.groupby(test.entry_time.dt.year)} if len(test) else {}
    checks["all_audit_years_positive"] = bool(year_rows) and all(r["mean_usd"] > 0 for r in year_rows.values())
    report = {"symbol": symbol, "target_usd": target_usd, "planned_stop_loss_usd": loss_usd,
        "initial_account_usd": 1000., "breakeven_target_rate_full_exit": loss_usd / (loss_usd + target_usd),
        "status": "RETROSPECTIVE_RESEARCH", "trade_permission": False,
        "source_hashes": {str(p): digest(p) for p in paths},
        "script_sha256": digest(Path(__file__)),
        "selected_on_development": selected, "audit": summary,
        "day_test": base_test, "stress_day_test": stress_test, "by_year": year_rows,
        "checks": checks, "research_screen_passed": all(checks.values()), "development_trials": rows,
        "limitations": ["2023+ archive already inspected in prior research; not a blind forward test.",
            "Estimated 2026 spread + 0.7 pip commission applied to historical prices; stress cost doubled.",
            "Fixed $100 risk budget is a hypothetical per-trade scale; brokerage leverage/lot steps are not simulated.",
            "Account balance simulation stops once balance falls below planned risk; it does not restart.",
            "Gap fills and stress costs can exceed the planned stop loss; $100 is not a guaranteed maximum.",
            "No automatic order or signal promotion; successful retrospective screens still need forward verification."]}
    folder = Path(output) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "report.json", report)
    test.to_csv(folder / "audit_trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: selected={selected['key']} audit={summary} screen={report['research_screen_passed']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=MAJORS)
    parser.add_argument("--target-usd", type=float, default=30.)
    parser.add_argument("--loss-usd", type=float, default=100.)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-08", "target_usd": args.target_usd,
        "loss_usd": args.loss_usd, "development": ["2016-01-01", "2023-01-01"],
        "audit_start": "2023-01-01", "stops_atr": [1.5, 3., 6.], "holding_hours": [1, 4],
        "entry_session_utc": [7, 17], "selection": "max worst-subperiod stress mean; at least 100 trades in each subperiod",
        "audit_previously_inspected": True, "script_sha256": digest(Path(__file__))})
    rows = []
    for symbol in args.symbols:
        report = run_pair(symbol, target_usd=args.target_usd, loss_usd=args.loss_usd)
        rows.append({"symbol": symbol, "rule": report["selected_on_development"]["rule"],
            **report["audit"], "research_screen_passed": report["research_screen_passed"]})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
