"""Frozen, offline hourly crossover experiment; never changes the live app."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_decision_core import stationary_bootstrap_mean_test
from forex_indicators import compute_atr

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "simple_baseline"


def repair_small_envelope_errors(frame):
    out = frame.copy()
    ceiling = out[["Open", "Close", "Low"]].max(axis=1)
    floor = out[["Open", "Close", "High"]].min(axis=1)
    error = pd.concat([(ceiling - out.High).clip(lower=0), (out.Low - floor).clip(lower=0)], axis=1).max(axis=1)
    if (error > .00001 + 1e-12).any():
        raise ValueError("OHLC envelope error exceeds recorded repair tolerance")
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    return out, int((error > 0).sum())


def prepare_bars(frame, rules):
    out = frame.copy()
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    prices = out[["Open", "High", "Low", "Close"]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Invalid OHLC data")
    if ((out.High < out[["Open", "Close", "Low"]].max(axis=1)) | (out.Low > out[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC data")
    fast = out.Close.ewm(span=rules["fast_ema"], min_periods=rules["ema_min_periods"], adjust=False).mean()
    slow = out.Close.ewm(span=rules["slow_ema"], min_periods=rules["ema_min_periods"], adjust=False).mean()
    out["signal"] = np.select([
        (fast > slow) & (fast.shift(1) <= slow.shift(1)),
        (fast < slow) & (fast.shift(1) >= slow.shift(1)),
    ], [1, -1], default=0)
    out["atr"] = compute_atr(out, rules["atr_period"])
    return out


def exit_fill(position, opening, high, low, timed_out):
    side, stop, target = position["side"], position["stop"], position["target"]
    if side * (opening - stop) <= 0:
        return opening, "stop_gap"
    if side * (opening - target) >= 0:
        return target, "target_gap"
    if timed_out:
        return opening, "time"
    stop_hit = low <= stop if side == 1 else high >= stop
    target_hit = high >= target if side == 1 else low <= target
    if stop_hit:
        return stop, "stop"
    if target_hit:
        return target, "target"
    return None, None


def simulate(bars, start, end, rules, pip, cost_pips):
    """Indicators use past data; orders use only the previous completed bar."""
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    times = bars.index
    selected = np.flatnonzero((times >= start) & (times < end))
    balance = float(rules["initial_balance_usd"])
    position = None
    rows = []
    equity = [balance]
    counters = {"bars": len(selected), "crossovers": 0, "gap_skips": 0, "entries": 0}
    values = bars[["Open", "High", "Low", "Close", "signal", "atr"]].to_numpy()

    def close(price, time, reason):
        nonlocal position, balance
        gross_pips = position["side"] * (price - position["entry"]) / pip
        net_r = (gross_pips - cost_pips) / position["risk_pips"]
        risk_usd = balance * rules["risk_pct_including_cost"] / 100
        pnl = risk_usd * net_r
        balance += pnl
        rows.append({
            "entry_time": str(position["time"]), "exit_time": str(time),
            "side": "LONG" if position["side"] == 1 else "SHORT",
            "entry": position["entry"], "exit": price,
            "stop": position["stop"], "target": position["target"],
            "reason": reason, "net_r": net_r, "net_pips": gross_pips - cost_pips,
            "pnl_usd": pnl, "balance": balance,
        })
        equity.append(balance)
        position = None

    for i in selected:
        opening, high, low, closing, _, _ = values[i]
        occupied = position is not None
        if occupied:
            price, reason = exit_fill(position, opening, high, low, i - position["index"] >= rules["max_holding_bars"])
            if price is not None:
                close(price, times[i], reason)
        if not occupied and i > 0 and times[i - 1] >= start:
            side, atr = values[i - 1, 4:6]
            if side != 0:
                counters["crossovers"] += 1
                if times[i] - times[i - 1] > pd.Timedelta(hours=rules["max_signal_gap_hours"]):
                    counters["gap_skips"] += 1
                    continue
                if np.isfinite(atr) and atr > 0 and balance > 0:
                    distance = atr * rules["stop_atr"]
                    position = {
                        "index": i, "time": times[i], "side": int(side),
                        "entry": opening, "stop": opening - side * distance,
                        "target": opening + side * distance * rules["target_r_before_cost"],
                        "risk_pips": distance / pip + cost_pips,
                    }
                    counters["entries"] += 1
                    price, reason = exit_fill(position, opening, high, low, False)
                    if price is not None:
                        close(price, times[i], reason)
    if position is not None:
        close(values[selected[-1], 3], times[selected[-1]], "period_end")
    return pd.DataFrame(rows), np.array(equity), counters


def metrics(trades, equity, stats):
    peak = np.maximum.accumulate(equity)
    r = trades.net_r.to_numpy() if len(trades) else np.array([])
    pnl = trades.pnl_usd.to_numpy() if len(trades) else np.array([])
    losses = -pnl[pnl < 0].sum()
    test = stationary_bootstrap_mean_test(
        r, simulations=stats["stationary_bootstrap_simulations"],
        mean_block_length=stats["mean_block_length"], seed=stats["seed"],
    )
    def finite(value):
        return float(value) if value is not None and np.isfinite(value) else None
    p = finite(test.get("p_value"))
    return {
        "trades": len(trades), "net_pnl_usd": float(pnl.sum()),
        "return_pct": float((equity[-1] / equity[0] - 1) * 100),
        "max_drawdown_pct": float(((peak - equity) / peak * 100).max()),
        "avg_net_r": finite(r.mean()) if len(r) else None,
        "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses else None,
        "win_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "ci_low": finite(test.get("ci_low")), "ci_high": finite(test.get("ci_high")),
        "adjusted_p": min(1., p * stats["multiple_symbol_adjustment"]) if p is not None else None,
    }


def passes(row, stats):
    return (row["trades"] >= stats["minimum_trades_per_evaluation_period"]
            and row["avg_net_r"] is not None and row["avg_net_r"] > 0
            and row["ci_low"] is not None and row["ci_low"] > 0
            and row["adjusted_p"] is not None and row["adjusted_p"] <= .05)


def main():
    protocol_bytes = (OUTPUT / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    rules, stats = protocol["rules"], protocol["statistical_check"]
    results, audits, verdicts = [], [], []
    for symbol in protocol["symbols"]:
        path = ROOT / "data" / "historical_1h" / f"{symbol}.csv"
        source = path.read_bytes()
        raw = pd.read_csv(path)
        raw.index = pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True)
        raw.columns = [name.capitalize() for name in raw.columns]
        cleaned, repairs = repair_small_envelope_errors(raw)
        bars = prepare_bars(cleaned, rules)
        if bars.index[0] > pd.Timestamp("2009-01-01", tz="UTC") or bars.index[-1] < pd.Timestamp("2025-12-30", tz="UTC"):
            raise ValueError(f"Insufficient period coverage: {symbol}")
        audits.append({"symbol": symbol, "sha256": hashlib.sha256(source).hexdigest(), "bars": len(bars), "start": str(bars.index[0]), "end": str(bars.index[-1]), "envelope_repairs": repairs, "bars_by_year": {str(year): int(count) for year, count in bars.groupby(bars.index.year).size().items()}, "gaps_over_2h": int((bars.index.to_series().diff() > pd.Timedelta(hours=2)).sum())})
        symbol_rows = {}
        for period, (start, end) in protocol["periods"].items():
            for cost in (rules["round_trip_cost_pips"], rules["stress_cost_pips"]):
                trades, equity, counts = simulate(bars, start, end, rules, get_pip_size(symbol), cost)
                row = {"symbol": symbol, "period": period, "cost_pips": cost, **metrics(trades, equity, stats)}
                symbol_rows[period, cost] = row
                results.append({**row, "counts": counts})
                trades.to_csv(OUTPUT / f"{symbol}_{period}_{cost:g}pip_trades.csv", index=False)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        base, stress = rules["round_trip_cost_pips"], rules["stress_cost_pips"]
        passed = all(passes(symbol_rows[p, base], stats) for p in ("validation", "final_test"))
        stressed = symbol_rows["final_test", stress]["avg_net_r"]
        verdicts.append({"symbol": symbol, "research_screen_passed": bool(passed and stressed is not None and stressed > 0)})
    payload = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(), "data_amendment_sha256": hashlib.sha256((OUTPUT / "data_amendment.json").read_bytes()).hexdigest(), "data_audit": audits, "results": results, "verdicts": verdicts}
    (OUTPUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "counts"} for row in results]).to_csv(OUTPUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
