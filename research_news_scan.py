"""Frozen, offline test of the 15M + 5M bias state at scheduled US releases; never changes the live app.

Rules live in research/news_scan/protocol.json and were written before any
outcome was computed. Run: python -X utf8 research_news_scan.py
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from forex_config import get_pip_size
from forex_decision_core import stationary_bootstrap_mean_test
from forex_indicators import add_indicators, compute_atr

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "news_scan"
APP = ROOT / "forex_web_app_streamlit_v14_alert_decision.py"
NEW_YORK = pytz.timezone("America/New_York")

STATE_THRESHOLD = 25.0
STOP_ATR = 1.5
TARGET_R = 1.5
MAX_HOLD = pd.Timedelta(hours=4)
RANGE_LOOKBACK_BARS = 288
NFP_MIN_RANGE_RATIO = 3.0
COSTS = (1.5, 3.0, 5.0)
DECISION_COST = 3.0
VARIANT_DELAY = {"before": pd.Timedelta(0), "after": pd.Timedelta(minutes=15)}
FIVE = pd.Timedelta(minutes=5)
FIFTEEN = pd.Timedelta(minutes=15)


def load_app_score():
    """Compile only the app's score function; importing the app would start Streamlit."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    wanted = {"score_series_for_backtest", "_utc_index_series"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + body, type_ignores=[])
    namespace = {"np": np, "pd": pd, "add_indicators": add_indicators}
    exec(compile(module, str(APP), "exec"), namespace)
    return namespace["score_series_for_backtest"]


def nfp_release_date(year: int, month: int) -> date:
    """Third Friday after the Saturday ending the Sunday-Saturday week that contains the 12th."""
    twelfth = date(year, month, 12)
    saturday = twelfth + timedelta(days=(5 - twelfth.weekday()) % 7)
    return saturday + timedelta(days=20)


def new_york_to_utc(day: date, hour: int, minute: int) -> pd.Timestamp:
    local = NEW_YORK.localize(datetime(day.year, day.month, day.day, hour, minute))
    return pd.Timestamp(local).tz_convert("UTC")


def load_bars(symbol: str, folder: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / folder / f"{symbol}.csv")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{symbol} {folder}: duplicate timestamps")
    return df[["Open", "High", "Low", "Close"]]


def range_ratio(index: pd.DatetimeIndex, ranges: np.ndarray, when: pd.Timestamp) -> float:
    """Range of the 5M bar starting at `when` over the median of the preceding day of bars."""
    pos = index.searchsorted(when)
    if pos >= len(index) or index[pos] != when or pos < RANGE_LOOKBACK_BARS:
        return float("nan")
    typical = float(np.median(ranges[pos - RANGE_LOOKBACK_BARS:pos]))
    return float(ranges[pos] / typical) if typical > 0 else float("nan")


def build_events(eurusd_5m: pd.DataFrame, last_usable: pd.Timestamp) -> pd.DataFrame:
    index = eurusd_5m.index
    ranges = (eurusd_5m.High - eurusd_5m.Low).to_numpy()
    rows = []
    year, month = 2008, 1
    while True:
        when = new_york_to_utc(nfp_release_date(year, month), 8, 30)
        if when > last_usable:
            break
        ratio = range_ratio(index, ranges, when)
        rows.append({"type": "NFP", "time": when, "range_ratio": ratio, "kept": bool(ratio >= NFP_MIN_RANGE_RATIO)})
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    fomc = pd.read_csv(ROOT / "data" / "news" / "fomc_statements.csv", parse_dates=["date"])
    for day in fomc["date"]:
        if day < pd.Timestamp("2013-01-01") or day.weekday() != 2:
            continue
        when = new_york_to_utc(day.date(), 14, 0)
        if when > last_usable:
            continue
        rows.append({"type": "FOMC", "time": when, "range_ratio": range_ratio(index, ranges, when), "kept": True})
    return pd.DataFrame(rows)


def state_from_scores(score_15m: float, score_5m: float) -> int:
    if np.isnan(score_15m) or np.isnan(score_5m):
        return 0
    if score_15m >= STATE_THRESHOLD and score_5m >= STATE_THRESHOLD:
        return 1
    if score_15m <= -STATE_THRESHOLD and score_5m <= -STATE_THRESHOLD:
        return -1
    return 0


def simulate_trade(bars: pd.DataFrame, entry_time: pd.Timestamp, side: int, stop_distance: float) -> tuple[float, float, str]:
    """Return (entry, exit, reason) on 5M bars; stop first when both levels sit inside one bar."""
    window = bars.loc[entry_time:entry_time + MAX_HOLD - FIVE]
    opens, highs, lows, closes = (window[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    entry = float(opens[0])
    stop = entry - side * stop_distance
    target = entry + side * stop_distance * TARGET_R
    for i in range(len(window)):
        if i > 0:
            if side * (opens[i] - stop) <= 0:
                return entry, float(opens[i]), "stop_gap"
            if side * (opens[i] - target) >= 0:
                return entry, float(target), "target_gap"
        stop_hit = lows[i] <= stop if side == 1 else highs[i] >= stop
        target_hit = highs[i] >= target if side == 1 else lows[i] <= target
        if stop_hit:
            return entry, float(stop), "stop"
        if target_hit:
            return entry, float(target), "target"
    return entry, float(closes[-1]), "time"


def price_at(bars: pd.DataFrame, when: pd.Timestamp) -> float:
    return float(bars.at[when, "Open"]) if when in bars.index else float("nan")


def evaluate_pair(symbol, bars_5m, bars_15m, score_5m, score_15m, atr_15m, events):
    pip = get_pip_size(symbol)
    rows, skipped = [], 0
    for event in events.itertuples():
        for sample, shift in (("news", pd.Timedelta(0)), ("control", pd.Timedelta(days=7))):
            release = event.time - shift
            for variant, delay in VARIANT_DELAY.items():
                decision = release + delay
                bar_15m, bar_5m = decision - FIFTEEN, decision - FIVE
                if bar_15m not in score_15m.index or bar_5m not in score_5m.index or decision not in bars_5m.index:
                    skipped += 1
                    continue
                side = state_from_scores(float(score_15m.at[bar_15m]), float(score_5m.at[bar_5m]))
                row = {
                    "type": event.type, "release": event.time, "sample": sample, "variant": variant,
                    "symbol": symbol, "decision": decision, "side": side,
                }
                atr = float(atr_15m.get(bar_15m, np.nan))
                if side != 0 and np.isfinite(atr) and atr > 0:
                    stop_distance = STOP_ATR * atr
                    entry, exit_price, reason = simulate_trade(bars_5m, decision, side, stop_distance)
                    gross = side * (exit_price - entry) / pip
                    stop_pips = stop_distance / pip
                    row.update({
                        "stop_pips": round(stop_pips, 2), "exit_reason": reason, "gross_pips": round(gross, 2),
                        "move_1h_pips": round(side * (price_at(bars_5m, decision + pd.Timedelta(hours=1)) - entry) / pip, 2),
                        "move_4h_pips": round(side * (price_at(bars_5m, decision + MAX_HOLD) - entry) / pip, 2),
                        **{f"net_r_{c}": (gross - c) / stop_pips for c in COSTS},
                    })
                rows.append(row)
    return rows, skipped


def event_level_test(trades: pd.DataFrame, cost: float) -> dict:
    per_event = trades.groupby("release")[f"net_r_{cost}"].mean()
    test = stationary_bootstrap_mean_test(per_event.tolist(), simulations=2000, mean_block_length=1.0, seed=42)
    return {"events": int(len(per_event)), "mean_r": test["observed_mean"], "ci_low": test["ci_low"],
            "ci_high": test["ci_high"], "p_value": test["p_value"]}


def summarize(results: pd.DataFrame, subperiods: dict) -> tuple[pd.DataFrame, dict]:
    rows, verdicts = [], {}
    for (event_type, variant, sample), group in results.groupby(["type", "variant", "sample"]):
        trades = group[group.side != 0].dropna(subset=["gross_pips"])
        row = {
            "type": event_type, "variant": variant, "sample": sample,
            "pair_events": len(group), "events": group.release.nunique(),
            "events_with_signal": trades.release.nunique(), "trades": len(trades),
            "long": int((trades.side == 1).sum()), "short": int((trades.side == -1).sum()),
            "target_rate": float(trades.exit_reason.isin(["target", "target_gap"]).mean()) if len(trades) else np.nan,
            "mean_move_1h_pips": trades.move_1h_pips.mean(), "mean_move_4h_pips": trades.move_4h_pips.mean(),
        }
        for cost in COSTS:
            row[f"win_rate_{cost}"] = float((trades[f"net_r_{cost}"] > 0).mean()) if len(trades) else np.nan
            test = event_level_test(trades, cost) if len(trades) else {}
            for key in ("mean_r", "ci_low", "ci_high", "p_value"):
                row[f"{key}_{cost}"] = test.get(key, np.nan)
        for name, (start, end) in subperiods.items():
            part = trades[(trades.release >= pd.Timestamp(start, tz="UTC")) & (trades.release < pd.Timestamp(end, tz="UTC"))]
            row[f"mean_r_{DECISION_COST}_{name}"] = part.groupby("release")[f"net_r_{DECISION_COST}"].mean().mean() if len(part) else np.nan
            row[f"events_{name}"] = part.release.nunique()
        rows.append(row)
    summary = pd.DataFrame(rows)

    for (event_type, variant), group in summary.groupby(["type", "variant"]):
        news = group[group["sample"] == "news"].iloc[0]
        control = group[group["sample"] == "control"].iloc[0]
        checks = {
            "mean_positive": bool(news[f"mean_r_{DECISION_COST}"] > 0),
            "ci_low_positive": bool(news[f"ci_low_{DECISION_COST}"] > 0),
            "both_subperiods_positive": all(bool(news[f"mean_r_{DECISION_COST}_{name}"] > 0) for name in subperiods),
            "beats_control": bool(news[f"mean_r_{DECISION_COST}"] > control[f"mean_r_{DECISION_COST}"]),
        }
        verdicts[f"{event_type}_{variant}"] = {"checks": checks, "promising": all(checks.values())}
    return summary, verdicts


def main() -> None:
    protocol_bytes = (OUTPUT / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    score = load_app_score()

    eurusd_5m = load_bars("EURUSD", "historical_5m")
    last_usable = eurusd_5m.index[-1] - MAX_HOLD - pd.Timedelta(hours=1)
    events = build_events(eurusd_5m, last_usable)
    kept = events[events.kept].reset_index(drop=True)
    print(f"Events: {len(events)} built, {len(kept)} kept "
          f"(NFP {int((kept.type == 'NFP').sum())}, FOMC {int((kept.type == 'FOMC').sum())})")

    all_rows, skipped = [], {}
    for symbol in protocol["symbols"]:
        bars_5m = eurusd_5m if symbol == "EURUSD" else load_bars(symbol, "historical_5m")
        bars_15m = load_bars(symbol, "historical_15m")
        score_5m, score_15m = score(bars_5m), score(bars_15m)
        atr_15m = compute_atr(bars_15m, 14)
        rows, skipped[symbol] = evaluate_pair(symbol, bars_5m, bars_15m, score_5m, score_15m, atr_15m, kept)
        all_rows.extend(rows)
        print(f"{symbol}: {len(rows)} pair-events, {skipped[symbol]} skipped for missing bars")

    results = pd.DataFrame(all_rows)
    summary, verdicts = summarize(results, protocol["statistics"]["subperiods"])

    events.to_csv(OUTPUT / "events.csv", index=False)
    results.to_csv(OUTPUT / "trades.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    payload = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "events_built": int(len(events)),
        "events_kept": {t: int((kept.type == t).sum()) for t in ("NFP", "FOMC")},
        "nfp_excluded_by_timing_check": [str(t) for t in events[(events.type == "NFP") & ~events.kept].time],
        "fomc_median_range_ratio": float(events[events.type == "FOMC"].range_ratio.median()),
        "nfp_median_range_ratio_kept": float(kept[kept.type == "NFP"].range_ratio.median()),
        "skipped_for_missing_bars": skipped,
        "verdicts": verdicts,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
