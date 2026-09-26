"""Meta-labeling the 15M + 5M alignment signal; offline research that never changes the live app.

Rules are in research/meta_label/protocol.json and were written before any stage ran.
Run the stages in order; `build` never prints outcomes from the final period and
`final` refuses to run twice:

    python -X utf8 research_meta_label.py build
    python -X utf8 research_meta_label.py select
    python -X utf8 research_meta_label.py final
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forex_config import get_pip_size, symbol_pair
from forex_decision_core import stationary_bootstrap_mean_test
from forex_indicators import add_indicators, compute_atr
from forex_ml_tournament import make_model
from research_news_scan import build_events, load_app_score

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "meta_label"
DATASET = OUTPUT / "dataset.csv.gz"
EXCLUDED = {"EURZAR"}

STATE_THRESHOLD = 25.0
APP_ENTRY_THRESHOLD = 60.0
STOP_ATR = 1.5
TARGET_R = 1.5
COST, STRESS_COST = 1.5, 3.0
LAST_BAR_OFFSET = np.timedelta64(4 * 60 - 5, "m")  # last 5M bar opening inside the 4-hour window
NEWS_WINDOW = pd.Timedelta(minutes=60)
FIVE = pd.Timedelta(minutes=5)
BAR = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
STALE = {"15m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=2), "4h": pd.Timedelta(hours=8)}
SPLITS = {"train": ("2008-01-01", "2019-01-01"), "validation": ("2019-01-01", "2023-01-01"), "final": ("2023-01-01", "2100-01-01")}
COVERAGES = (0.1, 0.2, 0.3, 0.5)
MIN_VALIDATION_TRADES = 1000
FEATURES = [
    "side", "s5", "s15", "s1h", "s4h", "stop_pips", "cost_fraction", "atr_ratio", "rv_ratio",
    "rsi_signed", "bb_signed", "ema200_signed", "hour_sin", "hour_cos", "weekday", "near_usd_news",
]


def load_bars(symbol: str, folder: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(ROOT / "data" / folder / f"{symbol}.csv")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index()[["Open", "High", "Low", "Close"]]
    if df.index.has_duplicates:
        raise ValueError(f"{symbol} {folder}: duplicate timestamps")
    values = df.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{symbol} {folder}: invalid prices")
    high = df[["High", "Open", "Close"]].max(axis=1)
    low = df[["Low", "Open", "Close"]].min(axis=1)
    repaired = int(((high != df.High) | (low != df.Low)).sum())
    return df.assign(High=high, Low=low), repaired


def resample_bars(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """UTC-aligned higher-timeframe bars from a complete lower-timeframe archive (data_amendment.json)."""
    out = bars.resample(rule, label="left", closed="left").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return out.dropna()


def native_coverage(symbol: str, bars_15m: pd.DataFrame) -> dict:
    counts = {key: len(pd.read_csv(ROOT / "data" / folder / f"{symbol}.csv", usecols=["timestamp"]))
              for key, folder in (("1h", "historical_1h"), ("4h", "historical_4h"))}
    return {"1h": round(counts["1h"] / len(resample_bars(bars_15m, "1h")), 3),
            "4h": round(counts["4h"] / len(resample_bars(bars_15m, "4h")), 3)}


def closed_values(times: pd.DatetimeIndex, series: pd.Series, bar: pd.Timedelta, stale: pd.Timedelta) -> np.ndarray:
    """Value of the last bar closed by each time in `times`; NaN when that bar is older than `stale`."""
    right = pd.DataFrame({"available": series.index + bar, "value": series.to_numpy()}).sort_values("available")
    left = pd.DataFrame({"time": times})
    merged = pd.merge_asof(left, right, left_on="time", right_on="available", direction="backward", tolerance=stale)
    return merged["value"].to_numpy()


def primary_signals(state: np.ndarray) -> np.ndarray:
    previous = np.concatenate([[0], state[:-1]])
    return np.flatnonzero((state != 0) & (state != previous))


def alignment_state(score_5m: np.ndarray, score_15m: np.ndarray) -> np.ndarray:
    long = (score_5m >= STATE_THRESHOLD) & (score_15m >= STATE_THRESHOLD)
    short = (score_5m <= -STATE_THRESHOLD) & (score_15m <= -STATE_THRESHOLD)
    return np.where(long, 1, np.where(short, -1, 0))


def simulate(opens, highs, lows, closes, times, entry: int, side: int, stop_distance: float) -> tuple[float, int, str]:
    """Return (exit price, exit bar position, reason); stop first when both levels are in one bar."""
    entry_price = opens[entry]
    stop = entry_price - side * stop_distance
    target = entry_price + side * stop_distance * TARGET_R
    last = int(np.searchsorted(times, times[entry] + LAST_BAR_OFFSET, side="right")) - 1
    for i in range(entry, last + 1):
        if i > entry:
            if side * (opens[i] - stop) <= 0:
                return opens[i], i, "stop_gap"
            if side * (opens[i] - target) >= 0:
                return target, i, "target_gap"
        if (lows[i] <= stop) if side == 1 else (highs[i] >= stop):
            return stop, i, "stop"
        if (highs[i] >= target) if side == 1 else (lows[i] <= target):
            return target, i, "target"
    return closes[last], last, "time"


def near_news_flags(symbol: str, times: pd.DatetimeIndex, releases: pd.DatetimeIndex) -> np.ndarray:
    if "USD" not in symbol_pair(symbol) or releases.empty:
        return np.zeros(len(times), dtype=int)
    pos = releases.searchsorted(times)
    after = releases[np.minimum(pos, len(releases) - 1)]
    before = releases[np.maximum(pos - 1, 0)]
    gap = np.minimum(np.abs((after - times).total_seconds()), np.abs((times - before).total_seconds()))
    return (gap <= NEWS_WINDOW.total_seconds()).astype(int)


def build_pair(symbol: str, score, releases: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    bars = {}
    repairs = {}
    for key, folder in (("5m", "historical_5m"), ("15m", "historical_15m")):
        bars[key], repairs[key] = load_bars(symbol, folder)
    bars["1h"], bars["4h"] = resample_bars(bars["15m"], "1h"), resample_bars(bars["15m"], "4h")
    repairs["native_archive_coverage"] = native_coverage(symbol, bars["15m"])
    m5 = bars["5m"]
    times = m5.index
    close_times = times + FIVE
    pip = get_pip_size(symbol)

    s5 = score(m5).reindex(times).to_numpy()
    ind15 = add_indicators(bars["15m"])
    ind1h = add_indicators(bars["1h"])
    atr15 = compute_atr(bars["15m"], 14)
    band = (ind15.Close - ind15.BBLow) / (ind15.BBUp - ind15.BBLow).replace(0, np.nan)
    ema_distance = (ind1h.Close - ind1h.EMA200) / ind1h.ATR14.replace(0, np.nan)
    at15 = {
        "s15": score(bars["15m"]), "atr": atr15, "atr_ratio": atr15 / atr15.rolling(1920, min_periods=480).median(),
        "rsi": ind15.RSI14, "band": band,
    }
    merged = {name: closed_values(close_times, series, BAR["15m"], STALE["15m"]) for name, series in at15.items()}
    merged["s1h"] = closed_values(close_times, score(bars["1h"]), BAR["1h"], STALE["1h"])
    merged["ema200"] = closed_values(close_times, ema_distance, BAR["1h"], STALE["1h"])
    merged["s4h"] = closed_values(close_times, score(bars["4h"]), BAR["4h"], STALE["4h"])
    returns = m5.Close.pct_change().pow(2)
    rv_ratio = (np.sqrt(returns.rolling(48).mean()) / np.sqrt(returns.rolling(5760, min_periods=1440).mean())).to_numpy()

    state = alignment_state(np.nan_to_num(s5), np.nan_to_num(merged["s15"]))
    opens, highs, lows, closes = (m5[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    time_values = times.values  # datetime64[ns] UTC; to_numpy() would give tz-aware objects
    rows, free_from = [], np.datetime64("NaT")
    for i in primary_signals(state):
        entry = i + 1
        if entry >= len(times) or times[entry] - times[i] != FIVE:
            continue
        if not np.isnat(free_from) and time_values[entry] < free_from:
            continue
        atr = merged["atr"][i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        side = int(state[i])
        stop_distance = STOP_ATR * atr
        exit_price, exit_pos, reason = simulate(opens, highs, lows, closes, time_values, entry, side, stop_distance)
        free_from = time_values[exit_pos] + np.timedelta64(5, "m")
        gross = side * (exit_price - opens[entry]) / pip
        stop_pips = stop_distance / pip
        rows.append({
            "symbol": symbol, "entry_time": times[entry], "exit_time": times[exit_pos] + FIVE, "side": side,
            "s5": side * s5[i], "s15": side * merged["s15"][i], "s1h": side * merged["s1h"][i], "s4h": side * merged["s4h"][i],
            "stop_pips": stop_pips, "cost_fraction": COST / stop_pips, "atr_ratio": merged["atr_ratio"][i],
            "rv_ratio": rv_ratio[i], "rsi_signed": side * (merged["rsi"][i] - 50), "bb_signed": side * (merged["band"][i] - .5),
            "ema200_signed": side * merged["ema200"][i], "exit_reason": reason, "gross_pips": gross,
            "net_r": (gross - COST) / stop_pips, "net_r_stress": (gross - STRESS_COST) / stop_pips,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, repairs
    hours = frame.entry_time.dt.hour
    frame["hour_sin"], frame["hour_cos"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    frame["weekday"] = frame.entry_time.dt.dayofweek
    frame["near_usd_news"] = near_news_flags(symbol, pd.DatetimeIndex(frame.entry_time), releases)
    return frame, repairs


def assign_split(frame: pd.DataFrame) -> pd.Series:
    split = pd.Series("purged", index=frame.index)
    for name, (start, end) in SPLITS.items():
        start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        split[(frame.entry_time >= start) & (frame.exit_time <= end)] = name
    return split


def app_rule_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame.s4h >= STATE_THRESHOLD) & (frame.s1h >= STATE_THRESHOLD) & (frame.s5 >= APP_ENTRY_THRESHOLD)


def make_candidate(name: str):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.1, max_iter=400))
    return make_model("hist_boost")


def day_level_test(trades: pd.DataFrame, column: str) -> dict:
    if trades.empty:
        return {"days": 0, "mean_r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    daily = trades.groupby(trades.entry_time.dt.floor("D"))[column].mean().sort_index()
    test = stationary_bootstrap_mean_test(daily.tolist(), simulations=2000, mean_block_length=5.0, seed=42)
    return {"days": int(len(daily)), "mean_r": test["observed_mean"], "ci_low": test["ci_low"],
            "ci_high": test["ci_high"], "p_value": test["p_value"]}


def describe(trades: pd.DataFrame) -> dict:
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades.net_r > 0).mean()) if len(trades) else np.nan,
        "mean_r": float(trades.net_r.mean()) if len(trades) else np.nan,
        "mean_r_stress": float(trades.net_r_stress.mean()) if len(trades) else np.nan,
    }


def run_build() -> None:
    score = load_app_score()
    eurusd, _ = load_bars("EURUSD", "historical_5m")
    events = build_events(eurusd, eurusd.index[-1])
    releases = pd.DatetimeIndex(events[events.kept].time).sort_values()
    frames, repairs = [], {}
    for path in sorted((ROOT / "data" / "historical_5m").glob("*.csv")):
        symbol = path.stem
        if symbol in EXCLUDED:
            continue
        frame, repairs[symbol] = build_pair(symbol, score, releases)
        frames.append(frame)
        print(f"{symbol}: {len(frame)} signals", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data["split"] = assign_split(data)
    data["label"] = (data.net_r > 0).astype(int)
    data.to_csv(DATASET, index=False, compression="gzip")
    counts = data.split.value_counts().to_dict()
    (OUTPUT / "build.json").write_text(json.dumps({
        "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
        "signals_by_split": {k: int(v) for k, v in counts.items()},
        "signals_by_symbol": {k: int(v) for k, v in data.symbol.value_counts().sort_index().items()},
        "ohlc_bars_repaired_and_native_archive_coverage": repairs,
        "news_releases_used": int(len(releases)),
    }, indent=2), encoding="utf-8")
    print("Signals by split:", counts)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["entry_time", "exit_time"])


def run_select() -> None:
    frozen = OUTPUT / "selected.json"
    if frozen.exists():
        raise ValueError("Selection already frozen; refuse to reselect")
    data = load_dataset()
    train, validation = data[data.split == "train"], data[data.split == "validation"]
    rows = []
    fitted = {}
    for name in ("logistic", "hist_boost"):
        model = make_candidate(name)
        model.fit(train[FEATURES], train.label)
        fitted[name] = model
        p = model.predict_proba(validation[FEATURES])[:, 1]
        for coverage in COVERAGES:
            threshold = float(np.quantile(p, 1 - coverage))
            kept = validation[p >= threshold]
            rows.append({"model": name, "coverage": coverage, "threshold": threshold, **describe(kept)})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "validation.csv", index=False)
    eligible = table[table.trades >= MIN_VALIDATION_TRADES].sort_values("mean_r", ascending=False)
    best = eligible.iloc[0]
    joblib.dump({"model": fitted[best.model], "features": FEATURES}, OUTPUT / "model.joblib")
    selected = {
        "model": str(best.model), "coverage": float(best.coverage), "threshold": float(best.threshold),
        "validation": {k: float(best[k]) for k in ("trades", "win_rate", "mean_r", "mean_r_stress")},
        "validation_all_signals": describe(validation),
        "validation_app_mtf_rule": describe(validation[app_rule_mask(validation)]),
        "train_signals": int(len(train)),
    }
    frozen.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(json.dumps(selected, indent=2))


def run_final() -> None:
    if (OUTPUT / "results.json").exists():
        raise ValueError("Final evaluation already ran; results are not recomputed or tuned")
    selected = json.loads((OUTPUT / "selected.json").read_text(encoding="utf-8"))
    bundle = joblib.load(OUTPUT / "model.joblib")
    data = load_dataset()
    final = data[data.split == "final"].copy()
    final["probability"] = bundle["model"].predict_proba(final[bundle["features"]])[:, 1]
    groups = {
        "kept": final[final.probability >= selected["threshold"]],
        "all_signals": final,
        "app_mtf_rule": final[app_rule_mask(final)],
    }
    summary = {name: {**describe(t), "day_test": day_level_test(t, "net_r"),
                      "day_test_stress": day_level_test(t, "net_r_stress")} for name, t in groups.items()}
    kept = summary["kept"]
    checks = {
        "mean_positive_with_ci": bool(kept["mean_r"] > 0 and kept["day_test"]["ci_low"] > 0),
        "beats_all_signals": bool(kept["mean_r"] > summary["all_signals"]["mean_r"]),
        "beats_app_mtf_rule": bool(kept["mean_r"] > summary["app_mtf_rule"]["mean_r"]),
        "positive_at_stress_cost": bool(kept["mean_r_stress"] > 0),
        "at_least_300_trades": bool(kept["trades"] >= 300),
    }
    by_year = pd.concat({name: t.groupby(t.entry_time.dt.year).net_r.agg(["count", "mean"]) for name, t in groups.items()}, axis=1)
    by_year.to_csv(OUTPUT / "final_by_year.csv")
    groups["kept"].groupby("symbol").net_r.agg(["count", "mean"]).to_csv(OUTPUT / "final_kept_by_symbol.csv")
    result = {"selected": selected, "final": summary, "checks": checks, "passes": all(checks.values())}
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(by_year.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "select", "final"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    {"build": run_build, "select": run_select, "final": run_final}[args.stage]()
