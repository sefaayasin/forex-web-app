"""Does the 72h volatility forecast help size 15M + 5M trades? Offline research; never changes the live app.

Rules are in research/vol_sizing/protocol.json and were written before any result was computed.
Needs research/meta_label/dataset.csv.gz (python -X utf8 research_meta_label.py build) and
data/historical_1h_from_15m (python rebuild_hourly_from_15m.py).

Run: python -X utf8 research_vol_sizing.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forex_ml_live import VOLATILITY_CONFIDENCE_THRESHOLD
from forex_ml_tournament import build_features, load_hourly
from research_meta_label import app_rule_mask

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "vol_sizing"
DATASET = ROOT / "research" / "meta_label" / "dataset.csv.gz"
HOURLY = ROOT / "data" / "historical_1h_from_15m"
MODELS = ROOT / "data" / "ml" / "tournament"
START = pd.Timestamp("2023-01-01", tz="UTC")
STALE = pd.Timedelta(hours=3)
HIGH_MULTIPLIER = 0.5
GAP_LOSS_R = -1.05
BOOTSTRAPS = 2000
REGIMES = ("high", "uncertain", "normal")


def hourly_probabilities(symbol: str) -> pd.Series:
    """Out-of-sample 72h high-volatility probability for every hourly bar from late 2022."""
    bundle = joblib.load(MODELS / f"{symbol}_high_volatility_research.joblib")
    bars = load_hourly(HOURLY / f"{symbol}.csv", quarantine=True)
    features, _ = build_features(bars, symbol)
    features = features[features.index >= START - pd.Timedelta(days=40)]
    return pd.Series(bundle["model"].predict_proba(features[bundle["columns"]])[:, 1], index=features.index)


def attach_probability(trades: pd.DataFrame, probabilities: pd.Series) -> pd.Series:
    """Probability from the last hourly bar closed at each entry (bar open + 1h <= entry)."""
    right = pd.DataFrame({"available": probabilities.index + pd.Timedelta(hours=1), "p": probabilities.to_numpy()})
    left = trades[["entry_time"]].reset_index().sort_values("entry_time")
    merged = pd.merge_asof(left, right, left_on="entry_time", right_on="available", direction="backward", tolerance=STALE)
    return merged.set_index("index")["p"].reindex(trades.index)


def regime(probability: pd.Series) -> pd.Series:
    t = VOLATILITY_CONFIDENCE_THRESHOLD
    return pd.Series(np.select([probability >= t, probability <= 1 - t], ["high", "normal"], "uncertain"), index=probability.index)


def day_aggregates(trades: pd.DataFrame, value: pd.Series, days: pd.Index, group: str) -> np.ndarray:
    """Per-day count, sum and sum of squares of `value` for trades in regime `group` (rows follow `days`)."""
    part = value[trades.regime == group]
    day = trades.loc[part.index, "day"]
    agg = pd.DataFrame({"n": 1.0, "s1": part, "s2": part ** 2, "day": day}).groupby("day").sum()
    return agg.reindex(days, fill_value=0.0)[["n", "s1", "s2"]].to_numpy()


def statistic(agg: np.ndarray, weights: np.ndarray, kind: str) -> float:
    n, s1, s2 = weights @ agg
    mean = s1 / n
    return float(mean if kind == "mean" else np.sqrt(max(s2 / n - mean ** 2, 0.0)))


def bootstrap_difference(trades: pd.DataFrame, value: pd.Series, kind: str, seed: int = 42) -> dict:
    """High minus normal, resampling whole trading days so same-day trades stay together."""
    days = pd.Index(sorted(trades.day.unique()))
    high, normal = day_aggregates(trades, value, days, "high"), day_aggregates(trades, value, days, "normal")
    ones = np.ones(len(days))
    observed = statistic(high, ones, kind) - statistic(normal, ones, kind)
    rng = np.random.default_rng(seed)
    draws = [
        statistic(high, w, kind) - statistic(normal, w, kind)
        for w in (np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)).astype(float) for _ in range(BOOTSTRAPS))
    ]
    low, high_ci = np.quantile(draws, [.025, .975])
    return {"difference": observed, "ci_low": float(low), "ci_high": float(high_ci)}


def regime_table(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in REGIMES:
        t = trades[trades.regime == name]
        rows.append({"regime": name, "trades": len(t), "share": len(t) / len(trades),
                     "stop_rate": t.stopped.mean(), "target_rate": t.exit_reason.isin(["target", "target_gap"]).mean(),
                     "mean_r": t.net_r.mean(), "std_r": t.net_r.std(ddof=0), "gap_loss_share": t.gap_loss.mean(),
                     "median_stop_pips": t.stop_pips.median()})
    return pd.DataFrame(rows).set_index("regime")


def max_drawdown(daily: pd.Series) -> float:
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min())


def sizing_comparison(trades: pd.DataFrame) -> dict:
    multiplier = np.where(trades.regime == "high", HIGH_MULTIPLIER, 1.0)
    multiplier = multiplier / multiplier.mean()
    out = {"multiplier_high": float(multiplier[trades.regime.to_numpy() == "high"][0]) if (trades.regime == "high").any() else None,
           "multiplier_other": float(multiplier[trades.regime.to_numpy() != "high"][0])}
    for name, weights in (("fixed", np.ones(len(trades))), ("regime_sized", multiplier)):
        daily = (trades.net_r * weights).groupby(trades.day).mean()
        out[name] = {"days": int(len(daily)), "mean_daily_r": float(daily.mean()), "std_daily_r": float(daily.std(ddof=0)),
                     "max_drawdown_r": max_drawdown(daily), "worst_day_r": float(daily.min())}
    return out


def evaluate(trades: pd.DataFrame) -> dict:
    table = regime_table(trades)
    return {
        "regimes": json.loads(table.to_json(orient="index")),
        "high_minus_normal": {
            "stop_rate": bootstrap_difference(trades, trades.stopped.astype(float), "mean"),
            "mean_r": bootstrap_difference(trades, trades.net_r, "mean"),
            "std_r": bootstrap_difference(trades, trades.net_r, "std"),
            "gap_loss_share": bootstrap_difference(trades, trades.gap_loss.astype(float), "mean"),
        },
        "sizing": sizing_comparison(trades),
    }


def main() -> None:
    protocol_bytes = (OUTPUT / "protocol.json").read_bytes()
    data = pd.read_csv(DATASET, parse_dates=["entry_time", "exit_time"])
    trades = data[data.split == "final"].copy()
    trades["p_high_vol"] = np.nan
    for symbol, index in trades.groupby("symbol").groups.items():
        trades.loc[index, "p_high_vol"] = attach_probability(trades.loc[index], hourly_probabilities(symbol))
        print(f"{symbol}: probabilities attached", flush=True)
    missing = int(trades.p_high_vol.isna().sum())
    trades = trades.dropna(subset=["p_high_vol"])
    trades["regime"] = regime(trades.p_high_vol)
    trades["stopped"] = trades.exit_reason.isin(["stop", "stop_gap"])
    trades["gap_loss"] = trades.net_r < GAP_LOSS_R
    trades["day"] = trades.entry_time.dt.floor("D")

    primary = evaluate(trades)
    h1 = primary["high_minus_normal"]["stop_rate"]
    fixed, sized = primary["sizing"]["fixed"], primary["sizing"]["regime_sized"]
    checks = {
        "H1_supported": bool(h1["ci_low"] > 0),
        "H2_supported": bool(sized["std_daily_r"] < fixed["std_daily_r"] and sized["max_drawdown_r"] > fixed["max_drawdown_r"]
                             and sized["mean_daily_r"] >= fixed["mean_daily_r"]),
    }
    results = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "trades": int(len(trades)), "dropped_without_forecast": missing,
        "primary_all_signals": primary,
        "info_app_mtf_rule_subset": evaluate(trades[app_rule_mask(trades)]),
        "checks": checks,
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(regime_table(trades).round(4).to_string())
    print(json.dumps({k: results[k] for k in ("trades", "dropped_without_forecast", "checks")}, indent=2))
    print(json.dumps(primary["high_minus_normal"], indent=2))
    print(json.dumps(primary["sizing"], indent=2))


if __name__ == "__main__":
    main()
