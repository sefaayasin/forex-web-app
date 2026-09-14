"""Offline, chronological ML research. Never grants permission to trade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from forex_indicators import add_indicators

ROOT = Path(__file__).resolve().parent


def load_bars(path, price_step=.00001):
    raw = pd.read_csv(path)
    raw.index = pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True)
    raw = raw.rename(columns=str.capitalize)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unordered timestamps: {path}")
    prices = raw[["Open", "High", "Low", "Close"]]
    high_error = prices.max(axis=1) - raw.High
    low_error = raw.Low - prices.min(axis=1)
    invalid = (~np.isfinite(prices).all(axis=1) | (prices <= 0).any(axis=1)
               | (high_error > price_step * 1.01) | (low_error > price_step * 1.01))
    if invalid.any():
        raise ValueError(f"Invalid OHLC rows: {int(invalid.sum())}: {path}")
    raw.attrs["repaired_bounds"] = int(((high_error > 0) | (low_error > 0)).sum())
    raw["High"] = prices.max(axis=1)
    raw["Low"] = prices.min(axis=1)
    return raw


def features(bars, news_path):
    ind = add_indicators(bars)
    atr = ind.ATR14.replace(0, np.nan)
    x = pd.DataFrame(index=bars.index)
    for lag in (1, 4, 24, 120):
        x[f"return_{lag}"] = ind.Close.pct_change(lag)
    for span in (20, 50, 200):
        x[f"ema_{span}_atr"] = (ind.Close - ind[f"EMA{span}"]) / atr
    x["rsi"] = ind.RSI14
    x["atr_pct"] = atr / ind.Close
    x["macd_atr"] = ind.MACDHist / atr
    x["range_atr"] = (ind.High - ind.Low) / atr
    x["hour_sin"] = np.sin(2 * np.pi * x.index.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.index.hour / 24)
    technical = list(x.columns)
    news = pd.read_csv(news_path)
    # Only a statement date is available: expose at next UTC midnight.
    # Do not pretend observation dates in FRED or COT are publication times.
    news["available_at"] = pd.to_datetime(news.date, utc=True) + pd.Timedelta(days=1)
    news = news.sort_values("available_at").drop_duplicates("available_at", keep="last")
    decisions = pd.DataFrame({"decision_at": bars.index + pd.Timedelta(hours=1)})
    aligned = pd.merge_asof(decisions, news, left_on="decision_at", right_on="available_at",
                            direction="backward", tolerance=pd.Timedelta(days=90))
    x["fomc_sentiment"] = aligned.sentiment_score.to_numpy()
    x["fomc_age_days"] = ((aligned.decision_at - aligned.available_at).dt.total_seconds() / 86400).to_numpy()
    return x.replace([np.inf, -np.inf], np.nan), technical


def targets(bars, horizon):
    """Decision after bar close, fill next open, exit horizon-th bar close."""
    return bars.Close.shift(-horizon) / bars.Open.shift(-1) - 1


def split_positions(n, horizon):
    # Three expanding development windows followed by an untouched final 20%.
    edges = [int(n * fraction) for fraction in (.4, .533333, .666667, .8, 1)]
    for fold, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        train = np.arange(200, start - horizon)
        test = np.arange(start, stop - horizon)
        if len(train) < 500 or len(test) < 100:
            raise ValueError("Insufficient history for purged walk-forward evaluation")
        yield fold, train, test


def strategy_metrics(probability, returns, positions, horizon, cost, threshold=.60):
    # Fixed independent entry schedule shared by all models/baselines.
    take = np.arange(0, len(positions), horizon)
    p = probability[take]
    direction = np.where(p >= threshold, 1, np.where(p <= 1 - threshold, -1, 0))
    net = direction * returns[take] - (direction != 0) * cost[take]
    traded = net[direction != 0]
    equity = np.r_[0., np.cumsum(net)]
    return {"trades": int(len(traded)), "mean_net_bps": float(traded.mean() * 1e4) if len(traded) else None,
            "total_net_bps": float(net.sum() * 1e4),
            "max_drawdown_bps": float(np.max(np.maximum.accumulate(equity) - equity) * 1e4)}


def run(symbol="EURUSD", data_dir=ROOT / "data", output=ROOT / "data/ml", horizon=24, cost_pips=1.5):
    if horizon < 1 or not np.isfinite(cost_pips) or cost_pips < 0:
        raise ValueError("Horizon must be positive and cost finite/nonnegative")
    data_dir, output = Path(data_dir), Path(output)
    bars = load_bars(data_dir / "historical_1h" / f"{symbol}.csv", .001 if symbol.endswith("JPY") else .00001)
    x, technical = features(bars, data_dir / "news/fomc_statements.csv")
    future = targets(bars, horizon)
    y = (future > 0).astype(int)
    pip = .01 if symbol.endswith("JPY") else .0001
    costs = cost_pips * pip / bars.Open.shift(-1)
    rows, predictions = [], []
    for fold, train, test in split_positions(len(bars), horizon):
        print(f"{symbol}: fold {fold + 1}/4", flush=True)
        for name, columns in (("technical", technical), ("technical_fomc", list(x.columns))):
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                        learning_rate=.05, l2_regularization=2., early_stopping=False, random_state=42)
            with threadpool_limits(limits=2):
                model.fit(x.iloc[train][columns], y.iloc[train])
                p = model.predict_proba(x.iloc[test][columns])[:, 1]
            actual = y.iloc[test]
            row = {"fold": fold + 1, "phase": "holdout" if fold == 3 else "development",
                   "model": name, "train_end": str(bars.index[train[-1]]),
                   "test_start": str(bars.index[test[0]]), "test_end": str(bars.index[test[-1]]),
                   "auc": float(roc_auc_score(actual, p)) if actual.nunique() == 2 else None,
                   "brier": float(brier_score_loss(actual, p)),
                   "baseline_brier": float(brier_score_loss(actual, np.full(len(test), y.iloc[train].mean())))}
            row.update(strategy_metrics(p, future.iloc[test].to_numpy(), test, horizon, costs.iloc[test].to_numpy()))
            rows.append(row)
            predictions.append(pd.DataFrame({"timestamp": bars.index[test], "fold": fold + 1,
                "model": name, "probability_up": p, "forward_return": future.iloc[test].to_numpy()}))
        baseline = strategy_metrics(np.ones(len(test)), future.iloc[test].to_numpy(), test, horizon, costs.iloc[test].to_numpy())
        rows.append({"fold": fold + 1, "phase": "holdout" if fold == 3 else "development", "model": "always_long", **baseline})
    report = {"symbol": symbol, "timeframe": "1h", "bars": len(bars), "start": str(bars.index.min()),
        "end": str(bars.index.max()), "horizon_bars": horizon, "round_trip_cost_pips": cost_pips,
        "threshold": .60, "status": "RESEARCH_ONLY", "trade_permission": False,
        "repaired_ohlc_bounds": bars.attrs["repaired_bounds"],
        "limitations": ["FOMC date-only availability assumed next UTC midnight; not verified publication timestamps.",
            "FRED/events/impact/COT excluded: missing publication timestamps and/or historical vintages.",
            "Probabilities are uncalibrated direction estimates, not trade success probabilities.",
            "Fixed costs omit variable spread, swap and broker execution; 24 bars is not always 24 clock hours.",
            "Does not yet measure incremental performance on existing TREND/RANGE entries.",
            "Repeated inspection of holdout makes it development data; reserve new data before deployment."],
        "results": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{symbol}_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(predictions).to_csv(output / f"{symbol}_oos.csv", index=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"])
    parser.add_argument("--cost-pips", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    for symbol in args.symbols:
        report = run(symbol.upper(), horizon=args.horizon, cost_pips=args.cost_pips)
        print(pd.DataFrame(report["results"]).to_string(index=False))
