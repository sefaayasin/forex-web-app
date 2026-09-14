"""Reproducible model tournament: selection on development only, one final audit.

Run `python forex_ml_tournament.py` to screen seven majors, freeze two global
configurations, audit all sufficiently long pairs, and generate HTML/PNG reports.
Outputs are research artifacts, never broker orders or live trading permission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from forex_indicators import add_indicators
from forex_ml import ROOT, features

OUT = ROOT / "data/ml/tournament"
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
MODELS = ["logistic", "hist_boost", "extra_trees", "catboost", "xgboost"]
BUNDLES = ["price", "context", "fomc", "cot_exploratory"]
HORIZONS = [4, 24, 72]
TASKS = ["direction", "high_volatility"]
FOLDS = [("2016-01-01", "2019-01-01"), ("2019-01-01", "2023-01-01")]
FINAL_START = pd.Timestamp("2023-01-01", tz="UTC")
SEED = 2718


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_hourly(path, quarantine=False):
    """Sort only in memory; reject duplicates and substantial OHLC corruption."""
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.pop("timestamp"), unit="ms", utc=True)
    reordered = not d.index.is_monotonic_increasing
    if d.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    d = d.sort_index().rename(columns=str.capitalize)
    p = d[["Open", "High", "Low", "Close"]]
    step = .001 if Path(path).stem.endswith("JPY") else .00001
    high_error, low_error = p.max(axis=1) - d.High, d.Low - p.min(axis=1)
    bad = (~np.isfinite(p).all(axis=1) | (p <= 0).any(axis=1)
           | (high_error > step * 1.01) | (low_error > step * 1.01))
    if bad.any() and not quarantine:
        raise ValueError(f"{int(bad.sum())} invalid OHLC rows")
    if (~np.isfinite(p).all(axis=1) | (p <= 0).any(axis=1)).any():
        raise ValueError("nonfinite or nonpositive prices")
    repaired = int(((high_error > 0) | (low_error > 0)).sum())
    d["High"], d["Low"] = p.max(axis=1), p.min(axis=1)
    d.attrs.update(reordered=reordered, repaired_bounds=repaired,
                   quarantined_rows=int(bad.sum()), bad_positions=np.flatnonzero(bad).tolist(),
                   gaps_over_4_days=int((d.index.to_series().diff() > pd.Timedelta(days=4)).sum()))
    return d


def asof_values(index, available, values, tolerance):
    left = pd.DataFrame({"decision_at": index + pd.Timedelta(hours=1)})
    right = values.copy()
    right["available_at"] = pd.to_datetime(available, utc=True).to_numpy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    right = right.sort_values("available_at").drop_duplicates("available_at", keep="last")
    aligned = pd.merge_asof(left, right, left_on="decision_at", right_on="available_at",
                            direction="backward", tolerance=pd.Timedelta(days=tolerance))
    result = aligned[values.columns].copy()
    result.index = index
    return result


def build_features(bars, symbol, data_dir=ROOT / "data"):
    x, technical = features(bars, Path(data_dir) / "news/fomc_statements.csv")
    price = list(technical)
    ind = add_indicators(bars)
    ret = bars.Close.pct_change()
    atr = ind.ATR14.replace(0, np.nan)
    for window in (4, 24, 72, 120, 480):
        x[f"rv_{window}"] = np.sqrt(ret.pow(2).rolling(window).mean())
        x[f"momentum_atr_{window}"] = (bars.Close - bars.Close.shift(window)) / atr
        x[f"channel_{window}"] = ((bars.Close - bars.Low.rolling(window).min()) /
            (bars.High.rolling(window).max() - bars.Low.rolling(window).min()).replace(0, np.nan))
    for window in (4, 24, 72, 120):
        x[f"rv_ratio_{window}"] = x[f"rv_{window}"] / x.rv_480.replace(0, np.nan)
    x["body_atr"] = (bars.Close - bars.Open) / atr
    x["upper_wick_atr"] = (bars.High - bars[["Open", "Close"]].max(axis=1)) / atr
    x["lower_wick_atr"] = (bars[["Open", "Close"]].min(axis=1) - bars.Low) / atr
    x["weekday_sin"] = np.sin(2 * np.pi * x.index.dayofweek / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * x.index.dayofweek / 7)
    x["gap_hours"] = x.index.to_series().diff().dt.total_seconds() / 3600
    # Higher timeframe context built from completed hourly observations only.
    # No left-labeled daily/weekly candle can expose its future close.
    context = [c for c in x if not c.startswith("fomc_")]
    fomc = list(x.columns)
    cot = pd.read_csv(Path(data_dir) / "news/cot/cot_currency_positioning.csv")
    for side, ccy in (("base", symbol[:3]), ("quote", symbol[3:])):
        subset = cot.loc[cot.currency == ccy].copy().sort_values("date").drop_duplicates("date")
        if subset.empty:
            continue
        net = subset.noncommercial_net / subset.open_interest.replace(0, np.nan)
        values = pd.DataFrame({f"cot_{side}_net_oi": net,
            f"cot_{side}_change": net.diff(),
            f"cot_{side}_z": (net - net.rolling(52, min_periods=20).mean()) /
                              net.rolling(52, min_periods=20).std().replace(0, np.nan)})
        # An explicit sensitivity experiment, NOT a verified historical release feed.
        available = pd.to_datetime(subset.date, utc=True) + pd.Timedelta(days=14)
        x = x.join(asof_values(x.index, available, values, tolerance=35))
    return x.replace([np.inf, -np.inf], np.nan), {
        "price": price, "context": context, "fomc": fomc, "cot_exploratory": list(x.columns)}


def make_targets(bars, horizon):
    returns = bars.Close.shift(-horizon) / bars.Open.shift(-1) - 1
    hourly_sq = bars.Close.pct_change().pow(2)
    future_vol = np.sqrt(hourly_sq.rolling(horizon).mean().shift(-horizon))
    historical_vol = np.sqrt(hourly_sq.rolling(480).mean())
    end = bars.index.to_series().shift(-horizon) + pd.Timedelta(hours=1)
    # Reject missing-history jumps and unfinished outcomes without inventing prices.
    elapsed = (end - bars.index.to_series()).dt.total_seconds() / 3600
    valid = returns.notna() & future_vol.notna() & historical_vol.gt(0) & elapsed.le(horizon + 96)
    bad = pd.Series(False, index=bars.index)
    bad.iloc[bars.attrs.get("bad_positions", [])] = True
    # Quarantine any target crossing a bad bar plus a 1000-bar feature warmup.
    affected_history = bad.rolling(1001, min_periods=1).max().astype(bool)
    affected_future = bad.rolling(horizon + 1).max().shift(-horizon).fillna(1).astype(bool)
    valid &= ~affected_history & ~affected_future
    return pd.DataFrame({"direction": (returns > 0).astype(int),
        "high_volatility": (future_vol > historical_vol).astype(int),
        "forward_return": returns, "label_end": end, "valid": valid,
        "vol_persistence": (np.sqrt(hourly_sq.rolling(horizon).mean()) > historical_vol).astype(int),
        "momentum_baseline": (bars.Close > bars.Close.shift(horizon)).astype(int)}, index=bars.index)


def positions(bars, target, start, stop, stride=4):
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    stop = stop.tz_localize("UTC") if stop.tzinfo is None else stop
    mask = (bars.index >= start) & (bars.index < stop) & target.valid & (target.label_end < stop)
    # Fixed wall-clock sampling, independent of labels and prices.
    return np.flatnonzero(mask & (bars.index.hour % stride == 0))


def nonoverlap_positions(index, end):
    selected, next_time = [], None
    for i, t in enumerate(index):
        if next_time is None or t >= next_time:
            selected.append(i)
            next_time = end.iloc[i]
    return np.asarray(selected, dtype=int)


def make_model(name):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=.1, max_iter=400, random_state=SEED))
    if name == "hist_boost":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, learning_rate=.05,
                    min_samples_leaf=100, l2_regularization=5., early_stopping=False, random_state=SEED)
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(n_estimators=80, max_depth=7, min_samples_leaf=60,
                                 max_features=.8, n_jobs=2, random_state=SEED))
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=120, depth=4, learning_rate=.05, l2_leaf_reg=5,
            thread_count=2, random_seed=SEED, verbose=False, allow_writing_files=False)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.05, min_child_weight=50,
            reg_lambda=5, subsample=1., colsample_bytree=1., tree_method="hist", n_jobs=2, random_state=SEED)
    raise ValueError(name)


def fit_predict(model, x, y, train, test, columns):
    if np.unique(y.iloc[train]).size != 2:
        raise ValueError("training target has only one class")
    with threadpool_limits(limits=2):
        model.fit(x.iloc[train][columns], y.iloc[train])
        return model.predict_proba(x.iloc[test][columns])[:, 1]


def score(y, p, prevalence, baseline):
    pred = p >= .5
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else .5,
        "brier": float(brier_score_loss(y, p)),
        "majority_accuracy": float(accuracy_score(y, np.full(len(y), prevalence >= .5))),
        "baseline_brier": float(brier_score_loss(y, np.full(len(y), prevalence))),
        "persistence_accuracy": float(accuracy_score(y, baseline)),
        "persistence_balanced_accuracy": float(balanced_accuracy_score(y, baseline)),
        "positive_rate": float(np.mean(y))}


def selection_score(frame):
    """Equal weight per currency/fold; penalize unstable balanced accuracy."""
    keys = ["task", "horizon", "bundle", "model"]
    ranked = frame.groupby(keys).agg(accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), ba_std=("balanced_accuracy", "std"),
        auc=("auc", "mean"), brier=("brier", "mean"), baseline_brier=("baseline_brier", "mean"),
        majority_accuracy=("majority_accuracy", "mean"),
        persistence_balanced_accuracy=("persistence_balanced_accuracy", "mean"), evaluations=("n", "count")).reset_index()
    ranked["selection_score"] = ranked.balanced_accuracy - .5 * ranked.ba_std.fillna(0)
    return ranked.sort_values("selection_score", ascending=False)


def bootstrap_difference(y, p, baseline, block=10, repetitions=1000):
    """Moving-block accuracy advantage CI; no iid claims on overlapping rows."""
    diff = ((p >= .5) == y).astype(float) - (baseline == y).astype(float)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(diff), size=(repetitions, int(np.ceil(len(diff) / block))))
    idx = (starts[..., None] + np.arange(block)) % len(diff)
    means = diff[idx.reshape(repetitions, -1)[:, :len(diff)]].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def choose_abstention(output, selected, majors):
    """Select confidence thresholds on development only, requiring broad coverage.

    This never replaces all-observation accuracy. Abstention statistics must
    always include coverage and the class mix of the selected observations.
    """
    rows = []
    for symbol in majors:
        bars = load_hourly(ROOT / "data/historical_1h" / f"{symbol}.csv", quarantine=True)
        x, bundles = build_features(bars, symbol)
        for task, config in selected.items():
            target = make_targets(bars, config["horizon"])
            for fold, (start, stop) in enumerate(FOLDS):
                train = positions(bars, target, "2008-01-01", start)
                test = positions(bars, target, start, stop)
                p = fit_predict(make_model(config["model"]), x, target[task], train, test, bundles[config["bundle"]])
                take = nonoverlap_positions(bars.index[test], target.label_end.iloc[test])
                y, p = target[task].iloc[test].to_numpy()[take], p[take]
                for threshold in (.5, .55, .6, .65, .7, .75, .8, .85, .9):
                    mask = np.abs(p - .5) >= threshold - .5
                    if mask.sum() < 30 or np.unique(y[mask]).size < 2:
                        continue
                    rows.append({"task": task, "symbol": symbol, "fold": fold, "threshold": threshold,
                        "coverage": float(mask.mean()), "n": int(mask.sum()),
                        "accuracy": float(accuracy_score(y[mask], p[mask] >= .5)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[mask], p[mask] >= .5))})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "abstention_development.csv", index=False)
    for task in TASKS:
        d = frame[frame.task == task]
        summary = d.groupby("threshold").agg(mean=("balanced_accuracy", "mean"),
            std=("balanced_accuracy", "std"), min_coverage=("coverage", "min"), count=("n", "count"))
        eligible = summary[(summary.min_coverage >= .2) & (summary["count"] == len(majors) * len(FOLDS))].copy()
        eligible["score"] = eligible["mean"] - .5 * eligible["std"].fillna(0)
        threshold = float(eligible.score.idxmax()) if len(eligible) else .5
        selected[task]["confidence_threshold"] = threshold
    return selected


def trade_scores(bars, target, ix, probability, threshold, cost_pips):
    take = nonoverlap_positions(bars.index[ix], target.label_end.iloc[ix])
    p = probability[take]
    mask = np.abs(p - .5) >= threshold - .5
    direction = np.where(p >= .5, 1, -1)
    pip = .01 if bars.attrs["symbol"].endswith("JPY") else .0001
    costs = cost_pips * pip / bars.Open.shift(-1).iloc[ix].to_numpy()[take]
    ret = target.forward_return.iloc[ix].to_numpy()[take]
    net = (direction * ret - costs)[mask] * 1e4
    ys = target.direction.iloc[ix].to_numpy()[take]
    return {"threshold": threshold, "cost_pips": cost_pips, "trades": int(mask.sum()),
        "coverage": float(mask.mean()),
        "selected_accuracy": float(((p[mask] >= .5) == ys[mask]).mean()) if mask.any() else None,
        "mean_net_bps": float(net.mean()) if len(net) else None,
        "sum_net_bps": float(net.sum()), "net_win_rate": float((net > 0).mean()) if len(net) else None}


def protocol(output, majors, models, horizons):
    value = {"version": 1, "seed": SEED, "selection_symbols": majors, "models": models,
        "horizons": horizons, "tasks": TASKS, "bundles": BUNDLES, "development_folds": FOLDS,
        "final_start": str(FINAL_START), "sampling": "every 4th UTC hour; label-end purged at every boundary",
        "selection": "mean balanced accuracy minus 0.5 fold/currency standard deviation; one global configuration per target",
        "cot_policy": "14-day assumed delay sensitivity only; cannot win deployable selection",
        "final_policy": "No tuning on final results; prior EURUSD/GBPUSD/USDJPY final periods were previously inspected",
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "data/historical_1h").glob("*.csv"))},
        "news_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "data/news/fomc_statements.csv", ROOT / "data/news/cot/cot_currency_positioning.csv"]},
        "versions": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn", "catboost", "xgboost"]}}
    target = output / "protocol.json"
    value = json.loads(json.dumps(value))
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != value:
            raise ValueError("Protocol/data changed: use a new --output directory")
    else:
        write_json(target, value)
    return value


def run(output=OUT, majors=MAJORS, models=MODELS, horizons=HORIZONS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = protocol(output, majors, models, horizons)
    if (output / "complete.json").exists():
        print("Completed experiment exists; generating its report without another final evaluation.", flush=True)
        from forex_ml_visuals import generate
        generate(output)
        return
    log_path = output / "development.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    done = {(r["symbol"], r["task"], r["horizon"], r["bundle"], r["model"], r["fold"]) for r in records}
    started = time.monotonic()
    total = len(majors) * len(horizons) * len(TASKS) * len(BUNDLES) * len(models) * len(FOLDS)
    for symbol in majors:
        bars = load_hourly(ROOT / "data/historical_1h" / f"{symbol}.csv", quarantine=True)
        x, bundles = build_features(bars, symbol)
        for horizon in horizons:
            target = make_targets(bars, horizon)
            for fold, (start, stop) in enumerate(FOLDS):
                tr = positions(bars, target, "2008-01-01", start)
                te = positions(bars, target, start, stop)
                for task in TASKS:
                    y = target[task]
                    baseline = target.vol_persistence if task == "high_volatility" else target.momentum_baseline
                    for bundle in BUNDLES:
                        for name in models:
                            key = (symbol, task, horizon, bundle, name, fold)
                            if key in done:
                                continue
                            p = fit_predict(make_model(name), x, y, tr, te, bundles[bundle])
                            row = dict(zip(["symbol", "task", "horizon", "bundle", "model", "fold"], key))
                            row.update(score(y.iloc[te].to_numpy(), p, y.iloc[tr].mean(), baseline.iloc[te].to_numpy()))
                            with log_path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                            records.append(row)
                            done.add(key)
                print(f"Screen {len(done)}/{total}: {symbol} horizon={horizon} fold={fold + 1}; elapsed={time.monotonic()-started:.0f}s", flush=True)
    development = pd.DataFrame(records)
    development.to_csv(output / "development.csv", index=False)
    rankings = selection_score(development)
    rankings.to_csv(output / "rankings.csv", index=False)
    selected = {}
    for task in TASKS:
        eligible = rankings[(rankings.task == task) & (rankings.bundle != "cot_exploratory")]
        best = eligible.iloc[0]
        selected[task] = {k: int(best[k]) if k == "horizon" else str(best[k]) for k in ["horizon", "bundle", "model"]}
        selected[task]["development_balanced_accuracy"] = float(best.balanced_accuracy)
    freeze = output / "selected.json"
    if freeze.exists():
        frozen = json.loads(freeze.read_text())
        if any(any(frozen[t][k] != v for k, v in c.items()) for t, c in selected.items()):
            raise ValueError("Frozen selection differs: refuse to retune after final audit")
        selected = frozen
    else:
        selected = choose_abstention(output, selected, majors)
    write_json(freeze, selected)
    print("Selection frozen: " + json.dumps(selected), flush=True)
    final_audit(output, selected)
    write_json(output / "complete.json", {"status": "RESEARCH_ONLY", "trade_permission": False,
        "development_fits": len(records), "protocol_sha256": sha256(output / "protocol.json"),
        "seconds": time.monotonic() - started})
    from forex_ml_visuals import generate
    generate(output)


def final_audit(output, selected):
    all_metrics, all_trades, quality, predictions, importances = [], [], [], [], []
    for path in sorted((ROOT / "data/historical_1h").glob("*.csv")):
        symbol = path.stem
        try:
            bars = load_hourly(path, quarantine=True)
            bars.attrs["symbol"] = symbol
            if (bars.index < FINAL_START).sum() < 20000 or (bars.index >= FINAL_START).sum() < 5000:
                raise ValueError("insufficient pre-2023 or final history")
            x, bundles = build_features(bars, symbol)
            quality.append({"symbol": symbol, "status": "included", **bars.attrs})
        except ValueError as exc:
            quality.append({"symbol": symbol, "status": "excluded", "reason": str(exc)})
            continue
        for task, config in selected.items():
            horizon, columns = config["horizon"], bundles[config["bundle"]]
            target = make_targets(bars, horizon)
            train = positions(bars, target, "2008-01-01", FINAL_START)
            test = positions(bars, target, FINAL_START, bars.index[-1] + pd.Timedelta(hours=1))
            model = make_model(config["model"])
            p = fit_predict(model, x, target[task], train, test, columns)
            baseline = target.vol_persistence if task == "high_volatility" else target.momentum_baseline
            # Headline metrics and CI use non-overlapping observations.
            take = nonoverlap_positions(bars.index[test], target.label_end.iloc[test])
            yy = target[task].iloc[test].to_numpy()[take]
            pp = p[take]
            bb = baseline.iloc[test].to_numpy()[take]
            confidence_mask = np.abs(pp - .5) >= config["confidence_threshold"] - .5
            metric = {"symbol": symbol, "task": task, **config,
                "previously_inspected": symbol in ["EURUSD", "GBPUSD", "USDJPY"],
                "accuracy_lift_ci_low": bootstrap_difference(yy, pp, bb)[0],
                "accuracy_lift_ci_high": bootstrap_difference(yy, pp, bb)[1]}
            metric.update(score(yy, pp, target[task].iloc[train].mean(), bb))
            metric.update(confident_n=int(confidence_mask.sum()), confident_coverage=float(confidence_mask.mean()),
                confident_accuracy=float(accuracy_score(yy[confidence_mask], pp[confidence_mask] >= .5)) if confidence_mask.any() else None,
                confident_balanced_accuracy=float(balanced_accuracy_score(yy[confidence_mask], pp[confidence_mask] >= .5)) if confidence_mask.any() else None,
                confident_majority_accuracy=float(accuracy_score(yy[confidence_mask], np.full(confidence_mask.sum(), target[task].iloc[train].mean() >= .5))) if confidence_mask.any() else None,
                confident_positive_rate=float(yy[confidence_mask].mean()) if confidence_mask.any() else None,
                confident_baseline_accuracy=float(accuracy_score(yy[confidence_mask], bb[confidence_mask])) if confidence_mask.any() else None)
            all_metrics.append(metric)
            part = pd.DataFrame({"timestamp": bars.index[test], "symbol": symbol, "task": task,
                "truth": target[task].iloc[test].to_numpy(), "probability": p,
                "baseline": baseline.iloc[test].to_numpy(), "nonoverlap": False,
                "forward_return": target.forward_return.iloc[test].to_numpy()})
            part.loc[take, "nonoverlap"] = True
            predictions.append(part)
            if task == "direction":
                for threshold in (.5, .55, .6, .65, .7):
                    for cost in (0., 1.5, 3.):
                        all_trades.append({"symbol": symbol, **trade_scores(bars, target, test, p, threshold, cost)})
            # Fixed illustrative symbol; interpretability is not used to select a model.
            if symbol == "EURUSD":
                from sklearn.inspection import permutation_importance
                with threadpool_limits(limits=2):
                    imp = permutation_importance(model, x.iloc[test[take]][columns], yy,
                        n_repeats=3, scoring="balanced_accuracy", random_state=SEED, n_jobs=1)
                importances.extend({"task": task, "feature": c, "importance": float(v)}
                                   for c, v in zip(columns, imp.importances_mean))
                joblib.dump({"model": model, "columns": columns, "config": config,
                    "task": task, "symbol": symbol, "trained_before": str(FINAL_START),
                    "status": "RESEARCH_ONLY"}, output / f"EURUSD_{task}_research.joblib")
            print(f"Final {symbol} {task}: accuracy={metric['accuracy']:.3f}, balanced={metric['balanced_accuracy']:.3f}, n={len(yy)}", flush=True)
    pd.DataFrame(all_metrics).to_csv(output / "final_metrics.csv", index=False)
    pd.DataFrame(all_trades).to_csv(output / "cost_sensitivity.csv", index=False)
    pd.DataFrame(importances).to_csv(output / "feature_importance.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "final_predictions.csv.gz", index=False, compression="gzip")
    write_json(output / "data_quality.json", quality)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--symbols", nargs="+", default=MAJORS)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    args = parser.parse_args()
    if any(h < 4 or h % 4 for h in args.horizons):
        parser.error("Horizons must be positive multiples of four")
    run(args.output, args.symbols, args.models, args.horizons)
