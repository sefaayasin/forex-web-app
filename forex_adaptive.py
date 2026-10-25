"""Local, periodically relearned 15-minute entry research; no broker orders.

python -X utf8 forex_adaptive.py --symbols EURUSD GBPUSD USDJPY
python -X utf8 forex_adaptive.py --watch 600
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from forex_config import get_pip_size
from forex_decision_core import stationary_bootstrap_mean_test
from forex_indicators import add_indicators
from forex_ml_tournament import load_hourly
from research_news_scan import load_app_score

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research/adaptive_entry"
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
VERSION = 1
COST, STRESS_COST = 1.5, 3.0
HOLD = pd.Timedelta(hours=4)
FIVE, FIFTEEN = pd.Timedelta(minutes=5), pd.Timedelta(minutes=15)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resample_complete(bars, hours):
    grouped = bars.resample(f"{hours}h", label="left", closed="left")
    out = grouped.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return out[grouped.Close.count() == hours * 4].dropna()


def entry_features(bars):
    """Features indexed by completed 15M bar OPEN time; no future data."""
    ind = add_indicators(bars)
    atr = ind.ATR14.replace(0, np.nan)
    x = pd.DataFrame(index=bars.index)
    groups = {}
    for lag in (1, 4, 16, 96):
        x[f"return_{lag}"] = bars.Close.pct_change(lag)
    ret = bars.Close.pct_change()
    x["rv_ratio"] = np.sqrt(ret.pow(2).rolling(16).mean() / ret.pow(2).rolling(480).mean())
    x["atr_pct"] = atr / bars.Close
    x["body_atr"] = (bars.Close - bars.Open) / atr
    x["range_atr"] = (bars.High - bars.Low) / atr
    decisions = bars.index + FIFTEEN
    x["hour_sin"] = np.sin(2 * np.pi * decisions.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * decisions.hour / 24)
    groups["price"] = list(x.columns)
    for span in (20, 50, 200):
        x[f"ema_{span}_atr"] = (bars.Close - ind[f"EMA{span}"]) / atr
    x["rsi"] = (ind.RSI14 - 50) / 50
    x["macd_atr"] = ind.MACDHist / atr
    x["bb_position"] = (bars.Close - ind.BBMid) / (ind.BBUp - ind.BBLow).replace(0, np.nan)
    score = load_app_score()
    x["score15"] = score(bars) / 100
    for hours in (1, 4):
        higher = resample_complete(bars, hours)
        s = score(higher)
        aligned = pd.merge_asof(
            pd.DataFrame({"decision": decisions}),
            pd.DataFrame({"available": higher.index + pd.Timedelta(hours=hours), "score": s.to_numpy()}),
            left_on="decision", right_on="available", direction="backward",
            tolerance=pd.Timedelta(hours=hours * 2))
        x[f"score{hours}h"] = aligned.score.to_numpy() / 100
    groups["technical"] = list(x.columns)
    up, down = bars.High.diff(), -bars.Low.diff()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr
    dx = (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    x["adx"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    x["di_balance"] = plus - minus
    low, high = bars.Low.rolling(14).min(), bars.High.rolling(14).max()
    x["stochastic"] = (bars.Close - low) / (high - low).replace(0, np.nan) - .5
    low, high = bars.Low.rolling(96).min(), bars.High.rolling(96).max()
    x["donchian"] = (bars.Close - low) / (high - low).replace(0, np.nan) - .5
    x["efficiency"] = bars.Close.diff(16).abs() / bars.Close.diff().abs().rolling(16).sum().replace(0, np.nan)
    groups["extended"] = list(x.columns)
    # Attach side later. Signed features let the model learn support/opposition
    # to a candidate, rather than confounding LONG and SHORT inputs.
    return x.replace([np.inf, -np.inf], np.nan), groups, atr


def trade_outcome(values, entry, side, distance, target_distance=None):
    """48 contiguous 5M bars; adverse gaps at open, simultaneous hit -> stop."""
    price = values[entry, 0]
    target_distance = distance * 1.5 if target_distance is None else target_distance
    stop, target = price - side * distance, price + side * target_distance
    for i in range(entry, entry + 48):
        opening, high, low, close = values[i]
        if i > entry and side * (opening - stop) <= 0:
            return opening, i, "stop_gap"
        if i > entry and side * (opening - target) >= 0:
            return target, i, "target_gap"
        if (low <= stop if side == 1 else high >= stop):
            return stop, i, "stop"
        if (high >= target if side == 1 else low <= target):
            return target, i, "target"
    return close, i, "time"


def build_dataset(symbol, data_dir=ROOT / "data", net_target_ratio=None):
    paths = [Path(data_dir) / f"historical_{tf}" / f"{symbol}.csv" for tf in ("15m", "5m")]
    bars, fills = [load_hourly(p, quarantine=True) for p in paths]
    for frame, minutes in ((bars, 15), (fills, 5)):
        if (frame.index.asi8 % pd.Timedelta(minutes=minutes).value != 0).any():
            raise ValueError("Off-grid bar timestamps")
    bad15 = pd.Series(False, index=bars.index)
    bad15.iloc[bars.attrs.get("bad_positions", [])] = True
    # Include the long 4H EMA context after a corrupted 15M bar, too.
    affected15 = bad15.rolling(4001, min_periods=1).max().astype(bool).to_numpy()
    bad5 = np.zeros(len(fills), dtype=bool)
    bad5[fills.attrs.get("bad_positions", [])] = True
    x, groups, atr = entry_features(bars)
    scores = x.score15.to_numpy() * 100
    states = np.select([scores >= 25, scores <= -25], [1, -1], default=0)
    previous = np.r_[0, states[:-1]]
    candidates = np.flatnonzero((states != 0) & (states != previous))
    values = fills[["Open", "High", "Low", "Close"]].to_numpy()
    times = fills.index
    pip = get_pip_size(symbol)
    rows, positions, blocked_until = [], [], None
    counters = {"candidates": len(candidates), "occupied": 0, "missing_path": 0, "warmup": 0}
    for i in candidates:
        when = bars.index[i] + FIFTEEN
        if blocked_until is not None and when < blocked_until:
            counters["occupied"] += 1
            continue
        if i < 480 or affected15[i] or not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0:
            counters["warmup"] += 1
            continue
        j = times.searchsorted(when)
        # Check the entire fixed outcome window for every candidate, regardless
        # of its eventual exit. Incomplete/weekend paths are never backfilled.
        if (j + 48 > len(times) or times[j] != when
                or times[j+47] != when + HOLD - FIVE or bad5[j:j+48].any()):
            counters["missing_path"] += 1
            continue
        side, distance = int(states[i]), float(atr.iloc[i] * 1.5)
        stop_pips = distance / pip
        target_distance = None if net_target_ratio is None else pip * (COST + net_target_ratio * (stop_pips + COST))
        exit_price, exit_index, reason = trade_outcome(values, j, side, distance, target_distance)
        gross = side * (exit_price - values[j, 0]) / pip
        risk_pips = stop_pips if net_target_ratio is None else stop_pips + COST
        rows.append({"symbol": symbol, "entry_time": when, "exit_time": times[exit_index] + FIVE,
            "label_end": when + HOLD, "side": side, "stop_pips": stop_pips,
            "gross_pips": gross, "net_r": (gross - COST) / risk_pips,
            "net_r_stress": (gross - STRESS_COST) / risk_pips, "exit_reason": reason})
        positions.append(i)
        # Freeze a common candidate schedule for all filters/baselines. An
        # omitted trade does not introduce new, outcome-dependent opportunities.
        blocked_until = times[exit_index] + FIVE
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"No usable candidates: {symbol}")
    features = x.iloc[positions].reset_index(drop=True)
    unsigned = {"rv_ratio", "atr_pct", "range_atr", "hour_sin", "hour_cos", "adx", "efficiency"}
    for c in features:
        if c not in unsigned:
            features[c] *= data.side
    data = pd.concat([data, features], axis=1)
    data["cost_fraction"] = COST / data.stop_pips
    for columns in groups.values():
        columns.extend(["side", "cost_fraction"])
    data["label"] = ((data.net_r > 0) if net_target_ratio is None else
        data.exit_reason.isin(["target", "target_gap"])).astype(int)
    audit = {"bars15": len(bars), "bars5": len(fills), "start": str(bars.index[0]),
        "end": str(bars.index[-1] + FIFTEEN), "counters": {k: int(v) for k, v in counters.items()},
        "repaired_bounds": [bars.attrs["repaired_bounds"], fills.attrs["repaired_bounds"]],
        "quarantined_rows": [bars.attrs["quarantined_rows"], fills.attrs["quarantined_rows"]],
        "source_hashes": {str(p): digest(p) for p in paths}}
    return data, groups, audit


def before(frame, start, stop):
    return frame[(frame.entry_time >= start) & (frame.entry_time < stop) & (frame.label_end < stop)]


def model_factory(name):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=.1, max_iter=400, random_state=42))
    return HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7,
        learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=42)


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def select_model(data, groups, cutoff):
    """Train -> 6M model selection -> 6M calibration/threshold -> future audit."""
    train_end = cutoff - pd.DateOffset(months=12)
    select_end = cutoff - pd.DateOffset(months=6)
    validation = before(data, train_end, select_end)
    calibration = before(data, select_end, cutoff)
    if min(len(validation), len(calibration)) < 100:
        raise ValueError(f"Insufficient validation history before {cutoff}")
    trials, fitted = [], []
    for years in (3, 8):
        train = before(data, train_end - pd.DateOffset(years=years), train_end)
        if len(train) < 500 or train.label.nunique() < 2:
            continue
        for group in groups:
            for name in (("logistic", "hist_boost") if group == "extended" else ("logistic",)):
                columns = groups[group]
                model = model_factory(name)
                model.fit(train[columns], train.label)
                p = model.predict_proba(validation[columns])[:, 1]
                trials.append({"model": name, "bundle": group, "years": years,
                    "validation_brier": float(brier_score_loss(validation.label, p))})
                fitted.append((model, columns, train))
    if not trials:
        raise ValueError("No eligible training window")
    best = int(np.argmin([r["validation_brier"] for r in trials]))
    model, columns, train = fitted[best]
    raw = model.predict_proba(calibration[columns])[:, 1]
    calibrator = None
    if calibration.label.nunique() == 2:
        calibrator = LogisticRegression(C=1., random_state=42).fit(logit(raw), calibration.label)
        p = calibrator.predict_proba(logit(raw))[:, 1]
    else:
        p = np.full(len(calibration), calibration.label.mean())
    options = []
    for coverage in (.1, .2, .3, .5, 1.):
        threshold = float(np.quantile(p, 1 - coverage))
        kept = calibration[p >= threshold]
        if len(kept) >= 50:
            options.append({"threshold": threshold, "coverage": coverage,
                "trades": len(kept), "mean_r": float(kept.net_r.mean()),
                "mean_r_stress": float(kept.net_r_stress.mean())})
    # Learning must be able to say "no useful trades" rather than force a bet.
    eligible = [o for o in options if o["mean_r"] > 0 and o["mean_r_stress"] > 0]
    selected = max(eligible, key=lambda o: o["mean_r_stress"]) if eligible else None
    diagnostic = max(options, key=lambda o: o["mean_r_stress"]) if options else None
    weights = []
    if trials[best]["model"] == "logistic":
        # Calibration rescales (and can reverse) the original log-odds.
        scale = float(calibrator.coef_[0, 0]) if calibrator is not None else 0.
        coef = model[-1].coef_[0] * scale
        total = np.abs(coef).sum()
        weights = [{"feature": c, "coefficient": float(v),
                    "relative_weight_pct": float(abs(v) / total * 100) if total else 0.}
                   for c, v in zip(columns, coef)]
    return {"model": model, "calibrator": calibrator, "constant": float(calibration.label.mean()),
        "columns": columns, "cutoff": str(cutoff), "config": trials[best],
        "threshold": selected["threshold"] if selected else None,
        "diagnostic_threshold": diagnostic["threshold"] if diagnostic else None,
        "baseline_probability": float(train.label.mean()), "weights": weights,
        "selection_trials": trials, "threshold_trials": options,
        "periods": {"train_start": str(train.entry_time.min()), "train_last_label": str(train.label_end.max()),
            "validation_start": str(train_end), "calibration_start": str(select_end),
            "calibration_last_label": str(calibration.label_end.max())}}


def probabilities(bundle, data):
    raw = bundle["model"].predict_proba(data[bundle["columns"]])[:, 1]
    if bundle["calibrator"] is None:
        return np.full(len(data), bundle["constant"])
    return bundle["calibrator"].predict_proba(logit(raw))[:, 1]


def metrics(data):
    if data.empty:
        return {"trades": 0, "win_rate": None, "mean_r": None, "mean_r_stress": None, "max_drawdown_r": 0.}
    equity = np.r_[0., data.net_r.cumsum().to_numpy()]
    return {"trades": len(data), "win_rate": float(data.net_r.gt(0).mean()),
        "target_rate": float(data.exit_reason.isin(["target", "target_gap"]).mean()) if "exit_reason" in data else None,
        "mean_r": float(data.net_r.mean()),
        "mean_r_stress": float(data.net_r_stress.mean()),
        "max_drawdown_r": float((np.maximum.accumulate(equity) - equity).max())}


def public_bundle(bundle):
    return {k: v for k, v in bundle.items() if k not in {"model", "calibrator"}}


def update_symbol(symbol, output=OUTPUT, data_dir=ROOT / "data", start="2023-01-01", months=3, net_target_ratio=None):
    symbol = symbol.upper().removesuffix("=X")
    if len(symbol) != 6 or not symbol.isalpha():
        raise ValueError("Expected six-letter FX symbol")
    if months not in (1, 3, 6, 12):
        raise ValueError("Retraining interval must be 1, 3, 6 or 12 months")
    if net_target_ratio is not None and (not np.isfinite(net_target_ratio) or net_target_ratio <= 0):
        raise ValueError("Net target/risk ratio must be finite and positive")
    folder = Path(output) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    sources = [Path(data_dir) / f"historical_{tf}" / f"{symbol}.csv" for tf in ("15m", "5m")]
    identity = {"version": VERSION, "start": start, "months": months,
        "cost": COST, "stress_cost": STRESS_COST, "net_target_ratio": net_target_ratio,
        "code_hashes": {p.name: digest(p) for p in (Path(__file__), ROOT / "forex_indicators.py", ROOT / "forex_analysis.py",
            ROOT / "forex_ml_tournament.py", ROOT / "research_news_scan.py")},
        "score_code": hashlib.sha256(ast.dump(ast.Module(body=[n for n in ast.parse(
            (ROOT / "forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name in {"score_series_for_backtest", "_utc_index_series"}],
            type_ignores=[])).encode()).hexdigest(),
        "sources": {str(p): digest(p) for p in sources}}
    report_path = folder / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("identity") == identity:
            print(f"{symbol}: unchanged archive/protocol, cached report", flush=True)
            return previous
    print(f"{symbol}: building closed-bar entry dataset", flush=True)
    data, groups, audit = build_dataset(symbol, data_dir, net_target_ratio)
    data.to_csv(folder / "dataset.csv.gz", index=False, compression="gzip")
    cutoff = pd.Timestamp(start, tz="UTC")
    last = data.label_end.max()
    predictions, periods = [], []
    with threadpool_limits(limits=2):
        while cutoff < last:
            stop = min(cutoff + pd.DateOffset(months=months), last + FIVE)
            test = before(data, cutoff, stop).copy()
            if len(test):
                bundle = select_model(data, groups, cutoff)
                test["probability"] = probabilities(bundle, test)
                test["baseline_probability"] = bundle["baseline_probability"]
                threshold = bundle["threshold"]
                test["selected"] = False if threshold is None else test.probability >= threshold
                diagnostic = bundle["diagnostic_threshold"]
                test["diagnostic_selected"] = False if diagnostic is None else test.probability >= diagnostic
                test["period"] = str(cutoff)
                predictions.append(test)
                row = {"cutoff": str(cutoff), "test_end": str(stop), "selection": public_bundle(bundle),
                    "all": metrics(test), "selected": metrics(test[test.selected]),
                    "diagnostic": metrics(test[test.diagnostic_selected]),
                    "brier": float(brier_score_loss(test.label, test.probability)),
                    "baseline_brier": float(brier_score_loss(test.label, test.baseline_probability))}
                periods.append(row)
                print(f"{symbol}: {cutoff.date()} {bundle['config']['model']}/{bundle['config']['bundle']}; "
                      f"candidates={len(test)}, selected={int(test.selected.sum())}", flush=True)
            cutoff += pd.DateOffset(months=months)
        if not predictions:
            raise ValueError("No out-of-sample periods in archive")
        # As-of update only uses matured outcomes. Appends in the archive trigger
        # rebuilding and another bounded search; unchanged files skip all fitting.
        latest_cutoff = last + FIVE
        latest = select_model(data, groups, latest_cutoff)
    oos = pd.concat(predictions).sort_values("entry_time").reset_index(drop=True)
    oos.to_csv(folder / "oos.csv.gz", index=False, compression="gzip")
    kept = oos[oos.selected]
    daily = kept.groupby(kept.entry_time.dt.floor("D")).net_r.mean().sort_index()
    bootstrap = stationary_bootstrap_mean_test(daily.tolist(), simulations=1000, seed=42)
    bootstrap = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in bootstrap.items()}
    selected_metrics = metrics(kept)
    # Retrospective re-use of inspected history cannot authorize live trading.
    report = {"symbol": symbol, "identity": identity, "audit": audit, "net_target_ratio": net_target_ratio,
        "label_definition": "net_positive" if net_target_ratio is None else "target_before_stop_or_timeout",
        "status": "RESEARCH_ONLY", "trade_permission": False,
        "evaluation": "Retrospective purged walk-forward; 2023+ history was already inspected in earlier research.",
        "all_candidates": metrics(oos), "selected": selected_metrics,
        "diagnostic": metrics(oos[oos.diagnostic_selected]),
        "coverage": float(oos.selected.mean()),
        "brier": float(brier_score_loss(oos.label, oos.probability)),
        "baseline_brier": float(brier_score_loss(oos.label, oos.baseline_probability)),
        "auc": float(roc_auc_score(oos.label, oos.probability)) if oos.label.nunique() == 2 else None,
        "day_bootstrap": bootstrap, "periods": periods, "latest": public_bundle(latest)}
    if any(digest(p) != identity["sources"][str(p)] for p in sources):
        raise ValueError("Archive changed during training; retry after downloader finishes")
    joblib.dump(latest, folder / "latest.joblib.tmp")
    (folder / "latest.joblib.tmp").replace(folder / "latest.joblib")
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=MAJORS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--months", type=int, default=3, choices=(1, 3, 6, 12))
    parser.add_argument("--watch", type=int, default=0, help="Poll changed local archives every N seconds; process must remain running")
    parser.add_argument("--net-target-ratio", type=float, default=None, help="Net profit/planned loss; .3 matches $30/$100")
    args = parser.parse_args()
    if args.watch and args.watch < 60:
        parser.error("--watch must be at least 60 seconds")
    while True:
        summaries, errors = [], []
        for symbol in args.symbols:
            try:
                report = update_symbol(symbol, args.output, start=args.start, months=args.months, net_target_ratio=args.net_target_ratio)
            except (OSError, ValueError) as exc:
                errors.append({"symbol": symbol, "error": str(exc)})
                print(f"{symbol}: {exc}", flush=True)
                continue
            summaries.append({"symbol": symbol, **report["selected"], "coverage": report["coverage"],
                "brier": report["brier"], "baseline_brier": report["baseline_brier"],
                "diagnostic_mean_r": report["diagnostic"]["mean_r"],
                "baseline_mean_r": report["all_candidates"]["mean_r"]})
        args.output.mkdir(parents=True, exist_ok=True)
        if summaries:
            pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
        write_json(args.output / "monitor.json", {"checked_at": str(pd.Timestamp.now(tz="UTC")),
            "watch_interval_seconds": args.watch, "errors": errors, "symbols": args.symbols})
        print(pd.DataFrame(summaries).to_string(index=False), flush=True)
        if not args.watch:
            if errors:
                raise SystemExit(1)
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
