"""4-hour high-volatility model: development-only selection, then one final 2023+ audit.

Rules are in data/ml/vol4h/protocol.json and were written before either stage ran.
Run the stages separately so the threshold is frozen before any final result exists:

    python -X utf8 research_vol4h.py select
    python -X utf8 research_vol4h.py final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forex_ml_tournament import (
    FINAL_START,
    FOLDS,
    MAJORS,
    OUT as TOURNAMENT_DIR,
    ROOT,
    bootstrap_difference,
    build_features,
    fit_predict,
    load_hourly,
    make_model,
    make_targets,
    nonoverlap_positions,
    positions,
    score,
    write_json,
)

OUTPUT = ROOT / "data" / "ml" / "vol4h"
HOURLY_DIR = ROOT / "data" / "historical_1h"
TASK = "high_volatility"
LIVE_TASK = "high_volatility_4h"
HORIZON = 4
DEPLOYABLE_MODELS = {"logistic", "hist_boost", "extra_trees"}
THRESHOLDS = (.5, .55, .6, .65, .7, .75, .8, .85, .9)


def select_config() -> dict:
    rankings = pd.read_csv(TOURNAMENT_DIR / "rankings.csv")
    eligible = rankings[
        (rankings.task == TASK) & (rankings.horizon == HORIZON)
        & (rankings.bundle != "cot_exploratory") & rankings.model.isin(DEPLOYABLE_MODELS)
    ].sort_values("selection_score", ascending=False)
    best = eligible.iloc[0]
    return {"horizon": HORIZON, "bundle": str(best.bundle), "model": str(best.model),
            "development_balanced_accuracy": float(best.balanced_accuracy),
            "development_auc": float(best.auc),
            "development_persistence_balanced_accuracy": float(best.persistence_balanced_accuracy)}


def choose_threshold(config: dict) -> tuple[float, pd.DataFrame]:
    """The tournament's abstention rule, restricted to this one configuration."""
    rows = []
    for symbol in MAJORS:
        bars = load_hourly(HOURLY_DIR / f"{symbol}.csv", quarantine=True)
        x, bundles = build_features(bars, symbol)
        target = make_targets(bars, HORIZON)
        for fold, (start, stop) in enumerate(FOLDS):
            train = positions(bars, target, "2008-01-01", start)
            test = positions(bars, target, start, stop)
            p = fit_predict(make_model(config["model"]), x, target[TASK], train, test, bundles[config["bundle"]])
            take = nonoverlap_positions(bars.index[test], target.label_end.iloc[test])
            y, p = target[TASK].iloc[test].to_numpy()[take], p[take]
            for threshold in THRESHOLDS:
                mask = np.abs(p - .5) >= threshold - .5
                if mask.sum() < 30 or np.unique(y[mask]).size < 2:
                    continue
                rows.append({"symbol": symbol, "fold": fold, "threshold": threshold,
                    "coverage": float(mask.mean()), "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], p[mask] >= .5)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[mask], p[mask] >= .5))})
        print(f"Development threshold fits: {symbol}", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("threshold").agg(mean=("balanced_accuracy", "mean"), std=("balanced_accuracy", "std"),
                                             min_coverage=("coverage", "min"), count=("n", "count"))
    eligible = summary[(summary.min_coverage >= .2) & (summary["count"] == len(MAJORS) * len(FOLDS))].copy()
    eligible["score"] = eligible["mean"] - .5 * eligible["std"].fillna(0)
    return (float(eligible.score.idxmax()) if len(eligible) else .5), frame


def clock_probability(index: pd.DatetimeIndex, y: np.ndarray, train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Training-period high-volatility rate for each row's UTC hour (protocol_amendment.json)."""
    rate_by_hour = pd.Series(y[train]).groupby(index.hour[train]).mean()
    return rate_by_hour.reindex(index.hour[rows]).fillna(float(y[train].mean())).to_numpy()


def simple_logistic_probability(x: pd.DataFrame, y: np.ndarray, train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Info-only baseline: hour of day plus recent-to-long volatility ratios."""
    design = pd.get_dummies(pd.Series(x.index.hour, index=x.index), prefix="h").astype(float)
    design = design.reindex(columns=[f"h_{h}" for h in range(24)], fill_value=0.0)
    design[["rv_ratio_4", "rv_ratio_24"]] = x[["rv_ratio_4", "rv_ratio_24"]]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.1, max_iter=400))
    model.fit(design.iloc[train], y[train])
    return model.predict_proba(design.iloc[rows])[:, 1]


def run_select() -> None:
    frozen = OUTPUT / "selected.json"
    config = select_config()
    if frozen.exists():
        existing = json.loads(frozen.read_text(encoding="utf-8"))
        if any(existing[k] != config[k] for k in ("horizon", "bundle", "model")):
            raise ValueError("Frozen selection differs: refuse to reselect")
        print("Selection already frozen: " + json.dumps(existing))
        return
    threshold, frame = choose_threshold(config)
    frame.to_csv(OUTPUT / "abstention_development.csv", index=False)
    config["confidence_threshold"] = threshold
    write_json(frozen, config)
    print("Selection frozen: " + json.dumps(config))


def run_final() -> None:
    config = json.loads((OUTPUT / "selected.json").read_text(encoding="utf-8"))
    if (OUTPUT / "final_metrics.csv").exists():
        raise ValueError("Final audit already ran; results are not recomputed or tuned")
    metrics, quality = [], []
    for path in sorted(HOURLY_DIR.glob("*.csv")):
        symbol = path.stem
        try:
            bars = load_hourly(path, quarantine=True)
            bars.attrs["symbol"] = symbol
            if (bars.index < FINAL_START).sum() < 20000 or (bars.index >= FINAL_START).sum() < 5000:
                raise ValueError("insufficient pre-2023 or final history")
            x, bundles = build_features(bars, symbol)
        except ValueError as exc:
            quality.append({"symbol": symbol, "status": "excluded", "reason": str(exc)})
            continue
        quality.append({"symbol": symbol, "status": "included"})
        columns = bundles[config["bundle"]]
        target = make_targets(bars, HORIZON)
        train = positions(bars, target, "2008-01-01", FINAL_START)
        test = positions(bars, target, FINAL_START, bars.index[-1] + pd.Timedelta(hours=1))
        model = make_model(config["model"])
        p = fit_predict(model, x, target[TASK], train, test, columns)
        take = nonoverlap_positions(bars.index[test], target.label_end.iloc[test])
        yy, pp = target[TASK].iloc[test].to_numpy()[take], p[take]
        bb = target.vol_persistence.iloc[test].to_numpy()[take]
        prevalence = target[TASK].iloc[train].mean()
        confident = np.abs(pp - .5) >= config["confidence_threshold"] - .5
        ci_low, ci_high = bootstrap_difference(yy, pp, bb)
        metric = {"symbol": symbol, "task": LIVE_TASK, **config,
                  "previously_inspected": symbol in ["EURUSD", "GBPUSD", "USDJPY"],
                  "accuracy_lift_ci_low": ci_low, "accuracy_lift_ci_high": ci_high}
        metric.update(score(yy, pp, prevalence, bb))
        metric.update(
            confident_n=int(confident.sum()), confident_coverage=float(confident.mean()),
            confident_accuracy=float(accuracy_score(yy[confident], pp[confident] >= .5)) if confident.any() else None,
            confident_baseline_accuracy=float(accuracy_score(yy[confident], bb[confident])) if confident.any() else None,
            confident_majority_accuracy=float(accuracy_score(yy[confident], np.full(confident.sum(), prevalence >= .5))) if confident.any() else None,
            confident_positive_rate=float(yy[confident].mean()) if confident.any() else None,
        )
        y_all = target[TASK].to_numpy()
        clock_p = clock_probability(bars.index, y_all, train, test[take])
        simple_p = simple_logistic_probability(x, y_all, train, test[take])
        metric.update(
            clock_accuracy=float(accuracy_score(yy, clock_p >= .5)),
            clock_balanced_accuracy=float(balanced_accuracy_score(yy, clock_p >= .5)),
            clock_auc=float(roc_auc_score(yy, clock_p)),
            simple_balanced_accuracy=float(balanced_accuracy_score(yy, simple_p >= .5)),
            simple_auc=float(roc_auc_score(yy, simple_p)),
        )
        metrics.append(metric)
        joblib.dump({"model": model, "columns": columns, "config": config, "task": LIVE_TASK, "symbol": symbol,
                     "trained_before": str(FINAL_START), "status": "RESEARCH_ONLY"},
                    TOURNAMENT_DIR / f"{symbol}_{LIVE_TASK}_research.joblib")
        print(f"Final {symbol}: balanced={metric['balanced_accuracy']:.3f} "
              f"persistence={metric['persistence_balanced_accuracy']:.3f} n={len(yy)}", flush=True)

    frame = pd.DataFrame(metrics)
    frame.to_csv(OUTPUT / "final_metrics.csv", index=False)
    write_json(OUTPUT / "data_quality.json", quality)
    lift = frame.balanced_accuracy - frame.persistence_balanced_accuracy
    checks = {
        "full_sample_lift_at_least_2pp": bool(lift.mean() >= .02),
        "lift_ci_low_positive_in_half_of_pairs": bool((frame.accuracy_lift_ci_low > 0).mean() >= .5),
        "confident_beats_persistence": bool(frame.confident_accuracy.mean() > frame.confident_baseline_accuracy.mean()),
        "confident_beats_majority": bool(frame.confident_accuracy.mean() > frame.confident_majority_accuracy.mean()),
        "beats_clock_baseline": bool(frame.balanced_accuracy.mean() > frame.clock_balanced_accuracy.mean()),
    }
    summary = {
        "pairs": int(len(frame)),
        "balanced_accuracy": float(frame.balanced_accuracy.mean()),
        "persistence_balanced_accuracy": float(frame.persistence_balanced_accuracy.mean()),
        "auc": float(frame.auc.mean()),
        "accuracy": float(frame.accuracy.mean()),
        "persistence_accuracy": float(frame.persistence_accuracy.mean()),
        "majority_accuracy": float(frame.majority_accuracy.mean()),
        "positive_lift_ci_pairs": int((frame.accuracy_lift_ci_low > 0).sum()),
        "confident_coverage": float(frame.confident_coverage.mean()),
        "confident_accuracy": float(frame.confident_accuracy.mean()),
        "confident_baseline_accuracy": float(frame.confident_baseline_accuracy.mean()),
        "confident_majority_accuracy": float(frame.confident_majority_accuracy.mean()),
        "clock_balanced_accuracy": float(frame.clock_balanced_accuracy.mean()),
        "clock_auc": float(frame.clock_auc.mean()),
        "simple_balanced_accuracy": float(frame.simple_balanced_accuracy.mean()),
        "simple_auc": float(frame.simple_auc.mean()),
        "model_beats_clock_pairs": int((frame.balanced_accuracy > frame.clock_balanced_accuracy).sum()),
        "n_nonoverlap": int(frame.n.sum()),
        "checks": checks,
        "passes": all(checks.values()),
        "selected": config,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "final"])
    parser.add_argument("--hourly-dir", type=Path, default=HOURLY_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    HOURLY_DIR, OUTPUT = args.hourly_dir, args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_select() if args.stage == "select" else run_final()
