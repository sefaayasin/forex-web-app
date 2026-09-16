"""Freeze a deployable direction/high_volatility model for every pair.

forex_ml_tournament.py already ran the full model/bundle/horizon comparison
once and froze a single recipe per task in data/ml/tournament/selected.json
(development-only selection, then one final audit — see that file's docstring
for the full protocol). Its final_audit() step already re-fits that frozen
recipe on every pair in data/historical_1h/ to report honest 2023+ holdout
accuracy in final_metrics.csv, but only persisted the fitted model for
EURUSD. This script does not re-run any comparison — it reuses the exact
same frozen recipe and training split, and simply also saves the fitted
model for every other pair, so forex_ml_live.build_research_prediction
(already symbol-generic) can serve live LONG/SHORT predictions for them.

Run: python train_all_pair_direction_models.py
"""
from __future__ import annotations

import json

import joblib
from threadpoolctl import threadpool_limits

from forex_ml_tournament import (
    FINAL_START,
    OUT,
    ROOT,
    build_features,
    load_hourly,
    make_model,
    make_targets,
    positions,
)

MIN_PRE_FINAL_BARS = 20000  # same inclusion threshold forex_ml_tournament.final_audit uses


def main() -> None:
    selected = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    saved, skipped = [], []

    for path in sorted((ROOT / "data" / "historical_1h").glob("*.csv")):
        symbol = path.stem
        try:
            bars = load_hourly(path, quarantine=True)
        except ValueError as exc:
            skipped.append((symbol, f"veri kalitesi: {exc}"))
            continue
        bars.attrs["symbol"] = symbol
        if (bars.index < FINAL_START).sum() < MIN_PRE_FINAL_BARS:
            skipped.append((symbol, "2023 öncesi yeterli geçmiş yok"))
            continue

        x, bundles = build_features(bars, symbol)
        for task, config in selected.items():
            horizon, columns = config["horizon"], bundles[config["bundle"]]
            target = make_targets(bars, horizon)
            train = positions(bars, target, "2008-01-01", FINAL_START)
            y = target[task]
            if len(train) < 500 or y.iloc[train].nunique() != 2:
                skipped.append((symbol, f"{task}: eğitim örneği yetersiz"))
                continue

            model = make_model(config["model"])
            with threadpool_limits(limits=2):
                model.fit(x.iloc[train][columns], y.iloc[train])

            out_path = OUT / f"{symbol}_{task}_research.joblib"
            joblib.dump({
                "model": model,
                "columns": columns,
                "config": config,
                "task": task,
                "symbol": symbol,
                "trained_before": str(FINAL_START),
                "status": "RESEARCH_ONLY",
            }, out_path)
            saved.append(out_path.name)
            print(f"[OK] {symbol} {task} -> {out_path.name}", flush=True)

    print(f"\n{len(saved)} model dosyası kaydedildi (data/ml/tournament/).")
    if skipped:
        print(f"{len(skipped)} parite/görev atlandı:")
        for symbol, reason in skipped:
            print(f"  - {symbol}: {reason}")


if __name__ == "__main__":
    main()
