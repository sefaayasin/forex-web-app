"""Compare the live Yahoo hourly feed with the Dukascopy archive the models were trained on.

Yahoo's hourly high-low ranges differ from Dukascopy's by a stable, pair-specific
factor (up to ~2x wider for CAD/CHF crosses). Close-to-close moves agree, but
range-based features such as ATR do not, so the 72h volatility model can give
the opposite verdict on live data. For every pair this script:

1. estimates the range ratio on the overlap older than the last 37 days,
2. on the last 30 days of overlap, compares model verdicts on Yahoo bars
   (raw and range-calibrated) with verdicts on the archive at the same hours,
3. writes data/ml/live_calibration.json, which the app uses to calibrate
   Yahoo bars and to flag pairs whose calibrated agreement stays below 80%,
   plus diagnostics to research/live_skew/results.json.

Needs network access (Yahoo Finance) and data/historical_1h_from_15m. Rerun after
retraining, or when Yahoo data quality may have changed.

Run: python -X utf8 audit_live_data_skew.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from forex_config import get_pip_size
from forex_ml_live import (
    LIVE_CALIBRATION_PATH,
    VOLATILITY_CONFIDENCE_THRESHOLD,
    calibrate_ranges,
    load_research_model,
    normalize_to_utc_hourly,
)
from forex_ml_tournament import build_features, load_hourly

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "live_skew"
EVALUATION_DAYS = 30
CALIBRATION_GAP_DAYS = 7


def fetch_yahoo(symbol: str, period: str = "730d") -> pd.DataFrame:
    raw = yf.download(f"{symbol}=X", interval="60m", period=period, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return normalize_to_utc_hourly(raw) if not raw.empty else pd.DataFrame()


def median_range(bars: pd.DataFrame, index: pd.DatetimeIndex) -> float:
    return float(((bars.High - bars.Low) / bars.Close).loc[index].median())


def verdict(p: np.ndarray) -> np.ndarray:
    return np.where(p >= VOLATILITY_CONFIDENCE_THRESHOLD, "high",
                    np.where(p <= 1 - VOLATILITY_CONFIDENCE_THRESHOLD, "normal", "uncertain"))


def model_agreement(symbol: str, yahoo: pd.DataFrame, archive_features: pd.DataFrame, hours: pd.DatetimeIndex) -> dict:
    features, _ = build_features(yahoo, symbol)
    hours = hours.intersection(features.index)
    out = {"hours": int(len(hours))}
    volatility = load_research_model(symbol, "high_volatility")
    py = volatility["model"].predict_proba(features.loc[hours, volatility["columns"]])[:, 1]
    pa = volatility["model"].predict_proba(archive_features.loc[hours, volatility["columns"]])[:, 1]
    out["volatility_agreement"] = float((verdict(py) == verdict(pa)).mean())
    out["volatility_probability_mae"] = float(np.abs(py - pa).mean())
    direction = load_research_model(symbol, "direction")
    if direction is not None:
        dy = direction["model"].predict_proba(features.loc[hours, direction["columns"]])[:, 1]
        da = direction["model"].predict_proba(archive_features.loc[hours, direction["columns"]])[:, 1]
        out["direction_agreement"] = float(((dy >= .5) == (da >= .5)).mean())
    return out


def audit_pair(symbol: str) -> dict:
    yahoo = fetch_yahoo(symbol)
    if yahoo.empty:
        return {"error": "no Yahoo data"}
    archive = load_hourly(ROOT / "data" / "historical_1h_from_15m" / f"{symbol}.csv", quarantine=True)
    archive = archive[archive.index >= yahoo.index[0] - pd.Timedelta(days=60)][["Open", "High", "Low", "Close"]]
    overlap = yahoo.index.intersection(archive.index)
    evaluation_start = overlap.max() - pd.Timedelta(days=EVALUATION_DAYS)
    calibration = overlap[overlap < evaluation_start - pd.Timedelta(days=CALIBRATION_GAP_DAYS)]
    evaluation = overlap[overlap > evaluation_start]
    ratio = median_range(yahoo, calibration) / median_range(archive, calibration)

    archive_features, _ = build_features(archive, symbol)
    raw = model_agreement(symbol, yahoo, archive_features, evaluation)
    calibrated = model_agreement(symbol, calibrate_ranges(yahoo, ratio), archive_features, evaluation)
    pip = get_pip_size(symbol)
    close_gap = (yahoo.Close.loc[overlap] - archive.Close.loc[overlap]).abs() / pip
    return {
        "range_ratio": round(ratio, 4),
        "calibration_days": int((calibration.max() - calibration.min()).days),
        "evaluation_hours": calibrated["hours"],
        "evaluation_range_ratio_raw": round(median_range(yahoo, evaluation) / median_range(archive, evaluation), 4),
        "volatility_agreement": round(calibrated["volatility_agreement"], 4),
        "volatility_agreement_uncalibrated": round(raw["volatility_agreement"], 4),
        "volatility_probability_mae": round(calibrated["volatility_probability_mae"], 4),
        "direction_agreement": round(calibrated.get("direction_agreement", float("nan")), 4),
        "yahoo_hours_missing_vs_archive": round(1 - len(overlap) / max(len(archive.loc[overlap.min():overlap.max()]), 1), 4),
        "median_close_gap_pips": round(float(close_gap.median()), 2),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    symbols = sorted(p.name.split("_")[0] for p in (ROOT / "data" / "ml" / "tournament").glob("*_high_volatility_research.joblib"))
    pairs = {}
    for symbol in symbols:
        pairs[symbol] = audit_pair(symbol)
        print(symbol, json.dumps(pairs[symbol]), flush=True)
    as_of = str(pd.Timestamp.now(tz="UTC").floor("min"))
    (OUTPUT / "results.json").write_text(json.dumps({"as_of": as_of, "pairs": pairs}, indent=2), encoding="utf-8")
    calibration = {
        "as_of": as_of,
        "method": "range_ratio = median Yahoo/Dukascopy hourly (high-low)/close on the overlap older than "
                  f"{EVALUATION_DAYS + CALIBRATION_GAP_DAYS} days; agreements are measured on the last {EVALUATION_DAYS} days after calibration",
        "pairs": {s: {k: v[k] for k in ("range_ratio", "volatility_agreement", "volatility_agreement_uncalibrated", "direction_agreement")}
                  for s, v in pairs.items() if "error" not in v},
    }
    LIVE_CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
