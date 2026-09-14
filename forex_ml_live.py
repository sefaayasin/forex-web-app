"""Live scoring for the 2008+ research-grade direction/volatility models.

forex_ml_tournament.py trains these models offline on years of hourly history
plus FOMC statement sentiment (the "fomc" feature bundle) and freezes one
joblib bundle per task for EURUSD. Their measured skill is explicitly weak —
out-of-sample direction balanced accuracy is only ~52% (a coin flip is 50%),
barely above the persistence/majority baselines — so this module is meant to
produce a soft, informational note, never a standalone LONG/SHORT trigger.

This module has no Streamlit dependency so it can be tested and reused
without starting the web app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from forex_ml_tournament import build_features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data/ml/tournament"
MIN_BARS_REQUIRED = 500
DIRECTION_ACCURACY_NOTE = (
    "Araştırma modeli notu: 2008-2023 saatlik veri + FOMC metniyle eğitildi, sadece EURUSD için var. "
    "2023 sonrası testte yön isabeti yaklaşık %52 (yazı-tura %50); tek başına işlem sinyali değildir."
)


def _model_path(symbol: str, task: str) -> Path:
    return MODEL_DIR / f"{symbol}_{task}_research.joblib"


def load_research_model(symbol: str, task: str) -> Optional[dict]:
    path = _model_path(symbol.upper(), task)
    if not path.exists():
        return None
    return joblib.load(path)


def normalize_to_utc_hourly(bars: pd.DataFrame) -> pd.DataFrame:
    """Match the tz-aware UTC index forex_ml_tournament.build_features expects."""
    out = bars[["Open", "High", "Low", "Close"]].dropna().copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def build_research_prediction(symbol: str, task: str = "direction", bars: Optional[pd.DataFrame] = None) -> dict:
    """Score the latest closed hourly bar with the frozen research model.

    `bars` must be hourly OHLC with a UTC (or tz-naive UTC) index when
    provided directly, e.g. for tests; otherwise pass already-fetched hourly
    data from the caller since this module does not fetch market data itself.
    """
    base_symbol = str(symbol).replace("=X", "").upper()
    bundle = load_research_model(base_symbol, task)
    if bundle is None:
        return {
            "status": "unavailable",
            "text": f"{base_symbol} için araştırma modeli henüz yok (şu an sadece EURUSD).",
        }

    if bars is None or bars.empty:
        return {"status": "no_data", "text": "Canlı saatlik veri alınamadı."}

    normalized = normalize_to_utc_hourly(bars)
    if len(normalized) < MIN_BARS_REQUIRED:
        return {
            "status": "insufficient_data",
            "text": f"Özellik hesaplamak için en az {MIN_BARS_REQUIRED} saatlik mum gerekiyor; {len(normalized)} var.",
        }

    try:
        features_df, _ = build_features(normalized, base_symbol)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a status, not raised
        return {"status": "error", "text": f"Özellik hesaplanamadı: {exc}"}

    columns = bundle["columns"]
    missing_columns = [c for c in columns if c not in features_df.columns]
    if missing_columns:
        return {"status": "error", "text": "Model özellik sütunları güncel kodla uyuşmuyor."}

    row = features_df[columns].iloc[[-1]]
    if row.isna().any(axis=1).iloc[0]:
        return {"status": "insufficient_data", "text": "Son mumda eksik özellik değeri var."}

    model = bundle["model"]
    probability_up = float(model.predict_proba(row)[:, 1][0])

    return {
        "status": "ready",
        "task": task,
        "symbol": base_symbol,
        "as_of": features_df.index[-1],
        "probability_up": probability_up,
        "horizon_bars": bundle["config"]["horizon"],
        "model_name": bundle["config"]["model"],
        "note": DIRECTION_ACCURACY_NOTE if task == "direction" else (
            "Araştırma modeli notu: bu skor fiyat yönünü değil, önümüzdeki dönemin oynaklığının "
            "tarihi ortalamanın üstüne çıkma ihtimalini tahmin eder; işlem izni vermez."
        ),
    }


def research_signal_alignment(prediction: dict, side: str) -> Optional[dict]:
    """Describe whether the research model's direction view agrees with `side`.

    Returns None when there is nothing to say (model unavailable/not ready).
    """
    if prediction.get("status") != "ready" or prediction.get("task") != "direction":
        return None
    probability_up = float(prediction["probability_up"])
    side_upper = str(side).upper()
    if side_upper not in {"LONG", "SHORT"}:
        return None
    aligned_probability_pct = (probability_up if side_upper == "LONG" else 1.0 - probability_up) * 100
    return {
        "side": side_upper,
        "aligned": aligned_probability_pct >= 50.0,
        "aligned_probability_pct": aligned_probability_pct,
    }
