"""Live scoring for the 2008+ research-grade direction/volatility models.

forex_ml_tournament.py trains these models offline on years of hourly history
plus FOMC statement sentiment (the "fomc" feature bundle), and
train_all_pair_direction_models.py freezes one joblib bundle per pair and task.
Direction skill is explicitly weak — out-of-sample balanced accuracy is only
~52% (a coin flip is 50%) and negative per trade after 1.5 pip costs — so it is
only ever a soft, informational note, never a standalone LONG/SHORT trigger.
The 72-hour high-volatility model is the stronger result, and only when its
confidence clears the frozen threshold; it is surfaced as risk context, never
as a trade direction or permission.

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
    "Araştırma modeli notu: 2008-2023 saatlik veri + FOMC metniyle eğitildi. "
    "2023 sonrası testte yön isabeti yaklaşık %52 (yazı-tura %50) ve 1,5 pip maliyetle neredeyse "
    "tüm paritelerde işlem başına zararda; tek başına işlem sinyali değildir."
)
# selected.json'da dondurulan güven eşiği; bundle config'inde yoksa bu kullanılır.
VOLATILITY_CONFIDENCE_THRESHOLD = 0.65


def _model_path(symbol: str, task: str) -> Path:
    return MODEL_DIR / f"{symbol}_{task}_research.joblib"


def has_research_model(symbol: str, task: str) -> bool:
    return _model_path(str(symbol).replace("=X", "").upper(), task).exists()


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

    `probability_up` is the model's positive-class probability: price up for
    "direction", above-normal volatility for "high_volatility".
    """
    base_symbol = str(symbol).replace("=X", "").upper()
    bundle = load_research_model(base_symbol, task)
    if bundle is None:
        return {
            "status": "unavailable",
            "text": f"{base_symbol} için araştırma modeli bulunamadı.",
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
    try:
        probability_up = float(model.predict_proba(row)[:, 1][0])
    except Exception as exc:  # noqa: BLE001 - e.g. a scikit-learn version mismatch on the host
        return {"status": "error", "text": f"Model çalıştırılamadı (sürüm uyuşmazlığı olabilir): {exc}"}

    return {
        "status": "ready",
        "task": task,
        "symbol": base_symbol,
        "as_of": features_df.index[-1],
        "probability_up": probability_up,
        "horizon_bars": bundle["config"]["horizon"],
        "model_name": bundle["config"]["model"],
        "confidence_threshold": float(bundle["config"].get("confidence_threshold", VOLATILITY_CONFIDENCE_THRESHOLD)),
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


VOLATILITY_LEVEL_LABELS = {
    "high": "Yüksek oynaklık bekleniyor",
    "normal": "Olağan veya sakin oynaklık bekleniyor",
    "uncertain": "Belirsiz (model emin değil)",
}


def volatility_risk_view(prediction: dict) -> Optional[dict]:
    """Turn the high-volatility model's output into a risk label.

    The model only beat the persistence baseline clearly on predictions whose
    confidence cleared the frozen threshold (either side), so anything between
    is reported as uncertain rather than as a weak lean. Returns None when
    there is nothing to say (model unavailable/not ready/wrong task).
    """
    if prediction.get("status") != "ready" or prediction.get("task") != "high_volatility":
        return None
    probability_high = float(prediction["probability_up"])
    threshold = float(prediction.get("confidence_threshold", VOLATILITY_CONFIDENCE_THRESHOLD))
    if probability_high >= threshold:
        level = "high"
    elif probability_high <= 1.0 - threshold:
        level = "normal"
    else:
        level = "uncertain"
    return {
        "level": level,
        "label": VOLATILITY_LEVEL_LABELS[level],
        "probability_high_pct": probability_high * 100,
        "confident": level != "uncertain",
        "horizon_bars": prediction.get("horizon_bars"),
    }


def direction_cost_verdict(net_bps_low_cost: Optional[float], net_bps_high_cost: Optional[float]) -> str:
    """Summarize the direction model's 2023+ mean net result per trade after costs.

    Inputs are cost_sensitivity.csv's mean_net_bps at 1.5 and 3 pip total cost
    for the always-trade (0.5) threshold the live page uses.
    """
    if net_bps_low_cost is None or pd.isna(net_bps_low_cost):
        return "-"
    if net_bps_low_cost <= 0:
        return "Maliyet sonrası zararda"
    if net_bps_high_cost is None or pd.isna(net_bps_high_cost) or net_bps_high_cost <= 0:
        return "Sadece düşük maliyette artı"
    return "Maliyet sonrası artı"
