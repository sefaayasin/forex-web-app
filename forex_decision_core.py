"""Pure decision rules shared by live analysis and backtests.

This module intentionally has no Streamlit, network, or market-data dependency so
the trading decision can be regression-tested without starting the web app.
"""

from __future__ import annotations

import math


def classify_opportunity_readiness(
    radar_score: float,
    side: str,
    catalyst_matches: bool,
    structure_matches: bool,
    htf_conflict: bool,
    capacity_ratio: float,
    in_session: bool,
    candidate_threshold: float = 42.0,
    ready_threshold: float = 65.0,
) -> tuple[str, list[str]]:
    """Classify a radar snapshot without pretending its score is a probability.

    READY is deliberately reserved for a directional trigger with compatible
    structure, higher timeframes, session and realistic recent range.  WATCH is
    informative but must never be interpreted as permission to place an order.
    """
    normalized_side = str(side).upper()
    score = float(radar_score) if not _missing(radar_score) else 0.0
    if normalized_side not in {"LONG", "SHORT"} or score < float(candidate_threshold):
        return "NEUTRAL", ["Radar puanı aday eşiğinin altında"]

    blockers: list[str] = []
    if score < float(ready_threshold):
        blockers.append(f"Radar puanı {score:.0f}; hazır eşiği {float(ready_threshold):.0f}")
    if not catalyst_matches:
        blockers.append("15M tepki veya Bollinger kırılım tetiği yok")
    if not structure_matches:
        blockers.append("15M MA ve swing yapısı yönü doğrulamıyor")
    if htf_conflict:
        blockers.append("4H ve 1H yönleri çelişiyor")
    if _missing(capacity_ratio):
        blockers.append("Hedef kapasitesi ölçülemedi")
    elif float(capacity_ratio) > 0.80:
        blockers.append("Hedef tipik 4 saatlik hareketin %80'inden büyük")
    if not in_session:
        blockers.append("Seçili likit işlem seansı dışında")

    return ("WATCH", blockers) if blockers else ("READY", [])


def _missing(value: float) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def decide_mtf_signal(
    entry_score: float,
    h4_score: float,
    h1_score: float,
    m15_score: float,
    tf_name: str,
    threshold: float,
) -> tuple[str, str]:
    """Return LONG/SHORT/NONE using the canonical closed-bar MTF rule."""
    if _missing(entry_score) or _missing(h4_score) or _missing(h1_score):
        return "NONE", "Ana zaman dilimi skorları yetersiz"

    htf_long = float(h4_score) >= 25 and float(h1_score) >= 25
    htf_short = float(h4_score) <= -25 and float(h1_score) <= -25

    if tf_name == "5 Dakika":
        m15_long_ok = not _missing(m15_score) and float(m15_score) >= 25
        m15_short_ok = not _missing(m15_score) and float(m15_score) <= -25
    else:
        m15_long_ok = True
        m15_short_ok = True

    if htf_long and m15_long_ok and float(entry_score) >= float(threshold):
        return "LONG", "4H+1H long uyumlu; giriş skoru eşiği geçti"
    if htf_short and m15_short_ok and float(entry_score) <= -float(threshold):
        return "SHORT", "4H+1H short uyumlu; giriş skoru eşiği geçti"

    return "NONE", "MTF filtre veya giriş skoru uygun değil"
