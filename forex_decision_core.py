"""Pure decision rules shared by live analysis and backtests.

This module intentionally has no Streamlit, network, or market-data dependency so
the trading decision can be regression-tested without starting the web app.
"""

from __future__ import annotations

import math


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
