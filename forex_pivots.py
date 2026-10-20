"""Klasik günlük pivot seviyeleri: P, R1, R2, S1, S2.

Bir günün seviyeleri, o günden önceki son hafta içi günlük mumdan hesaplanır
(Pazartesi Cuma'nın mumunu kullanır). Saf fonksiyonlar; uygulama ve
research_pivot_stops.py aynı hesabı kullanır.
"""

from __future__ import annotations

import pandas as pd

LEVEL_NAMES = ("S2", "S1", "P", "R1", "R2")


def _day_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Mumun kendi takvim günü; saat dilimi varsa duvar saati korunur (Yahoo Londra, Dukascopy UTC)."""
    index = pd.DatetimeIndex(index)
    return (index.tz_localize(None) if index.tz is not None else index).normalize()


def pivot_levels(daily: pd.DataFrame) -> pd.DataFrame:
    """Hafta içi her günlük mumun seviyeleri; index = mumun günü, `range` = o günün High - Low'u."""
    bars = daily.set_axis(_day_index(daily.index))
    bars = bars[bars.index.dayofweek < 5]
    high, low, close = bars["High"].astype(float), bars["Low"].astype(float), bars["Close"].astype(float)
    pivot = (high + low + close) / 3
    span = high - low
    return pd.DataFrame({
        "S2": pivot - span, "S1": 2 * pivot - high, "P": pivot, "R1": 2 * pivot - low, "R2": pivot + span,
        "range": span,
    })


def levels_for_day(daily: pd.DataFrame, day) -> dict | None:
    """`day` için geçerli seviyeler: `day`den önceki son hafta içi mumdan; yoksa None."""
    levels = pivot_levels(daily)
    day = pd.Timestamp(day)
    day = (day.tz_localize(None) if day.tzinfo is not None else day).normalize()
    earlier = levels[levels.index < day]
    if earlier.empty:
        return None
    row = earlier.iloc[-1]
    return {"source_day": earlier.index[-1], **{name: float(row[name]) for name in (*LEVEL_NAMES, "range")}}
