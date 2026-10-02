"""Bir yön sinyalinin ne kadar taze olduğunu ve ters yöne dönüş uyarılarını ölçer.

Radar puanı trendin şu anki gücünü ölçer; hareket uzadıkça 100'de kalır ve
"şimdi girmek için iyi an mı" sorusunu cevaplamaz. Bu modül o boşluğu doldurur:
yön ne zamandır var, o zamandan beri fiyat ne kadar gitti, kısa zaman
dilimlerinde ters dönüş işareti var mı.

Saf fonksiyonlar; Streamlit, ağ veya veri indirme yok. Eşikler açıklayıcı
sınırlardır, giriş kuralı değildir: research_freshness.py (2008-2026, 28 parite)
ne tazeliğin ne de dönüş uyarısının sonraki hareketi tahmin ettiğini buldu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIDE_THRESHOLD = 25.0  # 15M skorunun "bu yönde" sayıldığı sınır (decide_mtf_signal ile aynı)
FRESH_MAX_RATIO = 0.5  # sinyalden beri gidilen yol, tipik 4 saatlik hareketin yarısından azsa taze
LATE_MIN_RATIO = 1.0  # tipik 4 saatlik hareketin tamamı gidildiyse geç
STRETCH_LATE_ATR = 2.0  # fiyat EMA20'den 2 ATR'den fazla uzaklaştıysa geç
RSI_EXTREME = 70.0
WARNING_LOOKBACK_BARS = 3

FRESHNESS_TEXT = {
    "FRESH": "Taze",
    "MATURE": "Olgun",
    "LATE": "Geç",
    "NOT_ON_SIDE": "15M henüz bu yönde değil",
    "UNKNOWN": "Hesaplanamadı",
}


def _sign(side: str) -> int:
    return {"LONG": 1, "SHORT": -1}.get(str(side).upper(), 0)


def signal_freshness(
    bars: pd.DataFrame,
    score: pd.Series,
    side: str,
    pip: float,
    typical_move_pips: float,
    threshold: float = SIDE_THRESHOLD,
) -> dict:
    """Kapanmış 15M mumlarına göre yönün yaşını ve sinyalden beri gidilen yolu döndürür.

    `bars` en az Close, EMA20 ve ATR14 içerir; `score` aynı mumların yön skorudur.
    Yönün başladığı an, skorun yön eşiğini en son geçtiği mumdur; gidilen yol o
    mumun kapanışından son kapanışa kadardır.
    """
    sign = _sign(side)
    result = {
        "state": "UNKNOWN", "start_time": None, "bars": 0, "started_before_data": False,
        "moved_pips": np.nan, "moved_ratio": np.nan, "stretch_atr": np.nan, "typical_move_pips": typical_move_pips,
    }
    score = score.reindex(bars.index).dropna() if not bars.empty else score.iloc[0:0]
    if sign == 0 or score.empty or pip <= 0:
        return result
    on_side = (score * sign) >= float(threshold)
    if not bool(on_side.iloc[-1]):
        result["state"] = "NOT_ON_SIDE"
        return result

    off_positions = np.flatnonzero(~on_side.to_numpy())
    start_pos = int(off_positions[-1]) + 1 if off_positions.size else 0
    start_time = score.index[start_pos]
    close = bars["Close"].astype(float)
    last = bars.iloc[-1]
    moved_pips = sign * (float(close.iloc[-1]) - float(close.loc[start_time])) / float(pip)
    atr = float(last["ATR14"])
    stretch_atr = sign * (float(last["Close"]) - float(last["EMA20"])) / atr if atr > 0 else np.nan
    moved_ratio = moved_pips / float(typical_move_pips) if pd.notna(typical_move_pips) and typical_move_pips > 0 else np.nan

    if (pd.notna(moved_ratio) and moved_ratio >= LATE_MIN_RATIO) or (pd.notna(stretch_atr) and stretch_atr >= STRETCH_LATE_ATR):
        state = "LATE"
    elif pd.notna(moved_ratio) and moved_ratio >= FRESH_MAX_RATIO:
        state = "MATURE"
    else:
        state = "FRESH"
    result.update(
        state=state, start_time=start_time, bars=len(score) - start_pos, started_before_data=start_pos == 0,
        moved_pips=moved_pips, moved_ratio=moved_ratio, stretch_atr=stretch_atr,
    )
    return result


def reversal_warnings(frame: pd.DataFrame, side: str, lookback: int = WARNING_LOOKBACK_BARS) -> list[str]:
    """Kapanmış mumlarda, verilen yönün tersine işaret eden uyarılar.

    `frame`, forex_analysis.market_structure_frame çıktısıdır; isteğe bağlı
    "Score" sütunu varsa o zaman diliminin skorunun ters tarafa geçmesi de uyarıdır.
    Uyumsuzluk ve formasyonlar son `lookback` mumda, momentum ve RSI son mumda aranır.
    """
    sign = _sign(side)
    if sign == 0 or frame is None or frame.empty:
        return []
    against = "BEARISH" if sign > 0 else "BULLISH"
    recent = frame.tail(max(int(lookback), 1))
    last = frame.iloc[-1]

    def seen(column: str, value: str) -> bool:
        return column in recent.columns and bool((recent[column].astype(str) == value).any())

    warnings = []
    if seen("RSIDivergence", against):
        warnings.append("RSI uyumsuzluğu")
    if seen("MACDDivergence", against):
        warnings.append("MACD uyumsuzluğu")
    if seen("BBPattern", "M_CONFIRMED" if sign > 0 else "W_CONFIRMED"):
        warnings.append("M tepe formasyonu" if sign > 0 else "W dip formasyonu")
    if str(last.get("MACDMomentumState", "")) == ("BULLISH_WEAKENING" if sign > 0 else "BEARISH_WEAKENING"):
        warnings.append("MACD momentumu zayıflıyor")
    rsi = last.get("RSI14", np.nan)
    if pd.notna(rsi) and sign * (float(rsi) - 50.0) >= RSI_EXTREME - 50.0:
        warnings.append(f"RSI {'aşırı alım' if sign > 0 else 'aşırı satım'} ({float(rsi):.0f})")
    score = last.get("Score", np.nan)
    if pd.notna(score) and sign * float(score) <= -SIDE_THRESHOLD:
        warnings.append(f"skor ters yöne döndü ({float(score):+.0f})")
    return warnings


def reversal_level(warnings_15m: list[str], warnings_5m: list[str]) -> str:
    """STRONG: 15M'de en az iki ayrı işaret ve 5M'de en az bir işaret; WEAK: herhangi bir işaret; NONE: hiç yok.

    Tek bir işaret tek başına anlamsız derecede sık çıkar: 2026 Haziran-Eylül, 5 parite,
    15M yönü açıkken 15M'de %60, 5M'de %57, 1M'de %61. STRONG aynı anların yaklaşık %16'sında
    çıkar. Bu oranlar uyarının ne kadar seyrek olduğunu ölçer, dönüşü tahmin ettiğini değil.
    """
    if len(warnings_15m) >= 2 and len(warnings_5m) >= 1:
        return "STRONG"
    if warnings_15m or warnings_5m:
        return "WEAK"
    return "NONE"
