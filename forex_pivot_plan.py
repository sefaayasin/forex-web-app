"""Ders kitabı pivot trader planı: yön, geri çekilmede giriş, stop ve hedef.

Kural (research/pivot_plan/protocol.json):
- Fiyat P'nin üstündeyse LONG, altındaysa SHORT.
- Giriş: fiyatın gerisindeki en yakın seviye (LONG'da alttaki, SHORT'ta üstteki); geri çekilmede limit emir.
- Hedef: fiyatın önündeki en yakın seviye. R3'ün üstünde / S3'ün altında seviye kalmadıysa BEKLE.
- Stop: giriş seviyesinin 1 ATR ötesi. Hedef stoptan yakınsa BEKLE.

Saf fonksiyonlar; uygulamanın Özet kutusu ve research_pivot_plan.py aynı kuralı kullanır.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from forex_pivots import LEVEL_NAMES

LADDER_NAMES = ("S3", "S2", "S1", "P", "R1", "R2", "R3")
STOP_ATR = 1.0

# research/pivot_plan/results.json: 2008-2026, 28 parite, günde 6 karar anı, dolan planlar.
MEASURED_NOTE = (
    "Bu kural 2008–2026'da 28 paritede denendi: dolan 100 planın 14'ü hedefe, 72'si stopa gitti; "
    "maliyetten sonra işlem başına ≈ −2,3 pip. Rastgele çizgilerle aynı. Plan, nereden girip çıkacağını "
    "önceden belirlemek içindir, kazanç beklentisi değildir. (research/pivot_plan)"
)


def atr14(bars: pd.DataFrame) -> np.ndarray:
    """15M ATR14: gerçek aralığın basit 14 mumluk ortalaması (planın stop mesafesi)."""
    prev_close = bars["Close"].shift(1)
    true_range = pd.concat([bars["High"] - bars["Low"], (bars["High"] - prev_close).abs(),
                            (bars["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(14).mean().to_numpy(float)


def ladder(levels: dict) -> dict[str, float]:
    """S3..R3; R3 = H + 2(P - L) = 2·R1 - S1 ve S3 = L - 2(H - P) = 2·S1 - R1."""
    out = {name: float(levels[name]) for name in LEVEL_NAMES}
    out["R3"] = 2 * out["R1"] - out["S1"]
    out["S3"] = 2 * out["S1"] - out["R1"]
    return {name: out[name] for name in LADDER_NAMES}


def pivot_plan(price: float, levels: dict, atr: float) -> dict:
    """{"side": LONG / SHORT / WAIT, "entry", "stop", "target", "entry_name", "target_name", "reason"}."""
    plan = {"side": "WAIT", "entry": math.nan, "stop": math.nan, "target": math.nan,
            "entry_name": "", "target_name": "", "reason": ""}
    if not levels or not (atr > 0) or not math.isfinite(price):
        plan["reason"] = "Seviye ya da ATR hesaplanamadı."
        return plan
    steps = ladder(levels)
    if price == steps["P"]:
        plan["reason"] = "Fiyat tam P'de; yön yok."
        return plan
    sign = 1 if price > steps["P"] else -1
    behind = [(name, value) for name, value in steps.items() if sign * (price - value) > 0]
    ahead = [(name, value) for name, value in steps.items() if sign * (value - price) > 0]
    entry_name, entry = max(behind, key=lambda item: sign * item[1])
    plan.update(side="LONG" if sign > 0 else "SHORT", entry=entry, entry_name=entry_name, stop=entry - sign * STOP_ATR * atr)
    if not ahead:
        plan.update(side="WAIT", reason=f"Fiyat {'R3 üstünde' if sign > 0 else 'S3 altında'}; önünde hedef seviye kalmadı.")
        return plan
    target_name, target = min(ahead, key=lambda item: sign * item[1])
    plan.update(target=target, target_name=target_name)
    if abs(target - entry) < abs(entry - plan["stop"]):
        plan.update(side="WAIT", reason="Hedef, stoptan daha yakın; ödül riske değmiyor.")
    return plan
