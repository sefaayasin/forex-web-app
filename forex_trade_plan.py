"""Özet sekmesinin "İşlem Planı" kartı: yön, stop, hedef, lot ve bu planın ölçülmüş geçmiş sonucu.

Yön, ML 4 saatlik yön modelinden gelir (elimizdeki tek ölçülmüş yön bilgisi; yazı-turadan 2-3 puan iyi).
Stop teknik stoptur (kapanmış 15M mumlarında ATR14 x 1,5). Hedef, kullanıcının dolar hedefinden hesaplanır.
"Geçmişte bu plan" satırları research_trade_plan.py'nin 2023-2026 ölçümüdür (data/ml/trade_plan_stats.json):
aynı paritede, aynı hedef/stop oranıyla, ML yönüne her saat girilseydi ne olurdu.

Şart kontrolü (haber, gün sonu spread saati, maliyet payı, piyasa kapalı) kaybı küçültmek içindir;
kazanma oranını artırdığı ölçülmedi.

Streamlit bağımlılığı yoktur; uygulamayı açmadan test edilebilir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

STATS_PATH = Path(__file__).resolve().parent / "data" / "ml" / "trade_plan_stats.json"
STOP_ATR = 1.5
MIN_LOT = 0.01
HIGH_COST_SHARE = 0.15  # forex_costs.HIGH_COST_SHARE
NEWS_BEFORE_MINUTES = 60
NEWS_AFTER_MINUTES = 15
HIGH_RISK_SHARE = 0.02  # hesabın %2'sinden büyük kayıp riski uyarısı


def load_plan_stats(path: Path = STATS_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def measured_outcome(stats: dict, pair: str, ratio: float) -> Optional[dict]:
    """En yakın ölçülmüş hedef/stop oranındaki sonuç (ML yönü ve ters yön)."""
    entry = stats.get("pairs", {}).get(pair) or stats.get("pooled")
    if not entry:
        return None
    ratios = [float(r) for r in entry["ratios"]]
    nearest = min(ratios, key=lambda r: abs(r - ratio))
    row = entry["ratios"][str(nearest)] if str(nearest) in entry["ratios"] else entry["ratios"][f"{nearest:g}"]
    return {"ratio": nearest, "exact": abs(nearest - ratio) <= 0.15, **row}


def trading_blockers(
    now: pd.Timestamp,
    news_minutes: Optional[int] = None,
    news_title: str = "",
    cost_share: Optional[float] = None,
) -> list[str]:
    """İşlem açmayı bekletmesi gereken durumlar (ölçülmüş maliyet ve spread davranışına göre)."""
    ny = now.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    blockers = []
    closed = ny.weekday() == 5 or (ny.weekday() == 6 and minutes < 17 * 60) or (ny.weekday() == 4 and minutes >= 16 * 60 + 45)
    if closed:
        blockers.append("Piyasa kapalı veya kapanmak üzere (hafta sonu).")
    elif 16 * 60 + 45 <= minutes < 18 * 60:
        blockers.append("Gün sonu (New York 17:00) spread saati: spread 3–7 katına çıkıyor; 18:00 NY'den sonra bak.")
    if news_minutes is not None and -NEWS_AFTER_MINUTES <= news_minutes <= NEWS_BEFORE_MINUTES:
        when = f"{news_minutes} dk sonra" if news_minutes >= 0 else f"{-news_minutes} dk önce"
        blockers.append(f"Yüksek etkili haber {when}{': ' + news_title if news_title else ''}. Haberde yönü sürpriz belirler.")
    if cost_share is not None and cost_share > HIGH_COST_SHARE:
        blockers.append(f"Maliyet planlanan kaybın %{cost_share * 100:.0f}'i (sınır %15); bu parite/saat bu stop için pahalı.")
    return blockers


def build_trade_plan(
    pair: str,
    price: Optional[float],
    pip: float,
    atr: Optional[float],
    probability_up: Optional[float],
    target_usd: float,
    loss_usd: float,
    account_usd: float,
    pip_value_per_lot: float,
    cost_pips: float,
    stats: dict,
    blockers: list[str],
) -> dict:
    """Yön, giriş/stop/hedef/lot ve bu planın ölçülmüş geçmiş sonucu."""
    if probability_up is None or probability_up != probability_up or probability_up == 0.5:
        return {"status": "no_side", "action": "BEKLE", "reason": "ML yön modelinin bu parite için şu an görüşü yok."}
    if price is None or atr is None or not atr > 0 or pip_value_per_lot <= 0 or loss_usd <= 0 or target_usd <= 0:
        return {"status": "no_data", "action": "BEKLE", "reason": "Fiyat, ATR veya hesap ayarı eksik; plan hesaplanamadı."}

    side = "LONG" if probability_up > 0.5 else "SHORT"
    direction = 1 if side == "LONG" else -1
    side_probability = probability_up if side == "LONG" else 1 - probability_up
    stop_pips = STOP_ATR * atr / pip
    lot = loss_usd / ((stop_pips + cost_pips) * pip_value_per_lot)
    target_pips = target_usd / (lot * pip_value_per_lot) + cost_pips
    ratio = target_pips / stop_pips
    measured = measured_outcome(stats, pair, ratio)
    expected_usd = measured["ml"]["net_r"] * stop_pips * lot * pip_value_per_lot if measured else None

    warnings = []
    if lot < MIN_LOT:
        warnings.append(f"Hesaplanan lot {lot:.3f}; çoğu broker'da en küçük lot {MIN_LOT}. Kayıp limitini büyütmeden bu plan açılamaz.")
    if account_usd > 0 and loss_usd / account_usd > HIGH_RISK_SHARE:
        warnings.append(f"Bu işlemde kayıp riski hesabın %{loss_usd / account_usd * 100:.0f}'i. "
                        f"Üst üste 5 stop hesabın %{min(100, loss_usd * 5 / account_usd * 100):.0f}'ini götürür.")
    if measured and not measured["exact"]:
        warnings.append(f"Hedef/stop oranı {ratio:.2f}; geçmiş sonuç en yakın ölçülen oran {measured['ratio']:g} için.")

    return {
        "status": "ready",
        "action": "BEKLE" if blockers else side,
        "side": side,
        "side_probability": side_probability,
        "entry": price,
        "stop": price - direction * stop_pips * pip,
        "target": price + direction * target_pips * pip,
        "stop_pips": stop_pips,
        "target_pips": target_pips,
        "lot": lot,
        "ratio": ratio,
        "measured": measured,
        "expected_usd": expected_usd,
        "blockers": blockers,
        "warnings": warnings,
    }
