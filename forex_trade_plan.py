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
import math

import pandas as pd
from forex_lot_calculator import floor_lot

STATS_PATH = Path(__file__).resolve().parent / "data" / "ml" / "trade_plan_stats.json"
STOP_ATR = 1.5
MIN_LOT = 0.01
HIGH_COST_SHARE = 0.15  # forex_costs.HIGH_COST_SHARE
NEWS_BEFORE_MINUTES = 60
NEWS_AFTER_MINUTES = 15
HIGH_RISK_SHARE = 0.02  # hesabın %2'sinden büyük kayıp riski uyarısı
PARTIAL_AT_R = 1.0  # isteğe bağlı parçalı çıkış: 1R'de yarısı (research/arda_ideas H7)


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
    return {"ratio": nearest, "exact": math.isclose(nearest, ratio, abs_tol=1e-8, rel_tol=0), **row}


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


def partial_exit(entry: float, direction: int, stop_pips: float, target_pips: float, pip: float,
                 lot: float, cost_pips: float, pip_value_per_lot: float) -> Optional[dict]:
    """İsteğe bağlı: fiyat 1R gidince yarısını kapat, stopu girişe çek.

    Hedef 1R'den uzak ve lot ikiye bölünebiliyorsa anlamlıdır. H4 işlemlerinde (28 parite,
    2008–2024) kazanan oranını %34'ten %49'a çıkardı ve en kötü ayı yumuşattı, ama ortalama
    sonucu biraz düşürdü; kâr artırmaz.
    """
    if target_pips <= PARTIAL_AT_R * stop_pips or lot < 2 * MIN_LOT:
        return None
    close_lot = floor_lot(lot / 2)
    return {
        "price": entry + direction * PARTIAL_AT_R * stop_pips * pip,
        "close_lot": close_lot,
        "keep_lot": round(lot - close_lot, 2),
        "locked_usd": close_lot * (PARTIAL_AT_R * stop_pips - cost_pips) * pip_value_per_lot,
    }


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
    side: Optional[str] = None,
) -> dict:
    """Yön, giriş/stop/hedef/lot ve bu planın ölçülmüş geçmiş sonucu.

    side: ML'nin görüşü yokken kullanılacak yön (Karar kutusu). ML varsa yön her zaman ML'den gelir;
    geçmiş sonuç yalnız ML yönü için ölçüldüğünden dışarıdan verilen yönde measured boş kalır.
    """
    has_ml = (probability_up is not None and math.isfinite(probability_up) and 0 < probability_up < 1
              and probability_up != 0.5)
    if not has_ml and side not in ("LONG", "SHORT"):
        return {"status": "no_side", "action": "BEKLE", "reason": "ML yön modelinin bu parite için şu an görüşü yok."}
    values = (price, atr, pip, pip_value_per_lot, loss_usd, target_usd, account_usd, cost_pips)
    if any(v is None or not math.isfinite(v) for v in values) or min(values[:-1]) <= 0 or cost_pips < 0:
        return {"status": "no_data", "action": "BEKLE", "reason": "Fiyat, ATR veya hesap ayarı eksik; plan hesaplanamadı."}

    if has_ml:
        side = "LONG" if probability_up > 0.5 else "SHORT"
    direction = 1 if side == "LONG" else -1
    side_probability = (probability_up if side == "LONG" else 1 - probability_up) if has_ml else None
    stop_pips = STOP_ATR * atr / pip
    lot = floor_lot(min(loss_usd, account_usd) / ((stop_pips + cost_pips) * pip_value_per_lot))
    if lot < MIN_LOT:
        warning = f"Bütçe en küçük lota ({MIN_LOT:.2f}) yetmiyor; kayıp limitini aşmadan bu plan açılamaz."
        return {"status": "no_data", "action": "BEKLE", "reason": warning, "warnings": [warning], "lot": lot}
    target_pips = target_usd / (lot * pip_value_per_lot) + cost_pips
    ratio = target_pips / stop_pips
    measured = measured_outcome(stats, pair, ratio) if has_ml else None
    expected_usd = measured["ml"]["net_r"] * stop_pips * lot * pip_value_per_lot if measured else None

    warnings = []
    if account_usd > 0 and loss_usd / account_usd > HIGH_RISK_SHARE:
        warnings.append(f"Bu işlemde kayıp riski hesabın %{loss_usd / account_usd * 100:.0f}'i. "
                        f"Üst üste 5 stop hesabın %{min(100, loss_usd * 5 / account_usd * 100):.0f}'ini götürür.")
    if measured and not measured["exact"]:
        warnings.append(f"Hedef/stop oranı {ratio:.2f}; geçmiş sonuç en yakın ölçülen oran {measured['ratio']:g} için.")

    evidence_blockers = []
    if measured is None:
        evidence_blockers.append("Bu hedef/stop planı için ölçülmüş giriş başarısı yok.")
    elif not measured["exact"]:
        evidence_blockers.append("Bu hedef/stop oranı doğrudan test edilmedi; yakın oranın sonucu giriş izni vermez.")
    elif measured["ml"]["net_r"] <= 0:
        evidence_blockers.append("Bu plan geçmişte maliyet sonrası zarar etti; yön görüşü giriş sinyali sayılmaz.")
    elif not measured["ml"].get("validated", False):
        evidence_blockers.append("Pozitif ortalama için bağımsız giriş doğrulaması yok; geçmiş ortalama tek başına yeterli değil.")
    all_blockers = list(blockers) + evidence_blockers
    actual_loss = (stop_pips + cost_pips) * lot * pip_value_per_lot
    return {
        "status": "ready",
        "action": "BEKLE" if all_blockers else side,
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
        "planned_loss_usd": actual_loss,
        "target_usd": target_usd,
        "breakeven_win_rate": actual_loss / (actual_loss + target_usd),
        "partial_exit": partial_exit(price, direction, stop_pips, target_pips, pip, lot, cost_pips, pip_value_per_lot),
        "blockers": all_blockers,
        "warnings": warnings,
    }
