"""Pivot yorumcusu: açıklanabilir koşullar; uyum puanı kazanma olasılığı değildir.

Giriş kararı güncel koşullardan üretilir. Eski pivot / ML testleri bu
birleşimin başarı ölçümü olarak kullanılamaz.
"""
from __future__ import annotations

import math
import pandas as pd

from forex_costs import cost_share_of_risk
from forex_trade_plan import trading_blockers

MIN_ALIGNMENT = 70
ENTRY_ZONE_ATR = 0.25
STRATEGY_ID = "pivot_commentator_v2"


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def recent_bar(stamp, now: pd.Timestamp, minutes: int, *, closed: bool = True) -> bool:
    if stamp is None or pd.isna(stamp):
        return False
    stamp = pd.Timestamp(stamp)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    age = (now - stamp).total_seconds() / 60
    return (minutes if closed else 0) <= age <= (2 * minutes if closed else minutes) + 10


def closed_bars(bars: pd.DataFrame, now: pd.Timestamp, minutes: int) -> pd.DataFrame:
    """Tamamlanmamış mumu zamana göre ayır; son kapalı mumu körlemesine silme."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    out = bars.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[out.index + pd.Timedelta(minutes=minutes) <= now]


def calendar_available(events, now: pd.Timestamp) -> bool:
    if not isinstance(events, pd.DataFrame) or events.empty:
        return False
    if not {"time", "currency", "impact"}.issubset(events.columns):
        return False
    times = pd.to_datetime(events["time"], utc=True, errors="coerce")
    # Eski / yalnız geçmiş günleri içeren takvimde "haber yok" diyemeyiz.
    return bool((times >= now).any() and (times < now.normalize() + pd.Timedelta(days=2)).any())


def advise(snapshot: dict, *, now: pd.Timestamp, news_known: bool,
           news_minutes=None, news_title: str = "", cost_pips=None) -> dict:
    """Tek bir kaynağın aynı taramasından gelen gözlemleri yorumlar."""
    side = snapshot.get("side", "WAIT")
    valid = side in {"LONG", "SHORT"}
    sign = 1 if side == "LONG" else -1
    checks = []

    def check(name, ok, detail, *, required=True):
        checks.append({"condition": name, "ok": bool(ok), "detail": detail, "required": required})

    scores = snapshot.get("scores", {})
    for tf in ("4H", "1H", "15M", "5M"):
        score = scores.get(tf)
        ok = valid and finite(score) and sign * float(score) >= 25
        check(f"{tf} yön uyumu", ok,
              f"Skor {float(score):+.0f}; {side} için eşik {sign * 25:+d}." if finite(score) else "Veri eksik.",
              required=tf != "5M")
    prediction = snapshot.get("ml", {})
    p = prediction.get("probability_up")
    ml_ready = (prediction.get("status") == "ready" and finite(p) and 0 <= float(p) <= 1
                and recent_bar(prediction.get("as_of"), now, 60))
    side_p = (float(p) if sign == 1 else 1 - float(p)) if ml_ready else None
    check("ML yön desteği", valid and side_p is not None and side_p >= .55,
          f"Modelin {side} yön puanı %{side_p * 100:.0f}; hedefe ulaşma ihtimali değildir."
          if side_p is not None else "Güncel ML tahmini yok.", required=False)
    distance = snapshot.get("distance_atr")
    check("Giriş bölgesinde", valid and finite(distance) and 0 <= float(distance) <= ENTRY_ZONE_ATR,
          f"Girişe {float(distance):.2f} ATR; sınır {ENTRY_ZONE_ATR:.2f}." if finite(distance) else "Giriş hesaplanamadı.")
    check("Kapanmış mumda tepki", valid and snapshot.get("response_side") == side,
          "15M tepki yönü: " + str(snapshot.get("response_side", "belirsiz")))
    warnings = snapshot.get("reversal_warnings")
    check("Ters dönüş uyarısı yok", valid and warnings is not None and not warnings,
          "; ".join(warnings) if warnings else ("Uyarı yok." if warnings is not None else "Dönüş verisi eksik."))
    alignment = round(100 * sum(c["ok"] for c in checks) / len(checks))
    blockers = []
    if not valid:
        blockers.append(snapshot.get("reason") or "Pivot yönü ve işlem seviyeleri oluşmadı.")
    if snapshot.get("error"):
        blockers.append(snapshot["error"])
    if not snapshot.get("fresh", False):
        blockers.append("Fiyat veya zaman dilimi verisi eski/eksik; güncel karar verilemiyor.")
    if not snapshot.get("pivot_fresh", False):
        blockers.append("Pivotun kaynak günlük mumu güncel değil veya doğrulanamadı.")
    if not news_known:
        blockers.append("Güncel ve önem dereceli haber takvimi doğrulanamadı.")
    pip = snapshot.get("pip")
    pivot_entry = snapshot.get("entry")
    entry = snapshot.get("price")
    stop, target = (snapshot.get(k) for k in ("stop", "target"))
    geometry = (valid and all(finite(v) for v in (entry, pivot_entry, stop, target, pip)) and float(pip) > 0
                and sign * (float(entry) - float(stop)) > 0 and sign * (float(target) - float(entry)) > 0)
    stop_pips = abs(float(entry) - float(stop)) / float(pip) if geometry else None
    target_pips = abs(float(target) - float(entry)) / float(pip) if geometry else None
    cost_known = finite(cost_pips) and float(cost_pips) >= 0
    cost = float(cost_pips) if cost_known else None
    share = cost_share_of_risk(cost, stop_pips) if cost_known and geometry else None
    blockers.extend(trading_blockers(now, news_minutes, news_title, share))
    if not cost_known:
        blockers.append("Bu parite için spread + komisyon tahmini yok.")
    if not geometry:
        blockers.append("Güncel fiyat, stop ve hedef aynı yönde geçerli bir giriş planı oluşturmuyor.")
    net_rr = (target_pips - cost) / (stop_pips + cost) if geometry and cost_known else None
    if net_rr is None or net_rr < 1:
        blockers.append("Maliyet sonrası hedef/risk oranı en az 1 olmalı.")
    blockers.extend(c["condition"] + ": " + c["detail"] for c in checks if c["required"] and not c["ok"])
    if alignment < MIN_ALIGNMENT:
        blockers.append(f"Teknik uyum {alignment}/100; aday eşiği {MIN_ALIGNMENT}/100.")
    if valid and side_p is not None and side_p <= .40:
        blockers.append(f"ML güçlü biçimde ters yönde: {side} yön desteği %{side_p * 100:.0f}.")
    technical_ready = not blockers
    action = f"{side} GİR" if technical_ready else "BEKLE"
    passed = [c["condition"] for c in checks if c["ok"]]
    explanation = (
        f"4H ve 1H ana yönü {side}; 15M aynı yönü destekliyor ve kapanmış mumda tepki var. "
        "Fiyat pivot giriş bölgesinde, kısa vadede ters dönüş uyarısı yok. "
        f"Haber ve saat kontrolü uygun; maliyet sonrası hedef/risk {net_rr:.2f}. "
        + (f"ML {side} yönünü destekliyor. " if side_p is not None and side_p >= .55 else
           "ML ek teyit vermiyor; karar teknik yapı ve giriş koşullarına dayanıyor. ")
        + f"Bu nedenle kararım: {action}."
    ) if technical_ready else (
        f"{side + ' yönü var, ancak' if valid else 'Şu an'} giriş koşulları tamamlanmadı. "
        + blockers[0]
    )
    return {
        **snapshot, "entry": entry, "pivot_entry": pivot_entry,
        "action": action, "alignment": alignment, "checks": checks,
        "technical_ready": technical_ready, "blockers": blockers,
        "probability": None, "net_rr": net_rr, "cost_pips": cost,
        "supporting_reasons": passed, "comment": explanation,
        "performance_note": "Bu birleşimin başarı oranı ölçülmedi; uyum puanı kazanma ihtimali değildir.",
        "next_step": (f"{action}: güncel fiyatı brokerda kontrol et; gösterilen stop ve hedefle planı uygula. "
                      "Fiyat giriş bölgesinden çıkarsa yeniden değerlendir."
                      if technical_ready else blockers[0]),
    }
