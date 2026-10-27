"""Özet sekmesinin "Karar" kutusu: bütün görüşleri okuyup LONG, SHORT ya da hedefe göre BEKLE der.

Kullanıcı yön için BEKLE değil net bir taraf istedi. GİR ise kart ve botun ortak kararıdır (entry_decision):
hedef kontrolü (target_check: ulaşma oranı < %60, başabaş oranının altında ya da maliyet hedefin yarısından
büyükse BEKLE), işlem planının BEKLE nedenleri (ölçülmüş net sonuç ve doğrulama) ve zamanlama engelleri.
Yön sırası, ölçülen en iyi bilgiye göre:
1. ML 4 saatlik yön modeli (research/consensus: tek başına %52; teknik görüşlerle çeliştiğinde ML'nin
   yönü ≈ %53).
2. ML'nin görüşü yoksa teknik görüşlerin çoğunluğu (Özet kartı, Piyasa tara 15M+5M, Pariteler).
3. O da yoksa fiyatın günlük pivot P'ye göre konumu.

Geçmiş oranlar kartta açıkça yazar; hiçbiri maliyetten sonra artıda değildir. Haber, gün sonu spread
saati ve pahalı maliyet de girişi engeller (27 Ekim 2026'dan beri; önceden yalnız uyarıydı).

Streamlit bağımlılığı yoktur; uygulamayı açmadan test edilebilir.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

SIDE_TEXT = {1: "LONG", -1: "SHORT", 0: "yön yok"}
DIRECTION_HOURS = 4  # ML yön modelinin ufku ve consensus ölçümünün süresi

# research/consensus/results.json (2023, 14 parite), giriş anından 4 saat sonra.
# "ml_vs_technical": teknik yöne girmek %46,6 ve −0,96 pip hareket verdi; ML yönü bunun tersidir.
# Maliyet aynı anların ortalaması (−0,96 − (−3,61) = 2,65 pip), dolayısıyla net ≈ 0,96 − 2,65.
MEASURED = {
    "ml_with_technical": {"hit": 50.3, "net": -2.15, "text": "ML ve üç teknik görüş aynı yöndeyken"},
    "ml_vs_technical": {"hit": 53.4, "net": -1.69, "text": "teknik görüşler tersini derken ML'nin yönünde"},
    "ml": {"hit": 52.0, "net": -1.94, "text": "ML yönünde"},
    "technical": {"hit": 49.0, "net": -2.74, "text": "teknik görüşlerin yönünde"},
    "pivot": {"hit": None, "net": -2.3, "text": "pivot planının yönünde (28 parite, 2008–2026)"},
}


MIN_HIT = 0.60  # hedefin stoptan önce gelme oranı en az bu olmalı (kullanıcı seçimi, 26 Ekim 2026)
MAX_COST_OF_TARGET = 0.5  # maliyet hedef mesafesinin yarısını geçerse hedef fazla küçük
HOLD_HOURS = 24  # trade_plan_stats ufku: 288 5M mum (bir işlem günü)


def _hit_curve(stats: dict, pair: str, use_ml: bool) -> Optional[tuple[list[float], list[float]]]:
    """Hedef/stop oranına göre hedefin stoptan önce gelme oranı (artan oran, azalan isabet).

    ML yönünde girilince ML satırı; yön başka kaynaktan geldiyse ML ve ters yönün ortalaması
    (bütün anları kapsadığı için yönsüz bir girişin oranıdır).
    """
    entry = stats.get("pairs", {}).get(pair) or stats.get("pooled")
    if not entry:
        return None
    rows = sorted((float(r), row) for r, row in entry["ratios"].items())
    hits = [row["ml"]["target_rate"] if use_ml else (row["ml"]["target_rate"] + row["opposite"]["target_rate"]) / 2
            for _, row in rows]
    return [r for r, _ in rows], hits


def breakeven_hit(target_pips: float, stop_pips: float, cost_pips: float) -> float:
    """Hedefte net (hedef − maliyet) kazanıp stopta (stop + maliyet) kaybeden düzende başabaş ulaşma oranı."""
    return (stop_pips + cost_pips) / (target_pips + stop_pips)


def target_check(stats: dict, pair: str, use_ml: bool, target_pips: float, stop_pips: float,
                 cost_pips: float, require_breakeven: bool = True) -> dict:
    """Bu dolar hedefi bu stopla 'ihtimalli' mi: ulaşma oranı ≥ %60 ve başabaş oranının üstünde, maliyet
    hedefin yarısından az. Yüksek ulaşma oranı tek başına yetmez: 0,3:1 hedefte başabaş ≈ %77.
    require_breakeven=False yalnız botun --deney kipi içindir (27 Ekim öncesi kural).

    Ölçülen aralığın dışında oran sınır olarak verilir: daha yakın hedef en az en yakın ölçülen kadar,
    daha uzak hedef en fazla onun kadar tutar (aynı stopta hedef yaklaştıkça stoptan önce gelme artar).
    """
    curve = _hit_curve(stats, pair, use_ml)
    reasons = []
    hit, bound, ok_range = None, "≈", None
    if curve and stop_pips > 0:
        ratios, hits = curve
        ratio = target_pips / stop_pips
        hit = float(np.interp(ratio, ratios, hits))
        bound = "≥" if ratio < ratios[0] else ("≤" if ratio > ratios[-1] else "≈")
        grid = np.linspace(0.0, ratios[-1], 801)  # ölçülenden küçük oranda isabet en az ilk ölçülen kadar
        floor = breakeven_hit(grid * stop_pips, stop_pips, cost_pips) if require_breakeven else 0.0
        passing = grid[np.interp(grid, ratios, hits) >= np.maximum(MIN_HIT, floor)]
        low = cost_pips / MAX_COST_OF_TARGET
        ok_range = (max(low, float(passing[0]) * stop_pips) if passing.size else low,
                    float(passing[-1]) * stop_pips if passing.size else None)
        needed = breakeven_hit(target_pips, stop_pips, cost_pips)
        if hit < MIN_HIT:
            reasons.append(f"Hedefe ulaşma oranı %{hit * 100:.0f}; en az %{MIN_HIT * 100:.0f} istiyorsun.")
        elif require_breakeven and hit < needed:
            reasons.append(f"Hedefe ulaşma oranı %{hit * 100:.0f}; bu hedef ve stopla başabaş için "
                           f"%{needed * 100:.0f} gerekiyor, beklenen sonuç maliyet sonrası eksi.")
    else:
        reasons.append("Bu parite ve hedef için ölçülmüş ulaşma oranı yok.")
    if target_pips > 0 and cost_pips / target_pips > MAX_COST_OF_TARGET:
        reasons.append(f"Maliyet ({cost_pips:.1f} pip) hedef mesafesinin %{cost_pips / target_pips * 100:.0f}'i; "
                       f"hedef fazla küçük.")
    return {"ok": not reasons, "hit": hit, "bound": bound, "reasons": reasons,
            "ok_pips": ok_range}  # (en küçük, en büyük) hedef mesafesi, pip; en büyük None ise hiçbir hedef yetmiyor


def entry_decision(plan: dict, check: Optional[dict], blockers: list[str]) -> dict:
    """Karar kutusu ve botun tek giriş kararı: hedef kontrolü, planın BEKLE nedenleri ve zamanlama engelleri.

    Plan ölçülmüş istatistikle kurulmuş olmalı; aksi halde planın kanıt engeli "ölçülmüş başarı yok" der.
    """
    if plan.get("status") != "ready" or check is None:
        return {"ok": False, "hit": None, "reasons": [plan.get("reason", "Plan hesaplanamadı.")]}
    reasons = list(dict.fromkeys(list(check["reasons"]) + list(plan.get("blockers", [])) + list(blockers)))
    return {**check, "ok": not reasons, "reasons": reasons}


def _pips(value: float) -> str:
    return f"{value:+.1f}".replace(".", ",").replace("-", "−")


def pivot_side(price: Optional[float], levels: Optional[dict]) -> int:
    """Fiyat günlük P'nin üstündeyse LONG, altındaysa SHORT (forex_pivot_plan ile aynı yön kuralı)."""
    if price is None or not levels or levels.get("P") is None or price == levels["P"]:
        return 0
    return 1 if price > levels["P"] else -1


def final_call(sides: dict[str, int], pivot: int, probability_up: Optional[float] = None) -> dict:
    """Her zaman bir yön: ML > teknik çoğunluk > pivot. Hiçbiri yoksa side=0 (veri yok)."""
    technical = [sides["radar"], sides["firsatlar"], sides["pariteler"]]
    tech_all = technical[0] if technical[0] != 0 and all(s == technical[0] for s in technical) else 0
    tech_sum = sum(technical)
    tech_lean = 1 if tech_sum > 0 else (-1 if tech_sum < 0 else 0)

    if sides["ml"] != 0:
        side = sides["ml"]
        if tech_all == side:
            key = "ml_with_technical"
        elif tech_all == -side:
            key = "ml_vs_technical"
        else:
            key = "ml"
        probability = probability_up if side == 1 else (1 - probability_up if probability_up is not None else None)
        source = "ML 4 saatlik yön" + (f" (%{probability * 100:.0f})" if probability is not None else "")
    elif tech_lean != 0:
        side, key, source = tech_lean, "technical", "Teknik görüşlerin çoğunluğu (bu paritede ML modeli yok)"
    elif pivot != 0:
        side, key, source = pivot, "pivot", "Fiyatın günlük pivot P'ye göre yeri (başka görüş yön göstermiyor)"
    else:
        return {"side": 0, "action": None, "reason": "Fiyat ve görüşler alınamadı; veri gelince yön verilir."}

    views = [
        ("Piyasa tara 15M+5M", sides["firsatlar"]),
        ("Pariteler 4H·1H·15M·5M", sides["pariteler"]),
        ("Özet kartı", sides["radar"]),
        ("ML yön", sides["ml"]),
        ("Pivot", pivot),
    ]
    agree = sum(1 for _, s in views if s == side)
    against = sum(1 for _, s in views if s == -side)
    m = MEASURED[key]
    hit = f"kazanma %{m['hit']:.0f}, ".replace(".", ",") if m["hit"] is not None else ""
    return {
        "side": side,
        "action": SIDE_TEXT[side],
        "source": source,
        "pattern": key,
        "views": [(name, SIDE_TEXT[s], "✓" if s == side else ("✗" if s == -side else "–")) for name, s in views],
        "agree": agree,
        "against": against,
        "evidence": (f"Geçmişte {m['text']} girmek ({DIRECTION_HOURS} saat sonra): {hit}maliyetten sonra ortalama "
                     f"{_pips(m['net'])} pip."),
    }
