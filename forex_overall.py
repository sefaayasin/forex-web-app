"""Özet sekmesinin "Genel Yorum" paneli: dört yön görüşünü tek yerde okur.

Görüşler Özet sekmesindekilerle aynıdır: radar kartı, Fırsatlar (15M/5M Güçlü Alım/Satış),
Pariteler (4H·1H·15M·5M aynı yön) ve ML 4 saatlik yön modeli. research/consensus testi
(2023, 14 parite, 85.839 saatlik an) dördü aynı yönü gösterdiğinde bile o yöne girmenin
maliyetten sonra zarar ettiğini gösterdi. Bu yüzden panel LONG/SHORT emri vermez: hangi
görüşün ne dediğini, uyuşup uyuşmadıklarını ve o durumun ölçülen sonucunu söyler.

Streamlit bağımlılığı yoktur; uygulamayı açmadan test edilebilir.
"""
from __future__ import annotations

from typing import Optional

RADAR_CANDIDATE = 42.0  # forex_decision_core.classify_opportunity_readiness aday eşiği
VIEW_LABELS = {
    "radar": "Özet kartı",
    "firsatlar": "Fırsatlar (15M + 5M)",
    "pariteler": "Pariteler (4H·1H·15M·5M)",
    "ml": "ML 4 saatlik yön",
}
TIMEFRAMES = ("4 Saat", "1 Saat", "15 Dakika", "5 Dakika")
SIDE_TEXT = {1: "LONG", -1: "SHORT", 0: "yön yok"}
NEWS_WINDOW_MINUTES = 240
HIGH_COST_SHARE = 0.15  # forex_costs.HIGH_COST_SHARE

# research/consensus/results.json (2023, 14 parite): 4 saat sonra, görüşün yönünde.
MEASURED = {
    "consensus": {"hit": 50.3, "net": -2.15, "move": 0.24},
    "conflict": {"hit": 46.6, "net": -3.61, "move": -0.96},
    "radar": {"hit": 49.0, "net": -2.74, "move": -0.20},
    "ml": {"hit": 52.0, "net": -1.94, "move": 0.59},
}
SOURCE_NOTE = (
    "Ölçüm: 2023, 14 parite, 85.839 saatlik an; giriş anından 4 saat sonrası, ölçülen spread + komisyon düşülerek "
    "(research/consensus). Tahmin değil, geçmişte aynı durumun ortalama sonucu."
)


def _pips(value: float) -> str:
    return f"{value:+.1f}".replace(".", ",").replace("-", "−")


def label_side(label: str) -> int:
    text = str(label)
    return 1 if "Alım" in text else (-1 if "Satış" in text else 0)


def view_sides_now(
    tf_labels: dict[str, str],
    entry_labels: tuple[str, str],
    radar_side: str,
    radar_score: float,
    probability_up: Optional[float],
) -> dict[str, int]:
    """Her görüşün yönü (+1 LONG, -1 SHORT, 0 yok); research_consensus.view_sides ile aynı kurallar.

    tf_labels: zaman dilimi -> Bias etiketi (analyse_symbol özeti).
    entry_labels: global_bias'ın 15M ve 5M giriş etiketleri.
    """
    tf = [label_side(tf_labels.get(name, "")) for name in TIMEFRAMES]
    pariteler = tf[0] if tf[0] != 0 and all(side == tf[0] for side in tf) else 0
    strong_long = "Güçlü Alım Yönlü" in entry_labels
    strong_short = "Güçlü Satış Yönlü" in entry_labels
    firsatlar = 1 if strong_long and not strong_short else (-1 if strong_short and not strong_long else 0)
    radar_dir = {"LONG": 1, "SHORT": -1}.get(str(radar_side).upper(), 0)
    radar = radar_dir if float(radar_score or 0) >= RADAR_CANDIDATE else 0
    if probability_up is None or probability_up != probability_up or probability_up == 0.5:
        ml = 0
    else:
        ml = 1 if probability_up > 0.5 else -1
    return {"radar": radar, "firsatlar": firsatlar, "pariteler": pariteler, "ml": ml}


def overall_reading(
    sides: dict[str, int],
    news_minutes: Optional[int] = None,
    news_title: str = "",
    cost_share: Optional[float] = None,
    volatility_level: Optional[str] = None,
) -> dict:
    """Görüşlerin durumu, ölçülen sonucu, karar ve 'yine de girersen' uyarıları."""
    technical = [sides["radar"], sides["firsatlar"], sides["pariteler"]]
    tech_side = technical[0] if technical[0] != 0 and all(s == technical[0] for s in technical) else 0
    longs = sum(1 for s in sides.values() if s == 1)
    shorts = sum(1 for s in sides.values() if s == -1)

    if tech_side != 0 and sides["ml"] == tech_side:
        pattern, lean = "consensus", tech_side
        m = MEASURED["consensus"]
        headline = f"Dört görüş de {SIDE_TEXT[lean]}"
        evidence = (
            f"Dördü aynı yöndeyken o yöne girmek: kazanma %{m['hit']:.0f}, maliyetten sonra ortalama "
            f"{_pips(m['net'])} pip. Hepsinin aynı yönü göstermesi avantaj değil; maliyetten önce sonuç yazı-tura."
        )
    elif tech_side != 0 and sides["ml"] == -tech_side:
        pattern, lean = "conflict", tech_side
        m = MEASURED["conflict"]
        headline = f"Teknik görüşler {SIDE_TEXT[tech_side]}, ML {SIDE_TEXT[-tech_side]} — çelişki"
        evidence = (
            f"Bu durumda teknik yöne girmek en kötü sonucu verdi: kazanma %{m['hit']:.0f}, maliyetten sonra "
            f"{_pips(m['net'])} pip. ML'nin yönü de maliyeti karşılamadı (ML tek başına %{MEASURED['ml']['hit']:.0f}, "
            f"{_pips(MEASURED['ml']['net'])} pip)."
        )
    elif longs == 0 and shorts == 0:
        pattern, lean = "none", 0
        headline = "Belirgin bir yön yok"
        evidence = "Hiçbir görüş yön göstermiyor."
    else:
        pattern = "mixed"
        lean = 1 if longs > shorts else (-1 if shorts > longs else 0)
        m = MEASURED["radar"]
        if longs and shorts:
            headline = f"Görüşler çelişiyor ({longs} LONG · {shorts} SHORT)"
        else:
            headline = f"{max(longs, shorts)}/4 görüş {SIDE_TEXT[lean]}, kalanı yön göstermiyor"
        evidence = (
            f"Dördü birden aynı yönde değil. Ölçülen hiçbir durumda avantaj çıkmadı; örneğin Özet kartının yönüne "
            f"girmek tek başına: kazanma %{m['hit']:.0f}, maliyetten sonra {_pips(m['net'])} pip."
        )

    warnings = []
    if news_minutes is not None and 0 <= news_minutes <= NEWS_WINDOW_MINUTES:
        hours, minutes = divmod(int(news_minutes), 60)
        when = f"{hours} sa {minutes} dk" if hours else f"{minutes} dk"
        warnings.append(
            f"📰 {when} sonra yüksek etkili haber{(': ' + news_title) if news_title else ''}. Haberde yönü sürpriz "
            "belirler, spread birkaç saniyeliğine 5–25 katına çıkabilir; haberden önce pozisyon açma."
        )
    if cost_share is not None and cost_share > HIGH_COST_SHARE:
        warnings.append(
            f"💸 Maliyet planlanan kaybın %{cost_share * 100:.0f}'i. Bu oran yükseldikçe işlemin kazanması zorlaşır; "
            "daha düşük spreadli saat veya parite seç."
        )
    if volatility_level == "high":
        warnings.append(
            "🌪️ Önümüzdeki 72 saatte oynaklık yüksek bekleniyor. ATR stop bunu hesaba katar (stop genişler, lot "
            "küçülür); sabit pip stop kullanıyorsan kolay tetiklenir."
        )

    return {
        "pattern": pattern,
        "lean": lean,
        "lean_text": SIDE_TEXT[lean],
        "longs": longs,
        "shorts": shorts,
        "headline": headline,
        "evidence": evidence,
        "verdict": "Teknik görüşlere bakarak girme",
        "verdict_reason": (
            "Teknik görüşler, dördü bir araya gelince bile, maliyetten sonra kazandırmadı. Yön, stop, hedef ve "
            "planın geçmiş sonucu için aşağıdaki İşlem Planı'na bak."
        ),
        "views": [(VIEW_LABELS[name], SIDE_TEXT[sides[name]]) for name in VIEW_LABELS],
        "warnings": warnings,
    }
