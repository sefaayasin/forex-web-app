"""Presentation and counting helpers; these never grant trading permission."""


def engine_evidence_summary(engine, qualities):
    if engine not in {"TREND", "RANGE"}:
        return {"label": "Piyasa kararsız", "text": "Piyasa trend veya yatay olarak netleşmedi; şu anda aktif strateji seçilemiyor. Bu, testin yapılmadığı anlamına gelmez.", "css": "warn-box"}
    quality = qualities.get(engine) or {}
    name = "Trend" if engine == "TREND" else "Yatay piyasa"
    if not quality:
        return {"label": "Test yapılmalı", "text": f"{name} stratejisinin bu parite ve ayarlar için kayıtlı testi yok. Planı Kontrol Et ile hesapla.", "css": "warn-box"}
    edge = quality.get("edge") or {}
    count = quality.get("trade_count")
    if count is not None and count < quality.get("required_trades", 60):
        return {"label": "Örnek yetersiz", "text": f"{name} testinde {count} işlem oluştu; değerlendirme için en az {quality.get('required_trades', 60)} gerekiyor. Bu sonuç tek başına stratejinin iyi veya kötü olduğunu göstermez.", "css": "warn-box"}
    if quality.get("label") in {"İyi", "Orta"} and edge.get("label") == "DOĞRULANDI":
        return {"label": "Onaylandı", "text": f"{name} stratejisinin geçmiş performans ve kanıt kontrolleri geçti.", "css": "ok-box"}
    return {"label": "Onaylanmadı", "text": f"{name} testi tamamlandı; geçmiş performans ve kanıt koşullarının tamamı sağlanmadı.", "css": "warn-box"}


def funnel_rows(counts):
    """Counts must be sequential survivor counts, not independent filter effects."""
    previous = None
    rows = []
    for label, count in counts.items():
        rows.append({"Aşama": label, "Kalan": count, "Bu aşamada elenen": 0 if previous is None else previous - count})
        previous = count
    return rows
