"""Read the new internet-researched indicator tournament without training/I/O."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def render_new_indicators_panel():
    import streamlit as st

    folder = ROOT / "research/new_indicators"
    files = sorted(folder.glob("*/report.json"))
    if not files:
        return
    with st.expander("Yeni indikatörler — internet araştırması ve geçmiş test", expanded=False):
        st.caption("Fisher, Roofing/SuperSmoother, FRAMA, KAMA, Supertrend, ConnorsRSI ve LaguerreRSI. "
                   "Hedef 30 dolar; planlanan stop bütçeleri 20, 50 veya 100 dolar. Giriş 15 dakika.")
        try:
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
            table = pd.DataFrame([{
                "Parite": r["symbol"], "Geliştirmede seçilen": r["selected"]["rule"],
                "Stop bütçesi ($)": r["planned_stop_loss_usd"], "İşlem": r["audit"]["trades"],
                "Hedefe ulaştı (%)": round(100*r["audit"]["target_rate"], 1) if r["audit"]["target_rate"] is not None else None,
                "Ortalama net ($)": r["audit"]["mean_usd"],
                "İki kat masraf ($)": r["audit"]["stress_mean_usd"],
                "Araştırma ölçütleri geçti": r["screen_passed"],
            } for r in reports])
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("2016–2022 dönemi yöntemi seçer; 2023 sonrası dönem kontrol eder. "
                       "Bu geçmiş önceden incelendiği için yeni bir kör test değildir. "
                       "Dolarlar varsayımsal risk ölçeğidir; broker gerçekleşmesi/teminat ayrıca gerekir.")
            sources = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
            st.write("Formüllerin kaynakları")
            for source in sources["formula_sources"]:
                st.markdown(f"[{source['indicator']}]({source['url']})")
            chosen = st.selectbox("Yeni indikatör testi paritesi", files,
                format_func=lambda p: p.parent.name, key="new_indicators_report")
            report = json.loads(chosen.read_text(encoding="utf-8"))
            trials = [{"Kural": t["rule"], "Stop ATR": t["stop_atr"], "Saat": t["hours"],
                "Bütçe ($)": t["loss_usd"],
                "İlk geliştirme net ($)": t["subperiods"][0]["mean_usd"],
                "İkinci geliştirme net ($)": t["subperiods"][1]["mean_usd"],
                "En zayıf stres dönemi ($)": min(p["stress_mean_usd"] for p in t["subperiods"])
                    if all(p["stress_mean_usd"] is not None for p in t["subperiods"]) else None,
            } for t in report["development_trials"]]
            st.dataframe(pd.DataFrame(trials).sort_values("En zayıf stres dönemi ($)", ascending=False),
                         hide_index=True, use_container_width=True, height=260)
            st.download_button("Yeni indikatör raporunu indir", chosen.read_bytes(),
                file_name=f"{report['symbol']}_new_indicators.json", mime="application/json", key="new_indicators_download")
            news_files = sorted((ROOT / "research/news_surprise").glob("*/report.json"))
            if news_files:
                st.write("Fiyat dışı alternatif: açıklanan veri − beklenti")
                news = [json.loads(p.read_text(encoding="utf-8")) for p in news_files]
                st.dataframe(pd.DataFrame([{
                    "Parite": r["symbol"], "Önceki dönem net ($)": r["development"]["mean_usd"],
                    "Kontrol işlemi": r["audit"]["trades"], "Kontrol net ($)": r["audit"]["mean_usd"],
                    "İki kat masraf ($)": r["audit"]["stress_mean_usd"],
                    "%95 alt sınır ($)": r["day_test"]["ci_low"],
                } for r in news]), hide_index=True, use_container_width=True)
                st.caption("Hedef 30 dolar / planlanan stop 20 dolar; 2024–2025 önceki dönem, 2025–2026 kontrol. "
                           "Bazı pozitif ortalamalar var; az örnek ve negatif güven alt sınırları nedeniyle giriş onayı yok. "
                           "Takvim değerlerinin ilk yayın sürümleri doğrulanmamış; çalışma keşif amaçlıdır.")
                st.markdown("[Haber sürprizi araştırmasının kaynağı](https://www.nber.org/papers/w8959)")
            session_file = ROOT / "research/session_relative/summary.csv"
            if session_file.exists():
                st.write("Seans davranışı ve pariteler arası güç")
                session = pd.read_csv(session_file)
                st.dataframe(session[["symbol", "rule", "risk", "trades", "mean_usd",
                    "stress_mean_usd", "screen_passed"]].rename(columns={
                    "symbol": "Parite", "rule": "Yöntem", "risk": "Planlanan risk ($)",
                    "trades": "İşlem", "mean_usd": "Ortalama net ($)",
                    "stress_mean_usd": "İki kat masraf ($)", "screen_passed": "Ölçütleri geçti"}),
                    hide_index=True, use_container_width=True)
                st.caption("Londra açılışı, seans ortalamasına dönüş ve diğer altı pariteden dolar gücü. "
                    "Ayar seçimi 2016–2022; kontrol 2023 sonrası. Önceden incelenmiş geçmiş testidir.")
            level_files = sorted((ROOT / "research/price_levels").glob("*/report.json"))
            if level_files:
                st.write("Yuvarlak fiyat seviyeleri: kırılım ve dönüş")
                levels = [json.loads(p.read_text(encoding="utf-8")) for p in level_files]
                st.dataframe(pd.DataFrame([{
                    "Parite": r["symbol"], "Seçilen kural": r["selected"]["rule"],
                    "İşlem": r["audit"]["trades"], "Ortalama net ($)": r["audit"]["mean_usd"],
                    "0,5 pip masraf senaryosu ($)": r["fee_scenarios_pips"]["0.5"]["mean_usd"],
                    "Ölçütleri geçti": r["screen_passed"],
                } for r in levels]), hide_index=True, use_container_width=True)
                st.dataframe(pd.DataFrame([{
                    "Parite": r["symbol"], "Yıl": year, "İşlem": metrics["trades"],
                    "Ortalama net ($)": metrics["mean_usd"],
                } for r in levels for year, metrics in r["years"].items()]),
                    hide_index=True, use_container_width=True)
                st.caption("50/100 pip seviyeleri gerçek stop emirlerinin yerini göstermez. "
                    "Ayarlar 2016–2022 döneminde seçildi. Düşük masraf senaryosu gerçek hesap maliyeti değildir. "
                    "2026 sonuçları arşivin son tarihi olan 11 Eylül'e kadardır.")
            bracket_file = ROOT / "research/bracket_search/summary.csv"
            if bracket_file.exists():
                st.write("Yeni aramalar: gece, günlük, hafta sonu, COT, haber, 30 yeni indikatör, altın")
                bracket = pd.read_csv(bracket_file)
                st.dataframe(pd.DataFrame({
                    "Aile": bracket.label, "Ayar": bracket.configs,
                    "Geliştirmeyi geçen": bracket.development_picks,
                    "Doğrulamayı da geçen": bracket.validation_survivors,
                    "Kontrolde pozitif": bracket.audit_positive_survivors,
                    "Masraf / risk (%)": (100 * bracket.median_cost_share).round(1),
                    "Kontrol net ($/işlem)": bracket.net_usd_audit.round(2),
                }), hide_index=True, use_container_width=True)
                st.caption("Hedef 30 dolar net, planlanan stop 100 dolar. Geliştirme 2009–2017, doğrulama 2018–2022, "
                    "kontrol 2023 sonrası (altında 2001–2012 / 2013–2018 / 2019+). Hiçbir aile önceden yazılan ölçütleri geçmedi; giriş izni verilmedi. "
                    "0,3:1 oranda rastgele fiyatın hedef oranı %76,9'dur; yüksek isabet tek başına avantaj değildir.")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Yeni indikatör testleri okunamadı: {exc}")
