"""Read-only research results for the existing Streamlit application."""
import json
from pathlib import Path

import pandas as pd


def render_ml_panel():
    import streamlit as st

    render_tournament_panel()

    with st.expander("Makine öğrenmesi — araştırma sonuçları", expanded=False):
        st.caption("Teknik ve FOMC destekli modellerin geçmiş dönem karşılaştırması. Bu sonuçlar işlem izni vermez.")
        files = sorted((Path(__file__).resolve().parent / "data/ml").glob("*_report.json"))
        if not files:
            st.info("Henüz sonuç yok. Eğitim: python forex_ml.py --symbols EURUSD GBPUSD USDJPY")
            return
        selected = st.selectbox("ML araştırma paritesi", files, format_func=lambda p: p.stem.replace("_report", ""))
        try:
            report = json.loads(selected.read_text(encoding="utf-8"))
            st.write(f"Veri: {report['start']} → {report['end']} | {report['bars']:,} saatlik mum")
            st.caption(f"Gidiş-dönüş maliyet: {report['round_trip_cost_pips']} pip; ufuk: {report['horizon_bars']} mum; eşik: {report['threshold']}")
            st.dataframe(pd.DataFrame(report["results"]), hide_index=True, use_container_width=True)
            st.caption("AUC: yön ayrımı (0,50 referans). Brier: daha düşük daha iyi. Net bps: maliyet sonrası baz puan; portföy getirisi değildir.")
            st.write("Araştırmanın sınırları:")
            for limitation in report["limitations"]:
                st.write(f"• {limitation}")
            st.download_button("ML raporunu indir", selected.read_bytes(), file_name=selected.name, mime="application/json")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"ML raporu okunamadı: {exc}")


def render_tournament_panel():
    import streamlit as st

    folder = Path(__file__).resolve().parent / "data/ml/tournament"
    if not (folder / "complete.json").exists():
        return
    with st.expander("ML karşılaştırma laboratuvarı — modeller ve heatmap", expanded=True):
        try:
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            for column, row in zip(st.columns(2), summary):
                name = "Yön doğruluğu" if row["task"] == "direction" else "Oynaklık doğruluğu"
                column.metric(name, f"%{100 * row['accuracy']:.2f}")
                column.caption(f"Dengeli doğruluk: %{100 * row['balanced_accuracy']:.2f} · {row['pairs']} parite")
                if row["task"] == "high_volatility":
                    column.caption(f"Güven filtresiyle: %{100 * row['confident_accuracy']:.2f} doğruluk / %{100 * row['confident_coverage']:.1f} kapsam")
            st.caption("Oynaklık tahmini fiyat yönü veya işlem kazanma oranı değildir. Sonuçlar geliştirmede seçilmiş sabit modellerin 2023+ kontrolüdür; işlem izni vermez.")
            choices = {
                "Nihai dengeli doğruluk": "final_balanced_accuracy.png",
                "Yön modeli karşılaştırması": "development_direction.png",
                "Oynaklık modeli karşılaştırması": "development_high_volatility.png",
                "İşlem maliyeti": "cost_heatmap.png",
                "İşlem kapsamı": "coverage_heatmap.png",
                "Özellik korelasyonu": "feature_correlation.png",
                "Yıllara göre tutarlılık": "yearly_stability.png",
                "Güven filtresi ve kapsam": "confidence_accuracy.png",
                "Özellik gruplarının katkısı": "feature_group_lift.png",
            }
            selected = st.selectbox("Karşılaştırma", list(choices))
            st.image(str(folder / "figures" / choices[selected]), use_container_width=True)
            st.dataframe(pd.read_csv(folder / "final_metrics.csv"), hide_index=True, use_container_width=True)
            st.download_button("Etkileşimli HTML raporunu indir", (folder / "report.html").read_bytes(),
                file_name="forex_ml_karsilastirma.html", mime="text/html")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Karşılaştırma raporu okunamadı: {exc}")
