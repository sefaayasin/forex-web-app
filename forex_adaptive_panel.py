"""Read learning reports; locally update archives without importing the app."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def render_adaptive_entry_panel():
    import streamlit as st

    with st.expander("15 dakika — otomatik öğrenme ve giriş testleri", expanded=False):
        cash_folder = ROOT / "research/adaptive_cash_entry"
        cash_mode = any(cash_folder.glob("*/report.json"))
        folder = cash_folder if cash_mode else ROOT / "research/adaptive_entry"
        st.caption("Model; fiyat, teknik indikatörler ve ADX/Stochastic/Donchian gruplarını karşılaştırır. "
                   "Ağırlıklar ve eşik her üç ayda geçmiş sonuçlardan öğrenilir. Stop: 1,5 ATR; "
                   + ("net hedef/zarar: 30/100 dolar" if cash_mode else "hedef: 1,5 R")
                   + "; en fazla 4 saat. Sonuçlar araştırma amaçlıdır.")
        symbols = sorted({p.parent.name for p in folder.glob("*/report.json")} |
                         {p.stem for p in (ROOT / "data/historical_15m").glob("*.csv")})
        if not symbols:
            st.info("Yerel eğitim: python -X utf8 forex_adaptive.py")
            return
        symbol = st.selectbox("Öğrenen giriş modeli paritesi", symbols, key="adaptive_symbol")
        local = all((ROOT / f"data/historical_{tf}/{symbol}.csv").exists() for tf in ("5m", "15m"))
        if local and st.button("Yerel arşivden öğrenmeyi güncelle", key="adaptive_update"):
            try:
                from forex_adaptive import update_symbol
                with st.spinner("Geçmiş girişler test ediliyor, model ve ağırlıklar öğreniliyor…"):
                    update_symbol(symbol, output=folder, net_target_ratio=.3 if cash_mode else None)
                st.success("Öğrenme raporu güncellendi.")
            except (ImportError, OSError, ValueError) as exc:
                st.error(f"Eğitim tamamlanamadı: {exc}")
        path = folder / symbol / "report.json"
        if not path.exists():
            st.info(f"Bu parite için sonuç yok: python -X utf8 forex_adaptive.py --symbols {symbol}")
            return
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            st.write(f"Arşiv sonu: {report['audit']['end']}")
            st.caption("Model yalnız yerel arşivden öğrenir. Yeni veri eklendiğinde güncellenir; internetten veri indirmez.")
            all_rows, selected = report["all_candidates"], report["selected"]
            st.dataframe(pd.DataFrame([
                {"Grup": "Bütün giriş adayları", **all_rows},
                {"Grup": "Modelin uygun buldukları", **selected},
                {"Grup": "Kârlılık koşulu aranmayan teşhis filtresi", **report["diagnostic"]},
            ]), hide_index=True, use_container_width=True)
            st.caption("mean_r: işlem başına 1,5 pip sonrası R; mean_r_stress: 3 pip sonrası R. "
                       "Kapsam: %{:.1f}. R toplamları portföy getirisi değildir.".format(report["coverage"] * 100))
            if cash_mode:
                st.caption("Bu modelde 1 R planlanan maliyet dahil stop zararıdır; 100 dolar risk ölçeğinde hedef 30 dolar. "
                           "Sabit masraf senaryosu; gerçek lot ve teminat ayrıca hesaplanır.")
            if report["latest"]["threshold"] is None:
                st.info("Son öğrenme döneminde maliyet sonrası yeterli aday bulunamadı: model girişleri beklemeye aldı.")
            st.caption(f"Brier (düşük daha iyi): model {report['brier']:.4f}; "
                       f"eğitim kazanma oranı referansı {report['baseline_brier']:.4f}.")
            choices = {"Son öğrenme": report["latest"]}
            choices.update({p["cutoff"][:10]: p["selection"] for p in report["periods"]})
            chosen = st.selectbox("Öğrenilen ağırlıkların dönemi", list(choices), key="adaptive_weights_period")
            weights = choices[chosen]["weights"]
            if weights:
                st.write("Son modelin öğrendiği indikatör ağırlıkları")
                st.dataframe(pd.DataFrame(weights).sort_values("relative_weight_pct", ascending=False),
                             hide_index=True, use_container_width=True)
                st.caption("Yüzde, standartlaştırılmış katsayıların mutlak büyüklük payıdır; "
                           "isabet veya kazanma olasılığı değildir. Katsayı işareti etkinin yönünü gösterir.")
            else:
                st.caption("Son seçilen model ağaç modeli; tek bir sabit indikatör yüzdesi kullanmaz.")
            st.dataframe(pd.DataFrame([{"Dönem": p["cutoff"], "Model": p["selection"]["config"]["model"],
                "İndikatör grubu": p["selection"]["config"]["bundle"],
                "İşlem": p["selected"]["trades"], "Net R": p["selected"]["mean_r"],
                "Brier": p["brier"]} for p in report["periods"]]), hide_index=True, use_container_width=True)
            st.caption("2023 sonrası geçmiş daha önce incelendiği için bu sonuçlar yeni bir kör test sayılmaz. "
                       "Öğrenen model mevcut sinyal kurallarını ve işlem izinlerini değiştirmez.")
            st.download_button("Öğrenme raporunu indir", path.read_bytes(), file_name=f"{symbol}_learning.json",
                               mime="application/json", key="adaptive_download")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Öğrenme raporu okunamadı: {exc}")
