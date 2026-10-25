"""Display measured evidence for the user's cash target, without market I/O."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def render_cash_target_evidence(symbol, target_usd=30., loss_usd=100.):
    import streamlit as st

    path = ROOT / "research/cash_target" / str(symbol).removesuffix("=X").upper() / "report.json"
    if not path.exists():
        return
    with st.expander("Hedefinin geçmiş testi", expanded=False):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if abs(report["target_usd"] - target_usd) > .01 or abs(report["planned_stop_loss_usd"] - loss_usd) > .01:
                st.caption(f"Kayıtlı yeni giriş testi ${report['target_usd']:.0f} net hedef / "
                           f"${report['planned_stop_loss_usd']:.0f} planlanan stop için; seçtiğin bütçe farklı.")
                return
            audit = report["audit"]
            if not report["research_screen_passed"]:
                st.info("Bu bütçeyle yeni bir LONG/SHORT giriş yöntemi doğrulanmadı.")
            else:
                st.info("Geçmiş araştırma koşulları sağlandı; henüz ileriye dönük gerçek giriş doğrulaması yok.")
            st.dataframe(pd.DataFrame([{
                "Kontrol dönemi": "2023 – 11 Eylül 2026",
                "İşlem": audit["trades"],
                "30 $ hedefine ulaşan (%)": round(audit["target_rate"] * 100, 1) if audit["target_rate"] is not None else None,
                "Ortalama net sonuç ($)": audit["mean_usd"],
                "İki kat masrafla ($)": audit["stress_mean_usd"],
            }]), hide_index=True, use_container_width=True)
            st.caption("Bu ölçüm 13 farklı giriş kuralı, üç stop genişliği ve iki tutma süresi arasından "
                       "2016–2022 döneminde seçilen adayın sonraki dönem sonucudur. "
                       "Mevcut ML işlem planından farklı bir giriş stratejisidir.")
            st.caption("Hedef ve stopla biten işlemlerde +30/−100 dolar için başabaş isabet yaklaşık %77'dir. "
                       "Zaman aşımı işlemleri ayrıca değerlendirilir; hedefe ulaşma oranı yön olasılığı değildir.")
            st.caption("Dolarlar sabit risk bütçesinin varsayımsal ölçeğidir; gerçek lot, teminat ve broker gerçekleşmesi ayrıca gerekir.")
            st.download_button("Bütçe testini indir", path.read_bytes(), file_name=f"{report['symbol']}_cash_target.json",
                               mime="application/json", key="cash_target_download")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Hedef testi okunamadı: {exc}")
