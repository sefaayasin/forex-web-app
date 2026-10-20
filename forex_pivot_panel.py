"""Streamlit small-profit commentator. UI is separate from data providers and pure rules."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from forex_config import MAJOR_PAIRS, SYMBOL_LIST, price_decimals
from forex_pivot_advisor import calendar_available
from forex_pivot_live import live_plan

EVIDENCE = Path(__file__).resolve().parent / "research/pivot_advisor_small_profit/results.json"


def render_pivot_panel(*, scan, get_quote, get_calendar, get_news, get_cost, get_pip_value,
                       refresh, provider, account_usd, risk_pct=.5):
    st.markdown("#### 📐 Pivot Trader · küçük kâr yorumcusu")
    st.caption("LONG GİR / SHORT GİR yalnız giriş koşulları uygunsa gösterilir. Net hedef: 3 pip · azami süre: 4 saat.")
    buttons, budget = st.columns([1, 2])
    if buttons.button("🔄 Yorumcuyu yenile", key="pivot_live_refresh", use_container_width=True):
        refresh()
    max_loss = budget.number_input("İşlem başına planlanan en yüksek zarar ($)", min_value=0.,
                                   value=round(min(float(account_usd) * float(risk_pct) / 100, float(account_usd)), 2),
                                   step=1., key="pivot_live_max_loss")
    symbols = st.multiselect("İzlenecek pariteler", SYMBOL_LIST, default=MAJOR_PAIRS,
                            key="pivot_live_symbols", format_func=lambda value: value.replace("=X", ""))
    st.warning("GİR kararı zararsız işlem garantisi değildir. Küçük kâr hedefinde bir stop, birçok kazancı silebilir. "
               "Lot zarar bütçesine göre hesaplanır; gap, kayma ve değişen maliyet planlanan kaybı aşırabilir.")
    if not symbols:
        st.info("Bir parite seçerek yorumcuyu başlat.")
        return
    try:
        events, _, fetched_at, _ = get_calendar()
        now = pd.Timestamp.now(tz="UTC")
        known = calendar_available(events, now) and pd.Timedelta(0) <= now - fetched_at <= pd.Timedelta(minutes=15)
    except Exception:
        events, known = None, False
    anchors = st.session_state.setdefault("pivot_live_anchors", {})
    readings = []
    with st.spinner("Güncel fiyat, trend, ML ve haberler değerlendiriliyor..."):
        for symbol in symbols:
            snapshot = scan(symbol, provider)
            quote = get_quote(symbol, provider) or {}
            now = pd.Timestamp.now(tz="UTC")
            minutes, title = get_news(symbol, events) if known else (None, "")
            cost = get_cost(symbol, quote, now)
            value = get_pip_value(symbol, quote.get("price"))
            key = f"{provider}:{symbol}:{snapshot.get('side')}:{snapshot.get('signal_bar', snapshot.get('as_of'))}"
            reading = live_plan(snapshot, quote, now=now, news_known=known, news_minutes=minutes,
                                news_title=title, cost_pips=cost, account_usd=account_usd,
                                max_loss_usd=float(max_loss), pip_value_per_lot=value,
                                anchor_price=anchors.get(key))
            if reading["technical_ready"]:
                anchors.setdefault(key, quote["price"])
            readings.append(reading)
    if len(anchors) > 300:
        st.session_state["pivot_live_anchors"] = dict(list(anchors.items())[-150:])
    readings.sort(key=lambda r: (r["technical_ready"], r["alignment"]), reverse=True)
    entries = [r for r in readings if r["technical_ready"]]
    if entries:
        best = entries[0]
        st.success(f"{best['symbol'].replace('=X', '')} · {best['action']}")
        st.write(best["comment"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Net hedef (tahmini)", f"+{best['expected_profit_usd']:.2f} $", help="3 net pip, gösterilen lot ve maliyet tahminiyle.")
        c2.metric("Stopta planlanan kayıp", f"−{best['planned_loss_usd']:.2f} $")
        c3.metric("Hesaplanan lot", f"{best['lot']:.2f}")
    else:
        st.info("BEKLE · Şu an giriş koşulları tamamlanan plan yok.")
        st.write(readings[0]["symbol"].replace("=X", "") + ": " + readings[0]["comment"])
    st.caption(f"Son kontrol: {pd.Timestamp.now(tz='Europe/Istanbul'):%d.%m %H:%M:%S} İstanbul · "
               "30 saniyede bir yenilenir. Karar için fiyat zamanına ve geçerlilik saatine bak; eski kartla giriş yapma.")
    table = []
    for r in readings:
        decimals = price_decimals(r["symbol"])
        number = lambda key: f"{r[key]:.{decimals}f}" if r.get(key) is not None and pd.notna(r[key]) else "—"
        table.append({"Parite": r["symbol"].replace("=X", ""), "Karar": r["action"],
                      "Referans giriş": number("entry"), "3 pip net hedef": number("target"),
                      "Stop": number("stop"), "Lot": f"{r['lot']:.2f}" if r["lot"] else "—",
                      "Net hedef $": round(r["expected_profit_usd"], 2) if r["expected_profit_usd"] is not None else None,
                      "Stop kaybı $": round(r["planned_loss_usd"], 2) if r["planned_loss_usd"] is not None else None})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    for r in readings:
        with st.expander(f"{r['symbol'].replace('=X', '')} · {r['action']} · gerekçe ve seviyeler", expanded=r["technical_ready"]):
            decimals = price_decimals(r["symbol"])
            st.write(r["comment"])
            if r["valid_until"] is not None:
                st.caption(f"Fiyat zamanı: {pd.Timestamp(r['quote_time']).tz_convert('Europe/Istanbul'):%H:%M:%S} · "
                           f"En geç yeniden kontrol: {r['valid_until'].tz_convert('Europe/Istanbul'):%H:%M:%S} İstanbul")
            if r.get("broker_entry") is not None:
                st.write(f"Broker giriş: {r['broker_entry']:.{decimals}f} · "
                         f"Broker TP: {r['broker_target']:.{decimals}f} · Broker SL: {r['broker_stop']:.{decimals}f}")
                st.caption("SHORT seviyeleri ASK tetiklemesine göre spread kadar düzeltilmiştir; kotasyon değişirse tekrar kontrol et.")
            else:
                st.caption("Fiyatlar gösterge seviyesidir; broker alış/satış fiyatı ve maliyetleri farklı olabilir. "
                           "Broker fiyatı farklıysa bu planla körlemesine giriş yapma.")
            if r.get("cost_pips") is not None:
                st.write(f"Tahmini spread + komisyon: {r['cost_pips']:.1f} pip. Net 3 pip için bu maliyet hedef mesafesine eklenir.")
            if r["small_target_rr"] is not None and r["small_target_rr"] > 0:
                st.write(f"Bir stop yaklaşık {1/r['small_target_rr']:.1f} küçük kârı siler. Süre sonu: girişten itibaren 4 saat.")
            for blocker in r["blockers"]:
                st.write("• " + blocker)
            st.dataframe(pd.DataFrame([{"Kontrol": c["condition"], "Sonuç": "OK" if c["ok"] else "Eksik / ters",
                                        "Açıklama": c["detail"]} for c in r["checks"]]), hide_index=True, use_container_width=True)
    with st.expander("Geçmiş küçük hedef denemesi ve sınırları"):
        try:
            evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))["net_3_pip"]["all"]
            st.write(f"1 Ekim 2024–11 Eylül 2026, yedi ana parite: net 3 pip hedefi "
                     f"{evidence['targets']}/{evidence['n']} işlemde stop öncesi görüldü.")
        except (OSError, ValueError, KeyError):
            st.write("Geçmiş test raporu bu sunucuda bulunamadı.")
        st.write("Örnek çok küçük ve daha önce incelenmiş veri üzerinde. Bu, gelecekteki kazanma oranı değildir. "
                 "Canlı fiyat tazeliği ve fiyatı kovalamama kontrolleri eklenmiştir; bu ek kontroller bağımsız test edilmedi. "
                 "Pip hedefi, tahmini maliyet ve lotla dolar tutarına çevrilir; dolar kârı sabit veya garanti değildir.")
