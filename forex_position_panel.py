"""Trade-plan visual watchlist. Saved levels never follow refreshed signals."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from forex_config import SYMBOL_LIST, price_decimals
from forex_position_chart import make_plan, observe_plan, plan_from_reading, position_figure
from forex_storage import archive_tracked_plan, save_tracked_plan, tracked_plans


def render_position_tracker(*, readings, provider, get_bars, get_quote):
    st.markdown("#### İşlem takip grafiği")
    st.caption("Kırmızı alan stop mesafesini, yeşil alan hedef mesafesini gösterir. "
               "Takibe alınan giriş, stop ve hedef sabit kalır; fiyat 30 saniyede bir kontrol edilir.")
    st.caption(f"Veri kaynağı: {provider} · Takip listesi her veri kaynağı için ayrı tutulur.")
    plans = tracked_plans(provider)
    active_symbols = {plan["symbol"] for plan in plans}
    candidates = []
    for reading in readings:
        try:
            preview = plan_from_reading(reading, provider, now=pd.Timestamp.now(tz="UTC"))
        except (ValueError, TypeError, KeyError):
            continue
        candidates.append((reading, preview))

    if candidates:
        options = {p["symbol"]: (r, p) for r, p in candidates}
        selected = st.selectbox("Grafikte önizlenecek plan", list(options),
            format_func=lambda s: f"{s.replace('=X', '')} · {options[s][0]['action']}", key="position_preview_symbol")
        reading, preview = options[selected]
        ready = reading.get("technical_ready", False)
        if selected not in active_symbols:
            _render_chart(preview, get_bars=get_bars, get_quote=get_quote, preview=True,
                          action=reading.get("action", "BEKLE"))
        st.button("Bu planı takibe al", key="position_follow_candidate",
            disabled=not ready or selected in active_symbols, use_container_width=True,
            on_click=save_tracked_plan, args=(preview,))
        if not ready:
            st.caption("Taslak plan · BEKLE kararı sürüyor. Bu görünüm yeni giriş önerisi değildir.")
        elif selected in active_symbols:
            st.caption("Bu paritenin kayıtlı planı aşağıda. Yeni analiz kayıtlı seviyeleri değiştirmez.")

    with st.expander("Kendi planını ekle", expanded=not candidates and not plans):
        st.caption("Kendi giriş, stop ve hedef seviyelerini görsel takip için kaydet. "
                   "Bu kayıt bir sinyal onayı veya broker emri değildir.")
        with st.form("position_manual_form"):
            c1, c2 = st.columns(2)
            symbol = c1.selectbox("Takip paritesi", SYMBOL_LIST, format_func=lambda s: s.replace("=X", ""))
            side = c2.selectbox("Plan yönü", ["LONG", "SHORT"])
            c1, c2, c3 = st.columns(3)
            entry = c1.number_input("Giriş fiyatı", min_value=0., value=0., format="%.5f", step=.0001)
            stop = c2.number_input("Stop fiyatı", min_value=0., value=0., format="%.5f", step=.0001)
            target = c3.number_input("Hedef fiyatı", min_value=0., value=0., format="%.5f", step=.0001)
            lot = st.number_input("Lot (isteğe bağlı)", min_value=0., value=0., step=.01)
            submitted = st.form_submit_button("Seviyeleri kaydet ve takip et", use_container_width=True)
        if submitted:
            try:
                plan = make_plan(symbol, provider, side, entry, stop, target,
                                 lot=lot, now=pd.Timestamp.now(tz="UTC"))
                if symbol in active_symbols:
                    st.warning("Bu parite zaten takipte. Yeni plan için önce mevcut takibi bitir.")
                else:
                    save_tracked_plan(plan)
                    st.success("Seviyeler kaydedildi.")
                    plans = tracked_plans(provider)
            except ValueError as exc:
                st.error(str(exc))
    if not plans:
        st.info("Takipte plan yok. Hazır planı takibe al veya kendi seviyelerini ekle.")
        return

    st.markdown("**Takipteki planlar**")
    options = {plan["id"]: plan for plan in plans}
    selected = st.selectbox("Takipteki işlem", list(options), key="position_watch_id",
        format_func=lambda key: f"{options[key]['symbol'].replace('=X', '')} · {options[key]['side']}")
    plan = options[selected]
    _render_chart(plan, get_bars=get_bars, get_quote=get_quote)
    c1, c2 = st.columns(2)
    if c1.button("Takibi bitir", key=f"position_archive_{plan['id']}", use_container_width=True):
        archive_tracked_plan(plan["id"])
        st.rerun()
    c2.download_button("Takip listesini indir", json.dumps(plans, ensure_ascii=False, indent=2),
        file_name="islem_takibi.json", mime="application/json", use_container_width=True)
    st.caption("Takibi bitirmek brokerdaki pozisyonu kapatmaz. Kayıtlar bu uygulamanın yerel veritabanında tutulur; "
               "bulut sunucusu yenilenirse kaybolabilir. Listeyi indirerek saklayabilirsin.")


def _render_chart(plan, *, get_bars, get_quote, preview=False, action=""):
    now = pd.Timestamp.now(tz="UTC")
    try:
        bars = get_bars(plan["symbol"], plan["provider"])
    except Exception:
        bars = pd.DataFrame()
    try:
        quote = dict(get_quote(plan["symbol"], plan["provider"]) or {})
        quote.setdefault("provider", plan["provider"])
    except Exception:
        quote = {}
    view = observe_plan(plan, quote, now=now)
    heading = f"{plan['symbol'].replace('=X', '')} · {plan['side']}"
    st.markdown(f"**{heading} · {'Taslak / ' + action if preview else view['status']}**")
    if plan.get("source") == "pivot":
        st.caption("4H / 1H yön kontrolü → 15M tepki → 5M takip grafiği")
    else:
        st.caption("Kendi seviyelerin · 5M takip grafiği")
    st.plotly_chart(position_figure(bars, plan, quote=quote, now=now), use_container_width=True,
        config={"displaylogo": False, "scrollZoom": False},
        key=f"position_chart_{'preview' if preview else plan['id']}_{plan['symbol']}")
    if not preview:
        decimals = price_decimals(plan["symbol"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Güncel referans fiyat" if view["fresh"] else "Son alınan fiyat",
                  f"{view['price']:.{decimals}f}" if view["price"] is not None else "—")
        c2.metric("Girişten fiyat hareketi", f"{view['move_pips']:+.1f} pip" if view["move_pips"] is not None else "—")
        c3.metric("Hedefe mesafe", f"{view['target_distance_pips']:.1f} pip" if view["target_distance_pips"] is not None else "—")
        stamp = pd.Timestamp(plan["tracked_at"]).tz_convert("Europe/Istanbul")
        lot_text = f"{plan['lot']:.2f}" if plan.get("lot") else "girilmedi"
        st.caption(f"Takip başlangıcı: {stamp:%d.%m.%Y %H:%M} İstanbul · "
                   f"Kaynak: {plan['provider']} · Lot: {lot_text}")
    st.caption("Giriş / stop / hedef kayıtlı referans seviyeleridir. Fiyat hareketi masrafları içermez; "
               "broker kâr/zararı veya gerçekleşmiş TP/SL sonucu değildir.")
