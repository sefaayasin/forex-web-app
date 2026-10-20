"""Presentation helpers for the SEFA dashboard; no market or trading rules."""

from __future__ import annotations

from html import escape
from pathlib import Path
import base64
import json
import math

import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st


def apply_dashboard_theme() -> None:
    st.markdown(
        f"<style>{Path(__file__).with_name('forex_dashboard.css').read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
    template = go.layout.Template(pio.templates["plotly_dark"])
    template.layout.update(
        paper_bgcolor="#081722", plot_bgcolor="#081722",
        font=dict(color="#b2c4d3", family="Inter, Segoe UI, sans-serif"),
        colorway=["#00d4b1", "#00b4f0", "#e9b95b", "#a998f6", "#f36b80"],
        xaxis=dict(gridcolor="#142b39", zerolinecolor="#234353"),
        yaxis=dict(gridcolor="#142b39", zerolinecolor="#234353"),
        hoverlabel=dict(bgcolor="#102b3a", font_color="#eef7ff"),
    )
    template.data.candlestick = [go.Candlestick(
        increasing_line_color="#00d4a5", decreasing_line_color="#f36b80",
        increasing_fillcolor="#00d4a5", decreasing_fillcolor="#f36b80",
    )]
    pio.templates["sefa"] = template
    pio.templates.default = "sefa"


def render_brand(location: str = "sidebar") -> None:
    st.markdown(
        f'<div class="sefa-brand sefa-brand-{escape(location)}"><svg viewBox="0 0 44 44" aria-hidden="true">'
        '<path d="M3 30h7v12H3zm12-9h7v21h-7zm12-9h7v30h-7z" fill="#00c9a7"/>'
        '<path d="M4 23 16 14l9 2L39 3m-9 0h9v9" fill="none" stroke="#00e2bf" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        '<div><strong>SEFA</strong><span>FOREX ANALYZER</span></div></div>'
        + ('<div class="sefa-sidebar-label">İŞLEM MASASI</div>' if location == "sidebar" else ''),
        unsafe_allow_html=True,
    )




@st.cache_data(show_spinner=False)
def _hero_background() -> str:
    image = Path(__file__).with_name('assets') / 'sefa-market-globe.png'
    return 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')


def render_dashboard_header() -> None:
    st.markdown(
        f'<style>.sefa-hero {{ background-image: linear-gradient(90deg, #03131f 0%, #03131fbb 29%, #03131f00 72%), url("{_hero_background()}"); }}</style>'
        '<section class="sefa-hero"><div class="sefa-hero-copy">'
        '<div class="sefa-eyebrow"><span></span> SEFA PİYASA ÇALIŞMA ALANI</div>'
        '<h1><span>Piyasaları</span><em>Daha İyi Anlayın</em></h1>'
        '<p>Piyasa verileri, teknik analiz ve risk araçları tek bir yerde.</p>'
        '<div class="sefa-hero-actions"><a class="sefa-cta" href="#sefa-analysis" target="_self">Analize Başla <span>→</span></a>'
        '<a class="sefa-cta sefa-cta-secondary" href="#sefa-markets" target="_self">Piyasaları İncele</a></div>'
        '<div class="sefa-hero-features"><span>◉ Çoklu Zaman Dilimi</span><span>◇ Teknik Analiz Araçları</span>'
        '<span>♧ Strateji Testleri</span><span>▤ Ekonomik Takvim</span></div>'
        '</div></section>',
        unsafe_allow_html=True,
    )


def render_market_tiles(symbol: str, changes) -> None:
    """Display the existing change windows without additional price requests."""
    if changes.empty:
        st.caption("Fiyat özeti için veri bekleniyor.")
        return
    tiles = []
    for _, row in changes.iterrows():
        value = row["Değişim %"]
        available = value is not None and value == value
        tone = "up" if available and float(value) >= 0 else "down" if available else "neutral"
        text = f"{float(value):+.2f}%" if available else "—"
        tiles.append(
            f'<div class="sefa-market-tile {tone}"><div><strong>{escape(symbol.replace("=X", ""))}</strong>'
            f'<span>{escape(str(row["Pencere"]))}</span></div><b>{text}</b></div>'
        )
    st.markdown(f'<div class="sefa-market-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


LEVEL_COLORS = {"R": "#f36b80", "S": "#00d4a5", "P": "#e9b95b"}


def market_candles(frame, symbol: str, interval: str, levels: dict | None = None) -> go.Figure:
    """Use the same source as the intraday view, including uploaded broker data.

    `levels` ({name: price}) are drawn as dotted lines when they are near the visible candles.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[.8, .2], vertical_spacing=.02)
    if frame.empty:
        fig.add_annotation(text="Mum grafiği için fiyat verisi bekleniyor", showarrow=False)
    else:
        aggregation = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in frame:
            aggregation["Volume"] = "sum"
        candles = frame.resample(interval).agg(aggregation).dropna(subset=["Open", "High", "Low", "Close"]).tail(80)
        fig.add_trace(go.Candlestick(
            x=candles.index, open=candles.Open, high=candles.High,
            low=candles.Low, close=candles.Close, name=symbol.replace("=X", ""),
        ), row=1, col=1)
        if "Volume" in candles:
            colors = ["#145447" if close >= opening else "#573040"
                      for opening, close in zip(candles.Open, candles.Close)]
            fig.add_trace(go.Bar(x=candles.index, y=candles.Volume, marker_color=colors,
                                 name="Kaynak hacmi"), row=2, col=1)
        if levels:
            low, high = float(candles.Low.min()), float(candles.High.max())
            margin = (high - low) * 0.5
            for name, price in levels.items():
                if low - margin <= price <= high + margin:
                    fig.add_hline(y=price, line_dash="dot", line_width=1, line_color=LEVEL_COLORS[name[0]],
                                  annotation_text=name, annotation_position="top left",
                                  annotation_font_color=LEVEL_COLORS[name[0]], row=1, col=1)
        fig.update_yaxes(side="right", row=1, col=1)
    fig.update_layout(height=340, showlegend=False, margin=dict(l=12, r=12, t=12, b=12),
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    return fig


QUOTE_SYMBOLS = ("EURUSD=X", "GBPUSD=X", "USDJPY=X", "XAUUSD=X")
QUOTE_NAMES = {"EURUSD=X": "Euro / ABD Doları", "GBPUSD=X": "İngiliz Sterlini / ABD Doları",
               "USDJPY=X": "ABD Doları / Japon Yeni", "XAUUSD=X": "Altın / ABD Doları"}
QUOTE_FLAGS = {"EURUSD=X": "EU", "GBPUSD=X": "GB", "USDJPY=X": "JP", "XAUUSD=X": "AU"}


def quote_snapshot(frame: pd.DataFrame) -> dict:
    """A quote and change against the last available point at least 24h earlier."""
    if frame.empty or "Close" not in frame:
        return {}
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna().sort_index()
    close = close[close.map(lambda v: math.isfinite(v) and v > 0)]
    if close.empty:
        return {}
    cutoff = close.index[-1] - pd.Timedelta(hours=24)
    previous = close[close.index <= cutoff]
    change = (float(close.iloc[-1] / previous.iloc[-1]) - 1) * 100 if not previous.empty else None
    window = close[close.index >= cutoff]
    sampled = window.iloc[::max(1, len(window) // 35)].tail(40).tolist()
    if sampled[-1] != float(close.iloc[-1]):
        sampled.append(float(close.iloc[-1]))
    return {"price": float(close.iloc[-1]), "change": change, "series": sampled,
            "time": close.index[-1], "low": float(window.min()), "high": float(window.max())}


def _sparkline(values: list, tone: str) -> str:
    if len(values) < 2:
        return '<div class="sefa-spark-empty">Veri bekleniyor</div>'
    low, high = min(values), max(values)
    extent = high - low or 1
    points = ' '.join(f'{i * 112 / (len(values)-1):.1f},{40 - (v-low) * 34 / extent:.1f}' for i, v in enumerate(values))
    return f'<svg class="sefa-spark {tone}" viewBox="0 0 116 46" aria-hidden="true"><polyline points="{points}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>'


def render_quote_strip(quotes: dict, provider: str) -> None:
    cards = []
    for symbol in QUOTE_SYMBOLS:
        quote = quotes.get(symbol, {})
        value, change = quote.get("price"), quote.get("change")
        tone = "neutral" if change is None else "up" if change >= 0 else "down"
        decimals = 2 if symbol in ("USDJPY=X", "XAUUSD=X") else 4
        price = f'{value:,.{decimals}f}' if value is not None else '—'
        pct = f'{change:+.2f}%' if change is not None else '24s değişim yok'
        when = quote.get("time")
        timestamp = when.tz_convert("Europe/Istanbul").strftime("%d.%m %H:%M") if when is not None and when.tzinfo is not None else 'Kaynakta veri yok'
        name = 'COMEX altın vadelisi' if symbol == 'XAUUSD=X' and provider == 'Yahoo Finance' else QUOTE_NAMES[symbol]
        cards.append(
            f'<article class="sefa-quote {tone}"><div class="sefa-quote-heading"><span class="sefa-flags">{QUOTE_FLAGS[symbol]}</span>'
            f'<div><strong>{symbol.replace("=X", "")}</strong><small>{escape(name)}</small></div></div>'
            f'<div class="sefa-quote-values"><div><b>{price}</b><span>{pct}</span></div>{_sparkline(quote.get("series", []), tone)}</div>'
            f'<footer>Son veri · {timestamp}</footer></article>'
        )
    st.markdown('<div id="sefa-markets" class="sefa-quote-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_recent_signals(history: pd.DataFrame) -> None:
    rows = []
    for _, row in history.head(5).iterrows():
        side = str(row.get('Yön', ''))
        tone = 'up' if side == 'LONG' else 'down' if side == 'SHORT' else 'neutral'
        label = 'AL' if side == 'LONG' else 'SAT' if side == 'SHORT' else 'BEKLE'
        try:
            payload = json.loads(str(row.get('Detay', '{}')))
        except (ValueError, TypeError):
            payload = {}
        levels = payload.get('levels') or {}
        entry = levels.get('entry')
        price = f'{float(entry):.4f}' if isinstance(entry, (int, float)) and math.isfinite(entry) else '—'
        when = pd.to_datetime(row.get('Zaman'), utc=True, errors='coerce')
        time = when.tz_convert('Europe/Istanbul').strftime('%d.%m %H:%M') if pd.notna(when) else '—'
        rows.append(f'<tr><td>{escape(str(row.get("Sembol", "")).replace("=X", ""))}</td>'
                    f'<td><span class="sefa-badge {tone}">{label}</span></td><td>{price}</td><td>{time}</td></tr>')
    empty = '<div class="sefa-empty"><span>◎</span><b>Henüz kayıtlı sinyal yok</b><p>Onaylanan sinyaller burada listelenir.<br>Piyasa taramasıyla analizleri yenileyebilirsiniz.</p></div>'
    table = '<table class="sefa-table"><thead><tr><th>Sembol</th><th>Sinyal</th><th>Giriş</th><th>Saat</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    st.markdown('<div class="sefa-signals"><div class="sefa-panel-heading"><h3>Son Sinyaller</h3><span>Kayıtlı alarmlar</span></div>'
                + (table if rows else empty) + '</div>', unsafe_allow_html=True)


def render_desk_overview(quotes: dict, events: pd.DataFrame | None, *, account_size: float, risk_pct: float) -> None:
    upcoming_rows = []
    if events is not None and not events.empty:
        times = pd.to_datetime(events['time'], utc=True, errors='coerce')
        upcoming = events.assign(_time=times).loc[times >= pd.Timestamp.now(tz='UTC')].sort_values('_time').head(5)
        for _, row in upcoming.iterrows():
            at = row['_time'].tz_convert('Europe/Istanbul')
            impact = int(row.get('impact', 0))
            upcoming_rows.append(f'<tr><td>{at:%d.%m}<br>{at:%H:%M}</td><td>{escape(str(row.get("currency", "")))}</td>'
                                 f'<td><span class="sefa-impact">{"●" * max(0, min(3, impact))}</span></td>'
                                 f'<td>{escape(str(row.get("event", "")))}</td><td>{escape(str(row.get("forecast", "")))}</td></tr>')
    calendar = '<table class="sefa-table sefa-calendar-mini"><thead><tr><th>Saat</th><th>Para</th><th>Önem</th><th>Veri</th><th>Beklenti</th></tr></thead><tbody>' + ''.join(upcoming_rows) + '</tbody></table>' if upcoming_rows else '<div class="sefa-empty-small">Yaklaşan etkinlik bulunmuyor veya takvim verisi alınamadı.</div>'
    market_rows = []
    for symbol, quote in sorted(quotes.items(), key=lambda item: item[1].get('change') or 0, reverse=True):
        change = quote.get('change')
        if change is None:
            continue
        tone = 'up' if change >= 0 else 'down'
        market_rows.append(f'<tr><td>{escape(symbol.replace("=X", ""))}</td><td class="{tone}">{change:+.2f}%</td>'
                           f'<td><div class="sefa-mini-track"><span class="{tone}" style="width:{min(100, abs(change)*100):.1f}%"></span></div></td></tr>')
    market = '<table class="sefa-table"><thead><tr><th>Sembol</th><th>24s değişim</th><th></th></tr></thead><tbody>' + ''.join(market_rows) + '</tbody></table>' if market_rows else '<div class="sefa-empty-small">Piyasa özeti için fiyat verisi bekleniyor.</div>'
    st.markdown(
        '<div class="sefa-overview-grid"><section class="sefa-panel"><div class="sefa-panel-heading"><h3>▣ Risk Yönetimi</h3></div>'
        f'<div class="sefa-risk-values"><div><small>Hesap bakiyesi</small><b>${account_size:,.0f}</b></div><div><small>İşlem riski</small><b>%{risk_pct:g}</b></div></div>'
        f'<div class="sefa-risk-total"><small>İşlem başına risk bütçesi</small><strong>${account_size*risk_pct/100:,.2f}</strong></div>'
        '<a href="#sefa-risk-tools" target="_self" class="sefa-inline-link">Lot ve işlem planını incele →</a></section>'
        '<section class="sefa-panel"><div class="sefa-panel-heading"><h3>▤ Ekonomik Takvim</h3><span>İstanbul saati</span></div>' + calendar + '</section>'
        '<section class="sefa-panel"><div class="sefa-panel-heading"><h3>▥ Piyasa Özeti</h3></div>' + market + '</section></div>',
        unsafe_allow_html=True,
    )
