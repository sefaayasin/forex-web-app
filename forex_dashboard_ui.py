"""Presentation helpers for the SEFA dashboard; no market or trading rules."""

from html import escape
from pathlib import Path

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


def render_brand() -> None:
    st.markdown(
        '<div class="sefa-brand"><svg viewBox="0 0 44 44" aria-hidden="true">'
        '<path d="M3 30h7v12H3zm12-9h7v21h-7zm12-9h7v30h-7z" fill="#00c9a7"/>'
        '<path d="M4 23 16 14l9 2L39 3m-9 0h9v9" fill="none" stroke="#00e2bf" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        '<div><strong>SEFA</strong><span>FOREX ANALYZER</span></div></div>'
        '<div class="sefa-sidebar-label">İŞLEM MASASI</div>',
        unsafe_allow_html=True,
    )


def render_dashboard_header() -> None:
    # Decorative vector artwork, deliberately separate from actual market charts.
    globe = '''<svg class="sefa-globe" viewBox="0 0 600 280" aria-hidden="true">
    <defs><radialGradient id="ocean"><stop stop-color="#07415b"/>
    <stop offset=".8" stop-color="#052131"/><stop offset="1" stop-color="#008cae"/></radialGradient>
    <linearGradient id="fade"><stop stop-color="#00d5ae" stop-opacity="0"/>
    <stop offset="1" stop-color="#00d5ae" stop-opacity=".8"/></linearGradient>
    <clipPath id="sphere"><circle cx="385" cy="215" r="180"/></clipPath></defs>
    <circle cx="385" cy="215" r="182" fill="#00bafa" opacity=".12"/>
    <circle cx="385" cy="215" r="180" fill="url(#ocean)"/>
    <g clip-path="url(#sphere)" fill="none" stroke="#0bb4d0" stroke-opacity=".25">
    <ellipse cx="385" cy="215" rx="75" ry="180"/><ellipse cx="385" cy="215" rx="140" ry="180"/>
    <ellipse cx="385" cy="215" rx="180" ry="48"/><ellipse cx="385" cy="215" rx="180" ry="110"/>
    <path d="M205 215h360M385 35v360M230 105l310 220M230 325l310-220"/>
    <path d="m290 74 30-10 24 12 19-3 18 20-14 20-25 6-5 22-29 9-12-18-25-12z
    m75 82 28-16 25 13 25 31-17 32-8 40-23 28-17-24 4-35-22-27z
    m49-87 27-8 31 13 25-2 31 25-20 22-29-4-15 20-30-5-9-24-19-10z"
    fill="#00c8c0" fill-opacity=".2" stroke-opacity=".5"/></g>
    <path d="m50 260 38-17 25 7 32-35 28 12 37-40 27 16 34-50 27 20 35-41 28 13 31-35 28 12 34-46 27 15 40-47 40 12"
    fill="none" stroke="url(#fade)" stroke-width="3"/>
    <g stroke="#00d6b0" fill="#00d6b0">
    <path d="M278 172v49m24-75v50m25-69v46m25-61v49m25-80v47m25-55v49m25-24v52m25-81v50m25-65v45m25-78v53m25-66v47"/>
    <path d="M273 185h10v23h-10zm24-26h10v25h-10zm25-20h10v20h-10zm25-15h10v23h-10zm25-31h10v23h-10zm25-9h10v24h-10zm25 19h10v28h-10zm25-34h10v25h-10zm25-14h10v21h-10zm25-34h10v31h-10zm25-13h10v22h-10z"/></g></svg>'''
    st.markdown(
        '<div class="sefa-topbar"><span>GENEL BAKIŞ <i>/</i> Piyasa çalışma alanı</span>'
        '<span class="sefa-topbar-badge">Çoklu zaman dilimi analizi</span></div>'
        '<section class="sefa-hero"><div class="sefa-hero-copy">'
        '<div class="sefa-eyebrow"><span></span> PİYASAYA DAHA GENİŞ BAKIN</div>'
        '<h1>Piyasaları Daha İyi<br><em>Anlayın.</em></h1>'
        '<p>Teknik analiz, yapay zekâ destekli araştırmalar ve risk araçları.<br>'
        'Bir sonraki adımınız için her şey tek bir yerde.</p>'
        '<div class="sefa-hero-tags"><span>4H / 1H ana yön</span><span>15M / 5M giriş takibi</span></div>'
        f'</div>{globe}</section>',
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


def market_candles(frame, symbol: str, interval: str) -> go.Figure:
    """Use the same source as the intraday view, including uploaded broker data."""
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
        fig.update_yaxes(side="right", row=1, col=1)
    fig.update_layout(height=340, showlegend=False, margin=dict(l=12, r=12, t=12, b=12),
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    return fig
