# Streamlit Forex Analyzer Pro (Decision Support + Backtest + Risk)
# ================================================================
# Bu uygulama yatırım tavsiyesi değildir. Kendi işlemlerinizde kullanmadan önce
# mutlaka demo hesapta ve farklı piyasa koşullarında test edin.
#
# Özellikler:
# - Çoklu zaman dilimi analizi: 4H / 1H / 15M / 5M
# - RSI, MACD, Bollinger Bands, EMA50/EMA200, ATR, Ichimoku
# - "Kesin Al/Sat" yerine bias yaklaşımı: Güçlü Alım Yönlü / Alım Yönlü / İşlem Yok / Satış Yönlü / Güçlü Satış Yönlü
# - 4H + 1H ana yön filtresi, 15M/5M giriş zamanlama filtresi
# - ATR tabanlı SL/TP, Risk/Reward ve yaklaşık lot hesabı
# - Multi-timeframe backtest: canlı sistemdeki 4H + 1H ana yön filtresi ve 15M/5M giriş teyidi ile uyumlu çalışır
# - Parite tarayıcı, seans filtresi, cooldown filtresi, long/short ayrı performans ve işlem günlüğü
# - Basit İşlem Modu, Pozisyon Takip Modu, pratik/güvenli filtre ve daha anlaşılır backtest onayı
# - Streamlit Cloud uyumlu: pandas yeni sürümlerde "4h" kullanılır, st.rerun() kullanılır

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
import json
import logging
import os
from pathlib import Path
import sqlite3
from tempfile import gettempdir
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from forex_decision_core import decide_mtf_signal

# Bazı Windows/sandbox kurulumlarında yfinance kullanıcı profilindeki SQLite
# cache'ine yazamaz. İzinli geçici dizin veri indirme hatasını önler.
YF_CACHE_DIR = Path(gettempdir()) / "forex_analyzer_yfinance_cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    brier_score_loss = None
    roc_auc_score = None
    SKLEARN_AVAILABLE = False

TR_TZ = pytz.timezone("Europe/Istanbul")
APP_DIR = Path(__file__).resolve().parent
APP_DB_PATH = APP_DIR / "forex_analyzer.db"
APP_LOG_PATH = APP_DIR / "forex_analyzer.log"

logging.basicConfig(
    filename=APP_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("forex_analyzer")

# =============================================================================
# UI CONFIG
# =============================================================================

st.set_page_config(page_title="Forex Analyzer Pro", layout="wide")

st.markdown(
    """
    <style>
        .small-muted { color:#6c757d; font-size:0.9rem; }
        .simple-card {
            padding: 20px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.16);
            margin: 14px 0 18px 0;
            box-shadow: 0 8px 28px rgba(0,0,0,0.18);
        }
        .simple-buy { background: #d1e7dd; color: #0f5132 !important; }
        .simple-sell { background: #f8d7da; color: #842029 !important; }
        .simple-wait { background: #fff3cd; color: #664d03 !important; }
        .simple-pass { background: #e9ecef; color: #212529 !important; }
        .simple-card.simple-buy, .simple-card.simple-buy * { color: #0f5132 !important; }
        .simple-card.simple-sell, .simple-card.simple-sell * { color: #842029 !important; }
        .simple-card.simple-wait, .simple-card.simple-wait * { color: #664d03 !important; }
        .simple-card.simple-pass, .simple-card.simple-pass * { color: #212529 !important; }
        .simple-card ol { margin: 6px 0 0 22px; padding: 0; }
        .simple-card li { margin-bottom: 4px; }
        .simple-action { font-size: 2.2rem; font-weight: 900; margin-bottom: 6px; }
        .simple-subtitle { font-size: 1.05rem; font-weight: 700; margin-bottom: 12px; }
        .simple-levels {
            display: grid;
            grid-template-columns: repeat(4, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 12px;
        }
        .simple-level {
            background: rgba(255,255,255,0.55);
            padding: 10px;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.08);
        }
        .simple-level b { display:block; font-size:0.84rem; opacity:0.75; margin-bottom:3px; }
        .simple-level span { font-size:1.18rem; font-weight:800; }
        .position-box {
            padding: 14px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.15);
            background: #f8f9fa;
            color: #212529 !important;
            font-weight: 500;
        }
        .position-box, .position-box * { color: #212529 !important; }
        .risk-box {
            padding: 14px;
            border-radius: 12px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            color: #212529 !important;
            font-weight: 500;
        }
        .risk-box, .risk-box * {
            color: #212529 !important;
        }
        .warn-box {
            padding: 12px;
            border-radius: 12px;
            background: #fff3cd;
            border: 1px solid #ffe69c;
            color: #664d03 !important;
            font-weight: 500;
        }
        .ok-box {
            padding: 12px;
            border-radius: 12px;
            background: #d1e7dd;
            border: 1px solid #badbcc;
            color: #0f5132 !important;
            font-weight: 500;
        }
        .bad-box {
            padding: 12px;
            border-radius: 12px;
            background: #f8d7da;
            border: 1px solid #f5c2c7;
            color: #842029 !important;
            font-weight: 500;
        }
        .warn-box b, .ok-box b, .bad-box b, .risk-box b {
            color: inherit !important;
        }
        .decision-shell {
            padding: 18px 20px;
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            background: #f8f9fa;
            margin: 10px 0 14px 0;
        }
        .decision-shell h2 {
            margin: 0 0 4px 0;
            font-size: 2.0rem;
            line-height: 1.08;
        }
        .decision-shell p { margin: 0; font-size: 1rem; }
        .decision-buy { background:#d1e7dd; }
        .decision-sell { background:#f8d7da; }
        .decision-wait { background:#fff3cd; }
        .decision-pass { background:#e9ecef; }
        .decision-buy, .decision-buy * { color:#0f5132 !important; }
        .decision-sell, .decision-sell * { color:#842029 !important; }
        .decision-wait, .decision-wait * { color:#664d03 !important; }
        .decision-pass, .decision-pass * { color:#212529 !important; }
        .check-grid {
            display:grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 10px;
            margin: 8px 0 18px 0;
        }
        .check-item {
            padding: 11px 12px;
            border-radius: 8px;
            background:#ffffff;
            border: 1px solid #e9ecef;
            color:#212529 !important;
            min-height: 70px;
        }
        .check-item b { display:block; font-size:0.86rem; margin-bottom:4px; }
        .check-item span { display:block; font-size:0.9rem; color:#495057 !important; }
        .check-item, .check-item * { color:#212529 !important; }
        .check-item span { color:#495057 !important; }
        .check-ok { border-color:#badbcc; background:#f0f8f4; }
        .check-warn { border-color:#ffe69c; background:#fff9e6; }
        .check-bad { border-color:#f5c2c7; background:#fff1f2; }
        .entry-signal-shell {
            padding: 16px 18px;
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.08);
            margin: 4px 0 12px 0;
        }
        .entry-signal-title {
            font-size: 0.86rem;
            font-weight: 800;
            opacity: 0.78;
            margin-bottom: 4px;
        }
        .entry-signal-action {
            font-size: 1.75rem;
            font-weight: 900;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .entry-signal-summary {
            font-size: 0.98rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .entry-signal-meta {
            display:grid;
            grid-template-columns: repeat(4, minmax(130px, 1fr));
            gap: 8px;
        }
        .entry-signal-meta div {
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 8px;
            padding: 9px 10px;
        }
        .entry-signal-meta b {
            display:block;
            font-size: 0.78rem;
            opacity: 0.72;
            margin-bottom: 3px;
        }
        .entry-signal-meta span {
            display:block;
            font-size: 0.95rem;
            font-weight: 800;
        }
        .entry-step-grid {
            display:grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 10px;
            margin: 8px 0 18px 0;
        }
        .entry-step {
            padding: 11px 12px;
            border-radius: 8px;
            background:#ffffff;
            border: 1px solid #e9ecef;
            color:#212529 !important;
            min-height: 74px;
        }
        .entry-step b { display:block; font-size:0.86rem; margin-bottom:4px; color:#212529 !important; }
        .entry-step span { display:block; font-size:0.9rem; color:#495057 !important; }
        .entry-step-ok { border-color:#badbcc; background:#f0f8f4; }
        .entry-step-warn { border-color:#ffe69c; background:#fff9e6; }
        .entry-step-bad { border-color:#f5c2c7; background:#fff1f2; }
        .entry-signal-buy { background:#d1e7dd; color:#0f5132 !important; }
        .entry-signal-sell { background:#f8d7da; color:#842029 !important; }
        .entry-signal-wait { background:#fff3cd; color:#664d03 !important; }
        .entry-signal-pass { background:#e9ecef; color:#212529 !important; }
        .entry-signal-shell.entry-signal-buy, .entry-signal-shell.entry-signal-buy * { color:#0f5132 !important; }
        .entry-signal-shell.entry-signal-sell, .entry-signal-shell.entry-signal-sell * { color:#842029 !important; }
        .entry-signal-shell.entry-signal-wait, .entry-signal-shell.entry-signal-wait * { color:#664d03 !important; }
        .entry-signal-shell.entry-signal-pass, .entry-signal-shell.entry-signal-pass * { color:#212529 !important; }
        .entry-signal-shell .entry-signal-meta div {
            background: rgba(255,255,255,0.72);
            color:#212529 !important;
        }
        .entry-signal-shell .entry-signal-meta b {
            color:#495057 !important;
        }
        .entry-signal-shell .entry-signal-meta span {
            color:#212529 !important;
        }
        .logic-note {
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid #ffe69c;
            background: #fff9e6;
            color: #664d03 !important;
            font-weight: 500;
            margin: 0 0 14px 0;
        }
        .logic-note, .logic-note * { color:#664d03 !important; }
        .logic-note b { font-weight: 900; }
        .logic-note-ok {
            border-color:#badbcc;
            background:#f0f8f4;
            color:#0f5132 !important;
        }
        .logic-note-ok, .logic-note-ok * { color:#0f5132 !important; }
        .logic-note-pass {
            border-color:#f5c2c7;
            background:#fff1f2;
            color:#842029 !important;
        }
        .logic-note-pass, .logic-note-pass * { color:#842029 !important; }
        .action-row {
            display:flex;
            gap:10px;
            align-items:stretch;
            flex-wrap:wrap;
            margin: 4px 0 14px 0;
        }

        .alert-section-title {
            font-size:1.35rem;
            font-weight:900;
            margin: 20px 0 10px 0;
        }
        .alert-summary-row {
            display:grid;
            grid-template-columns: repeat(4, minmax(160px, 1fr));
            gap:12px;
            margin: 12px 0 18px 0;
        }
        .alert-summary-box {
            padding: 14px 16px;
            border-radius: 14px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            color:#212529 !important;
        }
        .alert-summary-box, .alert-summary-box * { color:#212529 !important; }
        .alert-summary-box b { display:block; font-size:0.85rem; opacity:.72; }
        .alert-summary-box span { display:block; font-size:1.35rem; font-weight:950; margin-top:2px; }
        .alert-card {
            min-height: 190px;
            padding: 14px;
            border-radius: 16px;
            border: 1px solid rgba(0,0,0,.10);
            margin-bottom: 12px;
            box-shadow: 0 8px 22px rgba(0,0,0,.12);
        }
        .alert-card, .alert-card * { color: #212529 !important; }
        .alert-buy { background: #d1e7dd; border-color:#badbcc; }
        .alert-sell { background: #f8d7da; border-color:#f5c2c7; }
        .alert-wait { background: #fff3cd; border-color:#ffe69c; }
        .alert-card-header {
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:10px;
            margin-bottom: 10px;
        }
        .alert-symbol { font-size:1.35rem; font-weight:950; }
        .alert-status { font-size:1.15rem; font-weight:950; }
        .alert-status-buy { color:#0f5132 !important; }
        .alert-status-sell { color:#842029 !important; }
        .alert-status-wait { color:#664d03 !important; }
        .alert-mini-grid {
            display:grid;
            grid-template-columns: repeat(4, 1fr);
            gap:6px;
            margin: 8px 0 10px 0;
        }
        .alert-mini {
            padding:7px 5px;
            border-radius:10px;
            background: rgba(255,255,255,.62);
            border: 1px solid rgba(0,0,0,.08);
            text-align:center;
            font-size:.78rem;
            font-weight:800;
        }
        .alert-reason { font-size:.88rem; font-weight:650; min-height:42px; }
        .alert-decision {
            margin-top:8px;
            padding:8px 10px;
            border-radius:12px;
            background: rgba(255,255,255,.70);
            border: 1px solid rgba(0,0,0,.08);
            font-size:.88rem;
            font-weight:900;
        }
        .alert-decision small {
            display:block;
            font-size:.76rem;
            font-weight:700;
            opacity:.80;
            margin-top:2px;
        }
        .alert-meta {
            display:flex;
            justify-content:space-between;
            gap:8px;
            margin-top:10px;
            font-size:.82rem;
            font-weight:800;
            opacity:.9;
        }
        @media (max-width: 1100px) {
            .alert-summary-row { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
        }

        @media (max-width: 900px) {
            .simple-levels, .check-grid, .entry-signal-meta, .entry-step-grid {
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }
        }
        @media (max-width: 520px) {
            .simple-levels, .check-grid, .entry-signal-meta, .entry-step-grid {
                grid-template-columns: 1fr;
            }
            .simple-action, .decision-shell h2 {
                font-size: 1.55rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SYMBOLS
# =============================================================================

SYMBOL_LIST = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURJPY=X", "EURNZD=X",
    "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDNZD=X",
    "CADCHF=X", "CADJPY=X", "CHFJPY=X",
    "NZDCAD=X", "NZDCHF=X", "NZDJPY=X",
    "EURZAR=X",
]


MAJOR_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]

MINOR_PAIRS = [s for s in SYMBOL_LIST if s not in MAJOR_PAIRS]

ALERT_PAIR_GROUPS = {
    "Major": MAJOR_PAIRS,
    "Minör": MINOR_PAIRS,
}

TIMEFRAMES = {
    "4 Saat": {"interval": "4h", "period": "60d", "weight": 4},
    "1 Saat": {"interval": "60m", "period": "30d", "weight": 3},
    "15 Dakika": {"interval": "15m", "period": "10d", "weight": 2},
    "5 Dakika": {"interval": "5m", "period": "5d", "weight": 1},
}

BACKTEST_PERIODS = {
    "5 Dakika": "5d",
    "15 Dakika": "30d",
    "1 Saat": "90d",
    "4 Saat": "120d",
}

PRICE_CHANGE_WINDOWS = {
    "Son 5 dakika": 5,
    "Son 10 dakika": 10,
    "Son 15 dakika": 15,
    "Son 30 dakika": 30,
    "Son 1 saat": 60,
    "Son 4 saat": 240,
    "Son 1 gün": 1440,
}

TRADING_SESSIONS = {
    "Tüm Gün": None,
    # Saatler ilgili merkezin yerel saatidir. pytz yaz/kış saatini otomatik uygular.
    "Asya": {"timezone": "Asia/Tokyo", "start": 9, "end": 18},
    "Londra": {"timezone": "Europe/London", "start": 8, "end": 17},
    "New York": {"timezone": "America/New_York", "start": 8, "end": 17},
    "Londra + New York Kesişimi": "OVERLAP",
}

# =============================================================================
# HELPERS
# =============================================================================

BIAS_LABELS = [
    "Güçlü Alım Yönlü",
    "Alım Yönlü",
    "İşlem Yok",
    "Satış Yönlü",
    "Güçlü Satış Yönlü",
]

BIAS_TO_SCORE = {
    "Güçlü Alım Yönlü": 2,
    "Alım Yönlü": 1,
    "İşlem Yok": 0,
    "Satış Yönlü": -1,
    "Güçlü Satış Yönlü": -2,
}


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper().replace("/", "")
    if symbol and not symbol.endswith("=X") and len(symbol) == 6:
        symbol += "=X"
    return symbol


def symbol_pair(symbol: str) -> tuple[str, str]:
    s = symbol.upper().replace("=X", "").replace("/", "").replace("-", "")
    if len(s) >= 6:
        return s[:3], s[-3:]
    return s[:3], s[-3:]


def get_pip_size(symbol: str) -> float:
    base, quote = symbol_pair(symbol)
    if quote == "JPY":
        return 0.01
    if base in {"XAU", "XAG"}:
        return 0.1
    if base in {"BTC", "ETH"} or quote in {"BTC", "ETH"}:
        return 1.0
    return 0.0001


def price_decimals(symbol: str) -> int:
    base, quote = symbol_pair(symbol)
    if quote == "JPY":
        return 3
    return 5


def estimate_pip_value_per_lot_usd(symbol: str, reference_price: Optional[float] = None) -> Optional[float]:
    """Standart 100k lot için yaklaşık USD pip değeri üretir."""
    base, quote = symbol_pair(symbol)
    contract_units = 100_000
    quote_pip_value = get_pip_size(symbol) * contract_units

    if quote == "USD":
        return float(quote_pip_value)

    if base == "USD" and reference_price and reference_price > 0:
        return float(quote_pip_value / reference_price)

    direct = fetch_last_price(f"{quote}USD=X")
    if direct and direct > 0:
        return float(quote_pip_value * direct)

    inverse = fetch_last_price(f"USD{quote}=X")
    if inverse and inverse > 0:
        return float(quote_pip_value / inverse)

    return None


def to_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TR_TZ)
    else:
 …58601 tokens truncated…reshold_pct), ml_filter_enabled)

if is_new_position_decision(str(simple_decision.get("action", ""))):
    alert_payload = {
        "symbol": symbol,
        "action": simple_decision.get("action"),
        "reason": simple_decision.get("reason"),
        "levels": simple_decision.get("levels", {}),
        "candle_time": entry_signal_tracker.get("last_closed_time", "-"),
    }
    is_new_alert = record_alert_once(
        symbol, str(entry_signal_tracker.get("side", "SIGNAL")), str(entry_signal_tracker.get("last_closed_time", "-")), alert_payload
    )
    if is_new_alert and webhook_url.strip():
        ok, notification_text = send_webhook_notification(webhook_url, alert_payload)
        if not ok:
            st.warning(f"Webhook gönderilemedi: {notification_text}")

st.header("Tek Karar")
health_cols = st.columns(4)
health_cols[0].metric("Veri Sağlığı", current_data_health.get("status", "-"))
health_cols[1].metric("Piyasa Rejimi", market_regime.get("label", "-"))
health_cols[2].metric("Açık Risk", f"%{current_portfolio_status.get('total_risk_pct', 0):.2f}")
health_cols[3].metric("Haber Filtresi", "BLOK" if current_news_status.get("blocks_trade") else "AÇIK")
for title, status in [("Veri", current_data_health), ("Haber", current_news_status), ("Portföy", current_portfolio_status)]:
    if status.get("blocks_trade"):
        st.warning(f"{title}: {status.get('text', '-')}")
render_top_decision_panel(simple_decision)
render_market_model_card(market_model_status)
render_ml_prediction_card(ml_prediction, float(ml_threshold_pct), ml_filter_enabled)
if beginner_mode:
    render_simple_decision_card(simple_decision)
    render_beginner_path(summary_df, matched_quality, entry_signal_tracker, selected_tf)
    with st.expander("Neden böyle dedi?", expanded=False):
        render_signal_summary_card(simple_decision, entry_signal_tracker, market_regime, signal_mode)
        render_entry_alarm_box(entry_signal_tracker)
        render_entry_signal_tracker(entry_signal_tracker)
else:
    render_signal_summary_card(simple_decision, entry_signal_tracker, market_regime, signal_mode)
    render_wait_reason_box(simple_decision, entry_signal_tracker)
    render_entry_alarm_box(entry_signal_tracker)
    render_direction_trade_explanation(
        final_label=final_label,
        final_score=final_score,
        selected_tf=selected_tf,
        simple_decision=simple_decision,
        matched_quality=matched_quality,
        allowed_quality_labels=allowed_quality_labels,
        entry_signal_tracker=entry_signal_tracker,
    )
    render_readiness_checklist(
        build_readiness_items(
            summary=summary_df,
            selected_tf=selected_tf,
            bt_tf=bt_tf,
            matched_quality=matched_quality,
            allowed_quality_labels=allowed_quality_labels,
            setup=preview_setup,
            practical_signal_mode=practical_signal_mode,
            signal_mode=signal_mode,
        )
    )
    render_entry_signal_tracker(entry_signal_tracker)

action_col, quality_col, risk_col = st.columns([1.2, 1.0, 1.0])
with action_col:
    main_run_bt_requested = st.button(
        "Yeniden Hesapla",
        type="primary",
        use_container_width=True,
        disabled=bt_tf != selected_tf,
        key="main_run_bt",
    )
with quality_col:
    st.metric("Strateji Kalitesi", current_quality_info.get("label", "Bekliyor"))
with risk_col:
    st.metric("İşlem Riski", f"{account_size * (risk_pct / 100):.2f}")

if main_run_bt_requested:
    run_and_store_backtest()
    st.rerun()

if not beginner_mode:
    with st.expander("Kararın adımları", expanded=False):
        render_simple_decision_card(simple_decision)

if "scanner_df" in st.session_state and isinstance(st.session_state["scanner_df"], pd.DataFrame):
    with st.expander("Parite Tarayıcı Sonuçları", expanded=False):
        scanner_view = st.session_state["scanner_df"].copy()
        st.dataframe(scanner_view, use_container_width=True, height=360)
        st.download_button(
            "Tarayıcı Sonucunu CSV İndir",
            data=scanner_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="forex_pair_scanner.csv",
            mime="text/csv",
        )
        st.caption("5 Dakika verisi Yahoo tarafında kısa geçmiş sunduğu için bazı paritelerde örnek sayısı yetersiz kalabilir.")

if show_position_tracker:
    st.header("Pozisyon Takip Modu")
    with st.expander("Açık pozisyonumu takip et", expanded=False):
        default_side_tracker = preview_setup.side if preview_setup is not None else ("LONG" if "Alım" in final_label else "SHORT")
        default_entry_tracker = float(preview_setup.entry) if preview_setup is not None else (float(price) if price is not None else 0.0)
        default_stop_tracker = float(preview_setup.stop) if preview_setup is not None else 0.0
        default_target_tracker = float(preview_setup.target) if preview_setup is not None else 0.0
        dec_tracker = price_decimals(symbol)
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        with pc1:
            pos_side = st.selectbox("Pozisyon Yönü", ["LONG", "SHORT"], index=0 if default_side_tracker == "LONG" else 1, key="pos_side")
        with pc2:
            pos_entry = st.number_input("Giriş fiyatım", min_value=0.0, value=float(default_entry_tracker), step=get_pip_size(symbol), format=f"%.{dec_tracker}f", key="pos_entry")
        with pc3:
            pos_lot = st.number_input("Lot", min_value=0.0, value=float(preview_setup.estimated_lot) if preview_setup else 0.0, step=0.01, key="pos_lot")
        with pc4:
            pos_stop = st.number_input("Stop", min_value=0.0, value=float(default_stop_tracker), step=get_pip_size(symbol), format=f"%.{dec_tracker}f", key="pos_stop")
        with pc5:
            pos_target = st.number_input("Kâr Al", min_value=0.0, value=float(default_target_tracker), step=get_pip_size(symbol), format=f"%.{dec_tracker}f", key="pos_target")
        tracker_result = build_position_tracker_result(
            symbol=symbol,
            side=pos_side,
            entry=float(pos_entry),
            current_price=price,
            stop=float(pos_stop),
            target=float(pos_target),
            lot=float(pos_lot),
            pip_value_per_lot=float(pip_value_per_lot),
            final_label=final_label,
            momentum_model=market_model_status,
        )
        render_position_tracker_result(tracker_result, price, symbol)

left_col, right_col = st.columns([2.2, 1.0])

with left_col:
    fig, chart_df = plot_main_figure(symbol, chart_tf)
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Piyasa Yönü")
    st.plotly_chart(gauge_figure(final_label, final_score), use_container_width=True)
    st.caption("Bu gösterge sadece yön gücüdür; LONG/SHORT kararı için İşlem Kararı ve Canlı Giriş Takibi geçmeli.")
    regime_css = "ok-box" if market_regime.get("state") == "ok" else "warn-box"
    st.markdown(
        f"<div class='{regime_css}'><b>Piyasa Tipi: {market_regime.get('label', '-')}</b><br>{market_regime.get('text', '-')}</div>",
        unsafe_allow_html=True,
    )

    if final_label == "İşlem Yok":
        st.markdown(f"<div class='warn-box'><b>Yön: {final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)
    elif "Alım" in final_label:
        st.markdown(f"<div class='ok-box'><b>Yön: {final_label}</b><br>{filter_note}<br><br>Bu tek başına işlem açma onayı değildir.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bad-box'><b>Yön: {final_label}</b><br>{filter_note}<br><br>Bu tek başına işlem açma onayı değildir.</div>", unsafe_allow_html=True)

    st.subheader("Risk Planı")

    # Risk Planı, yukarıda hesaplanan aynı sembol + aynı giriş zaman dilimi backtest kalitesine bağlıdır.

    if bt_tf != selected_tf:
        st.markdown(
            "<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
            "Risk planını onaylamak için Backtest zaman dilimi ile Grafik/Giriş zaman dilimi aynı olmalı.</div>",
            unsafe_allow_html=True,
        )
    elif matched_quality is None:
        if current_quality_info["status"] == "preview":
            st.markdown(
                "<div class='warn-box'><b>Ön Risk Planı</b><br>"
                "Backtest onayı yok; bu seviyeler yalnızca ön izleme/demo takibi içindir.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
                "Bu sembol ve giriş zaman dilimi için önce sidebar üzerinden 'Yeniden Hesapla' butonuna bas.</div>",
                unsafe_allow_html=True,
            )
    elif ml_filter_enabled and ml_blocks_trade:
        st.markdown(
            f"<div class='bad-box'><b>Risk Planı Kilitli — ML Filtresi</b><br>{escape(str(ml_block_reason))}</div>",
            unsafe_allow_html=True,
        )
    elif current_quality_info["status"] == "blocked":
        strict_note = "Güvenli mod açık olduğu için daha yüksek kalite gerekir." if (strict_safety_mode or signal_mode == "Güvenli Sinyal") else "İşlem için en az Orta kalite gerekir."
        st.markdown(
            f"<div class='{matched_quality['css']}'><b>PAS GEÇ — Strateji Kalitesi: {matched_quality['label']}</b><br>"
            f"{matched_quality['text']}<br>{strict_note}</div>",
            unsafe_allow_html=True,
        )
    elif (strict_safety_mode or signal_mode == "Güvenli Sinyal") and final_label not in {"Güçlü Alım Yönlü", "Güçlü Satış Yönlü"}:
        st.markdown(
            f"<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
            f"{signal_mode} için yönün Güçlü Alım veya Güçlü Satış olması gerekir. Mevcut yön: {final_label}</div>",
            unsafe_allow_html=True,
        )
    else:
        quality_css = "ok-box" if current_quality_info["status"] == "approved" else "warn-box"
        quality_title = "Backtest Onayı" if current_quality_info["status"] == "approved" else "Düşük Güvenli Plan"
        quality_text = matched_quality["text"] if matched_quality else current_quality_info["text"]
        st.markdown(
            f"<div class='{quality_css}'><b>{quality_title}: {current_quality_info['label']}</b><br>"
            f"{quality_text}</div>",
            unsafe_allow_html=True,
        )
        setup = build_trade_setup(
            symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult,
            pip_value_per_lot, spread_pips, entry_price=price, stop_mode=stop_mode,
            target_mode=target_mode, swing_lookback=int(swing_lookback),
        )
        if setup is None:
            st.info("Şu anda işlem planı üretmiyorum. Ana yön veya veri koşulları yeterli değil.")
        else:
            dec = price_decimals(symbol)
            st.markdown(
                f"""
                <div class='risk-box'>
                <b>Yön:</b> {setup.side}<br>
                <b>İşlem Tipi:</b> {setup.action}<br>
                <b>Entry:</b> {setup.entry:.{dec}f}<br>
                <b>SL:</b> {setup.stop:.{dec}f} ({setup.stop_pips:.1f} pip)<br>
                <b>TP:</b> {setup.target:.{dec}f} ({setup.target_pips:.1f} pip)<br>
                <b>RR:</b> {setup.rr:.2f}<br>
                <b>Risk:</b> {setup.risk_amount:.2f}<br>
                <b>Yaklaşık Lot:</b> {setup.estimated_lot:.2f}<br><br>
                <b>Aktivasyon:</b> {setup.activation_rule}<br>
                <b>Teyit:</b> {setup.confirmation_rule}<br>
                <b>İptal:</b> {setup.invalidation_rule}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(setup.note)

with st.expander("Teknik Detaylar", expanded=False):
    st.subheader("Çoklu Zaman Dilimi Karar Tablosu")
    st.dataframe(summary_df, use_container_width=True, height=190)
    st.subheader("Skor Detayı")
    st.dataframe(detail_df, use_container_width=True)
    st.subheader("Giriş Tetikleyici Paneli")
    st.plotly_chart(plot_live_trigger(symbol, chart_tf, final_label), use_container_width=True)

st.divider()
st.header("Backtest")
st.caption("Bu MTF backtest, canlı sistemle aynı ana mantığı kullanır: 4H + 1H yön filtresi, 5M için 15M teyidi, sinyal barı kapandıktan sonra sonraki bar açılışı. Aynı mumda hem TP hem SL görülürse muhafazakâr olarak SL kabul edilir.")

saved_bt = st.session_state.get("last_bt_result")
saved_bt_key = st.session_state.get("last_bt_key")
saved_quality = st.session_state.get("last_bt_quality")

if saved_bt is not None and saved_bt_key == current_bt_key:
    bt = saved_bt
    c1, c2 = st.columns([1.0, 2.0])
    with c1:
        st.subheader("Performans")
        st.dataframe(bt.metrics, use_container_width=True, hide_index=True)
        if saved_quality:
            st.markdown(
                f"<div class='{saved_quality['css']}'><b>Strateji Kalitesi: {saved_quality['label']}</b><br>{saved_quality['text']}</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.plotly_chart(plot_equity_curve(bt.equity), use_container_width=True)

    if not bt.trades.empty:
        s1, s2 = st.columns([1.2, 1.0])
        with s1:
            st.subheader("Long / Short Ayrı Performans")
            st.dataframe(side_performance_table(bt.trades), use_container_width=True, hide_index=True)
        with s2:
            st.subheader("İşlem Süresi Özeti")
            st.dataframe(trade_duration_table(bt.trades), use_container_width=True, hide_index=True)
        wf_report = st.session_state.get("last_wf_report")
        if walk_forward_enabled:
            st.subheader("Walk-forward Dönem Kararlılığı")
            if isinstance(wf_report, pd.DataFrame) and not wf_report.empty:
                st.dataframe(wf_report, use_container_width=True, hide_index=True)
            else:
                st.warning("Walk-forward dönemleri için yeterli işlem oluşmadı.")
        st.subheader("Monte Carlo Risk")
        monte_carlo_report = monte_carlo_risk_report(bt.trades)
        if monte_carlo_report.empty:
            st.info("Monte Carlo için en az 10 işlem gerekli.")
        else:
            st.dataframe(monte_carlo_report, use_container_width=True, hide_index=True)

    st.subheader("İşlem Listesi")
    if bt.trades.empty:
        st.info("Bu ayarlarla işlem oluşmadı veya yeterli veri yok.")
    else:
        view = bt.trades.copy()
        for col in ["Entry", "Exit", "SL", "TP"]:
            view[col] = view[col].astype(float).round(price_decimals(symbol))
        for col in ["Pips", "PnL", "Balance", "Lot", "Risk Amount", "Entry Score", "4H Score", "1H Score", "15M Score"]:
            view[col] = view[col].astype(float).round(2)
        st.dataframe(view.tail(100), use_container_width=True, height=360)
else:
    st.info("Backtest sonuçlarını görmek ve Risk Planı'nı kalite kontrolüne bağlamak için otomatik plan kontrolünü aç veya 'Yeniden Hesapla' butonuna bas.")

st.divider()
st.header("İşlem Günlüğü")
st.caption(f"Günlük SQLite ile kalıcı tutulur: {APP_DB_PATH.name}")

init_trade_journal()
journal_setup = build_trade_setup(
    symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult,
    pip_value_per_lot, spread_pips, entry_price=price, stop_mode=stop_mode,
    target_mode=target_mode, swing_lookback=int(swing_lookback),
)
default_side = journal_setup.side if journal_setup is not None else ("LONG" if "Alım" in final_label else "SHORT")
default_entry = float(journal_setup.entry) if journal_setup is not None else (float(price) if price is not None else 0.0)
default_sl = float(journal_setup.stop) if journal_setup is not None else 0.0
default_tp = float(journal_setup.target) if journal_setup is not None else 0.0

with st.form("trade_journal_form"):
    j1, j2, j3, j4 = st.columns(4)
    with j1:
        journal_side = st.selectbox("Yön", ["LONG", "SHORT"], index=0 if default_side == "LONG" else 1)
    with j2:
        journal_result = st.selectbox("Sonuç", ["Açık", "TP", "SL", "Manuel Kâr", "Manuel Zarar", "İptal"], index=0)
    with j3:
        journal_entry = st.number_input("Gerçek Entry", min_value=0.0, value=float(default_entry), step=get_pip_size(symbol), format="%.5f")
    with j4:
        journal_exit = st.number_input("Gerçek Exit", min_value=0.0, value=0.0, step=get_pip_size(symbol), format="%.5f")

    j5, j6, j7 = st.columns(3)
    with j5:
        journal_lot = st.number_input("Gerçek Lot", min_value=0.0, value=float(journal_setup.estimated_lot) if journal_setup else 0.0, step=0.01)
    with j6:
        journal_sl = st.number_input("Plan SL", min_value=0.0, value=float(default_sl), step=get_pip_size(symbol), format="%.5f")
    with j7:
        journal_tp = st.number_input("Plan TP", min_value=0.0, value=float(default_tp), step=get_pip_size(symbol), format="%.5f")

    journal_notes = st.text_area("Not", value="")
    submit_journal = st.form_submit_button("Günlüğe Ekle")

    if submit_journal:
        manual_pips = calculate_manual_pips(symbol, journal_side, journal_entry, journal_exit) if journal_exit > 0 else None
        stop_pips_for_r = abs(journal_entry - journal_sl) / get_pip_size(symbol) if journal_sl > 0 else None
        realized_r = (
            float(manual_pips) / float(stop_pips_for_r)
            if manual_pips is not None and stop_pips_for_r is not None and stop_pips_for_r > 0
            else None
        )
        add_trade_journal_entry({
            "Tarih": datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "Sembol": symbol,
            "Zaman Dilimi": selected_tf,
            "Genel Bias": final_label,
            "Skor": round(float(final_score), 2),
            "Backtest Kalitesi": matched_quality["label"] if "matched_quality" in locals() and matched_quality else "-",
            "Yön": journal_side,
            "Sonuç": journal_result,
            "Plan Entry": round(float(default_entry), price_decimals(symbol)) if default_entry else None,
            "Gerçek Entry": journal_entry,
            "Gerçek Exit": journal_exit if journal_exit > 0 else None,
            "Plan SL": journal_sl if journal_sl > 0 else None,
            "Plan TP": journal_tp if journal_tp > 0 else None,
            "Lot": journal_lot,
            "Risk %": float(risk_pct),
            "Risk Tutarı": float(account_size * risk_pct / 100),
            "Pips": None if manual_pips is None else round(float(manual_pips), 2),
            "R": None if realized_r is None else round(float(realized_r), 3),
            "Not": journal_notes,
        })
        st.success("İşlem günlüğe eklendi.")

journal_df = journal_dataframe()
if journal_df.empty:
    st.info("Henüz işlem günlüğü kaydı yok.")
else:
    st.dataframe(journal_df.head(100), use_container_width=True, height=300)
    st.download_button(
        "İşlem Günlüğünü CSV İndir",
        data=journal_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="forex_trade_journal.csv",
        mime="text/csv",
    )
    if st.button("İşlem Günlüğünü Temizle"):
        clear_trade_journal()
        st.rerun()

with st.expander("Alarm Geçmişi", expanded=False):
    alert_history = alert_history_dataframe()
    if alert_history.empty:
        st.info("Henüz tekilleştirilmiş alarm kaydı yok.")
    else:
        st.dataframe(alert_history, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    """
    **Kullanım Notu:** Bu sistem emir vermek için değil, karar disiplinini korumak için tasarlanmıştır.
    Yeni Başlayan Modu açıksa 4H ve 1H sadece ana yönü belirler, 15M giriş zamanıdır, 5M karar verici olarak gösterilmez.
    Ekrandaki tek karar kartı LONG / SHORT / BEKLE / PAS GEÇ sonucunu verir; teknik detaylar yalnızca kontrol amaçlıdır.
    """
)
