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
        df.index = df.index.tz_convert(TR_TZ)
    return df


def _fix_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).title() for c in df.columns]
    else:
        df.columns = [str(c).title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spreadpoints"] if c in df.columns]
    df = df[keep_cols]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def _to_istanbul_timestamp(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(TR_TZ)


def _to_utc_timestamp(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _is_in_local_session(ts, session: dict) -> bool:
    local_time = _to_utc_timestamp(ts).tz_convert(pytz.timezone(session["timezone"]))
    local_hour = local_time.hour + local_time.minute / 60
    start_hour = float(session["start"])
    end_hour = float(session["end"])
    if start_hour < end_hour:
        return start_hour <= local_hour < end_hour
    return local_hour >= start_hour or local_hour < end_hour


def is_in_trading_session(ts, session_name: str) -> bool:
    """Seansı merkezin yerel saatinde değerlendirir; DST kaymasını otomatik düzeltir."""
    session = TRADING_SESSIONS.get(session_name)
    if session is None:
        return True
    if session == "OVERLAP":
        return _is_in_local_session(ts, TRADING_SESSIONS["Londra"]) and _is_in_local_session(
            ts, TRADING_SESSIONS["New York"]
        )
    return _is_in_local_session(ts, session)


def session_description(session_name: str) -> str:
    session = TRADING_SESSIONS.get(session_name)
    if session is None:
        return "Tüm gün aktif"
    now = pd.Timestamp.now(tz="UTC")
    day_start = now.normalize()
    active_hours = [
        hour for hour in range(24)
        if is_in_trading_session(day_start + pd.Timedelta(hours=hour), session_name)
    ]
    if not active_hours:
        return "Bugün kesişim saati yok"
    istanbul_hours = [
        _to_istanbul_timestamp(day_start + pd.Timedelta(hours=hour)).hour for hour in active_hours
    ]
    start_hour = istanbul_hours[0]
    end_hour = (istanbul_hours[-1] + 1) % 24
    return f"Bugün İstanbul saatine göre yaklaşık {start_hour:02d}:00–{end_hour:02d}:00 (DST otomatik)"


def recommended_spread_pips(symbol: str) -> float:
    """Canlı spread yokken yalnızca muhafazakâr bir başlangıç maliyeti önerir."""
    s = normalize_symbol(symbol)
    if s in MAJOR_PAIRS:
        return 1.0
    if s == "EURZAR=X":
        return 25.0
    return 2.0


def observed_broker_spread_pips(symbol: str, tf_name: str) -> Optional[float]:
    if st.session_state.get("data_provider") != "MetaTrader 5":
        return None
    prm = TIMEFRAMES[tf_name]
    df = _fetch_ohlc_mt5(symbol, prm["interval"], prm["period"])
    if df.empty or "Spreadpoints" not in df.columns:
        return None
    points = pd.to_numeric(df["Spreadpoints"], errors="coerce").tail(100).median()
    if pd.isna(points) or points < 0:
        return None
    return float(points) / 10.0

# =============================================================================
# DATA
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_ohlc_yahoo(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo Finance verisini çeker. 4h için 60m veriyi güvenli şekilde resample eder."""
    symbol = normalize_symbol(symbol)
    interval_l = interval.lower()

    if interval_l in {"4h", "4hr", "4hour"}:
        base = yf.download(symbol, interval="60m", period=period, progress=False, auto_adjust=False, threads=False)
        if base is None or base.empty:
            return pd.DataFrame()
        base = _fix_cols(base)
        if base.empty:
            return pd.DataFrame()

        # Pandas yeni sürümlerde büyük H kabul etmeyebilir; bu yüzden "4h" kullanıyoruz.
        out = (
            base.resample("4h", label="right", closed="right")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna(subset=["Open", "High", "Low", "Close"])
        )
        return out

    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return _fix_cols(df)


def _mt5_timeframe(interval: str):
    try:
        import MetaTrader5 as mt5
    except Exception:
        return None, None
    mapping = {
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "60m": mt5.TIMEFRAME_H1,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
    }
    return mt5, mapping.get(interval.lower())


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_ohlc_mt5(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Bağlı MetaTrader 5 terminalinden broker mumlarını alır; bağlantı yoksa boş döner."""
    mt5, timeframe = _mt5_timeframe(interval)
    if mt5 is None or timeframe is None:
        return pd.DataFrame()
    mt5_symbol = normalize_symbol(symbol).replace("=X", "")
    try:
        if not mt5.initialize():
            return pd.DataFrame()
        period_days = {"5d": 5, "10d": 10, "30d": 30, "60d": 60, "90d": 90, "120d": 120}.get(str(period), 30)
        start = datetime.now(tz=pytz.UTC) - timedelta(days=period_days)
        rates = mt5.copy_rates_from(mt5_symbol, timeframe, datetime.now(tz=pytz.UTC), 50_000)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        out = pd.DataFrame(rates)
        out["time"] = pd.to_datetime(out["time"], unit="s", utc=True)
        out = out[out["time"] >= pd.Timestamp(start)]
        out = out.set_index("time").rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume", "spread": "SpreadPoints"})
        return _fix_cols(out)
    except Exception as exc:
        LOGGER.exception("MT5 OHLC error for %s: %s", symbol, exc)
        return pd.DataFrame()
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    provider = st.session_state.get("data_provider", "Yahoo Finance")
    if provider == "MetaTrader 5":
        broker_df = _fetch_ohlc_mt5(symbol, interval, period)
        if not broker_df.empty:
            return broker_df
    return _fetch_ohlc_yahoo(symbol, interval, period)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_last_price(symbol: str) -> Optional[float]:
    symbol = normalize_symbol(symbol)
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None
        df = _fix_cols(df)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_price_change(symbol: str, lookback_minutes: int) -> Optional[dict]:
    """
    Sembol seçildiği andan değil, kullanıcının seçtiği geçmiş pencereye göre yüzde değişim hesaplar.
    Örn: Son 5 dakika, Son 1 saat.
    """
    symbol = normalize_symbol(symbol)
    try:
        # 1m veri Yahoo tarafında genelde son birkaç gün için erişilebilir.
        period = "5d" if lookback_minutes > 1440 else "2d"
        df = yf.download(symbol, period=period, interval="1m", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return None

        df = _fix_cols(df)
        if df.empty or len(df) < 2:
            return None

        df = _utc_index_df(df)
        close = df["Close"].astype(float).dropna()
        if close.empty:
            return None

        latest_time = close.index[-1]
        latest_price = float(close.iloc[-1])
        target_time = latest_time - pd.Timedelta(minutes=int(lookback_minutes))

        ref_candidates = close[close.index <= target_time]
        if ref_candidates.empty:
            return {
                "latest": latest_price,
                "reference": None,
                "pct": None,
                "latest_time": latest_time,
                "reference_time": None,
            }

        reference_time = ref_candidates.index[-1]
        reference_price = float(ref_candidates.iloc[-1])

        if reference_price == 0:
            pct = None
        else:
            pct = 100 * (latest_price - reference_price) / reference_price

        return {
            "latest": latest_price,
            "reference": reference_price,
            "pct": pct,
            "latest_time": latest_time,
            "reference_time": reference_time,
        }
    except Exception:
        return None


def data_health_status(symbol: str, tf_name: str) -> dict:
    """Verinin güncelliğini ve temel mum bütünlüğünü ölçer."""
    prm = TIMEFRAMES[tf_name]
    df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if df is None or df.empty:
        return {"status": "ERROR", "state": "bad", "text": "Fiyat verisi alınamadı.", "blocks_trade": True}
    utc_df = _utc_index_df(df)
    latest = utc_df.index[-1]
    now = pd.Timestamp.now(tz="UTC")
    age_minutes = max((now - latest).total_seconds() / 60, 0.0)
    expected = {"5 Dakika": 5, "15 Dakika": 15, "1 Saat": 60, "4 Saat": 240}.get(tf_name, 60)
    weekend = now.weekday() >= 5
    stale_limit = expected * 3 + 5
    stale = age_minutes > stale_limit and not weekend
    duplicate_count = int(utc_df.index.duplicated().sum())
    bad_ohlc = int(((utc_df["High"] < utc_df[["Open", "Close"]].max(axis=1)) | (utc_df["Low"] > utc_df[["Open", "Close"]].min(axis=1))).sum())
    if stale or duplicate_count or bad_ohlc:
        reasons = []
        if stale:
            reasons.append(f"veri {age_minutes:.0f} dk eski")
        if duplicate_count:
            reasons.append(f"{duplicate_count} tekrar zaman damgası")
        if bad_ohlc:
            reasons.append(f"{bad_ohlc} bozuk OHLC")
        return {"status": "STALE", "state": "bad", "text": ", ".join(reasons), "blocks_trade": True, "latest": latest}
    market_note = "Piyasa hafta sonu kapalı" if weekend else "Veri güncel"
    return {
        "status": "OK",
        "state": "ok",
        "text": f"{market_note}; son mum {age_minutes:.0f} dk önce.",
        "blocks_trade": bool(weekend),
        "latest": latest,
    }

# =============================================================================
# INDICATORS
# =============================================================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder RSI: klasik rolling ortalamaya göre daha stabil sinyal verir.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bbands(close: pd.Series, period: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    out["Tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["Kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["SpanA"] = ((out["Tenkan"] + out["Kijun"]) / 2).shift(26)
    out["SpanB"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"].astype(float)
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()
    out["RSI14"] = compute_rsi(close)
    out["MACD"], out["MACDSignal"], out["MACDHist"] = compute_macd(close)
    out["BBLow"], out["BBMid"], out["BBUp"] = compute_bbands(close)
    out["ATR14"] = compute_atr(out)
    out = compute_ichimoku(out)
    return out


def calculate_stop_target_distances(
    history: pd.DataFrame,
    entry: float,
    side: str,
    atr: float,
    atr_mult: float,
    rr: float,
    stop_mode: str = "ATR",
    target_mode: str = "Sabit R",
    swing_lookback: int = 10,
) -> tuple[float, float]:
    """Canlı ve backtest tarafından paylaşılan ATR/yapısal stop-hedef hesabı."""
    atr_distance = max(float(atr) * float(atr_mult), np.finfo(float).eps)
    recent = history.tail(max(int(swing_lookback), 2)) if history is not None else pd.DataFrame()
    structural_distance = atr_distance
    if not recent.empty:
        if side == "LONG":
            swing = float(recent["Low"].min())
            structural_distance = max(entry - (swing - atr * 0.15), atr * 0.5)
        else:
            swing = float(recent["High"].max())
            structural_distance = max((swing + atr * 0.15) - entry, atr * 0.5)
    if stop_mode == "Swing + ATR":
        stop_distance = structural_distance
    elif stop_mode == "Hibrit (uzak olan)":
        stop_distance = max(atr_distance, structural_distance)
    else:
        stop_distance = atr_distance

    target_distance = stop_distance * float(rr)
    if target_mode == "Yapı / minimum 1R" and not recent.empty:
        opposite_range = (
            float(recent["High"].max()) - entry
            if side == "LONG"
            else entry - float(recent["Low"].min())
        )
        target_distance = max(stop_distance, opposite_range)
    return float(stop_distance), float(target_distance)

# =============================================================================
# DECISION ENGINE
# =============================================================================

@dataclass
class BiasResult:
    label: str
    score: float
    trend_score: float
    momentum_score: float
    volatility_score: float
    explanation: str


def label_from_score(score: float) -> str:
    if score >= 65:
        return "Güçlü Alım Yönlü"
    if score >= 25:
        return "Alım Yönlü"
    if score <= -65:
        return "Güçlü Satış Yönlü"
    if score <= -25:
        return "Satış Yönlü"
    return "İşlem Yok"


def latest_valid_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    needed = ["Close", "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "MACDSignal", "MACDHist", "BBLow", "BBMid", "BBUp", "ATR14"]
    valid = df.dropna(subset=[c for c in needed if c in df.columns])
    if valid.empty:
        return None
    return valid.iloc[-1]


def evaluate_bias(df: pd.DataFrame) -> BiasResult:
    """Tek bar değil, trend + momentum + volatilite yapısına göre bias üretir."""
    if df.empty or len(df) < 60:
        return BiasResult("İşlem Yok", 0, 0, 0, 0, "Yeterli veri yok")

    ind = add_indicators(df)
    row = latest_valid_row(ind)
    if row is None:
        return BiasResult("İşlem Yok", 0, 0, 0, 0, "İndikatörler için yeterli veri yok")

    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    rsi = float(row["RSI14"])
    macd = float(row["MACD"])
    sig = float(row["MACDSignal"])
    hist = float(row["MACDHist"])
    bb_low = float(row["BBLow"])
    bb_mid = float(row["BBMid"])
    bb_up = float(row["BBUp"])

    # EMA eğimleri
    ema50_slope = float(ind["EMA50"].iloc[-1] - ind["EMA50"].iloc[-8]) if len(ind) > 8 else 0.0
    ema200_slope = float(ind["EMA200"].iloc[-1] - ind["EMA200"].iloc[-20]) if len(ind) > 20 else 0.0
    hist_delta = float(ind["MACDHist"].iloc[-1] - ind["MACDHist"].iloc[-4]) if len(ind) > 4 else 0.0

    trend_score = 0.0
    reasons: list[str] = []

    # Trend skoru
    if close > ema200:
        trend_score += 20
        reasons.append("Fiyat EMA200 üzerinde")
    else:
        trend_score -= 20
        reasons.append("Fiyat EMA200 altında")

    if ema50 > ema200:
        trend_score += 20
        reasons.append("EMA50 > EMA200")
    else:
        trend_score -= 20
        reasons.append("EMA50 < EMA200")

    if close > ema20 > ema50:
        trend_score += 15
        reasons.append("Kısa vadeli trend yukarı")
    elif close < ema20 < ema50:
        trend_score -= 15
        reasons.append("Kısa vadeli trend aşağı")

    if ema50_slope > 0 and ema200_slope >= 0:
        trend_score += 10
        reasons.append("EMA eğimleri pozitif")
    elif ema50_slope < 0 and ema200_slope <= 0:
        trend_score -= 10
        reasons.append("EMA eğimleri negatif")

    # Momentum skoru
    momentum_score = 0.0
    if 52 <= rsi <= 68:
        momentum_score += 20
        reasons.append("RSI alım momentumunda")
    elif rsi > 75:
        momentum_score += 5
        reasons.append("RSI çok yüksek, momentum var ama geri çekilme riski yüksek")
    elif 32 <= rsi <= 48:
        momentum_score -= 20
        reasons.append("RSI satış momentumunda")
    elif rsi < 25:
        momentum_score -= 5
        reasons.append("RSI çok düşük, satış baskısı var ama tepki riski yüksek")

    if macd > sig and hist > 0:
        momentum_score += 20
        reasons.append("MACD pozitif")
    elif macd < sig and hist < 0:
        momentum_score -= 20
        reasons.append("MACD negatif")

    if hist_delta > 0:
        momentum_score += 8
        reasons.append("MACD histogram güçleniyor")
    elif hist_delta < 0:
        momentum_score -= 8
        reasons.append("MACD histogram zayıflıyor")

    # Bollinger/volatilite skoru: tek başına LONG/SHORT değil, pozisyon kalitesi filtresi.
    volatility_score = 0.0
    if bb_up > bb_low:
        band_pos = (close - bb_low) / (bb_up - bb_low)
        if close > bb_mid and 0.45 <= band_pos <= 0.90:
            volatility_score += 10
            reasons.append("Fiyat BB orta band üstünde, üst banda aşırı yapışmamış")
        elif close < bb_mid and 0.10 <= band_pos <= 0.55:
            volatility_score -= 10
            reasons.append("Fiyat BB orta band altında, alt banda aşırı yapışmamış")
        elif band_pos > 0.95:
            volatility_score += 3
            reasons.append("Fiyat üst banda çok yakın, alımda takip riski var")
        elif band_pos < 0.05:
            volatility_score -= 3
            reasons.append("Fiyat alt banda çok yakın, satışta takip riski var")

    raw_score = trend_score + momentum_score + volatility_score
    score = float(np.clip(raw_score, -100, 100))
    label = label_from_score(score)
    explanation = "; ".join(reasons[:5])
    return BiasResult(label, score, trend_score, momentum_score, volatility_score, explanation)


def analyse_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []
    for tf_name, prm in TIMEFRAMES.items():
        df = fetch_ohlc(symbol, prm["interval"], prm["period"])
        if df.empty:
            rows.append([tf_name, "Veri yok", 0, "-"])
            continue

        # Kapanmamış son barı kullanmamak daha güvenli.
        if len(df) > 2:
            df_eval = df.iloc[:-1]
        else:
            df_eval = df

        result = evaluate_bias(df_eval)
        rows.append([tf_name, result.label, round(result.score, 1), result.explanation])
        detail_rows.append([
            tf_name,
            round(result.trend_score, 1),
            round(result.momentum_score, 1),
            round(result.volatility_score, 1),
            result.label,
        ])

    summary = pd.DataFrame(rows, columns=["Zaman Dilimi", "Bias", "Skor", "Açıklama"])
    detail = pd.DataFrame(detail_rows, columns=["Zaman Dilimi", "Trend", "Momentum", "Bollinger/Volatilite", "Sonuç"])
    return summary, detail


def global_bias(summary: pd.DataFrame, selected_tf: Optional[str] = None) -> tuple[str, float, str]:
    if summary.empty:
        return "İşlem Yok", 0.0, "Veri yok"

    total_weight = 0
    weighted_score = 0.0
    labels_by_tf = {}
    scores_by_tf = {}
    for _, row in summary.iterrows():
        tf = row["Zaman Dilimi"]
        label = row["Bias"]
        score = float(row["Skor"]) if pd.notna(row["Skor"]) else 0.0
        w = TIMEFRAMES.get(tf, {}).get("weight", 1)
        weighted_score += score * w
        total_weight += w
        labels_by_tf[tf] = label
        scores_by_tf[tf] = score

    final_score = weighted_score / total_weight if total_weight else 0.0
    final_label = label_from_score(final_score)

    # Ana yön filtresi: backtest ile aynı şekilde 4H ve 1H aynı yönde olmalı.
    h4 = labels_by_tf.get("4 Saat", "İşlem Yok")
    h1 = labels_by_tf.get("1 Saat", "İşlem Yok")
    h4_score = scores_by_tf.get("4 Saat", np.nan)
    h1_score = scores_by_tf.get("1 Saat", np.nan)
    m15_score = scores_by_tf.get("15 Dakika", np.nan)
    long_set = {"Alım Yönlü", "Güçlü Alım Yönlü"}
    short_set = {"Satış Yönlü", "Güçlü Satış Yönlü"}
    htf_long = not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score >= 25 and h1_score >= 25
    htf_short = not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score <= -25 and h1_score <= -25

    if (h4 in long_set and h1 in short_set) or (h4 in short_set and h1 in long_set):
        return "İşlem Yok", final_score, "4H ve 1H yönleri çelişiyor. İşlem filtresi devrede."

    if not htf_long and not htf_short:
        return "İşlem Yok", final_score, "4H ve 1H aynı yönde yeterli skor üretmiyor."

    if selected_tf == "5 Dakika":
        if htf_long and (pd.isna(m15_score) or m15_score < 25):
            return "İşlem Yok", final_score, "5M giriş için 15M alım yönü teyidi yok."
        if htf_short and (pd.isna(m15_score) or m15_score > -25):
            return "İşlem Yok", final_score, "5M giriş için 15M satış yönü teyidi yok."

    if htf_long and final_label in long_set:
        return final_label, final_score, "4H + 1H ana yön filtresi alım tarafında uygun."

    if htf_short and final_label in short_set:
        return final_label, final_score, "4H + 1H ana yön filtresi satış tarafında uygun."

    return "İşlem Yok", final_score, "Ağırlıklı skor, ana yön filtresiyle aynı yönde yeterli sinyal üretmiyor."

# =============================================================================
# TRADE SETUP / RISK
# =============================================================================

@dataclass
class TradeSetup:
    side: str
    entry: float
    stop: float
    target: float
    stop_pips: float
    target_pips: float
    rr: float
    risk_amount: float
    estimated_lot: float
    action: str
    activation_rule: str
    confirmation_rule: str
    invalidation_rule: str
    note: str


def build_trade_setup(
    symbol: str,
    selected_tf: str,
    global_label: str,
    account_size: float,
    risk_pct: float,
    rr: float,
    atr_mult: float,
    pip_value_per_lot: float,
    total_cost_pips: float = 0.0,
    entry_price: Optional[float] = None,
    stop_mode: str = "ATR",
    target_mode: str = "Sabit R",
    swing_lookback: int = 10,
) -> Optional[TradeSetup]:
    long_labels = {"Alım Yönlü", "Güçlü Alım Yönlü"}
    short_labels = {"Satış Yönlü", "Güçlü Satış Yönlü"}

    if global_label not in long_labels.union(short_labels):
        return None

    prm = TIMEFRAMES[selected_tf]
    df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if df.empty or len(df) < 80:
        return None

    # Kapanmış son bar üzerinden plan üret.
    df = add_indicators(df.iloc[:-1])
    row = latest_valid_row(df)
    if row is None:
        return None

    closed_bar_entry = float(row["Close"])
    entry = (
        float(entry_price)
        if entry_price is not None and np.isfinite(float(entry_price)) and float(entry_price) > 0
        else closed_bar_entry
    )
    atr = float(row["ATR14"])
    pip = get_pip_size(symbol)

    if not np.isfinite(atr) or atr <= 0:
        return None

    side = "LONG" if global_label in long_labels else "SHORT"
    stop_distance, target_distance = calculate_stop_target_distances(
        df, entry, side, atr, atr_mult, rr, stop_mode, target_mode, swing_lookback
    )
    stop_pips = stop_distance / pip
    target_pips = target_distance / pip

    if side == "LONG":
        stop = entry - stop_distance
        target = entry + target_distance
        action = "Breakout / trend devamı bekle"
        activation_rule = f"4H+1H long uyumu korunur ve kapanmış {selected_tf} giriş skoru eşiği geçerse plan aktif sayılır."
        confirmation_rule = "4H ve 1H alım yönünde kalmalı; 15M ters satışa dönerse bekle; 5M alım yönüne dönerse giriş kalitesi artar."
        invalidation_rule = f"Fiyat {stop:.{price_decimals(symbol)}f} altına iner veya 1H Satış/İşlem Yok'a dönerse plan iptal."
    else:
        stop = entry + stop_distance
        target = entry - target_distance
        action = "Breakdown / trend devamı bekle"
        activation_rule = f"4H+1H short uyumu korunur ve kapanmış {selected_tf} giriş skoru eşiği geçerse plan aktif sayılır."
        confirmation_rule = "4H ve 1H satış yönünde kalmalı; 15M ters alıma dönerse bekle; 5M satış yönüne dönerse giriş kalitesi artar."
        invalidation_rule = f"Fiyat {stop:.{price_decimals(symbol)}f} üstüne çıkar veya 1H Alım/İşlem Yok'a dönerse plan iptal."

    risk_amount = account_size * (risk_pct / 100)
    risk_pips = stop_pips + max(float(total_cost_pips), 0.0)
    if risk_pips <= 0 or pip_value_per_lot <= 0:
        estimated_lot = 0.0
    else:
        estimated_lot = risk_amount / (risk_pips * pip_value_per_lot)

    return TradeSetup(
        side=side,
        entry=entry,
        stop=stop,
        target=target,
        stop_pips=stop_pips,
        target_pips=target_pips,
        rr=rr,
        risk_amount=risk_amount,
        estimated_lot=estimated_lot,
        action=action,
        activation_rule=activation_rule,
        confirmation_rule=confirmation_rule,
        invalidation_rule=invalidation_rule,
        note="Lot hesabı stop mesafesi ve girilen toplam işlem maliyetini içerir; broker dolumu yine farklı olabilir.",
    )

# =============================================================================
# BACKTEST
# =============================================================================

@dataclass
class BacktestResult:
    metrics: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def _utc_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """Backtest hizalaması için tüm verileri UTC indeksine çeker."""
    if df.empty:
        return df
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def _utc_index_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    out = s.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def signal_from_score(score: float, threshold: float) -> str:
    if pd.isna(score):
        return "NONE"
    if score >= threshold:
        return "LONG"
    if score <= -threshold:
        return "SHORT"
    return "NONE"


def score_series_for_backtest(df: pd.DataFrame) -> pd.Series:
    """Backtest için vektörel yaklaşık skor üretir. evaluate_bias ile aynı felsefeyi kullanır."""
    ind = add_indicators(df)
    close = ind["Close"].astype(float)

    score = pd.Series(0.0, index=ind.index)

    # Trend
    score += np.where(close > ind["EMA200"], 20, -20)
    score += np.where(ind["EMA50"] > ind["EMA200"], 20, -20)
    score += np.where((close > ind["EMA20"]) & (ind["EMA20"] > ind["EMA50"]), 15, 0)
    score += np.where((close < ind["EMA20"]) & (ind["EMA20"] < ind["EMA50"]), -15, 0)

    ema50_slope = ind["EMA50"] - ind["EMA50"].shift(8)
    ema200_slope = ind["EMA200"] - ind["EMA200"].shift(20)
    score += np.where((ema50_slope > 0) & (ema200_slope >= 0), 10, 0)
    score += np.where((ema50_slope < 0) & (ema200_slope <= 0), -10, 0)

    # Momentum
    rsi = ind["RSI14"]
    score += np.where((rsi >= 52) & (rsi <= 68), 20, 0)
    score += np.where(rsi > 75, 5, 0)
    score += np.where((rsi >= 32) & (rsi <= 48), -20, 0)
    score += np.where(rsi < 25, -5, 0)

    score += np.where((ind["MACD"] > ind["MACDSignal"]) & (ind["MACDHist"] > 0), 20, 0)
    score += np.where((ind["MACD"] < ind["MACDSignal"]) & (ind["MACDHist"] < 0), -20, 0)

    hist_delta = ind["MACDHist"] - ind["MACDHist"].shift(4)
    score += np.where(hist_delta > 0, 8, 0)
    score += np.where(hist_delta < 0, -8, 0)

    # Bollinger
    denom = (ind["BBUp"] - ind["BBLow"]).replace(0, np.nan)
    band_pos = (close - ind["BBLow"]) / denom
    score += np.where((close > ind["BBMid"]) & (band_pos >= 0.45) & (band_pos <= 0.90), 10, 0)
    score += np.where((close < ind["BBMid"]) & (band_pos >= 0.10) & (band_pos <= 0.55), -10, 0)
    score += np.where(band_pos > 0.95, 3, 0)
    score += np.where(band_pos < 0.05, -3, 0)

    score = score.clip(-100, 100)
    return _utc_index_series(score.dropna())


def _filter_period_for_tf(tf_name: str, fallback_period: str) -> str:
    """
    Giriş zaman dilimi kısa olsa bile 4H/1H/15M filtre skorları daha uzun veriyle hesaplanır.
    Böylece EMA200 ve trend skorları 5M backtestte sadece birkaç günlük veriye sıkışmaz.
    """
    return {
        "4 Saat": "120d",
        "1 Saat": "90d",
        "15 Dakika": "30d",
    }.get(tf_name, fallback_period)


def _fetch_score_for_tf(symbol: str, tf_name: str, period: str, shift_closed_bar: bool = True) -> pd.Series:
    """Her zaman dilimi için skor üretir. shift_closed_bar=True, üst zaman diliminde ileri bakışı engeller."""
    prm = TIMEFRAMES[tf_name]
    df = fetch_ohlc(symbol, prm["interval"], period)
    if df.empty:
        return pd.Series(dtype=float)
    df = _utc_index_df(df)
    score = score_series_for_backtest(df)
    if shift_closed_bar:
        score = score.shift(1).dropna()
    return score


def _aligned_value(series: Optional[pd.Series], ts) -> float:
    if series is None or series.empty:
        return np.nan
    try:
        value = series.loc[ts]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return float(value)
    except Exception:
        return np.nan


def mtf_signal_decision(
    entry_score: float,
    h4_score: float,
    h1_score: float,
    m15_score: float,
    tf_name: str,
    threshold: float,
) -> tuple[str, str]:
    """Canlı ekran ve backtest için ortak, test edilebilir MTF karar kuralı."""
    return decide_mtf_signal(entry_score, h4_score, h1_score, m15_score, tf_name, threshold)


def run_backtest(
    symbol: str,
    tf_name: str,
    period: str,
    initial_balance: float,
    risk_pct: float,
    rr: float,
    atr_mult: float,
    signal_threshold: float,
    spread_pips: float,
    pip_value_per_lot: float,
    cooldown_bars: int = 0,
    session_filter: str = "Tüm Gün",
    max_same_direction_trades: int = 3,
    min_trades_required: int = 20,
    stop_mode: str = "ATR",
    target_mode: str = "Sabit R",
    swing_lookback: int = 10,
    max_holding_bars: int = 0,
    break_even_at_r: float = 0.0,
) -> BacktestResult:
    """
    Multi-timeframe backtest.
    - 4H ve 1H aynı yönde değilse işlem açmaz.
    - 5M giriş için ayrıca 15M yön teyidi ister.
    - Sinyal barı kapandıktan sonra sonraki bar açılışından giriş yapar.
    """
    prm = TIMEFRAMES[tf_name]
    raw = fetch_ohlc(symbol, prm["interval"], period)
    if raw.empty or len(raw) < 120:
        empty_metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["Yeterli veri yok"]})
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame())

    df = _utc_index_df(raw)
    df = add_indicators(df)
    entry_score_series = score_series_for_backtest(df)
    df = df.join(entry_score_series.rename("Score"), how="left")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "ATR14", "Score"])

    if len(df) < 80:
        empty_metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["İndikatörler sonrası yeterli veri yok"]})
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame())

    # Üst zaman dilimi skorlarını giriş zaman dilimine hizala.
    aligned_scores: dict[str, pd.Series] = {}
    for tf in ["4 Saat", "1 Saat", "15 Dakika"]:
        if tf == tf_name:
            continue
        score_period = _filter_period_for_tf(tf, period)
        score = _fetch_score_for_tf(symbol, tf, score_period, shift_closed_bar=True)
        if score.empty:
            aligned_scores[tf] = pd.Series(index=df.index, dtype=float)
        else:
            aligned_scores[tf] = score.reindex(df.index, method="ffill")

    pip = get_pip_size(symbol)
    balance = initial_balance
    equity_rows = []
    trades = []
    open_trade = None
    last_exit_i = -10**9
    last_entry_i = -10**9
    last_entry_side: Optional[str] = None
    same_direction_entries = 0

    def resolve_bar_exit(trade: dict, bar: pd.Series) -> tuple[Optional[str], Optional[float]]:
        side = trade["Side"]
        stop = trade["Stop"]
        target = trade["Target"]
        bar_open = float(bar["Open"])

        if side == "LONG":
            # Hafta sonu/haber boşluğunda stop seviyesinden daha kötü açılışı hesaba kat.
            if bar_open <= stop:
                return "SL-GAP", bar_open
            hit_stop = float(bar["Low"]) <= stop
            hit_target = float(bar["High"]) >= target
        else:
            if bar_open >= stop:
                return "SL-GAP", bar_open
            hit_stop = float(bar["High"]) >= stop
            hit_target = float(bar["Low"]) <= target

        if hit_stop and hit_target:
            return "SL", stop  # OHLC içinde sıra bilinmediği için muhafazakar kabul.
        if hit_stop:
            return "SL", stop
        if hit_target:
            return "TP", target
        return None, None

    def close_trade(trade: dict, exit_price: float, exit_time, exit_reason: str) -> None:
        nonlocal balance

        side = trade["Side"]
        entry = trade["Entry"]
        lot = trade["Lot"]
        risk_amount = trade["RiskAmount"]

        if side == "LONG":
            pnl_pips = (exit_price - entry) / pip
        else:
            pnl_pips = (entry - exit_price) / pip

        pnl_pips -= spread_pips
        pnl = pnl_pips * pip_value_per_lot * lot
        balance += pnl

        trades.append({
            "Entry Time": trade["EntryTime"],
            "Exit Time": exit_time,
            "Side": side,
            "Entry": entry,
            "Exit": exit_price,
            "SL": trade["Stop"],
            "TP": trade["Target"],
            "Pips": pnl_pips,
            "PnL": pnl,
            "Balance": balance,
            "Result": exit_reason,
            "Lot": lot,
            "Risk Amount": risk_amount,
            "Entry Score": trade["EntryScore"],
            "4H Score": trade["H4Score"],
            "1H Score": trade["H1Score"],
            "15M Score": trade["M15Score"],
            "MTF Reason": trade["Reason"],
        })

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        ts = df.index[i]
        prev_ts = df.index[i - 1]
        closed_this_bar = False

        if open_trade is not None:
            exit_reason, exit_price = resolve_bar_exit(open_trade, current)
            if exit_reason is None and max_holding_bars > 0 and (i - open_trade["EntryIndex"]) >= int(max_holding_bars):
                exit_reason, exit_price = "TIME", float(current["Open"])
            if exit_reason is not None:
                close_trade(open_trade, float(exit_price), ts, exit_reason)
                last_exit_i = i
                open_trade = None
                closed_this_bar = True
            elif break_even_at_r > 0 and not open_trade.get("BreakEvenMoved", False):
                trigger_distance = open_trade["InitialRiskDistance"] * float(break_even_at_r)
                reached = (
                    float(current["High"]) >= open_trade["Entry"] + trigger_distance
                    if open_trade["Side"] == "LONG"
                    else float(current["Low"]) <= open_trade["Entry"] - trigger_distance
                )
                if reached:
                    # Aynı mum içi sıra bilinmediğinden yeni stop bir sonraki mumdan itibaren geçerlidir.
                    open_trade["Stop"] = open_trade["Entry"]
                    open_trade["BreakEvenMoved"] = True

        if open_trade is None and not closed_this_bar:
            entry_score = float(previous["Score"])

            h4_score = entry_score if tf_name == "4 Saat" else _aligned_value(aligned_scores.get("4 Saat"), prev_ts)
            h1_score = entry_score if tf_name == "1 Saat" else _aligned_value(aligned_scores.get("1 Saat"), prev_ts)
            m15_score = entry_score if tf_name == "15 Dakika" else _aligned_value(aligned_scores.get("15 Dakika"), prev_ts)

            sig, reason = mtf_signal_decision(entry_score, h4_score, h1_score, m15_score, tf_name, signal_threshold)

            if sig != "NONE" and not is_in_trading_session(ts, session_filter):
                sig = "NONE"
                reason = f"Seans filtresi dışında: {session_filter}"

            if sig != "NONE" and cooldown_bars > 0 and (i - last_exit_i) <= cooldown_bars:
                sig = "NONE"
                reason = f"Cooldown filtresi: son işlemden sonra {cooldown_bars} mum bekleniyor"

            # Aynı yön filtresi: uzun süre sonra gelen yeni setup yeni trend dalgası kabul edilir.
            if sig != "NONE" and last_entry_side == sig and (i - last_entry_i) > max(cooldown_bars * 3, 20):
                same_direction_entries = 0

            if sig != "NONE" and last_entry_side == sig and same_direction_entries >= max_same_direction_trades:
                sig = "NONE"
                reason = f"Tekrar sinyal filtresi: aynı yönde maksimum {max_same_direction_trades} işlem sınırı"

            if sig != "NONE":
                atr = float(previous["ATR14"])
                entry = float(current["Open"])
                stop_distance, target_distance = calculate_stop_target_distances(
                    df.iloc[max(0, i - int(swing_lookback)):i],
                    entry,
                    sig,
                    atr,
                    atr_mult,
                    rr,
                    stop_mode,
                    target_mode,
                    swing_lookback,
                )
                stop_pips = stop_distance / pip
                risk_amount = balance * (risk_pct / 100)
                # Stop + tahmini işlem maliyeti birlikte seçilen risk yüzdesini aşmasın.
                risk_pips = stop_pips + max(float(spread_pips), 0.0)
                lot = risk_amount / (risk_pips * pip_value_per_lot) if risk_pips > 0 and pip_value_per_lot > 0 else 0.0

                if lot > 0 and np.isfinite(lot):
                    if sig == "LONG":
                        stop = entry - stop_distance
                        target = entry + target_distance
                    else:
                        stop = entry + stop_distance
                        target = entry - target_distance

                    candidate_trade = {
                        "EntryTime": ts,
                        "EntryIndex": i,
                        "InitialRiskDistance": stop_distance,
                        "BreakEvenMoved": False,
                        "Side": sig,
                        "Entry": entry,
                        "Stop": stop,
                        "Target": target,
                        "Lot": lot,
                        "RiskAmount": risk_amount,
                        "EntryScore": entry_score,
                        "H4Score": h4_score,
                        "H1Score": h1_score,
                        "M15Score": m15_score,
                        "Reason": reason,
                    }
                    if last_entry_side == sig:
                        same_direction_entries += 1
                    else:
                        last_entry_side = sig
                        same_direction_entries = 1
                    last_entry_i = i

                    # İşlem current bar açılışından girildiği için aynı bar içinde
                    # SL/TP görülürse sonucu bu mumda kapatmak gerekir.
                    immediate_reason, immediate_price = resolve_bar_exit(candidate_trade, current)
                    if immediate_reason is not None:
                        close_trade(candidate_trade, float(immediate_price), ts, immediate_reason)
                        last_exit_i = i
                        open_trade = None
                    else:
                        open_trade = candidate_trade

        equity_rows.append({"Time": ts, "Balance": balance})

    if open_trade is not None and not df.empty:
        final_ts = df.index[-1]
        final_close = float(df.iloc[-1]["Close"])
        close_trade(open_trade, final_close, final_ts, "EOD")
        open_trade = None
        if equity_rows:
            equity_rows[-1]["Balance"] = balance
        else:
            equity_rows.append({"Time": final_ts, "Balance": balance})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if trades_df.empty:
        metrics = pd.DataFrame({"Metrik": ["İşlem Sayısı", "Not"], "Değer": ["0", "MTF filtrelerle bu periyotta işlem oluşmadı"]})
        return BacktestResult(metrics, trades_df, equity_df)

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]
    total_pnl = trades_df["PnL"].sum()
    win_rate = 100 * len(wins) / len(trades_df)
    loss_sum = abs(losses["PnL"].sum()) if not losses.empty else 0.0
    profit_factor = wins["PnL"].sum() / loss_sum if loss_sum > 0 else np.nan
    r_multiples = trades_df["PnL"] / trades_df["Risk Amount"].replace(0, np.nan)
    avg_r = float(r_multiples.mean()) if r_multiples.notna().any() else np.nan
    std_r = float(r_multiples.std(ddof=1)) if r_multiples.notna().sum() > 1 else np.nan
    trade_sharpe = avg_r / std_r if pd.notna(std_r) and std_r > 0 else np.nan
    win_ci_low, win_ci_high = wilson_win_rate_interval(len(wins), len(trades_df))

    # Son %30, ayrı bir tarih bölümü olarak raporlanır. Parametre seçimi için kullanılmamalıdır.
    oos_start = max(1, int(len(trades_df) * 0.70))
    oos = trades_df.iloc[oos_start:].copy()
    oos_wins = oos[oos["PnL"] > 0]
    oos_losses = oos[oos["PnL"] <= 0]
    oos_loss_sum = abs(oos_losses["PnL"].sum()) if not oos_losses.empty else 0.0
    oos_pf = oos_wins["PnL"].sum() / oos_loss_sum if oos_loss_sum > 0 else np.nan
    oos_r = oos["PnL"] / oos["Risk Amount"].replace(0, np.nan)
    oos_avg_r = float(oos_r.mean()) if oos_r.notna().any() else np.nan

    if not equity_df.empty:
        eq = equity_df["Balance"]
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min())
        max_dd_idx = dd.idxmin()
        peak_at_dd = float(peak.loc[max_dd_idx]) if len(peak) > 0 else initial_balance
        max_dd_pct = 100 * max_dd / peak_at_dd if peak_at_dd else 0
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    metrics = pd.DataFrame([
        ["Backtest Tipi", "MTF filtreli"],
        ["İşlem Sayısı", len(trades_df)],
        ["Kazanan İşlem", len(wins)],
        ["Kaybeden İşlem", len(losses)],
        ["Win Rate", f"{win_rate:.2f}%"],
        ["Toplam PnL", f"{total_pnl:.2f}"],
        ["Son Bakiye", f"{balance:.2f}"],
        ["Profit Factor", "-" if pd.isna(profit_factor) else f"{profit_factor:.2f}"],
        ["Maks. Drawdown", f"{max_dd:.2f} ({max_dd_pct:.2f}%)"],
        ["Ortalama Pips", f"{trades_df['Pips'].mean():.2f}"],
        ["Ortalama R", "-" if pd.isna(avg_r) else f"{avg_r:.3f}R"],
        ["İşlem Bazlı Sharpe", "-" if pd.isna(trade_sharpe) else f"{trade_sharpe:.2f}"],
        ["Win Rate %95 GA", f"%{100 * win_ci_low:.1f} – %{100 * win_ci_high:.1f}"],
        ["Son %30 İşlem", len(oos)],
        ["Son %30 Profit Factor", "-" if pd.isna(oos_pf) else f"{oos_pf:.2f}"],
        ["Son %30 Ortalama R", "-" if pd.isna(oos_avg_r) else f"{oos_avg_r:.3f}R"],
    ], columns=["Metrik", "Değer"])
    metrics["Değer"] = metrics["Değer"].astype(str)

    return BacktestResult(metrics, trades_df, equity_df)


def wilson_win_rate_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Küçük örneklerde normal yaklaşımdan daha güvenli win-rate güven aralığı."""
    if total <= 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + (z * z / total)
    center = (p + z * z / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _trade_slice_stats(trades: pd.DataFrame) -> tuple[float, float, int]:
    if trades is None or trades.empty:
        return np.nan, np.nan, 0
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    loss_sum = abs(float(losses["PnL"].sum())) if not losses.empty else 0.0
    pf = float(wins["PnL"].sum()) / loss_sum if loss_sum > 0 else np.nan
    r = trades["PnL"] / trades["Risk Amount"].replace(0, np.nan)
    avg_r = float(r.mean()) if r.notna().any() else np.nan
    return pf, avg_r, len(trades)


def walk_forward_stability_report(trades: pd.DataFrame, folds: int = 4) -> pd.DataFrame:
    """Kronolojik işlem dizisini ardışık dönemlere bölerek rejimler arası kararlılığı ölçer."""
    columns = ["Fold", "Başlangıç", "Bitiş", "İşlem", "Profit Factor", "Ortalama R", "Win Rate", "Maks. DD R"]
    if trades is None or trades.empty or len(trades) < max(int(folds) * 3, 12):
        return pd.DataFrame(columns=columns)
    rows = []
    for fold_no, indices in enumerate(np.array_split(np.arange(len(trades)), int(folds)), start=1):
        part = trades.iloc[indices].copy()
        pf, avg_r, count = _trade_slice_stats(part)
        r = part["PnL"] / part["Risk Amount"].replace(0, np.nan)
        cumulative_r = r.fillna(0).cumsum()
        drawdown_r = cumulative_r - cumulative_r.cummax()
        win_rate = 100 * float((part["PnL"] > 0).mean())
        rows.append({
            "Fold": fold_no,
            "Başlangıç": str(part["Entry Time"].iloc[0]),
            "Bitiş": str(part["Exit Time"].iloc[-1]),
            "İşlem": count,
            "Profit Factor": np.nan if pd.isna(pf) else round(pf, 2),
            "Ortalama R": np.nan if pd.isna(avg_r) else round(avg_r, 3),
            "Win Rate": round(win_rate, 1),
            "Maks. DD R": round(float(drawdown_r.min()), 2),
        })
    return pd.DataFrame(rows, columns=columns)


def assess_walk_forward_stability(report: pd.DataFrame) -> tuple[bool, str]:
    if report is None or report.empty:
        return False, "Walk-forward için dönem başına yeterli işlem yok."
    avg_r = pd.to_numeric(report["Ortalama R"], errors="coerce")
    pf = pd.to_numeric(report["Profit Factor"], errors="coerce")
    positive_ratio = float((avg_r > 0).mean())
    median_pf = float(pf.median()) if pf.notna().any() else np.nan
    stable = positive_ratio >= 0.60 and pd.notna(median_pf) and median_pf >= 1.05
    return stable, f"Pozitif fold %{positive_ratio * 100:.0f}; medyan PF {'-' if pd.isna(median_pf) else f'{median_pf:.2f}'}."


def monte_carlo_risk_report(trades: pd.DataFrame, simulations: int = 500, ruin_level_r: float = -10.0) -> pd.DataFrame:
    """Tarihsel R sonuçlarını bootstrap ederek olası kayıp serisi ve drawdown dağılımını verir."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Metrik", "Değer"])
    r = (trades["PnL"] / trades["Risk Amount"].replace(0, np.nan)).dropna().to_numpy(dtype=float)
    if len(r) < 10:
        return pd.DataFrame(columns=["Metrik", "Değer"])
    rng = np.random.default_rng(42)
    max_drawdowns = []
    final_r = []
    ruined = 0
    for _ in range(int(simulations)):
        path = rng.choice(r, size=len(r), replace=True).cumsum()
        peaks = np.maximum.accumulate(np.insert(path, 0, 0.0))[1:]
        drawdown = path - peaks
        max_drawdowns.append(float(drawdown.min()))
        final_r.append(float(path[-1]))
        ruined += int(float(path.min()) <= float(ruin_level_r))
    return pd.DataFrame([
        ["Simülasyon", str(int(simulations))],
        ["Medyan Sonuç", f"{np.median(final_r):.2f}R"],
        ["%5 Kötü Sonuç", f"{np.percentile(final_r, 5):.2f}R"],
        ["Medyan Maks. Drawdown", f"{np.median(max_drawdowns):.2f}R"],
        ["%95 Kötü Drawdown", f"{np.percentile(max_drawdowns, 5):.2f}R"],
        [f"{ruin_level_r:.0f}R Risk of Ruin", f"%{100 * ruined / simulations:.1f}"],
    ], columns=["Metrik", "Değer"])


def assess_backtest_quality(bt: BacktestResult, min_trades_required: int = 20) -> tuple[str, str, str]:
    """Backtest sonucunu canlı karar ekranında kullanılabilir kalite etiketine çevirir."""
    if bt.trades is None or bt.trades.empty:
        return "Yetersiz", "warn-box", "MTF filtrelerle işlem oluşmadı veya veri yetersiz. Bu sonuçla gerçek işlem kararı verilmemeli."

    trades = bt.trades.copy()
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    pf = np.nan
    loss_sum = abs(losses["PnL"].sum()) if not losses.empty else 0.0
    if loss_sum > 0:
        pf = wins["PnL"].sum() / loss_sum

    total_pnl = float(trades["PnL"].sum())
    avg_pips = float(trades["Pips"].mean())
    trade_count = len(trades)
    oos_start = max(1, int(trade_count * 0.70))
    oos_pf, oos_avg_r, oos_count = _trade_slice_stats(trades.iloc[oos_start:])

    if bt.equity is not None and not bt.equity.empty:
        eq = bt.equity["Balance"]
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min())
        max_dd_idx = dd.idxmin()
        peak_at_dd = float(peak.loc[max_dd_idx]) if len(peak) > 0 else 0.0
        dd_pct = 100 * max_dd / peak_at_dd if peak_at_dd else 0.0
    else:
        dd_pct = 0.0

    if trade_count < int(min_trades_required):
        return "Yetersiz Örnek", "warn-box", f"Sadece {trade_count} işlem var. Minimum {int(min_trades_required)} işlem istendiği için sonuç henüz güvenilir sayılmadı. Daha uzun periyot veya daha yüksek zaman dilimi test edilmeli."

    if oos_count < max(5, int(min_trades_required * 0.20)):
        return "Yetersiz Örnek", "warn-box", f"Son tarih bölümünde yalnızca {oos_count} işlem var. Ayrı dönem kontrolü için veri yetersiz."

    if pd.isna(oos_avg_r) or oos_avg_r <= 0 or (pd.notna(oos_pf) and oos_pf < 1.0):
        return "Zayıf", "bad-box", f"Tüm dönem olumlu görünse bile son %30 tarih bölümünde avantaj doğrulanmadı (PF {'-' if pd.isna(oos_pf) else f'{oos_pf:.2f}'}, ortalama R {'-' if pd.isna(oos_avg_r) else f'{oos_avg_r:.3f}'})."

    if pd.notna(pf) and pf >= 1.30 and total_pnl > 0 and avg_pips > 0 and dd_pct > -15 and (pd.isna(oos_pf) or oos_pf >= 1.10):
        return "İyi", "ok-box", f"PF {pf:.2f}, son %30 PF {'-' if pd.isna(oos_pf) else f'{oos_pf:.2f}'} ve drawdown {dd_pct:.2f}%. Demo/broker doğrulaması yine gereklidir."

    if pd.notna(pf) and pf >= 1.10 and total_pnl > 0 and avg_pips > 0:
        return "Orta", "warn-box", f"PF {pf:.2f}. Sistem pozitif ama marj dar; spread/kayma sonucu bozabilir. Küçük risk veya demo daha uygun."

    return "Zayıf", "bad-box", f"PF {'-' if pd.isna(pf) else f'{pf:.2f}'}, toplam PnL {total_pnl:.2f}, ortalama pip {avg_pips:.2f}. Bu ayarla gerçek işlem için pas geçmek daha güvenli."


def make_backtest_key(
    symbol: str,
    tf_name: str,
    period: str,
    risk_pct: float,
    rr: float,
    atr_mult: float,
    signal_threshold: float,
    spread_pips: float,
    cooldown_bars: int,
    session_filter: str,
    max_same_direction_trades: int,
    min_trades_required: int,
) -> tuple:
    """
    Canlı risk planını hangi backtest sonucuna bağladığımızı anlamak için kullanılır.
    Hesap büyüklüğü ve pip değeri kaliteyi doğrudan değiştirmediği için anahtar dışında bırakıldı.
    """
    return (
        normalize_symbol(symbol),
        tf_name,
        str(period),
        round(float(risk_pct), 4),
        round(float(rr), 4),
        round(float(atr_mult), 4),
        round(float(signal_threshold), 4),
        round(float(spread_pips), 4),
        int(cooldown_bars),
        str(session_filter),
        int(max_same_direction_trades),
        int(min_trades_required),
    )


def get_matching_backtest_quality(current_key: tuple) -> Optional[dict]:
    saved_key = st.session_state.get("last_bt_key")
    saved_quality = st.session_state.get("last_bt_quality")
    if saved_key == current_key and saved_quality:
        return saved_quality
    return None


def side_performance_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Backtest işlemlerini LONG/SHORT bazında ayrı performans tablosuna çevirir."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Yön", "İşlem", "Win Rate", "PnL", "Profit Factor", "Ortalama Pips", "Maks. Ardışık Zarar"])

    rows = []
    for side in ["LONG", "SHORT"]:
        part = trades[trades["Side"] == side].copy()
        if part.empty:
            rows.append([side, 0, "-", "0.00", "-", "-", 0])
            continue

        wins = part[part["PnL"] > 0]
        losses = part[part["PnL"] <= 0]
        loss_sum = abs(losses["PnL"].sum()) if not losses.empty else 0.0
        pf = wins["PnL"].sum() / loss_sum if loss_sum > 0 else np.nan
        win_rate = 100 * len(wins) / len(part)

        # Maksimum ardışık zarar
        max_loss_streak = 0
        current_streak = 0
        for pnl in part["PnL"]:
            if pnl <= 0:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0

        rows.append([
            side,
            len(part),
            f"{win_rate:.2f}%",
            f"{part['PnL'].sum():.2f}",
            "-" if pd.isna(pf) else f"{pf:.2f}",
            f"{part['Pips'].mean():.2f}",
            max_loss_streak,
        ])

    return pd.DataFrame(rows, columns=["Yön", "İşlem", "Win Rate", "PnL", "Profit Factor", "Ortalama Pips", "Maks. Ardışık Zarar"])


def trade_duration_table(trades: pd.DataFrame) -> pd.DataFrame:
    """İşlem sürelerini özetler."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["Metrik", "Değer"])

    t = trades.copy()
    t["Entry Time"] = pd.to_datetime(t["Entry Time"], utc=True, errors="coerce")
    t["Exit Time"] = pd.to_datetime(t["Exit Time"], utc=True, errors="coerce")
    t["Duration Min"] = (t["Exit Time"] - t["Entry Time"]).dt.total_seconds() / 60

    return pd.DataFrame([
        ["Ortalama Süre (dk)", f"{t['Duration Min'].mean():.1f}"],
        ["Medyan Süre (dk)", f"{t['Duration Min'].median():.1f}"],
        ["En Uzun İşlem (dk)", f"{t['Duration Min'].max():.1f}"],
        ["En Kısa İşlem (dk)", f"{t['Duration Min'].min():.1f}"],
    ], columns=["Metrik", "Değer"])


def scan_symbol_live(symbol: str, change_window_minutes: int, selected_tf: Optional[str] = None) -> dict:
    """Tek sembol için canlı çoklu zaman dilimi özetini üretir."""
    summary, _ = analyse_symbol(symbol)
    label, score, note = global_bias(summary, selected_tf)
    price_info = fetch_price_change(symbol, change_window_minutes)
    pct = price_info.get("pct") if price_info else None

    return {
        "Sembol": symbol,
        "Genel Bias": label,
        "Skor": round(float(score), 1),
        "Değişim %": None if pct is None else round(float(pct), 2),
        "4H": summary.loc[summary["Zaman Dilimi"] == "4 Saat", "Bias"].iloc[0] if not summary.empty else "-",
        "1H": summary.loc[summary["Zaman Dilimi"] == "1 Saat", "Bias"].iloc[0] if not summary.empty else "-",
        "15M": summary.loc[summary["Zaman Dilimi"] == "15 Dakika", "Bias"].iloc[0] if not summary.empty else "-",
        "5M": summary.loc[summary["Zaman Dilimi"] == "5 Dakika", "Bias"].iloc[0] if not summary.empty else "-",
        "4H Skor": _summary_score(summary, "4 Saat"),
        "1H Skor": _summary_score(summary, "1 Saat"),
        "15M Skor": _summary_score(summary, "15 Dakika"),
        "5M Skor": _summary_score(summary, "5 Dakika"),
        "Not": note,
    }


def extract_metric(metrics: pd.DataFrame, metric_name: str) -> Optional[str]:
    if metrics is None or metrics.empty:
        return None
    row = metrics[metrics["Metrik"] == metric_name]
    if row.empty:
        return None
    return str(row["Değer"].iloc[0])


def run_symbol_scanner(
    symbols: list[str],
    change_window_minutes: int,
    include_backtest: bool,
    scanner_tf: str,
    scanner_period: str,
    initial_balance: float,
    risk_pct: float,
    rr: float,
    atr_mult: float,
    signal_threshold: float,
    spread_pips: float,
    pip_value_per_lot: float,
    cooldown_bars: int,
    session_filter: str,
    max_same_direction_trades: int,
    min_trades_required: int,
    signal_mode: str = "Dengeli Sinyal",
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Pariteler taranıyor...")

    for i, sym in enumerate(symbols, start=1):
        row = scan_symbol_live(sym, change_window_minutes, scanner_tf)

        if include_backtest:
            bt = run_backtest(
                symbol=sym,
                tf_name=scanner_tf,
                period=scanner_period,
                initial_balance=initial_balance,
                risk_pct=risk_pct,
                rr=rr,
                atr_mult=atr_mult,
                signal_threshold=signal_threshold,
                spread_pips=spread_pips,
                pip_value_per_lot=pip_value_per_lot,
                cooldown_bars=cooldown_bars,
                session_filter=session_filter,
                max_same_direction_trades=max_same_direction_trades,
                min_trades_required=min_trades_required,
            )
            q_label, _, _ = assess_backtest_quality(bt, min_trades_required=min_trades_required)
            row["Backtest Kalitesi"] = q_label
            row["PF"] = extract_metric(bt.metrics, "Profit Factor")
            row["Drawdown"] = extract_metric(bt.metrics, "Maks. Drawdown")
            row["İşlem Sayısı"] = extract_metric(bt.metrics, "İşlem Sayısı")
        else:
            row["Backtest Kalitesi"] = "-"
            row["PF"] = "-"
            row["Drawdown"] = "-"
            row["İşlem Sayısı"] = "-"

        opportunity, opportunity_reason, signal_score = scanner_opportunity_from_row(row, signal_mode)
        row["Fırsat"] = opportunity
        row["Fırsat Nedeni"] = opportunity_reason
        row["Sinyal Skoru"] = round(float(signal_score), 1)
        if opportunity == "PAS":
            row["Karar"] = "PAS"
        elif "DÜŞÜK GÜVEN" in opportunity:
            row["Karar"] = "DEMO/İZLE"
        elif "ÖN İZLEME" in opportunity:
            row["Karar"] = "ÖN İZLEME"
        else:
            row["Karar"] = "İZLE"

        rows.append(row)
        progress.progress(i / len(symbols), text=f"{sym} tarandı ({i}/{len(symbols)})")

    progress.empty()
    result = pd.DataFrame(rows)

    # En işe yarar sıralama: önce aksiyon alınabilecekler, sonra sinyal skoru.
    decision_order = {"İZLE": 0, "DEMO/İZLE": 1, "ÖN İZLEME": 2, "PAS": 3}
    result["_order"] = result["Karar"].map(decision_order).fillna(9)
    result = result.sort_values(["_order", "Sinyal Skoru", "Skor"], ascending=[True, False, False]).drop(columns=["_order"])
    return result



def _is_long_bias(label: str) -> bool:
    return str(label) in {"Alım Yönlü", "Güçlü Alım Yönlü"}


def _is_short_bias(label: str) -> bool:
    return str(label) in {"Satış Yönlü", "Güçlü Satış Yönlü"}


def _mini_tf_text(label: str) -> str:
    label = str(label)
    if "Alım" in label:
        return "LONG"
    if "Satış" in label:
        return "SHORT"
    if label == "Veri yok":
        return "YOK"
    return "BEKLE"


def alert_decision_from_row(
    row: dict,
    alert_entry_tf: str = "15 Dakika",
    signal_threshold: float = 60.0,
) -> tuple[str, str, float]:
    """Alarm ekranında da canlı/backtest ile aynı kapanmış-mum MTF kuralını kullanır."""
    score_key = {"5 Dakika": "5M Skor", "15 Dakika": "15M Skor", "1 Saat": "1H Skor"}.get(
        alert_entry_tf, "15M Skor"
    )
    entry_score = float(row.get(score_key, np.nan))
    h4_score = float(row.get("4H Skor", np.nan))
    h1_score = float(row.get("1H Skor", np.nan))
    m15_score = float(row.get("15M Skor", np.nan))
    decision, reason = mtf_signal_decision(
        entry_score, h4_score, h1_score, m15_score, alert_entry_tf, float(signal_threshold)
    )
    alert_score = 0.0 if pd.isna(entry_score) else abs(entry_score)
    if decision in {"LONG", "SHORT"}:
        return decision, reason, alert_score
    if h4_score >= 25 and h1_score >= 25:
        return "BEKLE", f"Ana yön long; {reason}", alert_score * 0.65
    if h4_score <= -25 and h1_score <= -25:
        return "BEKLE", f"Ana yön short; {reason}", alert_score * 0.65
    return "BEKLE", reason, alert_score * 0.35


def alert_board_single_decision(row: dict) -> tuple[str, str]:
    """Alarm kartı için sade tek karar üretir.

    Alarm ekranı hızlı takip ekranıdır. Backtest/risk planı her kartta çalışmadığı için
    LONG/SHORT alarmını doğrudan 'pozisyon aç' olarak değil, 'izle/adaya al' olarak gösterir.
    Eğer ileride row içinde Backtest Kalitesi gelirse, karar buna göre sertleştirilir.
    """
    alarm = str(row.get("Alarm", "BEKLE"))
    quality = str(row.get("Backtest Kalitesi", "") or "").strip()

    good_quality = quality in {"İyi", "Orta"}
    bad_quality = quality in {"Zayıf", "Kötü", "Yetersiz Örnek", "Yetersiz", "ML yetersiz örnek"}

    if alarm == "LONG":
        if good_quality:
            return "LONG AÇ ADAYI", f"Alarm long yönünde ve kalite {quality}."
        if bad_quality:
            return "PAS GEÇ", f"Alarm long olsa da kalite {quality}; İşlem Asistanı onayı olmadan açma."
        return "LONG İÇİN İZLE", "Alarm long yönünde. Detay için İşlem Asistanı ekranında risk/backtest kontrolü yap."

    if alarm == "SHORT":
        if good_quality:
            return "SHORT AÇ ADAYI", f"Alarm short yönünde ve kalite {quality}."
        if bad_quality:
            return "PAS GEÇ", f"Alarm short olsa da kalite {quality}; İşlem Asistanı onayı olmadan açma."
        return "SHORT İÇİN İZLE", "Alarm short yönünde. Detay için İşlem Asistanı ekranında risk/backtest kontrolü yap."

    return "BEKLE", "4H + 1H ve giriş teyidi aynı yönde net izin vermiyor."


def build_alert_board_rows(
    symbols: list[str], change_window_minutes: int, alert_entry_tf: str, signal_threshold: float
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Alarm ekranı hazırlanıyor...")
    for i, sym in enumerate(symbols, start=1):
        row = scan_symbol_live(sym, change_window_minutes, alert_entry_tf)
        decision, reason, alert_score = alert_decision_from_row(row, alert_entry_tf, signal_threshold)
        row["Alarm"] = decision
        row["Alarm Nedeni"] = reason
        row["Alarm Skoru"] = round(float(alert_score), 1)
        tek_karar, tek_karar_notu = alert_board_single_decision(row)
        row["Tek Karar"] = tek_karar
        row["Tek Karar Notu"] = tek_karar_notu
        rows.append(row)
        progress.progress(i / max(len(symbols), 1), text=f"{sym} kontrol edildi ({i}/{len(symbols)})")
    progress.empty()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"LONG": 0, "SHORT": 1, "BEKLE": 2}
    df["_order"] = df["Alarm"].map(order).fillna(9)
    return df.sort_values(["_order", "Alarm Skoru", "Skor"], ascending=[True, False, False]).drop(columns=["_order"])


def render_alert_card(row: dict) -> None:
    alarm = str(row.get("Alarm", "BEKLE"))
    css = {"LONG": "alert-buy", "SHORT": "alert-sell", "BEKLE": "alert-wait"}.get(alarm, "alert-wait")
    status_css = {"LONG": "alert-status-buy", "SHORT": "alert-status-sell", "BEKLE": "alert-status-wait"}.get(alarm, "alert-status-wait")
    icon = {"LONG": "🟢", "SHORT": "🔴", "BEKLE": "🟡"}.get(alarm, "🟡")
    symbol_txt = escape(str(row.get("Sembol", "-"))).replace("=X", "")
    reason = escape(str(row.get("Alarm Nedeni", "-")))
    tek_karar = escape(str(row.get("Tek Karar", "-")))
    tek_karar_notu = escape(str(row.get("Tek Karar Notu", "")))
    score = escape(str(row.get("Alarm Skoru", "-")))
    change = row.get("Değişim %", None)
    change_text = "-" if pd.isna(change) else f"{float(change):+.2f}%"
    h4 = _mini_tf_text(row.get("4H", "-"))
    h1 = _mini_tf_text(row.get("1H", "-"))
    m15 = _mini_tf_text(row.get("15M", "-"))
    m5 = _mini_tf_text(row.get("5M", "-"))

    st.markdown(
        f"""
        <div class="alert-card {css}">
            <div class="alert-card-header">
                <div class="alert-symbol">{symbol_txt}</div>
                <div class="alert-status {status_css}">{icon} {alarm}</div>
            </div>
            <div class="alert-mini-grid">
                <div class="alert-mini">4H<br>{escape(h4)}</div>
                <div class="alert-mini">1H<br>{escape(h1)}</div>
                <div class="alert-mini">15M<br>{escape(m15)}</div>
                <div class="alert-mini">5M<br>{escape(m5)}</div>
            </div>
            <div class="alert-reason">{reason}</div>
            <div class="alert-decision">Tek Karar: {tek_karar}<small>{tek_karar_notu}</small></div>
            <div class="alert-meta">
                <span>Skor: {score}</span>
                <span>Değişim: {escape(change_text)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_section(title: str, df: pd.DataFrame) -> None:
    st.markdown(f"<div class='alert-section-title'>{escape(title)}</div>", unsafe_allow_html=True)
    if df.empty:
        st.info("Bu grupta gösterilecek parite yok.")
        return
    rows = df.to_dict("records")
    cols_per_row = 4
    for start in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, row in zip(cols, rows[start:start + cols_per_row]):
            with col:
                render_alert_card(row)


def render_pair_alert_screen(
    change_window_minutes: int,
    change_window_label: str,
    alert_entry_tf: str,
    alert_groups: list[str],
    alert_sort_mode: str,
    signal_threshold: float,
    webhook_url: str = "",
) -> None:
    st.header("Parite Alarm Ekranı")
    st.caption("Major ve minör pariteleri tek bakışta LONG / SHORT / BEKLE olarak gösterir. Bu ekran hızlı takip içindir; gerçek işlem için İşlem Asistanı karar kartı ve demo doğrulama kullanılmalı.")

    selected_symbols: list[str] = []
    for group in alert_groups:
        selected_symbols.extend(ALERT_PAIR_GROUPS.get(group, []))
    selected_symbols = list(dict.fromkeys(selected_symbols))

    if not selected_symbols:
        st.warning("En az bir parite grubu seçmelisin.")
        return

    with st.spinner("Major/minör pariteler taranıyor..."):
        board = build_alert_board_rows(selected_symbols, change_window_minutes, alert_entry_tf, signal_threshold)

    if board.empty:
        st.warning("Alarm ekranı için veri alınamadı.")
        return

    for row in board[board["Alarm"].isin(["LONG", "SHORT"])].to_dict("records"):
        floor_rule = {"5 Dakika": "5min", "15 Dakika": "15min", "1 Saat": "1h"}.get(alert_entry_tf, "15min")
        candle_key = f"{alert_entry_tf}|{pd.Timestamp.now(tz='UTC').floor(floor_rule)}"
        payload = {"symbol": row.get("Sembol"), "side": row.get("Alarm"), "reason": row.get("Alarm Nedeni"), "score": row.get("Alarm Skoru")}
        if record_alert_once(str(row.get("Sembol")), str(row.get("Alarm")), candle_key, payload) and webhook_url.strip():
            send_webhook_notification(webhook_url, payload)

    if alert_sort_mode == "Önce LONG/SHORT":
        order = {"LONG": 0, "SHORT": 1, "BEKLE": 2}
        board = board.assign(_sort=board["Alarm"].map(order).fillna(9)).sort_values(["_sort", "Alarm Skoru"], ascending=[True, False]).drop(columns=["_sort"])
    elif alert_sort_mode == "Sadece LONG-SHORT üstte":
        order = {"LONG": 0, "SHORT": 0, "BEKLE": 1}
        board = board.assign(_sort=board["Alarm"].map(order).fillna(9)).sort_values(["_sort", "Alarm Skoru"], ascending=[True, False]).drop(columns=["_sort"])
    else:
        board = board.sort_values("Alarm Skoru", ascending=False)

    al_count = int((board["Alarm"] == "LONG").sum())
    sat_count = int((board["Alarm"] == "SHORT").sum())
    wait_count = int((board["Alarm"] == "BEKLE").sum())

    st.markdown(
        f"""
        <div class="alert-summary-row">
            <div class="alert-summary-box"><b>LONG</b><span>🟢 {al_count}</span></div>
            <div class="alert-summary-box"><b>SHORT</b><span>🔴 {sat_count}</span></div>
            <div class="alert-summary-box"><b>BEKLE</b><span>🟡 {wait_count}</span></div>
            <div class="alert-summary-box"><b>Filtre</b><span>{escape(alert_entry_tf)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        f"Alarm mantığı: 4H + 1H ana yön aynı olmalı; {alert_entry_tf} giriş teyidi verir. "
        f"Değişim %, sol menüdeki '{change_window_label}' seçimine göre Yahoo 1 dakikalık kapanış verisinden hesaplanır."
    )

    major_df = board[board["Sembol"].isin(MAJOR_PAIRS)]
    minor_df = board[board["Sembol"].isin(MINOR_PAIRS)]

    if "Major" in alert_groups:
        render_alert_section("Major Pariteler", major_df)
    if "Minör" in alert_groups:
        render_alert_section("Minör Pariteler", minor_df)

    with st.expander("Tablo görünümü", expanded=False):
        table_cols = ["Sembol", "Alarm", "Tek Karar", "Tek Karar Notu", "Alarm Skoru", "Değişim %", "4H", "1H", "15M", "5M", "Alarm Nedeni"]
        st.dataframe(board[table_cols], use_container_width=True, height=420)
        st.download_button(
            "Alarm Tablosunu CSV İndir",
            data=board[table_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="forex_alert_board.csv",
            mime="text/csv",
        )


def init_trade_journal() -> None:
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS alert_history (
                alert_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        conn.commit()


def add_trade_journal_entry(entry: dict) -> None:
    init_trade_journal()
    payload = json.dumps(entry, ensure_ascii=False, default=str)
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO trade_journal(created_at, payload) VALUES (?, ?)",
            (datetime.now(tz=TR_TZ).isoformat(), payload),
        )
        conn.commit()


def journal_dataframe() -> pd.DataFrame:
    init_trade_journal()
    with sqlite3.connect(APP_DB_PATH) as conn:
        rows = conn.execute("SELECT id, created_at, payload FROM trade_journal ORDER BY id DESC").fetchall()
    records = []
    for row_id, created_at, payload in rows:
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        item["Kayıt ID"] = row_id
        item.setdefault("Kayıt Zamanı", created_at)
        records.append(item)
    return pd.DataFrame(records)


def clear_trade_journal() -> None:
    init_trade_journal()
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute("DELETE FROM trade_journal")
        conn.commit()


def record_alert_once(symbol: str, side: str, candle_time: str, payload: dict) -> bool:
    """Aynı sembol/yön/mum alarmını yalnızca bir kez kaydeder."""
    init_trade_journal()
    alert_key = f"{normalize_symbol(symbol)}|{side}|{candle_time}"
    try:
        with sqlite3.connect(APP_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO alert_history(alert_key, created_at, symbol, side, payload) VALUES (?, ?, ?, ?, ?)",
                (alert_key, datetime.now(tz=TR_TZ).isoformat(), normalize_symbol(symbol), side, json.dumps(payload, ensure_ascii=False, default=str)),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def alert_history_dataframe(limit: int = 200) -> pd.DataFrame:
    init_trade_journal()
    with sqlite3.connect(APP_DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT created_at AS Zaman, symbol AS Sembol, side AS Yön, payload AS Detay FROM alert_history ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )


def send_webhook_notification(webhook_url: str, payload: dict) -> tuple[bool, str]:
    if not webhook_url.strip():
        return False, "Webhook adresi tanımlı değil."
    try:
        request = Request(
            webhook_url.strip(),
            data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ForexAnalyzer/1.0"},
            method="POST",
        )
        with urlopen(request, timeout=8) as response:
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except (URLError, ValueError, TimeoutError) as exc:
        LOGGER.warning("Webhook notification failed: %s", exc)
        return False, str(exc)


def calculate_manual_pips(symbol: str, side: str, entry: float, exit_price: float) -> Optional[float]:
    if entry is None or exit_price is None or entry <= 0 or exit_price <= 0:
        return None
    pip = get_pip_size(symbol)
    if side == "LONG":
        return (exit_price - entry) / pip
    if side == "SHORT":
        return (entry - exit_price) / pip
    return None


def news_blackout_status(
    symbol: str,
    events: pd.DataFrame,
    before_minutes: int = 30,
    after_minutes: int = 20,
) -> dict:
    if events is None or events.empty:
        return {"blocks_trade": False, "state": "ok", "text": "Yüklü yüksek etkili haber yok."}
    required = {"time", "currency"}
    if not required.issubset({str(c).lower() for c in events.columns}):
        return {"blocks_trade": True, "state": "bad", "text": "Haber CSV sütunları: time,currency,title,impact olmalı."}
    e = events.copy()
    e.columns = [str(c).lower() for c in e.columns]
    e["time"] = pd.to_datetime(e["time"], utc=True, errors="coerce")
    e = e.dropna(subset=["time"])
    if "impact" in e.columns:
        e = e[e["impact"].astype(str).str.lower().isin({"high", "yüksek", "3"})]
    base, quote = symbol_pair(symbol)
    e = e[e["currency"].astype(str).str.upper().isin({base, quote})]
    now = pd.Timestamp.now(tz="UTC")
    active = e[(e["time"] >= now - pd.Timedelta(minutes=after_minutes)) & (e["time"] <= now + pd.Timedelta(minutes=before_minutes))]
    if active.empty:
        return {"blocks_trade": False, "state": "ok", "text": "Yakın yüksek etkili haber yok."}
    event = active.sort_values("time").iloc[0]
    title = str(event.get("title", "Yüksek etkili veri"))
    local_time = event["time"].tz_convert(TR_TZ).strftime("%H:%M")
    return {"blocks_trade": True, "state": "bad", "text": f"{event['currency']} haberi {local_time}: {title}"}


def portfolio_risk_status(
    journal: pd.DataFrame,
    symbol: str,
    proposed_risk_pct: float,
    max_total_risk_pct: float,
    max_currency_risk_pct: float,
    max_open_positions: int,
    daily_stop_r: float,
    weekly_stop_r: float,
) -> dict:
    if journal is None or journal.empty:
        journal = pd.DataFrame()
    result_col = journal.get("Sonuç", pd.Series(dtype=str)).astype(str)
    open_rows = journal[result_col == "Açık"].copy() if not journal.empty else pd.DataFrame()
    open_risks = pd.to_numeric(open_rows.get("Risk %", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    current_total = float(open_risks.sum())
    base, quote = symbol_pair(symbol)
    currency_risk = 0.0
    if not open_rows.empty and "Sembol" in open_rows.columns:
        for (_, row), row_risk in zip(open_rows.iterrows(), open_risks):
            row_base, row_quote = symbol_pair(str(row.get("Sembol", "")))
            if base in {row_base, row_quote} or quote in {row_base, row_quote}:
                currency_risk += float(row_risk)

    now_local = pd.Timestamp.now(tz=TR_TZ)
    times = pd.to_datetime(journal.get("Kayıt Zamanı", pd.Series(dtype=str)), utc=True, errors="coerce") if not journal.empty else pd.Series(dtype="datetime64[ns, UTC]")
    r_values = pd.to_numeric(journal.get("R", pd.Series(dtype=float)), errors="coerce").fillna(0.0) if not journal.empty else pd.Series(dtype=float)
    local_dates = times.dt.tz_convert(TR_TZ) if not times.empty else times
    daily_r = float(r_values[local_dates.dt.date == now_local.date()].sum()) if not times.empty else 0.0
    week_start = (now_local - pd.Timedelta(days=now_local.weekday())).normalize()
    weekly_r = float(r_values[local_dates >= week_start].sum()) if not times.empty else 0.0

    reasons = []
    if len(open_rows) >= int(max_open_positions):
        reasons.append(f"açık pozisyon limiti {max_open_positions}")
    if current_total + proposed_risk_pct > max_total_risk_pct + 1e-9:
        reasons.append(f"toplam risk %{current_total + proposed_risk_pct:.2f} > %{max_total_risk_pct:.2f}")
    if currency_risk + proposed_risk_pct > max_currency_risk_pct + 1e-9:
        reasons.append(f"ilişkili para riski %{currency_risk + proposed_risk_pct:.2f} > %{max_currency_risk_pct:.2f}")
    if daily_r <= -abs(daily_stop_r):
        reasons.append(f"günlük sonuç {daily_r:.2f}R")
    if weekly_r <= -abs(weekly_stop_r):
        reasons.append(f"haftalık sonuç {weekly_r:.2f}R")
    return {
        "blocks_trade": bool(reasons),
        "state": "bad" if reasons else "ok",
        "text": "; ".join(reasons) if reasons else f"Açık risk %{current_total:.2f}; günlük {daily_r:.2f}R; haftalık {weekly_r:.2f}R.",
        "open_positions": len(open_rows),
        "total_risk_pct": current_total,
        "daily_r": daily_r,
        "weekly_r": weekly_r,
    }


def _metric_to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).replace("%", "").replace(",", ".").strip()
        if text == "-":
            return None
        # "-123.45 (-12.30%)" gibi değerlerde ilk sayıyı alır.
        return float(text.split()[0])
    except Exception:
        return None


def backtest_quality_allowed(matched_quality: Optional[dict], strict_safety_mode: bool) -> bool:
    if not matched_quality:
        return False
    allowed = {"İyi"} if strict_safety_mode else {"İyi", "Orta"}
    return matched_quality.get("label") in allowed


SIGNAL_MODES = ["Dengeli Sinyal", "Hızlı Sinyal", "Güvenli Sinyal"]
LOW_SAMPLE_QUALITIES = {"Yetersiz", "Yetersiz Örnek"}
TRADE_DIRECTIONS = {"Alım Yönlü", "Güçlü Alım Yönlü", "Satış Yönlü", "Güçlü Satış Yönlü"}


def signal_mode_settings(signal_mode: str) -> dict:
    settings = {
        "Hızlı Sinyal": {
            "allow_low_sample_signal": True,
            "allow_preview_without_backtest": True,
            "requires_strong_direction": False,
            "confidence_adjust": -5,
            "description": "Daha erken sinyal verir; düşük örnek ve ön sinyal durumlarını ayrıca işaretler.",
        },
        "Dengeli Sinyal": {
            "allow_low_sample_signal": True,
            "allow_preview_without_backtest": False,
            "requires_strong_direction": False,
            "confidence_adjust": 0,
            "description": "Yön, mum kapanışı ve kalite dengesini varsayılan şekilde kullanır.",
        },
        "Güvenli Sinyal": {
            "allow_low_sample_signal": False,
            "allow_preview_without_backtest": False,
            "requires_strong_direction": True,
            "confidence_adjust": 8,
            "description": "Daha az sinyal verir; güçlü yön ve iyi kalite arar.",
        },
    }
    return settings.get(signal_mode, settings["Dengeli Sinyal"])


def allowed_quality_for_mode(strict_safety_mode: bool, signal_mode: str) -> set[str]:
    if strict_safety_mode or signal_mode == "Güvenli Sinyal":
        return {"İyi"}
    return {"İyi", "Orta"}


def quality_signal_status(
    matched_quality: Optional[dict],
    allowed_quality_labels: set[str],
    practical_signal_mode: bool,
    signal_mode: str,
) -> dict:
    settings = signal_mode_settings(signal_mode)
    if matched_quality is None:
        if practical_signal_mode and settings["allow_preview_without_backtest"]:
            return {
                "status": "preview",
                "state": "warn",
                "label": "Ön Sinyal",
                "text": "Plan kontrolü yok; sadece ön izleme sinyali.",
                "blocks_trade": False,
            }
        return {
            "status": "pending",
            "state": "warn",
            "label": "Bekliyor",
            "text": "Plan kontrolü bekliyor",
            "blocks_trade": True,
        }

    label = str(matched_quality.get("label", "-"))
    if label in allowed_quality_labels:
        return {
            "status": "approved",
            "state": "ok",
            "label": label,
            "text": f"Kalite: {label}",
            "blocks_trade": False,
        }

    if practical_signal_mode and settings["allow_low_sample_signal"] and label in LOW_SAMPLE_QUALITIES and allowed_quality_labels != {"İyi"}:
        return {
            "status": "blocked",
            "state": "bad",
            "label": label,
            "text": f"Yetersiz örnek: {label}. Yalnızca demo/izleme; gerçek işlem sinyali kapalı.",
            "blocks_trade": True,
        }

    return {
        "status": "blocked",
        "state": "bad",
        "label": label,
        "text": f"Kalite: {label}",
        "blocks_trade": True,
    }


def confidence_label(score: float) -> str:
    if score >= 80:
        return "Yüksek Güven"
    if score >= 60:
        return "Orta Güven"
    if score >= 40:
        return "Düşük Güven"
    return "Çok Düşük Güven"


def classify_market_regime(symbol: str, selected_tf: str) -> dict:
    prm = TIMEFRAMES[selected_tf]
    df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if df.empty or len(df) < 80:
        return {
            "label": "Bilinmiyor",
            "state": "warn",
            "text": "Piyasa tipi için yeterli veri yok.",
            "score_adjust": 0,
        }

    ind = add_indicators(df.iloc[:-1].tail(260))
    row = latest_valid_row(ind)
    if row is None:
        return {
            "label": "Bilinmiyor",
            "state": "warn",
            "text": "Piyasa tipi için indikatör verisi yetersiz.",
            "score_adjust": 0,
        }

    atr = float(row["ATR14"])
    if not np.isfinite(atr) or atr <= 0:
        return {
            "label": "Bilinmiyor",
            "state": "warn",
            "text": "ATR okunamadığı için piyasa tipi belirsiz.",
            "score_adjust": 0,
        }

    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema_gap_atr = abs(ema20 - ema50) / atr
    ema50_slope_atr = abs(float(ind["EMA50"].iloc[-1] - ind["EMA50"].iloc[-20])) / atr if len(ind) > 20 else 0.0
    bb_width_atr = abs(float(row["BBUp"] - row["BBLow"])) / atr if pd.notna(row.get("BBUp")) and pd.notna(row.get("BBLow")) else 0.0

    if ema50_slope_atr >= 1.0 and ema_gap_atr >= 0.35:
        return {
            "label": "Trend",
            "state": "ok",
            "text": f"Trend piyasası: EMA eğimi {ema50_slope_atr:.2f} ATR, EMA açıklığı {ema_gap_atr:.2f} ATR.",
            "score_adjust": 8,
        }

    if ema50_slope_atr <= 0.35 and ema_gap_atr <= 0.25 and bb_width_atr <= 3.5:
        return {
            "label": "Yatay",
            "state": "warn",
            "text": f"Yatay/sıkışık piyasa: EMA eğimi {ema50_slope_atr:.2f} ATR, EMA açıklığı {ema_gap_atr:.2f} ATR.",
            "score_adjust": -10,
        }

    return {
        "label": "Kararsız",
        "state": "warn",
        "text": f"Trend net değil: EMA eğimi {ema50_slope_atr:.2f} ATR, Bollinger genişliği {bb_width_atr:.2f} ATR.",
        "score_adjust": -2,
    }


def build_wait_reason(decision: dict, tracker: Optional[dict] = None) -> tuple[str, str, str]:
    action = str(decision.get("action", "BEKLE"))
    if tracker and tracker.get("signal_now"):
        return "Sinyal aktif", "Giriş şartları tamamlandı; risk ve spread kontrolü yapılmalı.", "ok"

    if tracker:
        blocker = str(tracker.get("primary_blocker", "")).strip()
        blocker_text = str(tracker.get("primary_blocker_text", "")).strip()
        if blocker or blocker_text:
            return blocker or "Bekleme sebebi", blocker_text or str(decision.get("reason", "")), "warn"

    if "PAS" in action:
        return "Pas geçme sebebi", str(decision.get("reason", "")), "pass"
    return "Bekleme sebebi", str(decision.get("reason", "")), "warn"


def signal_class_for_status(side: Optional[str], quality_status: str) -> tuple[str, str]:
    if quality_status == "approved":
        return ("simple-buy" if side == "LONG" else "simple-sell", "entry-signal-buy" if side == "LONG" else "entry-signal-sell")
    if quality_status in {"low", "preview"}:
        return "simple-wait", "entry-signal-wait"
    return "simple-pass", "entry-signal-pass"


def calculate_signal_confidence(
    final_score: float,
    quality_status: str,
    candle_ok: bool,
    direction_ok: bool,
    same_tf_ok: bool,
    market_regime: Optional[dict],
    signal_mode: str,
) -> float:
    score = min(55.0, abs(float(final_score)) * 0.55)
    if direction_ok:
        score += 12
    if same_tf_ok:
        score += 8
    if candle_ok:
        score += 12

    if quality_status == "approved":
        score += 18
    elif quality_status == "low":
        score += 5
    elif quality_status == "preview":
        score -= 8
    elif quality_status == "blocked":
        score -= 25
    elif quality_status == "pending":
        score -= 15

    if market_regime:
        score += float(market_regime.get("score_adjust", 0))
    score += float(signal_mode_settings(signal_mode).get("confidence_adjust", 0))
    return float(np.clip(score, 0, 100))


def scanner_opportunity_from_row(row: dict, signal_mode: str) -> tuple[str, str, float]:
    bias = str(row.get("Genel Bias", "İşlem Yok"))
    quality = str(row.get("Backtest Kalitesi", "-"))
    score = abs(float(row.get("Skor", 0) or 0))

    if bias == "İşlem Yok":
        return "PAS", "Yön yok", score

    if "Alım" in bias:
        side = "LONG"
    elif "Satış" in bias:
        side = "SHORT"
    else:
        return "PAS", "Yön okunamadı", score

    if quality in {"İyi", "Orta"}:
        return f"{side} ADAYI", f"Kalite: {quality}", score + (30 if quality == "İyi" else 20)

    if quality in LOW_SAMPLE_QUALITIES and signal_mode != "Güvenli Sinyal":
        return f"{side} DÜŞÜK GÜVEN", f"Örnek düşük: {quality}", score + 8

    if quality == "-":
        return f"{side} ÖN İZLEME", "Backtest hesaplanmadı", score + 5

    return "PAS", f"Kalite: {quality}", score - 10


def build_simple_trade_decision(
    symbol: str,
    selected_tf: str,
    bt_tf: str,
    final_label: str,
    final_score: float,
    filter_note: str,
    price: Optional[float],
    setup: Optional[TradeSetup],
    matched_quality: Optional[dict],
    strict_safety_mode: bool,
    practical_signal_mode: bool = True,
    signal_mode: str = "Dengeli Sinyal",
    allowed_quality_labels: Optional[set[str]] = None,
    entry_score: Optional[float] = None,
    signal_threshold: float = 60.0,
) -> dict:
    """Teknik ekranı acemi kullanıcı için LONG/SHORT/BEKLE/PAS GEÇ kararına indirger."""
    dec = price_decimals(symbol)
    base = {
        "action": "BEKLE",
        "class": "simple-wait",
        "subtitle": "Henüz net işlem yok.",
        "reason": filter_note,
        "steps": ["Yeni pozisyon açma.", "Pariteyi izlemeye devam et.", "Backtest ve ana yön uyumu oluşmadan işlem alma."],
        "levels": {},
    }
    if allowed_quality_labels is None:
        allowed_quality_labels = allowed_quality_for_mode(strict_safety_mode, signal_mode)
    quality_info = quality_signal_status(
        matched_quality=matched_quality,
        allowed_quality_labels=allowed_quality_labels,
        practical_signal_mode=practical_signal_mode,
        signal_mode=signal_mode,
    )

    if bt_tf != selected_tf:
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Backtest zaman dilimi ile giriş zaman dilimi eşleşmiyor.",
            "reason": "Basit karar için Backtest zaman dilimi, Grafik/Giriş zaman dilimi ile aynı olmalı.",
            "steps": ["Sidebar'dan backtest zaman dilimini giriş zaman dilimiyle aynı seç.", "Yeniden Hesapla butonuna bas.", "Sonra bu karttaki kararı takip et."],
        })
        return base

    if quality_info["status"] == "pending":
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Önce strateji kontrolü gerekiyor.",
            "reason": "Bu sembol ve zaman dilimi için backtest onayı yok.",
            "steps": ["Sidebar'dan Yeniden Hesapla butonuna bas.", "Strateji Kalitesi Orta veya İyi değilse yeni pozisyon açma.", "Sert Güvenli Mod açıksa sadece İyi kalite kabul edilir."],
        })
        return base

    if quality_info["status"] == "blocked":
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Kalite filtresi bu işlemi reddetti.",
            "reason": matched_quality.get("text", "Strateji kalitesi zayıf/yetersiz.") if matched_quality else quality_info["text"],
            "steps": ["Bu ayarla yeni pozisyon açma.", "Başka parite veya daha yüksek zaman dilimi dene.", "Kalite filtresi düzelmeden gerçek pozisyon açma."],
        })
        return base

    strong_required = strict_safety_mode or signal_mode_settings(signal_mode)["requires_strong_direction"]
    if strong_required and final_label not in {"Güçlü Alım Yönlü", "Güçlü Satış Yönlü"}:
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Ana sinyal yeterince güçlü değil.",
            "reason": f"{signal_mode} için 'Güçlü Alım' veya 'Güçlü Satış' gerekli. Mevcut: {final_label}.",
            "steps": ["Bu paritede şimdilik yeni pozisyon açma.", "4H ve 1H güçlü aynı yöne dönene kadar bekle.", "Tarayıcıdan daha net fırsat ara."],
        })
        return base

    if final_label == "İşlem Yok" or setup is None:
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Sistem işlem planı üretmiyor.",
            "reason": filter_note,
            "steps": ["Yeni pozisyon açma.", "Ana yön netleşene kadar bekle.", "Risk Planı oluşmadan emir girme."],
        })
        return base

    side = setup.side
    quality_text = quality_info.get("label", "-")
    levels = {
        "Giriş": f"{setup.entry:.{dec}f}",
        "Stop": f"{setup.stop:.{dec}f}",
        "Kâr Al": f"{setup.target:.{dec}f}",
        "Lot": f"{setup.estimated_lot:.2f}",
    }

    if side == "LONG":
        score_text = "-" if entry_score is None or pd.isna(entry_score) else f"{float(entry_score):.1f}"
        return {
            "action": "LONG İÇİN BEKLE",
            "class": "simple-wait",
            "subtitle": f"{symbol} long yönünde; kapanmış mum MTF skoru bekleniyor.",
            "reason": f"{selected_tf} skoru {score_text}; LONG eşiği {float(signal_threshold):.0f}. Backtest onayı: {quality_text}.",
            "steps": [
                f"{selected_tf} kapanmış mum skoru {float(signal_threshold):.0f} veya üstüne çıkmadan LONG açma.",
                f"LONG açarsan stop {setup.stop:.{dec}f}, kâr al {setup.target:.{dec}f}.",
                "4H ve 1H long uyumunu kaybederse planı iptal et.",
            ],
            "levels": levels,
        }

    if side == "SHORT":
        score_text = "-" if entry_score is None or pd.isna(entry_score) else f"{float(entry_score):.1f}"
        return {
            "action": "SHORT İÇİN BEKLE",
            "class": "simple-wait",
            "subtitle": f"{symbol} short yönünde; kapanmış mum MTF skoru bekleniyor.",
            "reason": f"{selected_tf} skoru {score_text}; SHORT eşiği -{float(signal_threshold):.0f}. Backtest onayı: {quality_text}.",
            "steps": [
                f"{selected_tf} kapanmış mum skoru -{float(signal_threshold):.0f} veya altına inmeden SHORT açma.",
                f"SHORT açarsan stop {setup.stop:.{dec}f}, kâr al {setup.target:.{dec}f}.",
                "4H ve 1H short uyumunu kaybederse planı iptal et.",
            ],
            "levels": levels,
        }

    return base


def render_simple_decision_card(decision: dict) -> None:
    """Basit karar kartını düz HTML olarak basar.

    HTML kompakt üretilir; Streamlit markdown içinde girintili HTML bazen
    kod bloğu gibi görünebildiği için burada çok satırlı/indentli HTML kullanılmaz.
    """
    card_class = escape(str(decision.get("class", "simple-wait")))
    action = escape(str(decision.get("action", "BEKLE")))
    subtitle = escape(str(decision.get("subtitle", "")))
    reason = escape(str(decision.get("reason", "")))

    levels_html = ""
    if decision.get("levels"):
        parts = []
        for key, value in decision["levels"].items():
            parts.append(
                "<div class='simple-level'>"
                f"<b>{escape(str(key))}</b>"
                f"<span>{escape(str(value))}</span>"
                "</div>"
            )
        levels_html = "<div class='simple-levels'>" + "".join(parts) + "</div>"

    steps = decision.get("steps", []) or []
    steps_html = "".join(f"<li>{escape(str(step))}</li>" for step in steps) if steps else "<li>Şu an yeni işlem açma.</li>"

    html = (
        f"<div class='simple-card {card_class}'>"
        f"<div class='simple-action'>{action}</div>"
        f"<div class='simple-subtitle'>{subtitle}</div>"
        f"<div><b>Sebep:</b> {reason}</div>"
        f"{levels_html}"
        f"<div style='margin-top:12px;'><b>Ne yapacağım?</b><ol>{steps_html}</ol></div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def decision_panel_class(decision: dict) -> str:
    css = str(decision.get("class", "simple-wait"))
    if css == "simple-buy":
        return "decision-buy"
    if css == "simple-sell":
        return "decision-sell"
    if css == "simple-pass":
        return "decision-pass"
    return "decision-wait"


def render_top_decision_panel(decision: dict) -> None:
    action = escape(str(decision.get("action", "BEKLE")))
    subtitle = escape(str(decision.get("subtitle", "")))
    reason = escape(str(decision.get("reason", "")))
    panel_class = decision_panel_class(decision)
    st.markdown(
        (
            f"<div class='decision-shell {panel_class}'>"
            f"<h2>{action}</h2>"
            f"<p><b>{subtitle}</b></p>"
            f"<p style='margin-top:6px;'>{reason}</p>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_signal_summary_card(decision: dict, tracker: dict, market_regime: dict, signal_mode: str) -> None:
    action = escape(str(decision.get("action", "BEKLE")))
    confidence = escape(f"{tracker.get('confidence_score', 0):.0f}/100 - {tracker.get('confidence_label', '-')}")
    trigger = escape(str(tracker.get("condition", "-")))
    alarm = escape(str(tracker.get("alarm_text", "-")))
    regime = escape(f"{market_regime.get('label', '-')} - {market_regime.get('text', '-')}")
    mode = escape(signal_mode)
    html = (
        "<div class='risk-box'>"
        f"<b>Sinyal Kartı:</b> {action}<br>"
        f"<b>Güven:</b> {confidence}<br>"
        f"<b>Mod:</b> {mode}<br>"
        f"<b>Giriş Alarmı:</b> {alarm}<br>"
        f"<b>Piyasa Tipi:</b> {regime}<br>"
        f"<b>Giriş Şartı:</b> {trigger}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_wait_reason_box(decision: dict, tracker: dict) -> None:
    title, text, state = build_wait_reason(decision, tracker)
    css = "logic-note-ok" if state == "ok" else ("logic-note-pass" if state == "pass" else "")
    st.markdown(
        f"<div class='logic-note {css}'><b>{escape(title)}</b><br>{escape(text)}</div>",
        unsafe_allow_html=True,
    )


def render_entry_alarm_box(tracker: dict) -> None:
    state = "logic-note-ok" if tracker.get("signal_now") else ""
    st.markdown(
        f"<div class='logic-note {state}'><b>Giriş Alarmı</b><br>{escape(str(tracker.get('alarm_text', '-')))}</div>",
        unsafe_allow_html=True,
    )


def render_direction_trade_explanation(
    final_label: str,
    final_score: float,
    selected_tf: str,
    simple_decision: dict,
    matched_quality: Optional[dict],
    allowed_quality_labels: set[str],
    entry_signal_tracker: dict,
) -> None:
    """Genel yön ile gerçek işlem kararının neden farklı olabileceğini açıklar."""
    decision_action = str(simple_decision.get("action", "BEKLE"))
    quality_label = matched_quality.get("label") if matched_quality else None
    allowed_text = " veya ".join(sorted(allowed_quality_labels))

    if entry_signal_tracker.get("signal_now"):
        quality_status = entry_signal_tracker.get("quality_status", "approved")
        css = "logic-note logic-note-ok" if quality_status == "approved" else "logic-note"
        title = "Yön ve işlem kararı uyumlu" if quality_status == "approved" else "Sinyal var, güven düşük"
        text = (
            f"Piyasa yönü {final_label} ({final_score:.1f} skor). "
            f"Son kapanmış {selected_tf} MTF skoru giriş şartını geçti. "
            f"Güven: {entry_signal_tracker.get('confidence_label', '-')}."
        )
    elif quality_label and quality_label not in allowed_quality_labels:
        css = "logic-note logic-note-pass"
        title = "Yön var ama işlem izni yok"
        text = (
            f"Piyasa yönü {final_label} ({final_score:.1f} skor) sadece yön bilgisidir. "
            f"İşlem için strateji kalitesi {allowed_text} olmalı; şu an {quality_label}. "
            f"Bu yüzden karar: {decision_action}."
        )
    elif matched_quality is None:
        css = "logic-note"
        title = "Yön ayrı, plan onayı ayrı"
        text = (
            f"Piyasa yönü {final_label} ({final_score:.1f} skor) olabilir; ancak işlem için önce "
            "Planı Kontrol Et ile backtest/kalite onayı alınmalı."
        )
    elif not entry_signal_tracker.get("signal_now"):
        css = "logic-note"
        title = "Yön var, giriş mumu bekleniyor"
        text = (
            f"Piyasa yönü {final_label} ({final_score:.1f} skor). "
            "İşlem kararı için ayrıca Canlı Giriş Takibi bölümündeki mum kapanışı adımı geçmeli."
        )
    else:
        return

    st.markdown(
        f"<div class='{css}'><b>{escape(title)}</b><br>{escape(text)}</div>",
        unsafe_allow_html=True,
    )


def _summary_score(summary: pd.DataFrame, tf_name: str) -> float:
    if summary is None or summary.empty:
        return np.nan
    row = summary[summary["Zaman Dilimi"] == tf_name]
    if row.empty:
        return np.nan
    try:
        return float(row["Skor"].iloc[0])
    except Exception:
        return np.nan


def build_readiness_items(
    summary: pd.DataFrame,
    selected_tf: str,
    bt_tf: str,
    matched_quality: Optional[dict],
    allowed_quality_labels: set[str],
    setup: Optional[TradeSetup],
    practical_signal_mode: bool = True,
    signal_mode: str = "Dengeli Sinyal",
) -> list[dict]:
    data_ok = summary is not None and not summary.empty and (summary["Bias"] != "Veri yok").any()
    h4_score = _summary_score(summary, "4 Saat")
    h1_score = _summary_score(summary, "1 Saat")
    m15_score = _summary_score(summary, "15 Dakika")
    htf_long = not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score >= 25 and h1_score >= 25
    htf_short = not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score <= -25 and h1_score <= -25
    htf_ok = htf_long or htf_short

    if selected_tf == "5 Dakika":
        m15_ok = (htf_long and not pd.isna(m15_score) and m15_score >= 25) or (htf_short and not pd.isna(m15_score) and m15_score <= -25)
        m15_text = "Aynı yön teyidi var" if m15_ok else "Teyit bekleniyor"
        m15_state = "ok" if m15_ok else "warn"
    else:
        m15_text = "Bu girişte zorunlu değil"
        m15_state = "ok"

    if bt_tf != selected_tf:
        bt_state = "bad"
        bt_text = "Giriş zamanıyla eşleşmiyor"
    else:
        quality_info = quality_signal_status(
            matched_quality=matched_quality,
            allowed_quality_labels=allowed_quality_labels,
            practical_signal_mode=practical_signal_mode,
            signal_mode=signal_mode,
        )
        bt_state = quality_info["state"]
        bt_text = quality_info["text"]

    risk_state = "ok" if setup is not None and bt_state == "ok" else ("bad" if bt_state == "bad" else "warn")
    risk_text = "Seviyeler hazır" if risk_state == "ok" else ("Düşük güvenli plan" if setup is not None and bt_state == "warn" else "Risk planı kilitli")

    return [
        {"label": "Veri", "state": "ok" if data_ok else "bad", "text": "Fiyat verisi alındı" if data_ok else "Veri bekleniyor"},
        {"label": "Ana Yön", "state": "ok" if htf_ok else "warn", "text": "4H + 1H uyumlu" if htf_ok else "4H + 1H net değil"},
        {"label": "15M Teyit", "state": m15_state, "text": m15_text},
        {"label": "Backtest", "state": bt_state, "text": bt_text},
        {"label": "Risk Planı", "state": risk_state, "text": risk_text},
    ]


def _format_tracker_time(ts) -> str:
    if ts is None:
        return "-"
    try:
        return _to_istanbul_timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def _quality_state_text(matched_quality: Optional[dict], allowed_quality_labels: set[str]) -> tuple[str, str]:
    if matched_quality is None:
        return "warn", "Plan kontrolü bekliyor"
    label = matched_quality.get("label", "-")
    if label in allowed_quality_labels:
        return "ok", f"Kalite: {label}"
    return "bad", f"Kalite: {label}"


def build_entry_signal_tracker(
    symbol: str,
    selected_tf: str,
    bt_tf: str,
    final_label: str,
    setup: Optional[TradeSetup],
    matched_quality: Optional[dict],
    allowed_quality_labels: set[str],
    strict_safety_mode: bool,
    price: Optional[float],
    summary: pd.DataFrame,
    signal_threshold: float,
    final_score: float = 0.0,
    practical_signal_mode: bool = True,
    signal_mode: str = "Dengeli Sinyal",
    market_regime: Optional[dict] = None,
) -> dict:
    """Son kapanan giriş mumuna göre kullanıcıya net LONG/SHORT/BEKLE takibi verir."""
    dec = price_decimals(symbol)
    current_price_label = "-" if price is None else f"{price:.{dec}f}"
    current_price_value = float(price) if price is not None else None
    last_closed_time = "-"
    last_closed_close_label = "-"
    last_closed_close: Optional[float] = None

    prm = TIMEFRAMES[selected_tf]
    candle_df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if candle_df is not None and not candle_df.empty:
        candle_df = to_tz_index(candle_df)
        closed_i = -2 if len(candle_df) > 1 else -1
        closed_row = candle_df.iloc[closed_i]
        last_closed_time = _format_tracker_time(candle_df.index[closed_i])
        last_closed_close = float(closed_row["Close"])
        last_closed_close_label = f"{last_closed_close:.{dec}f}"
        current_price_value = float(candle_df["Close"].iloc[-1])
        current_price_label = f"{current_price_value:.{dec}f}"

    quality_info = quality_signal_status(
        matched_quality=matched_quality,
        allowed_quality_labels=allowed_quality_labels,
        practical_signal_mode=practical_signal_mode,
        signal_mode=signal_mode,
    )
    quality_state = quality_info["state"]
    quality_text = quality_info["text"]
    quality_status = quality_info["status"]
    # Gerçek pozisyon sinyali yalnızca doğrulanmış backtest kalitesiyle üretilebilir.
    quality_allows_signal = quality_status == "approved"
    same_tf_ok = bt_tf == selected_tf
    strong_required = strict_safety_mode or signal_mode_settings(signal_mode)["requires_strong_direction"]
    blocked_by_strong = strong_required and final_label not in {"Güçlü Alım Yönlü", "Güçlü Satış Yönlü"}
    entry_score = _summary_score(summary, selected_tf)
    h4_score = _summary_score(summary, "4 Saat")
    h1_score = _summary_score(summary, "1 Saat")
    m15_score = _summary_score(summary, "15 Dakika")
    technical_signal, technical_reason = mtf_signal_decision(
        entry_score=entry_score,
        h4_score=h4_score,
        h1_score=h1_score,
        m15_score=m15_score,
        tf_name=selected_tf,
        threshold=float(signal_threshold),
    )

    direction_ok = setup is not None and technical_signal == setup.side
    if blocked_by_strong:
        direction_ok = False

    side = setup.side if setup is not None else None
    side_word = "LONG AÇ" if side == "LONG" else ("SHORT AÇ" if side == "SHORT" else "BEKLE")
    trigger_level = "-" if setup is None else f"{setup.entry:.{dec}f}"
    if setup is None:
        trigger_condition = "Önce risk planı oluşmalı."
    elif side == "LONG":
        trigger_condition = f"Kapanmış {selected_tf} skoru ≥ {float(signal_threshold):.0f} ve 4H+1H LONG olmalı."
    else:
        trigger_condition = f"Kapanmış {selected_tf} skoru ≤ -{float(signal_threshold):.0f} ve 4H+1H SHORT olmalı."

    # Canlı ve backtest aynı çekirdek kararı kullanır. Fiyatı aynı mumun kendi
    # kapanışıyla karşılaştırmak yerine kapanmış mumun MTF skoru değerlendirilir.
    candle_ok = bool(last_closed_close is not None and setup is not None and technical_signal == side)

    risk_ok = setup is not None and same_tf_ok and quality_allows_signal and direction_ok
    signal_now = bool(risk_ok and candle_ok)
    confidence_score = calculate_signal_confidence(
        final_score=final_score,
        quality_status=quality_status,
        candle_ok=candle_ok,
        direction_ok=direction_ok,
        same_tf_ok=same_tf_ok,
        market_regime=market_regime,
        signal_mode=signal_mode,
    )
    decision_class, active_status_class = signal_class_for_status(side, quality_status)

    distance_to_trigger_pips = None

    if signal_now:
        action = f"ONAYLI {side_word} SİNYALİ"
        status_class = active_status_class
        summary = f"Son kapanan {selected_tf} MTF giriş şartını geçti. Güven: {confidence_label(confidence_score)}."
        final_step_text = f"{side_word} sinyali üretildi"
        primary_blocker = "Sinyal aktif"
        primary_blocker_text = "Giriş şartı tetiklendi; kalite ve risk notunu kontrol et."
    elif setup is None:
        action = "BEKLE"
        status_class = "entry-signal-wait"
        summary = "Henüz takip edilecek giriş seviyesi yok. Önce ana yön, backtest ve risk planı hazır olmalı."
        final_step_text = "Risk planı bekleniyor"
        primary_blocker = "Risk planı yok"
        primary_blocker_text = "Ana yön veya veri koşulları yeni pozisyon için giriş seviyesi üretmedi."
    elif not same_tf_ok:
        action = "BEKLE"
        status_class = "entry-signal-pass"
        summary = "Backtest zamanı ile giriş zamanı aynı olmadığı için yeni pozisyon sinyali kilitli."
        final_step_text = "Zaman dilimi eşleşmiyor"
        primary_blocker = "Zaman dilimi eşleşmiyor"
        primary_blocker_text = "Backtest zamanı, giriş zamanı ile aynı olmalı."
    elif not quality_allows_signal:
        if quality_status == "blocked":
            action = "PAS GEÇ"
            status_class = "entry-signal-pass"
            summary = f"Piyasa yönü güçlü olabilir ama {quality_text}. Bu yüzden sistem yeni pozisyon sinyali vermez."
            final_step_text = "Plan kalitesi yetersiz"
            primary_blocker = "Kalite filtresi reddetti"
            primary_blocker_text = quality_text
        else:
            action = "BEKLE"
            status_class = "entry-signal-wait"
            summary = "Plan kontrolü yapılmadan yeni pozisyon sinyali verilmez."
            final_step_text = "Plan kontrolü bekleniyor"
            primary_blocker = "Plan kontrolü bekleniyor"
            primary_blocker_text = "Planı Kontrol Et butonu ile kalite sonucu alınmalı."
    elif not direction_ok:
        action = "BEKLE"
        status_class = "entry-signal-wait"
        summary = "Ana yön koşulu giriş için yeterli değil."
        final_step_text = "Ana yön bekleniyor"
        primary_blocker = "Ana yön bekleniyor"
        primary_blocker_text = "4H + 1H ve seçilen sinyal modu aynı yönde yeterli güç üretmeli."
    else:
        action = "BEKLE"
        status_class = "entry-signal-wait"
        summary = f"Son kapanan {selected_tf} skoru MTF giriş şartını henüz geçmedi."
        final_step_text = "MTF giriş skoru bekleniyor"
        primary_blocker = "Giriş skoru bekleniyor"
        primary_blocker_text = technical_reason

    if setup is None:
        candle_text = "Giriş seviyesi yok"
    elif last_closed_close is None:
        candle_text = "Mum verisi bekleniyor"
    elif candle_ok:
        candle_text = f"Skor {entry_score:.1f}; {technical_signal} şartı geçti"
    else:
        score_text = "-" if pd.isna(entry_score) else f"{entry_score:.1f}"
        candle_text = f"Skor {score_text}; {technical_reason}"

    steps = [
        {
            "label": "1. Zaman",
            "state": "ok" if same_tf_ok else "bad",
            "text": "Giriş ve backtest aynı" if same_tf_ok else "Backtest zamanı farklı",
        },
        {"label": "2. Plan", "state": quality_state, "text": quality_text},
        {
            "label": "3. Ana Yön",
            "state": "ok" if direction_ok else ("bad" if blocked_by_strong else "warn"),
            "text": "Yön uygun" if direction_ok else "Yön bekleniyor",
        },
        {
            "label": "4. Giriş Skoru",
            "state": "ok" if candle_ok else "warn",
            "text": candle_text,
        },
        {
            "label": "5. Sinyal",
            "state": "ok" if signal_now else ("bad" if (not same_tf_ok or quality_status == "blocked") else "warn"),
            "text": final_step_text,
        },
    ]

    if setup is None:
        alarm_text = "Alarm kurulamadı; önce risk planı ve giriş seviyesi oluşmalı."
    elif signal_now:
        alarm_text = f"Alarm tetiklendi: {selected_tf} skoru {entry_score:.1f}; {technical_reason}."
    else:
        alarm_text = f"Alarm bekliyor: {symbol} {selected_tf}; {technical_reason}."

    levels = {}
    if setup is not None:
        levels = {
            "Giriş": f"{setup.entry:.{dec}f}",
            "Stop": f"{setup.stop:.{dec}f}",
            "Kâr Al": f"{setup.target:.{dec}f}",
            "Lot": f"{setup.estimated_lot:.2f}",
        }

    return {
        "action": action,
        "status_class": status_class,
        "signal_now": signal_now,
        "quality_status": quality_status,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label(confidence_score),
        "decision_class": decision_class,
        "side": side,
        "side_word": side_word,
        "summary": summary,
        "selected_tf": selected_tf,
        "condition": trigger_condition,
        "trigger_level": trigger_level,
        "entry_score": entry_score,
        "signal_threshold": float(signal_threshold),
        "technical_signal": technical_signal,
        "technical_reason": technical_reason,
        "alarm_text": alarm_text,
        "distance_to_trigger_pips": distance_to_trigger_pips,
        "primary_blocker": primary_blocker,
        "primary_blocker_text": primary_blocker_text,
        "last_closed_time": last_closed_time,
        "last_closed_close": last_closed_close_label,
        "current_price": current_price_label,
        "market_regime": market_regime or {},
        "levels": levels,
        "steps": steps,
    }


def apply_entry_signal_to_decision(decision: dict, tracker: dict) -> dict:
    if not tracker.get("signal_now"):
        return decision

    out = dict(decision)
    side = tracker.get("side")
    side_word = tracker.get("side_word", "LONG AÇ" if side == "LONG" else "SHORT AÇ")
    out.update({
        "action": tracker.get("action", f"SİSTEM {side_word} SİNYALİ"),
        "class": tracker.get("decision_class", "simple-buy" if side == "LONG" else "simple-sell"),
        "subtitle": f"{tracker.get('selected_tf', '')} kapanmış mum MTF giriş şartını teyit etti.",
        "reason": (
            f"Giriş skoru {tracker.get('entry_score', '-')} ve eşik {tracker.get('signal_threshold', '-')}. "
            f"{tracker.get('technical_reason', '')}. "
            f"Güven: {tracker.get('confidence_label', '-')}; durum: {tracker.get('quality_status', '-')}. "
            "Plan, ana yön ve kapanmış mum skoru adımları tamam."
        ),
    })
    if tracker.get("levels"):
        out["levels"] = tracker["levels"]
    if side == "LONG":
        out["steps"] = [
            f"{tracker.get('action', 'Sistem LONG AÇ sinyali')} üretildi; broker fiyatını ve spreadi kontrol et.",
            "İşleme girersen stop ve kâr al seviyelerini değiştirme.",
            "Stop seviyesine gelirse işlemden çık; stopu büyütme.",
        ]
    else:
        out["steps"] = [
            f"{tracker.get('action', 'Sistem SHORT AÇ sinyali')} üretildi; broker fiyatını ve spreadi kontrol et.",
            "İşleme girersen stop ve kâr al seviyelerini değiştirme.",
            "Stop seviyesine gelirse işlemden çık; stopu büyütme.",
        ]
    return out


def render_entry_signal_tracker(tracker: dict) -> None:
    status_class = escape(str(tracker.get("status_class", "entry-signal-wait")))
    action = escape(str(tracker.get("action", "BEKLE")))
    summary = escape(str(tracker.get("summary", "")))
    selected_tf = escape(str(tracker.get("selected_tf", "-")))
    condition = escape(str(tracker.get("condition", "-")))
    last_closed = escape(f"{tracker.get('last_closed_time', '-')} / {tracker.get('last_closed_close', '-')}")
    current_price = escape(str(tracker.get("current_price", "-")))

    st.markdown(
        (
            f"<div class='entry-signal-shell {status_class}'>"
            "<div class='entry-signal-title'>Canlı Giriş Takibi</div>"
            f"<div class='entry-signal-action'>{action}</div>"
            f"<div class='entry-signal-summary'>{summary}</div>"
            "<div class='entry-signal-meta'>"
            f"<div><b>Takip Edilen Mum</b><span>{selected_tf}</span></div>"
            f"<div><b>Giriş Şartı</b><span>{condition}</span></div>"
            f"<div><b>Son Kapanan Mum</b><span>{last_closed}</span></div>"
            f"<div><b>Anlık/Son Fiyat</b><span>{current_price}</span></div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    state_symbol = {"ok": "Geçti", "warn": "Bekle", "bad": "Kilitli"}
    parts = []
    for step in tracker.get("steps", []):
        state = str(step.get("state", "warn"))
        parts.append(
            f"<div class='entry-step entry-step-{escape(state)}'>"
            f"<b>{escape(str(step.get('label', 'Adım')))}: {escape(state_symbol.get(state, 'Bekle'))}</b>"
            f"<span>{escape(str(step.get('text', '')))}</span>"
            f"</div>"
        )
    st.markdown("<div class='entry-step-grid'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def render_readiness_checklist(items: list[dict]) -> None:
    parts = []
    state_symbol = {"ok": "Hazır", "warn": "Dikkat", "bad": "Kilitli"}
    for item in items:
        state = str(item.get("state", "warn"))
        parts.append(
            f"<div class='check-item check-{escape(state)}'>"
            f"<b>{escape(str(item.get('label', 'Kontrol')))}: {escape(state_symbol.get(state, 'Bekle'))}</b>"
            f"<span>{escape(str(item.get('text', '')))}</span>"
            f"</div>"
        )
    st.markdown("<div class='check-grid'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def _tf_score(summary: pd.DataFrame, tf_name: str) -> float:
    return _summary_score(summary, tf_name)


def _tf_bias(summary: pd.DataFrame, tf_name: str) -> str:
    if summary is None or summary.empty:
        return "Veri yok"
    row = summary[summary["Zaman Dilimi"] == tf_name]
    if row.empty:
        return "Veri yok"
    return str(row["Bias"].iloc[0])


def _beginner_side_from_scores(summary: pd.DataFrame) -> Optional[str]:
    h4 = _tf_score(summary, "4 Saat")
    h1 = _tf_score(summary, "1 Saat")
    if not pd.isna(h4) and not pd.isna(h1) and h4 >= 25 and h1 >= 25:
        return "LONG"
    if not pd.isna(h4) and not pd.isna(h1) and h4 <= -25 and h1 <= -25:
        return "SHORT"
    return None


def build_beginner_single_decision(
    symbol: str,
    summary: pd.DataFrame,
    selected_tf: str,
    setup: Optional[TradeSetup],
    matched_quality: Optional[dict],
    allowed_quality_labels: set[str],
    tracker: dict,
    price: Optional[float],
) -> dict:
    """Yeni başlayan kullanıcı için 4H/1H/15M/5M karmaşasını tek karara indirir."""
    dec = price_decimals(symbol)
    side = _beginner_side_from_scores(summary)
    quality_label = None if matched_quality is None else str(matched_quality.get("label", "-"))

    def levels_from_setup() -> dict:
        if setup is None:
            return {}
        return {
            "Giriş": f"{setup.entry:.{dec}f}",
            "Stop": f"{setup.stop:.{dec}f}",
            "Kâr Al": f"{setup.target:.{dec}f}",
            "Lot": f"{setup.estimated_lot:.2f}",
        }

    if side is None:
        return {
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Ana yön net değil.",
            "reason": "4H ve 1H aynı yönde güçlü sinyal üretmiyor. Alt zaman dilimleri ne derse desin işlem açma.",
            "steps": [
                "Yeni pozisyon açma.",
                "4H ve 1H aynı yöne dönene kadar bekle.",
                "15M veya 5M tek başına LONG/SHORT sebebi değildir.",
            ],
            "levels": {},
        }

    side_word = "LONG AÇ" if side == "LONG" else "SHORT AÇ"
    side_text = "alım" if side == "LONG" else "satış"
    m15_ok = tracker.get("technical_signal") == side

    if matched_quality is None:
        return {
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Strateji kontrolü yapılıyor.",
            "reason": "Plan kontrolü henüz tamamlanmadı. Otomatik kontrol açık değilse Yeniden Hesapla butonuna bas.",
            "steps": [
                "Şimdilik işlem açma.",
                "Strateji Kalitesi İyi/Orta olmadan gerçek işlem alma.",
                "Kontrol bitince bu kart tek karar verecek.",
            ],
            "levels": {},
        }

    if quality_label not in allowed_quality_labels:
        return {
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Backtest kalite filtresi işlemi reddetti.",
            "reason": f"Ana yön {side_text} olabilir ama strateji kalitesi {quality_label}. İşlem için kalite {', '.join(sorted(allowed_quality_labels))} olmalı.",
            "steps": [
                "Bu paritede bu ayarla işlem açma.",
                "Başka parite tara veya daha yüksek zaman dilimi dene.",
                "Kalite filtresi düzelmeden gerçek pozisyon açma.",
            ],
            "levels": {},
        }

    if setup is None:
        return {
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Yön var ama giriş planı yok.",
            "reason": "Risk planı üretilemedi. Veri veya ATR koşulları yeterli değil.",
            "steps": ["İşlem açma.", "Veri yenile veya başka parite dene.", "Risk planı oluşmadan işlem alma."],
            "levels": {},
        }

    if not m15_ok:
        return {
            "action": f"{side_word} İÇİN BEKLE",
            "class": "simple-wait",
            "subtitle": f"4H + 1H {side_text} yönünde ama 15M henüz hazır değil.",
            "reason": f"Ana yön var; giriş zamanı için kapanmış 15M skoru bekleniyor. {tracker.get('technical_reason', '')}",
            "steps": [
                f"{tracker.get('condition', '15M giriş skoru eşiği geçmeden işlem açma.')}",
                "4H ve 1H aynı yönde kalmalı.",
                "Kapanmış mum skoru oluşmadan acele etme.",
            ],
            "levels": levels_from_setup(),
        }

    if tracker.get("signal_now"):
        return {
            "action": side_word,
            "class": "simple-buy" if side == "LONG" else "simple-sell",
            "subtitle": f"Giriş şartı tamamlandı: {side_word} sinyali aktif.",
            "reason": f"4H+1H yön uygun, 15M teyit var, kalite {quality_label}. Broker fiyatı/spread kontrolü yapmadan emir verme.",
            "steps": [
                f"Broker fiyatı uygunsa {side_word} işlemi değerlendir.",
                f"Giriş: {setup.entry:.{dec}f} | Stop: {setup.stop:.{dec}f} | Kâr Al: {setup.target:.{dec}f}",
                "Stop seviyesini büyütme; plan bozulursa çık.",
            ],
            "levels": levels_from_setup(),
        }

    return {
        "action": f"{side_word} İÇİN BEKLE",
        "class": "simple-wait",
        "subtitle": f"Yön {side_text}; kapanmış mum giriş skoru henüz yeterli değil.",
        "reason": f"{tracker.get('technical_reason', 'MTF giriş skoru uygun değil')}. Şart: {tracker.get('condition', '-')}",
        "steps": [
            f"{tracker.get('condition', '15M giriş skoru eşiği geçmeden işlem açma.')}",
            f"Şart geçerse stop {setup.stop:.{dec}f}, kâr al {setup.target:.{dec}f} kullan.",
            "Mum kapanmadan oluşan geçici skoru sinyal kabul etme.",
        ],
        "levels": levels_from_setup(),
    }


def render_beginner_path(summary: pd.DataFrame, matched_quality: Optional[dict], tracker: dict, selected_tf: str) -> None:
    """Yeni başlayan modda sadece karar hunisini gösterir; 4 ayrı zaman dilimini yorumlatmaz."""
    side = _beginner_side_from_scores(summary)
    side_text = "Alım" if side == "LONG" else ("Satış" if side == "SHORT" else "Yok")
    m15_ok = tracker.get("technical_signal") == side
    quality_label = "Bekliyor" if matched_quality is None else str(matched_quality.get("label", "-"))
    quality_ok = quality_label in {"İyi", "Orta"}
    signal_now = bool(tracker.get("signal_now"))

    items = [
        {"label": "1. Ana Yön", "state": "ok" if side else "bad", "text": f"4H + 1H sonucu: {side_text}"},
        {"label": "2. Giriş Zamanı", "state": "ok" if m15_ok else "warn", "text": "15M teyit var" if m15_ok else "15M teyit bekleniyor"},
        {"label": "3. Strateji Kalitesi", "state": "ok" if quality_ok else "bad", "text": f"Kalite: {quality_label}"},
        {"label": "4. Son Mum", "state": "ok" if signal_now else "warn", "text": "Giriş şartı geçti" if signal_now else str(tracker.get("primary_blocker_text", "Kapanış bekleniyor"))},
        {"label": "5M", "state": "ok", "text": "Yeni başlayan modda karar verici değil"},
    ]
    render_readiness_checklist(items)


def build_position_tracker_result(
    symbol: str,
    side: str,
    entry: float,
    current_price: Optional[float],
    stop: float,
    target: float,
    lot: float,
    pip_value_per_lot: float,
    final_label: str,
) -> dict:
    dec = price_decimals(symbol)
    if current_price is None or entry <= 0:
        return {"action": "BİLGİ YOK", "class": "simple-wait", "text": "Güncel fiyat veya giriş fiyatı alınamadı.", "pips": None, "pnl": None}

    pips = calculate_manual_pips(symbol, side, entry, current_price)
    pnl = None if pips is None else pips * pip_value_per_lot * lot
    pip = get_pip_size(symbol)
    stop_distance_pips = abs(entry - stop) / pip if stop > 0 else None
    target_distance_pips = abs(target - entry) / pip if target > 0 else None
    r_multiple = None
    if pips is not None and stop_distance_pips and stop_distance_pips > 0:
        r_multiple = pips / stop_distance_pips
    if side == "LONG":
        to_stop_pips = (current_price - stop) / pip if stop > 0 else None
        to_target_pips = (target - current_price) / pip if target > 0 else None
    else:
        to_stop_pips = (stop - current_price) / pip if stop > 0 else None
        to_target_pips = (current_price - target) / pip if target > 0 else None

    opposite = False
    neutral = final_label == "İşlem Yok"
    if side == "LONG" and "Satış" in final_label:
        opposite = True
    if side == "SHORT" and "Alım" in final_label:
        opposite = True

    action = "POZİSYONU TUT"
    css = "simple-buy" if pips is not None and pips >= 0 else "simple-wait"
    reason = "Plan bozulmadı. Stop ve kâr al seviyelerini takip et."

    if side == "LONG":
        if stop > 0 and current_price <= stop:
            action, css, reason = "POZİSYONU KAPAT", "simple-sell", f"Fiyat stop seviyesine geldi/altına indi: {stop:.{dec}f}."
        elif target > 0 and current_price >= target:
            action, css, reason = "KÂR AL / POZİSYONU KAPAT", "simple-buy", f"Fiyat hedef seviyeye geldi/üstüne çıktı: {target:.{dec}f}."
        elif to_target_pips is not None and target_distance_pips and 0 <= to_target_pips <= max(target_distance_pips * 0.15, 2):
            action, css, reason = "KÂR AL SEVİYESİNE YAKLAŞTI", "simple-buy", "Fiyat hedefe yaklaştı; plan dışı acele etmeden kâr al/stop takibi yap."
        elif opposite:
            action, css, reason = "KAPATMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."
    else:
        if stop > 0 and current_price >= stop:
            action, css, reason = "POZİSYONU KAPAT", "simple-sell", f"Fiyat stop seviyesine geldi/üstüne çıktı: {stop:.{dec}f}."
        elif target > 0 and current_price <= target:
            action, css, reason = "KÂR AL / POZİSYONU KAPAT", "simple-buy", f"Fiyat hedef seviyeye geldi/altına indi: {target:.{dec}f}."
        elif to_target_pips is not None and target_distance_pips and 0 <= to_target_pips <= max(target_distance_pips * 0.15, 2):
            action, css, reason = "KÂR AL SEVİYESİNE YAKLAŞTI", "simple-buy", "Fiyat hedefe yaklaştı; plan dışı acele etmeden kâr al/stop takibi yap."
        elif opposite:
            action, css, reason = "KAPATMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."

    risk_note = "Stop ve hedef plana göre izleniyor."
    if to_stop_pips is not None and to_stop_pips <= 0:
        risk_note = "Stop seviyesi tetiklendi."
    elif to_stop_pips is not None and stop_distance_pips and to_stop_pips <= max(stop_distance_pips * 0.25, 2):
        risk_note = "Stopa yakın; stopu büyütme."
    elif r_multiple is not None and r_multiple >= 1:
        risk_note = "Pozisyon en az 1R kâr bölgesinde."

    return {
        "action": action,
        "class": css,
        "text": reason,
        "pips": pips,
        "pnl": pnl,
        "r_multiple": r_multiple,
        "to_stop_pips": to_stop_pips,
        "to_target_pips": to_target_pips,
        "risk_note": risk_note,
    }


def render_position_tracker_result(result: dict, current_price: Optional[float], symbol: str) -> None:
    dec = price_decimals(symbol)
    price_txt = "-" if current_price is None else f"{current_price:.{dec}f}"
    pips_txt = "-" if result.get("pips") is None else f"{result['pips']:+.1f} pip"
    pnl_txt = "-" if result.get("pnl") is None else f"{result['pnl']:+.2f}"
    r_txt = "-" if result.get("r_multiple") is None else f"{result['r_multiple']:+.2f}R"
    stop_txt = "-" if result.get("to_stop_pips") is None else f"{result['to_stop_pips']:.1f} pip"
    target_txt = "-" if result.get("to_target_pips") is None else f"{result['to_target_pips']:.1f} pip"

    card_class = escape(str(result.get("class", "simple-wait")))
    action = escape(str(result.get("action", "POZİSYONU TUT")))
    reason = escape(str(result.get("text", "")))
    risk_note = escape(str(result.get("risk_note", "")))
    html = (
        f"<div class='simple-card {card_class}'>"
        f"<div class='simple-action'>{action}</div>"
        f"<div class='simple-subtitle'>Açık pozisyon takip sonucu</div>"
        f"<div><b>Sebep:</b> {reason}</div>"
        f"<div style='margin-top:6px;'><b>Risk Notu:</b> {risk_note}</div>"
        f"<div class='simple-levels'>"
        f"<div class='simple-level'><b>Güncel Fiyat</b><span>{escape(price_txt)}</span></div>"
        f"<div class='simple-level'><b>Pip</b><span>{escape(pips_txt)}</span></div>"
        f"<div class='simple-level'><b>Tahmini PnL</b><span>{escape(pnl_txt)}</span></div>"
        f"<div class='simple-level'><b>R</b><span>{escape(r_txt)}</span></div>"
        f"<div class='simple-level'><b>Stop Mesafe</b><span>{escape(stop_txt)}</span></div>"
        f"<div class='simple-level'><b>Hedef Mesafe</b><span>{escape(target_txt)}</span></div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)



# =============================================================================
# PLOTS
# =============================================================================

def plot_main_figure(symbol: str, tf_name: str) -> tuple[go.Figure, pd.DataFrame]:
    prm = TIMEFRAMES[tf_name]
    df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=600, title="Veri alınamadı")
        return fig, df

    df = to_tz_index(add_indicators(df))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.04,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Fiyat", line=dict(width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BBUp"], name="BB Üst", line=dict(width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BBLow"], name="BB Alt", line=dict(width=0.8), fill="tonexty"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(width=1.2)), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", row=2, col=1)
    fig.add_hline(y=50, line_width=1, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    hist = df["MACDHist"].fillna(0)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=1.1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACDSignal"], name="Signal", line=dict(width=1.1)), row=3, col=1)

    fig.update_layout(
        height=720,
        margin=dict(l=30, r=20, t=45, b=30),
        title=f"{symbol} | {tf_name}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig, df


def gauge_figure(label: str, score: float) -> go.Figure:
    # Plotly gauge: -100 satış, 0 no trade, +100 alım.
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " skor"},
        title={"text": label},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"thickness": 0.28},
            "steps": [
                {"range": [-100, -65], "color": "#f8d7da"},
                {"range": [-65, -25], "color": "#fde2e1"},
                {"range": [-25, 25], "color": "#e9ecef"},
                {"range": [25, 65], "color": "#d1e7dd"},
                {"range": [65, 100], "color": "#badbcc"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_equity_curve(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity is None or equity.empty:
        fig.update_layout(height=300, title="Equity verisi yok")
        return fig
    fig.add_trace(go.Scatter(x=equity["Time"], y=equity["Balance"], mode="lines", name="Balance"))
    fig.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=30), title="Backtest Equity Curve")
    return fig


def plot_live_trigger(symbol: str, selected_tf: str, global_label: str) -> go.Figure:
    # 5M veya 1M değil; seçili kısa periyotta son sinyali gösterir.
    prm = TIMEFRAMES[selected_tf]
    df = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=320, title="Veri yok")
        return fig

    df = to_tz_index(add_indicators(df.tail(250)))
    score = score_series_for_backtest(df).reindex(df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))

    long_allowed = global_label in {"Alım Yönlü", "Güçlü Alım Yönlü"}
    short_allowed = global_label in {"Satış Yönlü", "Güçlü Satış Yönlü"}

    long_points = df[(score >= 60) & long_allowed]
    short_points = df[(score <= -60) & short_allowed]

    fig.add_trace(go.Scatter(x=long_points.index, y=long_points["Close"], mode="markers", name="Long trigger",
                             marker=dict(symbol="triangle-up", size=10)))
    fig.add_trace(go.Scatter(x=short_points.index, y=short_points["Close"], mode="markers", name="Short trigger",
                             marker=dict(symbol="triangle-down", size=10)))

    fig.update_layout(height=340, margin=dict(l=30, r=20, t=45, b=30), title=f"{selected_tf} Giriş Tetikleyici | Ana Yön Filtresi: {global_label}")
    return fig



# =============================================================================
# MACHINE LEARNING FILTER
# =============================================================================

ML_FEATURE_COLUMNS = [
    "side_long",
    "entry_score_aligned",
    "h4_score_aligned",
    "h1_score_aligned",
    "m15_score_aligned",
    "abs_entry_score",
    "agreement_count",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def _safe_float_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def build_ml_dataset_from_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Backtest işlemlerini ML eğitim verisine çevirir.
    Hedef: Bu sinyal TP/pozitif sonuç verdi mi?
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=ML_FEATURE_COLUMNS), pd.Series(dtype=int)

    needed = ["Side", "Entry Score", "4H Score", "1H Score", "15M Score", "Entry Time", "PnL"]
    missing = [c for c in needed if c not in trades.columns]
    if missing:
        return pd.DataFrame(columns=ML_FEATURE_COLUMNS), pd.Series(dtype=int)

    t = trades.copy()
    t["Entry Time"] = pd.to_datetime(t["Entry Time"], utc=True, errors="coerce")
    t = t.dropna(subset=["Entry Time", "Side", "PnL"])

    entry_score = _safe_float_col(t, "Entry Score")
    h4_score = _safe_float_col(t, "4H Score")
    h1_score = _safe_float_col(t, "1H Score")
    m15_score = _safe_float_col(t, "15M Score")
    pnl = _safe_float_col(t, "PnL")

    side_long = (t["Side"].astype(str).str.upper() == "LONG").astype(int)
    side_mult = np.where(side_long == 1, 1.0, -1.0)

    x = pd.DataFrame(index=t.index)
    x["side_long"] = side_long
    x["entry_score_aligned"] = entry_score * side_mult
    x["h4_score_aligned"] = h4_score * side_mult
    x["h1_score_aligned"] = h1_score * side_mult
    x["m15_score_aligned"] = m15_score * side_mult
    x["abs_entry_score"] = entry_score.abs()
    aligned_parts = pd.concat([
        x["entry_score_aligned"],
        x["h4_score_aligned"],
        x["h1_score_aligned"],
        x["m15_score_aligned"],
    ], axis=1)
    x["agreement_count"] = (aligned_parts >= 25).sum(axis=1)
    local_entry_time = t["Entry Time"].dt.tz_convert(TR_TZ)
    hour = local_entry_time.dt.hour + local_entry_time.dt.minute / 60
    weekday = local_entry_time.dt.weekday
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    y = (pnl > 0).astype(int)
    valid = x.replace([np.inf, -np.inf], np.nan).dropna().index
    x = x.loc[valid, ML_FEATURE_COLUMNS].astype(float)
    y = y.loc[valid].astype(int)
    return x, y


def train_ml_model_from_backtest(bt_result: Optional[BacktestResult], min_samples: int = 50) -> dict:
    """
    Backtest sonuçlarından basit bir RandomForest sınıflandırıcı eğitir.
    Not: Bu model karar verici değil, sinyal kalite filtresidir.
    """
    if not SKLEARN_AVAILABLE:
        return {
            "status": "not_available",
            "label": "ML pasif",
            "text": "scikit-learn kurulu değil. requirements.txt içine scikit-learn eklenmeli.",
        }

    if bt_result is None or bt_result.trades is None or bt_result.trades.empty:
        return {
            "status": "no_data",
            "label": "ML bekliyor",
            "text": "ML eğitimi için önce backtest sonucunda işlem oluşmalı.",
        }

    x, y = build_ml_dataset_from_trades(bt_result.trades)
    n = len(x)
    if n < int(min_samples):
        return {
            "status": "insufficient",
            "label": "ML yetersiz örnek",
            "text": f"ML eğitimi için {int(min_samples)} işlem isteniyor; mevcut örnek: {n}.",
            "sample_count": n,
        }

    if y.nunique() < 2:
        return {
            "status": "one_class",
            "label": "ML eğitilemedi",
            "text": "Backtest işlemlerinde tek sınıf var. Hem kazanan hem kaybeden örnek gerekli.",
            "sample_count": n,
        }

    train_end = max(15, int(n * 0.60))
    calibration_end = max(train_end + 8, int(n * 0.80))
    calibration_end = min(calibration_end, n - 8)
    x_train, x_calibration, x_test = x.iloc[:train_end], x.iloc[train_end:calibration_end], x.iloc[calibration_end:]
    y_train, y_calibration, y_test = y.iloc[:train_end], y.iloc[train_end:calibration_end], y.iloc[calibration_end:]

    if y_train.nunique() < 2 or y_calibration.nunique() < 2:
        return {
            "status": "class_imbalance",
            "label": "ML kalibre edilemedi",
            "text": "Kronolojik eğitim veya kalibrasyon bölümünde hem kazanan hem kaybeden işlem yok.",
            "sample_count": n,
        }

    quality_note = "" if y_test.nunique() == 2 else "Test bölümünde tek sınıf var; AUC yorumlanamaz."

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    calibration_scores = model.predict_proba(x_calibration)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(calibration_scores, y_calibration)

    raw_test_proba = model.predict_proba(x_test)[:, 1].reshape(-1, 1)
    proba = calibrator.predict_proba(raw_test_proba)[:, 1]
    pred = (proba >= 0.5).astype(int)

    baseline = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    baseline.fit(x_train, y_train)
    baseline_proba = baseline.predict_proba(x_test)[:, 1]

    accuracy = float(accuracy_score(y_test, pred)) if accuracy_score is not None and len(y_test) else np.nan
    brier = float(brier_score_loss(y_test, proba)) if brier_score_loss is not None and len(y_test) else np.nan
    try:
        auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else np.nan
        baseline_auc = float(roc_auc_score(y_test, baseline_proba)) if y_test.nunique() == 2 else np.nan
    except Exception:
        auc = np.nan
        baseline_auc = np.nan

    train_std = x_train.std(ddof=0).replace(0, np.nan)
    standardized_shift = ((x_test.mean() - x_train.mean()).abs() / train_std).replace([np.inf, -np.inf], np.nan)
    drift_score = float(standardized_shift.mean()) if standardized_shift.notna().any() else 0.0
    drift_label = "Yüksek" if drift_score >= 0.75 else ("Orta" if drift_score >= 0.35 else "Düşük")

    return {
        "status": "ready",
        "label": "ML hazır",
        "text": quality_note or "ML modeli backtest sinyallerinden eğitildi.",
        "model": model,
        "calibrator": calibrator,
        "feature_columns": ML_FEATURE_COLUMNS,
        "sample_count": n,
        "train_count": len(x_train),
        "calibration_count": len(x_calibration),
        "test_count": len(x_test),
        "test_accuracy": accuracy,
        "test_auc": auc,
        "baseline_auc": baseline_auc,
        "test_brier": brier,
        "drift_score": drift_score,
        "drift_label": drift_label,
        "historical_win_rate": float(y.mean()),
    }


def build_current_ml_feature(summary: pd.DataFrame, selected_tf: str, setup: Optional[TradeSetup], final_score: float) -> Optional[pd.DataFrame]:
    if setup is None:
        return None

    side = setup.side
    side_long = 1 if side == "LONG" else 0
    side_mult = 1.0 if side == "LONG" else -1.0

    entry_score = _tf_score(summary, selected_tf)
    if pd.isna(entry_score):
        entry_score = final_score

    h4_score = _tf_score(summary, "4 Saat")
    h1_score = _tf_score(summary, "1 Saat")
    m15_score = _tf_score(summary, "15 Dakika")

    now_local = pd.Timestamp.now(tz=TR_TZ)
    hour = now_local.hour + now_local.minute / 60
    weekday = now_local.weekday()
    vals = {
        "side_long": side_long,
        "entry_score_aligned": float(entry_score) * side_mult,
        "h4_score_aligned": float(0 if pd.isna(h4_score) else h4_score) * side_mult,
        "h1_score_aligned": float(0 if pd.isna(h1_score) else h1_score) * side_mult,
        "m15_score_aligned": float(0 if pd.isna(m15_score) else m15_score) * side_mult,
        "abs_entry_score": abs(float(entry_score)),
        "agreement_count": 0.0,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * weekday / 7)),
    }

    aligned_scores = [
        vals["entry_score_aligned"],
        vals["h4_score_aligned"],
        vals["h1_score_aligned"],
        vals["m15_score_aligned"],
    ]
    vals["agreement_count"] = float(sum(v >= 25 for v in aligned_scores))

    return pd.DataFrame([vals], columns=ML_FEATURE_COLUMNS).astype(float)


def build_live_ml_prediction(
    bt_result: Optional[BacktestResult],
    summary: pd.DataFrame,
    selected_tf: str,
    setup: Optional[TradeSetup],
    final_score: float,
    min_samples: int = 50,
) -> dict:
    model_info = train_ml_model_from_backtest(bt_result, min_samples=min_samples)
    if model_info.get("status") != "ready":
        return model_info

    x_live = build_current_ml_feature(summary, selected_tf, setup, final_score)
    if x_live is None:
        model_info.update({
            "status": "no_setup",
            "label": "ML bekliyor",
            "text": "ML olasılığı için önce LONG/SHORT yönünde risk planı oluşmalı.",
        })
        return model_info

    model = model_info["model"]
    calibrator = model_info.get("calibrator")
    try:
        raw_probability = float(model.predict_proba(x_live[ML_FEATURE_COLUMNS])[:, 1][0])
        probability = float(calibrator.predict_proba(np.array([[raw_probability]]))[:, 1][0])
    except Exception:
        probability = np.nan

    model_info["probability"] = probability
    model_info["probability_pct"] = None if pd.isna(probability) else probability * 100
    model_info["side"] = setup.side if setup is not None else None
    model_info["label"] = "ML tahmini hazır"
    model_info["text"] = (
        "Bu skor kronolojik eğitim, kalibrasyon ve test bölümleriyle hesaplandı. "
        "Kesinlik değil, yalnızca ek kalite filtresidir."
    )
    return model_info


def ml_should_block_trade(ml_prediction: dict, threshold_pct: float, filter_enabled: bool) -> tuple[bool, str]:
    if not filter_enabled:
        return False, "ML filtresi kapalı."

    status = ml_prediction.get("status")
    if status != "ready":
        return True, ml_prediction.get("text", "ML modeli hazır değil.")

    drift_score = ml_prediction.get("drift_score")
    if drift_score is not None and not pd.isna(drift_score) and float(drift_score) >= 0.75:
        return True, f"ML özellik dağılımı eğitimden uzaklaştı; drift skoru {float(drift_score):.2f}."

    prob = ml_prediction.get("probability_pct")
    if prob is None or pd.isna(prob):
        return True, "ML olasılığı hesaplanamadı."

    if float(prob) < float(threshold_pct):
        return True, f"ML güveni %{float(prob):.1f}; minimum eşik %{float(threshold_pct):.0f}."

    return False, f"ML güveni %{float(prob):.1f}; eşik geçildi."


def is_new_position_decision(action: str) -> bool:
    a = str(action).upper()
    if "PAS" in a:
        return False
    if "KAPAT" in a or "TUT" in a:
        return False
    return ("LONG" in a or "SHORT" in a)


def apply_ml_filter_to_decision(decision: dict, ml_prediction: dict, filter_enabled: bool, threshold_pct: float) -> dict:
    if not filter_enabled:
        return decision

    action = str(decision.get("action", ""))
    if not is_new_position_decision(action):
        return decision

    block, reason = ml_should_block_trade(ml_prediction, threshold_pct, filter_enabled)
    if not block:
        out = dict(decision)
        out["reason"] = f"{out.get('reason', '')} ML filtresi geçti: {reason}"
        return out

    out = dict(decision)
    out.update({
        "action": "PAS GEÇ",
        "class": "simple-pass",
        "subtitle": "ML filtresi yeni pozisyonu reddetti.",
        "reason": reason,
        "steps": [
            "Bu sinyalde yeni pozisyon açma.",
            "ML güveni eşik üstüne çıkmadan veya yeni backtest oluşmadan bekle.",
            "Başka pariteyi Alarm Ekranı veya İşlem Asistanı ile kontrol et.",
        ],
    })
    return out


def apply_operational_safety_filters(
    decision: dict,
    data_health: dict,
    news_status: dict,
    portfolio_status: dict,
    market_regime: dict,
    block_sideways: bool,
) -> dict:
    """Veri, haber, rejim ve portföy limitleri yeni pozisyon üzerinde son sözü söyler."""
    if not is_new_position_decision(str(decision.get("action", ""))):
        return decision
    blockers = []
    if data_health.get("blocks_trade"):
        blockers.append(f"Veri: {data_health.get('text', '-')}")
    if news_status.get("blocks_trade"):
        blockers.append(f"Haber: {news_status.get('text', '-')}")
    if portfolio_status.get("blocks_trade"):
        blockers.append(f"Portföy: {portfolio_status.get('text', '-')}")
    if block_sideways and market_regime.get("label") == "Yatay":
        blockers.append("Piyasa rejimi yatay; trend işlemi engellendi.")
    if not blockers:
        return decision
    out = dict(decision)
    out.update({
        "action": "PAS GEÇ",
        "class": "simple-pass",
        "subtitle": "Operasyonel güvenlik filtresi yeni pozisyonu engelledi.",
        "reason": " | ".join(blockers),
        "steps": ["Yeni pozisyon açma.", "Engel kalktıktan sonra kapanmış mumla sinyali yeniden hesapla.", "Mevcut stopları genişletme."],
    })
    return out


def render_ml_prediction_card(ml_prediction: dict, threshold_pct: float, filter_enabled: bool) -> None:
    if not filter_enabled:
        return

    status = ml_prediction.get("status", "unknown")
    label = escape(str(ml_prediction.get("label", "ML")))
    text = escape(str(ml_prediction.get("text", "")))

    if status == "ready":
        prob = ml_prediction.get("probability_pct")
        prob_txt = "-" if prob is None or pd.isna(prob) else f"%{float(prob):.1f}"
        acc = ml_prediction.get("test_accuracy")
        auc = ml_prediction.get("test_auc")
        baseline_auc = ml_prediction.get("baseline_auc")
        brier = ml_prediction.get("test_brier")
        drift_label = ml_prediction.get("drift_label", "-")
        drift_score = ml_prediction.get("drift_score")
        acc_txt = "-" if acc is None or pd.isna(acc) else f"%{float(acc)*100:.1f}"
        auc_txt = "-" if auc is None or pd.isna(auc) else f"{float(auc):.2f}"
        baseline_auc_txt = "-" if baseline_auc is None or pd.isna(baseline_auc) else f"{float(baseline_auc):.2f}"
        brier_txt = "-" if brier is None or pd.isna(brier) else f"{float(brier):.3f}"
        drift_txt = "-" if drift_score is None or pd.isna(drift_score) else f"{drift_label} ({float(drift_score):.2f})"
        sample_count = ml_prediction.get("sample_count", "-")
        css = "ok-box" if prob is not None and not pd.isna(prob) and float(prob) >= float(threshold_pct) else "bad-box"
        st.markdown(
            f"<div class='{css}'><b>{label}</b><br>"
            f"Pozitif işlem olasılığı: <b>{prob_txt}</b> | Minimum eşik: %{float(threshold_pct):.0f}<br>"
            f"Örnek: {sample_count} | Test doğruluk: {acc_txt} | RF AUC: {auc_txt} | Baseline AUC: {baseline_auc_txt}<br>"
            f"Brier: {brier_txt} | Özellik drift: {drift_txt}<br>"
            f"{text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='warn-box'><b>{label}</b><br>{text}</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# UI
# =============================================================================

with st.sidebar:
    st.header("Kontrol Paneli")

    screen_mode = st.radio("Ekran", ["İşlem Asistanı", "Parite Alarm Ekranı"], index=0)

    tf_options = list(TIMEFRAMES.keys())
    default_symbol = st.session_state.get("symbol", "EURUSD=X")
    selected_symbol = st.selectbox(
        "Parite",
        options=SYMBOL_LIST,
        index=SYMBOL_LIST.index(default_symbol) if default_symbol in SYMBOL_LIST else 0,
    )
    manual_symbol = st.text_input("Elle gir", value="", placeholder="EURUSD veya EURUSD=X")
    symbol = normalize_symbol(manual_symbol) if manual_symbol.strip() else selected_symbol
    st.session_state["symbol"] = symbol
    data_provider = st.selectbox("Veri kaynağı", ["Yahoo Finance", "MetaTrader 5"], index=0)
    st.session_state["data_provider"] = data_provider
    if data_provider == "MetaTrader 5":
        st.caption("MT5 terminali/kitaplığı hazır değilse Yahoo verisine otomatik dönülür.")

    chart_tf = st.radio("Grafik zamanı", tf_options, index=1)
    st.caption("Bu seçim grafiği değiştirir. Yeni Başlayan Modu açıksa işlem kararı yine 4H + 1H ana yön ve 15M giriş mantığıyla hesaplanır.")

    st.divider()
    st.subheader("Temel Risk")
    account_size = st.number_input("Hesap büyüklüğü", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("İşlem başına risk %", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    if risk_pct > 1.0:
        st.warning("%1 üzerindeki işlem riski kayıp serilerinde hesabı hızlı küçültebilir.")

    with st.expander("Gelişmiş risk", expanded=False):
        rr = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        atr_mult = st.number_input("ATR Stop Çarpanı", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        stop_mode = st.selectbox("Stop modeli", ["ATR", "Swing + ATR", "Hibrit (uzak olan)"], index=2)
        target_mode = st.selectbox("Hedef modeli", ["Sabit R", "Yapı / minimum 1R"], index=0)
        swing_lookback = st.number_input("Swing bakış mumu", min_value=3, max_value=100, value=10, step=1)
        max_holding_bars = st.number_input("Maksimum işlem süresi (mum, 0=kapalı)", min_value=0, max_value=500, value=0, step=5)
        break_even_at_r = st.number_input("Başabaş taşıma eşiği (R, 0=kapalı)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
        pip_value_estimate = estimate_pip_value_per_lot_usd(symbol, fetch_last_price(symbol))
        pip_value_default = round(float(pip_value_estimate), 2) if pip_value_estimate and pip_value_estimate > 0 else 10.0
        pip_value_per_lot = st.number_input(
            "1 lot için yaklaşık pip değeri",
            min_value=0.1,
            value=float(pip_value_default),
            step=0.5,
            key=f"pip_value_per_lot_{symbol}",
        )
        if pip_value_estimate and pip_value_estimate > 0:
            st.caption(f"USD hesap varsayımıyla otomatik tahmin: {pip_value_estimate:.2f}.")
        else:
            st.caption("Pip değeri otomatik tahmin edilemedi; brokerındaki değeri gir.")
        max_total_risk_pct = st.number_input("Maksimum toplam açık risk %", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        max_currency_risk_pct = st.number_input("İlişkili para birimi risk limiti %", min_value=0.5, max_value=10.0, value=1.5, step=0.5)
        max_open_positions = st.number_input("Maksimum açık pozisyon", min_value=1, max_value=20, value=3, step=1)
        daily_stop_r = st.number_input("Günlük kill-switch (R)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        weekly_stop_r = st.number_input("Haftalık kill-switch (R)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

    with st.expander("Ekran ve güvenlik", expanded=False):
        beginner_mode = st.checkbox(
            "Yeni Başlayan Modu (tek karar)",
            value=True,
            help="4H/1H/15M/5M ayrımını sana yorumlatmaz. 4H+1H ana yön, 15M giriş, 5M ise sadece arka planda kalır.",
        )
        auto_plan_control = st.checkbox(
            "Otomatik plan kontrolü",
            value=True,
            help="Parite, grafik zamanı veya risk ayarı değişince backtest/kalite kontrolünü otomatik yeniler.",
        )
        enable_simple_mode = st.checkbox("Basit İşlem Modu", value=True)
        practical_signal_mode = st.checkbox("Pratik Sinyal Modu", value=True)
        signal_mode = st.selectbox("Sinyal modu", SIGNAL_MODES, index=0)
        strict_safety_mode = st.checkbox("Sert Güvenli Mod", value=False)
        block_sideways = st.checkbox("Yatay piyasada trend işlemini engelle", value=True)
        show_position_tracker = st.checkbox("Pozisyon Takip Modu", value=True)
        change_window_label = st.selectbox("Yüzde değişim periyodu", list(PRICE_CHANGE_WINDOWS.keys()), index=1)
        change_window_minutes = PRICE_CHANGE_WINDOWS[change_window_label]

    with st.expander("Makine öğrenmesi", expanded=False):
        ml_filter_enabled = st.checkbox(
            "ML filtresi aktif",
            value=False,
            help="Açık olursa teknik sinyalin geçmiş benzer örneklerdeki başarı olasılığı hesaplanır. Eşik altında yeni pozisyon reddedilir.",
        )
        ml_threshold_pct = st.slider("Minimum ML güveni %", min_value=50, max_value=80, value=60, step=5)
        ml_min_samples = st.number_input("ML minimum işlem örneği", min_value=50, max_value=500, value=80, step=10)
        if not SKLEARN_AVAILABLE:
            st.warning("ML için scikit-learn kurulu değil. requirements.txt içine scikit-learn ekle.")
        else:
            st.caption("ML modeli, son backtestte oluşan işlemlerden otomatik eğitilir. Karar verici değil, ek kalite filtresidir.")


    with st.expander("Alarm ekranı ayarları", expanded=screen_mode == "Parite Alarm Ekranı"):
        alert_groups = st.multiselect("Gösterilecek gruplar", list(ALERT_PAIR_GROUPS.keys()), default=list(ALERT_PAIR_GROUPS.keys()))
        alert_entry_tf = st.selectbox("Alarm giriş teyidi", ["15 Dakika", "5 Dakika", "1 Saat"], index=0)
        alert_sort_mode = st.selectbox("Sıralama", ["Önce LONG/SHORT", "Sadece LONG-SHORT üstte", "En yüksek skor"], index=0)
        webhook_url = st.text_input("Webhook URL (opsiyonel)", value=os.getenv("FOREX_WEBHOOK_URL", ""), type="password")
        st.caption("Alarm ekranı hızlı takip içindir. Yeni başlayan kullanımda 15 Dakika önerilir.")

    with st.expander("Ekonomik haber filtresi", expanded=False):
        news_filter_enabled = st.checkbox("Yüksek etkili haber filtresi", value=True)
        news_before_minutes = st.number_input("Haber öncesi blok (dk)", min_value=0, max_value=240, value=30, step=5)
        news_after_minutes = st.number_input("Haber sonrası blok (dk)", min_value=0, max_value=240, value=20, step=5)
        news_file = st.file_uploader("Haber CSV yükle", type=["csv"], help="Sütunlar: time,currency,title,impact. time ISO/UTC olabilir.")
        news_events = pd.DataFrame()
        if news_file is not None:
            try:
                news_events = pd.read_csv(news_file)
                st.caption(f"{len(news_events)} haber kaydı yüklendi.")
            except Exception as exc:
                st.warning(f"Haber CSV okunamadı: {exc}")

    decision_tf = "15 Dakika" if beginner_mode else chart_tf
    if beginner_mode:
        st.caption("Yeni Başlayan Modu aktif: karar 4H+1H ana yön + 15M giriş mantığıyla tek sonuca indirilir. 5M yorumu sana gösterilmez.")

    with st.expander("Backtest ayarları", expanded=False):
        if enable_simple_mode:
            bt_tf = decision_tf
            st.caption(f"Backtest zamanı karar zamanı ile aynı: {bt_tf}")
        else:
            bt_tf = st.selectbox("Backtest zaman dilimi", tf_options, index=tf_options.index(decision_tf))
        default_period = BACKTEST_PERIODS.get(bt_tf, "30d")
        bt_period = st.text_input("Backtest period", value=default_period, key=f"bt_period_{bt_tf}", help="Örn: 5d, 30d, 90d, 120d")
        signal_threshold = st.slider("Sinyal eşiği", min_value=25, max_value=85, value=60, step=5)
        observed_spread = observed_broker_spread_pips(symbol, bt_tf)
        spread_default = observed_spread if observed_spread is not None else recommended_spread_pips(symbol)
        spread_pips = st.number_input(
            "Toplam işlem maliyeti (pip)",
            min_value=0.0,
            value=float(spread_default),
            step=0.1,
            key=f"spread_pips_{symbol}",
            help="Canlı bid/ask verisi olmadığı için spread + komisyon + tahmini kaymayı tek değer olarak gir.",
        )
        session_filter = st.selectbox("İşlem seansı", list(TRADING_SESSIONS.keys()), index=0)
        if observed_spread is not None:
            st.caption(f"Broker son mumlarından medyan spread: {observed_spread:.1f} pip.")
        st.caption(session_description(session_filter))
        cooldown_bars = st.number_input("Cooldown (mum)", min_value=0, max_value=200, value=5, step=1)
        max_same_direction_trades = st.number_input("Aynı yönde maksimum tekrar", min_value=1, max_value=10, value=2, step=1)
        min_trades_required = st.number_input("Minimum backtest işlem sayısı", min_value=20, max_value=500, value=40, step=10)
        walk_forward_enabled = st.checkbox("Walk-forward sağlamlık kontrolü", value=True)
        walk_forward_folds = st.slider("Walk-forward fold", min_value=3, max_value=8, value=4, step=1)

    run_bt_requested = st.button("Yeniden Hesapla", type="primary", use_container_width=True)

    settings_export = {
        "symbol": symbol, "chart_tf": chart_tf, "risk_pct": risk_pct, "rr": rr,
        "atr_mult": atr_mult, "stop_mode": stop_mode, "target_mode": target_mode,
        "swing_lookback": int(swing_lookback), "signal_threshold": int(signal_threshold),
        "total_cost_pips": spread_pips, "session": session_filter,
        "max_total_risk_pct": max_total_risk_pct, "max_currency_risk_pct": max_currency_risk_pct,
        "daily_stop_r": daily_stop_r, "weekly_stop_r": weekly_stop_r,
    }
    st.download_button(
        "Ayarları JSON İndir",
        data=json.dumps(settings_export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="forex_settings.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Parite tarayıcı", expanded=False):
        scanner_tf = st.selectbox("Tarayıcı backtest zamanı", tf_options, index=tf_options.index(chart_tf))
        scanner_period = st.text_input("Tarayıcı period", value=BACKTEST_PERIODS.get(scanner_tf, "30d"), key=f"scanner_period_{scanner_tf}")
        scanner_include_backtest = st.checkbox("Backtest kalitesi hesapla", value=False)
        scanner_limit = st.number_input("Maksimum parite", min_value=1, max_value=len(SYMBOL_LIST), value=min(12, len(SYMBOL_LIST)), step=1)
        run_scanner_requested = st.button("Pariteleri Tara", use_container_width=True)

    if st.button("Veriyi Yenile", use_container_width=True):
        _fetch_ohlc_yahoo.clear()
        _fetch_ohlc_mt5.clear()
        fetch_last_price.clear()
        fetch_price_change.clear()
        st.rerun()

# İşlem kararı için kullanılan zaman dilimi.
# Yeni Başlayan Modu açıksa karar zamanı sabit 15M'dir.
# Grafik zamanı ise chart_tf değişkeniyle bağımsız çalışır.
selected_tf = decision_tf

st.title("Forex Analyzer Pro")
st.caption("Eğitim ve karar destek amaçlıdır; yatırım tavsiyesi değildir. Gerçek işlem öncesi demo test ve broker verisiyle doğrulama yapın.")

st.info("Terim notu: LONG AÇ = yükseliş beklentisiyle yeni pozisyon açmak. SHORT AÇ = düşüş beklentisiyle yeni pozisyon açmak. POZİSYONU KAPAT = açık işlemi sonlandırmak. ML filtresi açıksa teknik sinyal ayrıca geçmiş benzer sinyallerle karşılaştırılır.")
if beginner_mode:
    st.info("Yeni Başlayan Modu aktif: 4H ana yön, 1H işlem izni, 15M giriş şartı olarak kullanılır. Sen sadece LONG / SHORT / BEKLE / PAS GEÇ kararını takip et.")
elif strict_safety_mode:
    st.info("Sert Güvenli Mod aktif: yalnızca güçlü yön + İyi backtest kalitesi olan işlemler için LONG/SHORT kartı gösterilir.")
else:
    mode_note = signal_mode_settings(signal_mode)["description"]
    if practical_signal_mode:
        st.info(f"Pratik Sinyal Modu aktif ({signal_mode}): {mode_note} Gerçek işlem öncesi demo/broker doğrulaması önerilir.")
    else:
        st.info(f"Standart Mod aktif ({signal_mode}): {mode_note}")


if screen_mode == "Parite Alarm Ekranı":
    render_pair_alert_screen(
        change_window_minutes=change_window_minutes,
        change_window_label=change_window_label,
        alert_entry_tf=alert_entry_tf,
        alert_groups=alert_groups,
        alert_sort_mode=alert_sort_mode,
        signal_threshold=float(signal_threshold),
        webhook_url=webhook_url,
    )
    st.stop()

current_bt_key = make_backtest_key(
    symbol=symbol,
    tf_name=bt_tf,
    period=bt_period,
    risk_pct=risk_pct,
    rr=rr,
    atr_mult=atr_mult,
    signal_threshold=float(signal_threshold),
    spread_pips=spread_pips,
    cooldown_bars=int(cooldown_bars),
    session_filter=session_filter,
    max_same_direction_trades=int(max_same_direction_trades),
    min_trades_required=int(min_trades_required),
) + (bool(walk_forward_enabled), int(walk_forward_folds), stop_mode, target_mode, int(swing_lookback), int(max_holding_bars), float(break_even_at_r))

def run_and_store_backtest() -> None:
    with st.spinner("Plan kontrol ediliyor..."):
        bt_result = run_backtest(
            symbol=symbol,
            tf_name=bt_tf,
            period=bt_period,
            initial_balance=account_size,
            risk_pct=risk_pct,
            rr=rr,
            atr_mult=atr_mult,
            signal_threshold=float(signal_threshold),
            spread_pips=spread_pips,
            pip_value_per_lot=pip_value_per_lot,
            cooldown_bars=int(cooldown_bars),
            session_filter=session_filter,
            max_same_direction_trades=int(max_same_direction_trades),
            min_trades_required=int(min_trades_required),
            stop_mode=stop_mode,
            target_mode=target_mode,
            swing_lookback=int(swing_lookback),
            max_holding_bars=int(max_holding_bars),
            break_even_at_r=float(break_even_at_r),
        )
        q_label, q_css, q_text = assess_backtest_quality(bt_result, min_trades_required=int(min_trades_required))
        wf_report = walk_forward_stability_report(bt_result.trades, int(walk_forward_folds)) if walk_forward_enabled else pd.DataFrame()
        if walk_forward_enabled:
            wf_ok, wf_text = assess_walk_forward_stability(wf_report)
            if not wf_ok:
                q_label, q_css = "Zayıf", "bad-box"
                q_text = f"Walk-forward sağlamlık kontrolü başarısız: {wf_text}"
            else:
                q_text = f"{q_text} Walk-forward: {wf_text}"
        st.session_state["last_bt_key"] = current_bt_key
        st.session_state["last_bt_result"] = bt_result
        st.session_state["last_wf_report"] = wf_report
        st.session_state["last_bt_quality"] = {
            "label": q_label,
            "css": q_css,
            "text": q_text,
        }


if auto_plan_control and st.session_state.get("last_bt_key") != current_bt_key:
    run_and_store_backtest()

if run_bt_requested:
    run_and_store_backtest()

if run_scanner_requested:
    scan_symbols = SYMBOL_LIST[:int(scanner_limit)]
    with st.spinner("Parite tarayıcı çalışıyor..."):
        scanner_df = run_symbol_scanner(
            symbols=scan_symbols,
            change_window_minutes=change_window_minutes,
            include_backtest=scanner_include_backtest,
            scanner_tf=scanner_tf,
            scanner_period=scanner_period,
            initial_balance=account_size,
            risk_pct=risk_pct,
            rr=rr,
            atr_mult=atr_mult,
            signal_threshold=float(signal_threshold),
            spread_pips=spread_pips,
            pip_value_per_lot=pip_value_per_lot,
            cooldown_bars=int(cooldown_bars),
            session_filter=session_filter,
            max_same_direction_trades=int(max_same_direction_trades),
            min_trades_required=int(min_trades_required),
            signal_mode=signal_mode,
        )
        st.session_state["scanner_df"] = scanner_df

# Top metrics
price_info = fetch_price_change(symbol, change_window_minutes)
price = price_info["latest"] if price_info and price_info.get("latest") is not None else fetch_last_price(symbol)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Sembol", symbol)
with m2:
    dec = price_decimals(symbol)
    st.metric("Güncel Fiyat", f"{price:.{dec}f}" if price is not None else "-")
with m3:
    if price_info and price_info.get("pct") is not None:
        st.metric(change_window_label, f"{price_info['pct']:+.2f}%")
    else:
        st.metric(change_window_label, "-")
with m4:
    st.metric("Pip Size", get_pip_size(symbol))

if beginner_mode:
    st.caption(f"Grafik zamanı: {chart_tf} | İşlem karar zamanı: {selected_tf} (Yeni Başlayan Modu)")
else:
    st.caption(f"Grafik zamanı / İşlem karar zamanı: {selected_tf}")

# Main analysis
summary_df, detail_df = analyse_symbol(symbol)
final_label, final_score, filter_note = global_bias(summary_df, selected_tf)

plan_bt_key = make_backtest_key(
    symbol=symbol,
    tf_name=selected_tf,
    period=bt_period,
    risk_pct=risk_pct,
    rr=rr,
    atr_mult=atr_mult,
    signal_threshold=float(signal_threshold),
    spread_pips=spread_pips,
    cooldown_bars=int(cooldown_bars),
    session_filter=session_filter,
    max_same_direction_trades=int(max_same_direction_trades),
    min_trades_required=int(min_trades_required),
) + (bool(walk_forward_enabled), int(walk_forward_folds), stop_mode, target_mode, int(swing_lookback), int(max_holding_bars), float(break_even_at_r))
matched_quality = get_matching_backtest_quality(plan_bt_key)
allowed_quality_labels = allowed_quality_for_mode(strict_safety_mode, signal_mode)
preview_setup = build_trade_setup(
    symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult,
    pip_value_per_lot, spread_pips, entry_price=price, stop_mode=stop_mode,
    target_mode=target_mode, swing_lookback=int(swing_lookback),
)
market_regime = classify_market_regime(symbol, selected_tf)
current_data_health = data_health_status(symbol, selected_tf)
current_news_status = (
    news_blackout_status(symbol, news_events, int(news_before_minutes), int(news_after_minutes))
    if news_filter_enabled
    else {"blocks_trade": False, "state": "ok", "text": "Haber filtresi kapalı."}
)
current_portfolio_status = portfolio_risk_status(
    journal_dataframe(),
    symbol=symbol,
    proposed_risk_pct=float(risk_pct),
    max_total_risk_pct=float(max_total_risk_pct),
    max_currency_risk_pct=float(max_currency_risk_pct),
    max_open_positions=int(max_open_positions),
    daily_stop_r=float(daily_stop_r),
    weekly_stop_r=float(weekly_stop_r),
)
current_quality_info = quality_signal_status(
    matched_quality=matched_quality,
    allowed_quality_labels=allowed_quality_labels,
    practical_signal_mode=practical_signal_mode,
    signal_mode=signal_mode,
)

simple_decision = build_simple_trade_decision(
    symbol=symbol,
    selected_tf=selected_tf,
    bt_tf=bt_tf,
    final_label=final_label,
    final_score=final_score,
    filter_note=filter_note,
    price=price,
    setup=preview_setup,
    matched_quality=matched_quality,
    strict_safety_mode=strict_safety_mode,
    practical_signal_mode=practical_signal_mode,
    signal_mode=signal_mode,
    allowed_quality_labels=allowed_quality_labels,
    entry_score=_summary_score(summary_df, selected_tf),
    signal_threshold=float(signal_threshold),
)
entry_signal_tracker = build_entry_signal_tracker(
    symbol=symbol,
    selected_tf=selected_tf,
    bt_tf=bt_tf,
    final_label=final_label,
    setup=preview_setup,
    matched_quality=matched_quality,
    allowed_quality_labels=allowed_quality_labels,
    strict_safety_mode=strict_safety_mode,
    price=price,
    summary=summary_df,
    signal_threshold=float(signal_threshold),
    final_score=final_score,
    practical_signal_mode=practical_signal_mode,
    signal_mode=signal_mode,
    market_regime=market_regime,
)
simple_decision = apply_entry_signal_to_decision(simple_decision, entry_signal_tracker)
if beginner_mode:
    simple_decision = build_beginner_single_decision(
        symbol=symbol,
        summary=summary_df,
        selected_tf=selected_tf,
        setup=preview_setup,
        matched_quality=matched_quality,
        allowed_quality_labels=allowed_quality_labels,
        tracker=entry_signal_tracker,
        price=price,
    )

bt_result_for_ml = st.session_state.get("last_bt_result") if st.session_state.get("last_bt_key") == plan_bt_key else None
ml_prediction = build_live_ml_prediction(
    bt_result=bt_result_for_ml,
    summary=summary_df,
    selected_tf=selected_tf,
    setup=preview_setup,
    final_score=final_score,
    min_samples=int(ml_min_samples),
)
simple_decision = apply_ml_filter_to_decision(
    decision=simple_decision,
    ml_prediction=ml_prediction,
    filter_enabled=ml_filter_enabled,
    threshold_pct=float(ml_threshold_pct),
)
simple_decision = apply_operational_safety_filters(
    decision=simple_decision,
    data_health=current_data_health,
    news_status=current_news_status,
    portfolio_status=current_portfolio_status,
    market_regime=market_regime,
    block_sideways=block_sideways,
)
ml_blocks_trade, ml_block_reason = ml_should_block_trade(ml_prediction, float(ml_threshold_pct), ml_filter_enabled)

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
