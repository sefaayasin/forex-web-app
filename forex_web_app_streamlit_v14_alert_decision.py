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
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from forex_config import (
    ALERT_PAIR_GROUPS,
    BACKTEST_PERIODS,
    INTRADAY_CHART_WINDOWS,
    MAJOR_PAIRS,
    MINOR_PAIRS,
    PRICE_CHANGE_WINDOWS,
    SYMBOL_LIST,
    TIMEFRAMES,
    TRADING_SESSIONS,
    get_pip_size,
    normalize_symbol,
    price_decimals,
    symbol_pair,
)
from forex_analysis import (
    BiasResult,
    MarketStructureResult,
    calculate_stop_target_distances,
    evaluate_bias,
    evaluate_market_structure,
    label_from_score,
    latest_valid_row,
    market_structure_frame,
)
from forex_indicators import (
    add_indicators,
    compute_atr,
    compute_bbands,
    compute_ichimoku,
    compute_macd,
    compute_rsi,
)
from forex_edge import build_edge_validation_report, edge_validation_table
from forex_storage import (
    APP_DB_PATH,
    add_trade_journal_entry,
    alert_history_dataframe,
    clear_trade_journal,
    init_trade_journal,
    journal_dataframe,
    record_alert_once,
    send_webhook_notification,
)

try:
    from forex_decision_core import classify_opportunity_readiness, decide_mtf_signal, position_level_event
except ImportError:
    # Streamlit Cloud bazen ana dosyayi yeni commit'ten, yardimci modulu ise
    # onceki build cache'inden yukleyebiliyor. Uygulamanin tamamen acilamaz
    # hale gelmemesi icin iki saf karar kurali burada da guvenli yedeklenir.
    def _decision_value_missing(value: float) -> bool:
        try:
            return np.isnan(float(value))
        except (TypeError, ValueError):
            return True

    def classify_opportunity_readiness(
        radar_score: float,
        side: str,
        catalyst_matches: bool,
        structure_matches: bool,
        htf_conflict: bool,
        capacity_ratio: float,
        in_session: bool,
        candidate_threshold: float = 42.0,
        ready_threshold: float = 65.0,
    ) -> tuple[str, list[str]]:
        normalized_side = str(side).upper()
        score = float(radar_score) if not _decision_value_missing(radar_score) else 0.0
        if normalized_side not in {"LONG", "SHORT"} or score < float(candidate_threshold):
            return "NEUTRAL", ["Radar puani aday esiginin altinda"]

        blockers: list[str] = []
        if score < float(ready_threshold):
            blockers.append(f"Radar puani {score:.0f}; hazir esigi {float(ready_threshold):.0f}")
        if not catalyst_matches:
            blockers.append("15M tepki veya Bollinger kirilim tetigi yok")
        if not structure_matches:
            blockers.append("15M MA ve swing yapisi yonu dogrulamiyor")
        if htf_conflict:
            blockers.append("4H ve 1H yonleri celisiyor")
        if _decision_value_missing(capacity_ratio):
            blockers.append("Hedef kapasitesi olculemedi")
        elif float(capacity_ratio) > 0.80:
            blockers.append("Hedef tipik 4 saatlik hareketin %80'inden buyuk")
        if not in_session:
            blockers.append("Secili likit islem seansi disinda")
        return ("WATCH", blockers) if blockers else ("READY", [])

    def decide_mtf_signal(
        entry_score: float,
        h4_score: float,
        h1_score: float,
        m15_score: float,
        tf_name: str,
        threshold: float,
    ) -> tuple[str, str]:
        if any(_decision_value_missing(value) for value in (entry_score, h4_score, h1_score)):
            return "NONE", "Ana zaman dilimi skorlari yetersiz"

        htf_long = float(h4_score) >= 25 and float(h1_score) >= 25
        htf_short = float(h4_score) <= -25 and float(h1_score) <= -25
        m15_long_ok = tf_name != "5 Dakika" or (
            not _decision_value_missing(m15_score) and float(m15_score) >= 25
        )
        m15_short_ok = tf_name != "5 Dakika" or (
            not _decision_value_missing(m15_score) and float(m15_score) <= -25
        )

        if htf_long and m15_long_ok and float(entry_score) >= float(threshold):
            return "LONG", "4H+1H long uyumlu; giris skoru esigi gecti"
        if htf_short and m15_short_ok and float(entry_score) <= -float(threshold):
            return "SHORT", "4H+1H short uyumlu; giris skoru esigi gecti"
        return "NONE", "MTF filtre veya giris skoru uygun degil"

    def position_level_event(side: str, current_price: float, stop: float, target: float) -> str:
        normalized_side = str(side).upper()
        if normalized_side == "LONG":
            if float(stop) > 0 and float(current_price) <= float(stop):
                return "STOP"
            if float(target) > 0 and float(current_price) >= float(target):
                return "TARGET"
        elif normalized_side == "SHORT":
            if float(stop) > 0 and float(current_price) >= float(stop):
                return "STOP"
            if float(target) > 0 and float(current_price) <= float(target):
                return "TARGET"
        return "NONE"

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
        .daily-desk {
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid #dfe3e8;
            background: linear-gradient(135deg, #f8f9fa 0%, #eef3f8 100%);
            color: #212529 !important;
            margin: 8px 0 14px 0;
        }
        .daily-desk, .daily-desk * { color:#212529 !important; }
        .daily-desk-title { font-size:1.15rem; font-weight:900; margin-bottom:5px; }
        .daily-desk-note { font-size:.92rem; color:#495057 !important; }
        .daily-progress-track {
            height: 12px;
            border-radius: 999px;
            background:#dee2e6;
            overflow:hidden;
            margin: 12px 0 8px 0;
        }
        .daily-progress-fill { height:100%; background:#198754; border-radius:999px; }
        .section-kicker { color:#6c757d; font-size:.82rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
        .opportunity-card {
            padding:18px;
            border-radius:14px;
            border:1px solid #dfe3e8;
            background:#f8f9fa;
            color:#212529 !important;
            min-height:310px;
        }
        .opportunity-card, .opportunity-card * { color:#212529 !important; }
        .opportunity-long { background:#e8f5ee; border-color:#badbcc; }
        .opportunity-short { background:#fbeaec; border-color:#f5c2c7; }
        .opportunity-neutral { background:#fff8e1; border-color:#ffe69c; }
        .opportunity-title { font-size:1.45rem; font-weight:950; margin:4px 0 8px 0; }
        .opportunity-score { font-size:2rem; font-weight:950; line-height:1; margin:10px 0; }
        .opportunity-line { margin-top:8px; font-size:.92rem; }
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


INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "60m": 60, "1h": 60, "4h": 240}


@st.cache_data(show_spinner=False)
def parse_broker_csv_bytes(content: bytes) -> pd.DataFrame:
    """MT5/standart OHLC CSV dışa aktarımlarını UTC indeksli forma çevirir."""
    if not content:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(BytesIO(content), sep=None, engine="python")
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    normalized = {str(col).strip().lower().replace("<", "").replace(">", ""): col for col in raw.columns}
    date_col = normalized.get("date")
    time_col = normalized.get("time") or normalized.get("datetime") or normalized.get("timestamp")
    if date_col is not None and time_col is not None and date_col != time_col:
        time_values = raw[date_col].astype(str) + " " + raw[time_col].astype(str)
    elif time_col is not None:
        time_values = raw[time_col]
    else:
        return pd.DataFrame()
    index = pd.to_datetime(time_values, utc=True, errors="coerce")
    aliases = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tickvol": "Volume", "tick_volume": "Volume", "volume": "Volume",
        "spread": "Spreadpoints", "spreadpoints": "Spreadpoints",
    }
    out = pd.DataFrame(index=index)
    for key, target in aliases.items():
        original = normalized.get(key)
        if original is not None and target not in out.columns:
            out[target] = pd.to_numeric(raw[original], errors="coerce").to_numpy()
    out = out[~out.index.isna()]
    return _fix_cols(out).sort_index()


def _resample_broker_csv(df: pd.DataFrame, base_interval: str, target_interval: str) -> pd.DataFrame:
    base_minutes = INTERVAL_MINUTES.get(str(base_interval).lower())
    target_minutes = INTERVAL_MINUTES.get(str(target_interval).lower())
    if df.empty or base_minutes is None or target_minutes is None or target_minutes < base_minutes:
        return pd.DataFrame()
    if target_minutes == base_minutes:
        return df.copy()
    rule = f"{target_minutes}min"
    aggregation = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if "Spreadpoints" in df.columns:
        aggregation["Spreadpoints"] = "median"
    return df.resample(rule, label="right", closed="right").agg(aggregation).dropna(subset=["Open", "High", "Low", "Close"])


def _fetch_ohlc_broker_csv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    csv_symbol = normalize_symbol(st.session_state.get("broker_csv_symbol", ""))
    if csv_symbol != normalize_symbol(symbol):
        return pd.DataFrame()
    source = st.session_state.get("broker_csv_df")
    if not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame()
    base_interval = str(st.session_state.get("broker_csv_interval", "15m"))
    out = _resample_broker_csv(_utc_index_df(source), base_interval, interval)
    if out.empty:
        return out
    cutoff = out.index[-1] - pd.Timedelta(days=_period_to_days(period, 30))
    return out[out.index >= cutoff].copy()


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
    provider = st.session_state.get("data_provider")
    if provider not in {"MetaTrader 5", "Broker CSV"}:
        return None
    prm = TIMEFRAMES[tf_name]
    df = (
        _fetch_ohlc_mt5(symbol, prm["interval"], prm["period"])
        if provider == "MetaTrader 5"
        else _fetch_ohlc_broker_csv(symbol, prm["interval"], prm["period"])
    )
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
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "60m": mt5.TIMEFRAME_H1,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
    }
    return mt5, mapping.get(interval.lower())


def _period_to_days(period: str, default: int = 30) -> int:
    """30d/6mo/1y gibi periyotları broker tarih aralığına çevirir."""
    text = str(period).strip().lower()
    try:
        if text.endswith("d"):
            return max(int(float(text[:-1])), 1)
        if text.endswith("mo"):
            return max(int(float(text[:-2]) * 30), 1)
        if text.endswith("y"):
            return max(int(float(text[:-1]) * 365), 1)
    except (TypeError, ValueError):
        pass
    return int(default)


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
        period_days = _period_to_days(period, 30)
        end = datetime.now(tz=pytz.UTC)
        start = end - timedelta(days=period_days)
        mt5.symbol_select(mt5_symbol, True)
        rates = mt5.copy_rates_range(mt5_symbol, timeframe, start, end)
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


@st.cache_data(ttl=5, show_spinner=False)
def fetch_mt5_quote(symbol: str) -> Optional[dict]:
    """Broker terminalinden yürütülebilir bid/ask ve anlık spreadi alır."""
    mt5, _ = _mt5_timeframe("15m")
    if mt5 is None:
        return None
    mt5_symbol = normalize_symbol(symbol).replace("=X", "")
    try:
        if not mt5.initialize():
            return None
        mt5.symbol_select(mt5_symbol, True)
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None or float(tick.bid) <= 0 or float(tick.ask) <= 0:
            return None
        pip = get_pip_size(symbol)
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "mid": (float(tick.bid) + float(tick.ask)) / 2.0,
            "spread_pips": (float(tick.ask) - float(tick.bid)) / pip,
            "time": pd.to_datetime(int(tick.time), unit="s", utc=True),
        }
    except Exception as exc:
        LOGGER.exception("MT5 quote error for %s: %s", symbol, exc)
        return None
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    provider = st.session_state.get("data_provider", "Yahoo Finance")
    if provider == "Broker CSV":
        return _fetch_ohlc_broker_csv(symbol, interval, period)
    if provider == "MetaTrader 5":
        broker_df = _fetch_ohlc_mt5(symbol, interval, period)
        return broker_df
    return _fetch_ohlc_yahoo(symbol, interval, period)


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_intraday_history_yahoo(symbol: str, period: str = "2d") -> pd.DataFrame:
    """Kısa vadeli fiyat, yüzde değişim ve grafik için ortak 1 dakikalık veri."""
    symbol = normalize_symbol(symbol)
    try:
        df = yf.download(symbol, period=period, interval="1m", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _fix_cols(df)
        return _utc_index_df(df) if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_intraday_history(symbol: str, period: str = "2d") -> pd.DataFrame:
    provider = st.session_state.get("data_provider", "Yahoo Finance")
    if provider == "Broker CSV":
        source = st.session_state.get("broker_csv_df")
        if not isinstance(source, pd.DataFrame) or source.empty:
            return pd.DataFrame()
        out = _utc_index_df(source)
        cutoff = out.index[-1] - pd.Timedelta(days=_period_to_days(period, 2))
        return out[out.index >= cutoff].copy()
    if provider == "MetaTrader 5":
        return _fetch_ohlc_mt5(symbol, "1m", period)
    return _fetch_intraday_history_yahoo(symbol, period)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_last_price(symbol: str) -> Optional[float]:
    df = fetch_intraday_history(symbol, "2d")
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])


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
        df = fetch_intraday_history(symbol, period)
        if df.empty or len(df) < 2:
            return None
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


def intraday_change_snapshot(symbol: str, windows: Optional[dict[str, int]] = None) -> pd.DataFrame:
    """Aynı son fiyata göre 1/2/4/8/12/24 saatlik değişimleri tek tabloda hesaplar."""
    windows = windows or INTRADAY_CHART_WINDOWS
    df = fetch_intraday_history(symbol, "2d")
    columns = ["Pencere", "Referans Zamanı", "Referans", "Son Fiyat", "Değişim %"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    close = df["Close"].astype(float).dropna()
    if close.empty:
        return pd.DataFrame(columns=columns)
    latest_time = close.index[-1]
    latest_price = float(close.iloc[-1])
    rows = []
    for label, minutes in windows.items():
        candidates = close[close.index <= latest_time - pd.Timedelta(minutes=int(minutes))]
        if candidates.empty:
            rows.append([label, None, np.nan, latest_price, np.nan])
            continue
        reference_time = candidates.index[-1]
        reference = float(candidates.iloc[-1])
        pct = 100 * (latest_price - reference) / reference if reference else np.nan
        rows.append([label, reference_time, reference, latest_price, pct])
    return pd.DataFrame(rows, columns=columns)


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

# Market analysis is implemented in forex_analysis.py.



def analyse_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []
    for tf_name, prm in TIMEFRAMES.items():
        df = fetch_ohlc(symbol, prm["interval"], prm["period"])
        if df.empty:
            rows.append([
                tf_name, "Veri yok", 0, "-", "NEUTRAL", "RANGE", "VERİ YETERSİZ", "NONE",
                "NEUTRAL", "NONE", "NONE", "VERİ YETERSİZ", "NONE", "NONE", "NONE",
                "NONE", np.nan, np.nan,
                "TRANSITION", "MIXED", "NONE", False, np.nan, np.nan,
            ])
            continue

        # Kapanmamış son barı kullanmamak daha güvenli.
        if len(df) > 2:
            df_eval = df.iloc[:-1]
        else:
            df_eval = df

        result = evaluate_bias(df_eval)
        structure_result = evaluate_market_structure(df_eval)
        rows.append([
            tf_name,
            result.label,
            round(result.score, 1),
            result.explanation,
            structure_result.ma_direction,
            structure_result.structure,
            structure_result.phase,
            structure_result.response_side,
            structure_result.rsi_regime,
            structure_result.rsi_divergence,
            structure_result.rsi_momentum_break,
            structure_result.bb_state,
            structure_result.bb_trend_signal,
            structure_result.bb_band_walk,
            structure_result.bb_pattern,
            structure_result.bb_mean_reversion_side,
            structure_result.bb_mid_target,
            structure_result.bb_width_percentile,
            structure_result.macd_regime,
            structure_result.macd_momentum_state,
            structure_result.macd_divergence,
            structure_result.macd_whipsaw,
            structure_result.macd_atr,
            structure_result.macd_hist_atr,
        ])
        detail_rows.append([
            tf_name,
            round(result.trend_score, 1),
            round(result.momentum_score, 1),
            round(result.volatility_score, 1),
            result.label,
        ])

    summary = pd.DataFrame(
        rows,
        columns=[
            "Zaman Dilimi", "Bias", "Skor", "Açıklama", "MA Yönü",
            "Market Yapısı", "Hareket Fazı", "Tepki Teyidi", "RSI Rejimi",
            "RSI Uyumsuzluğu", "RSI Momentum Kırılımı",
            "Bollinger Durumu", "BB Trend Sinyali", "Band Walk", "W/M Formasyonu",
            "Ortalama Dönüş Adayı", "BB Orta Bant Hedefi", "BB Genişlik Yüzdeliği",
            "MACD Rejimi", "MACD Histogram Durumu", "MACD Uyumsuzluğu", "MACD Whipsaw",
            "MACD / ATR", "MACD Histogram / ATR",
        ],
    )
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

    atr = ind["ATR14"].replace(0, np.nan)
    macd_atr = ind["MACD"] / atr
    hist_atr = ind["MACDHist"] / atr
    hist_sign = np.sign(ind["MACDHist"].fillna(0))
    hist_cross = ((hist_sign != hist_sign.shift(1)) & (hist_sign != 0) & (hist_sign.shift(1) != 0)).astype(int)
    whipsaw = (hist_cross.rolling(12).sum() >= 4) & (macd_atr.abs() <= 0.20) & (hist_atr.abs() <= 0.08)
    usable_macd = ~whipsaw.fillna(False)

    score += np.where(usable_macd & (ind["MACD"] > 0) & (ind["MACDSignal"] > 0), 10, 0)
    score += np.where(usable_macd & (ind["MACD"] < 0) & (ind["MACDSignal"] < 0), -10, 0)
    score += np.where(usable_macd & (ind["MACD"] > ind["MACDSignal"]) & (ind["MACDHist"] > 0), 10, 0)
    score += np.where(usable_macd & (ind["MACD"] < ind["MACDSignal"]) & (ind["MACDHist"] < 0), -10, 0)

    hist_delta = ind["MACDHist"].diff()
    score += np.where(usable_macd & (ind["MACDHist"] > 0) & (hist_delta > 0) & (hist_delta.shift(1) > 0), 6, 0)
    score += np.where(usable_macd & (ind["MACDHist"] < 0) & (hist_delta < 0) & (hist_delta.shift(1) < 0), -6, 0)

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
        "15 Dakika": "60d",
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


def _fetch_structure_for_tf(symbol: str, tf_name: str, period: str, shift_closed_bar: bool = True) -> pd.Series:
    """Üst zaman diliminin MA + swing yönünü giriş zamanına taşır."""
    prm = TIMEFRAMES[tf_name]
    df = fetch_ohlc(symbol, prm["interval"], period)
    if df.empty:
        return pd.Series(dtype=object)
    df = _utc_index_df(df)
    model = market_structure_frame(df)
    side = model["CombinedDirection"].astype(str)
    if shift_closed_bar:
        side = side.shift(1).dropna()
    return side


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


def _aligned_text(series: Optional[pd.Series], ts, default: str = "NONE") -> str:
    if series is None or series.empty:
        return default
    try:
        value = series.loc[ts]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return str(value)
    except Exception:
        return default


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
    market_structure_enabled: bool = True,
    entry_model: str = "Düzeltme + Tepki",
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
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
    structure_frame = market_structure_frame(df)
    for col in [
        "MADirection", "MarketStructure", "CombinedDirection", "ResponseSide", "CorrectionActive",
        "Retracement", "RSIRegime", "RSIDivergence", "RSIMomentumBreak",
        "BBState", "BBTrendSignal", "BBBandWalk", "BBPattern", "BBMeanReversionSide",
        "BBMidTarget", "BBWidthPercentile", "BBExtremeVolatility",
        "MACDRegime", "MACDMomentumState", "MACDDivergence", "MACDWhipsaw",
        "MACDATR", "MACDHistATR", "StrategyRegime", "RegimeEMAGapATR",
        "RegimeSlopeATR", "RegimeBBWidthATR",
    ]:
        df[col] = structure_frame[col]
    entry_score_series = score_series_for_backtest(df)
    df = df.join(entry_score_series.rename("Score"), how="left")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "ATR14", "Score"])

    if len(df) < 80:
        empty_metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["İndikatörler sonrası yeterli veri yok"]})
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame())

    # Üst zaman dilimi skorlarını giriş zaman dilimine hizala.
    aligned_scores: dict[str, pd.Series] = {}
    aligned_structures: dict[str, pd.Series] = {}
    for tf in ["4 Saat", "1 Saat", "15 Dakika"]:
        if tf == tf_name:
            continue
        score_period = _filter_period_for_tf(tf, period)
        score = _fetch_score_for_tf(symbol, tf, score_period, shift_closed_bar=True)
        if score.empty:
            aligned_scores[tf] = pd.Series(index=df.index, dtype=float)
        else:
            aligned_scores[tf] = score.reindex(df.index, method="ffill")
        structure = _fetch_structure_for_tf(symbol, tf, score_period, shift_closed_bar=True)
        if structure.empty:
            aligned_structures[tf] = pd.Series(index=df.index, dtype=object)
        else:
            aligned_structures[tf] = structure.reindex(df.index, method="ffill")

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
            "Engine": trade.get("Engine", "TREND_PULLBACK"),
            "Market Regime": trade.get("MarketRegime", "TREND"),
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

            if sig != "NONE" and market_structure_enabled:
                requested_side = sig
                entry_structure = str(previous.get("CombinedDirection", "NONE"))
                response_side = str(previous.get("ResponseSide", "NONE"))
                h4_structure = entry_structure if tf_name == "4 Saat" else _aligned_text(aligned_structures.get("4 Saat"), prev_ts)
                h1_structure = entry_structure if tf_name == "1 Saat" else _aligned_text(aligned_structures.get("1 Saat"), prev_ts)
                if h4_structure != sig or h1_structure != sig:
                    sig = "NONE"
                    reason = f"MA + market yapısı uyuşmuyor: 4H={h4_structure}, 1H={h1_structure}"
                elif entry_model == "Düzeltme + Tepki" and response_side != sig:
                    sig = "NONE"
                    reason = f"{tf_name} düzeltme sonrası {requested_side} tepki teyidi bekleniyor"
                elif entry_model == "Trend + Yapı" and entry_structure != sig:
                    sig = "NONE"
                    reason = f"{tf_name} MA + swing yapısı giriş yönünü doğrulamıyor"
                elif entry_model == "Bollinger Trend Devamı" and str(previous.get("BBTrendSignal", "NONE")) != sig:
                    sig = "NONE"
                    reason = f"{tf_name} Bollinger daralma sonrası yönlü volatilite açılımı bekleniyor"
                elif entry_model == "Hibrit (Tepki / Bollinger)" and response_side != sig and str(previous.get("BBTrendSignal", "NONE")) != sig:
                    sig = "NONE"
                    reason = f"{tf_name} corrective response veya Bollinger trend açılımı bekleniyor"

            if sig != "NONE" and rsi_regime_enabled:
                required_regime = "BULLISH" if sig == "LONG" else "BEARISH"
                actual_regime = str(previous.get("RSIRegime", "NEUTRAL"))
                if actual_regime != required_regime:
                    sig = "NONE"
                    reason = f"RSI 50 rejimi giriş yönünü doğrulamıyor: {actual_regime}"

            if sig != "NONE" and rsi_divergence_filter_enabled:
                requested_side = sig
                opposing_divergence = "BEARISH" if sig == "LONG" else "BULLISH"
                actual_divergence = str(previous.get("RSIDivergence", "NONE"))
                if actual_divergence == opposing_divergence:
                    sig = "NONE"
                    reason = f"Ters RSI uyumsuzluğu yeni {requested_side} girişini engelledi"

            if sig != "NONE" and bb_extreme_volatility_block and bool(previous.get("BBExtremeVolatility", False)):
                sig = "NONE"
                reason = "Bollinger genişliği tarihsel %95 bölgesinde; aşırı volatilite filtresi"

            if sig != "NONE" and macd_confirmation_enabled:
                required_macd = "BULLISH" if sig == "LONG" else "BEARISH"
                actual_macd = str(previous.get("MACDRegime", "TRANSITION"))
                if bool(previous.get("MACDWhipsaw", False)):
                    sig = "NONE"
                    reason = "MACD sıfır çevresinde whipsaw; kesişim sinyali reddedildi"
                elif actual_macd != required_macd:
                    sig = "NONE"
                    reason = f"MACD sıfır rejimi uygun değil: {actual_macd}, gerekli {required_macd}"

            if sig != "NONE" and macd_divergence_filter_enabled:
                requested_side = sig
                opposing_macd_divergence = "BEARISH" if sig == "LONG" else "BULLISH"
                actual_macd_divergence = str(previous.get("MACDDivergence", "NONE"))
                if actual_macd_divergence == opposing_macd_divergence:
                    sig = "NONE"
                    reason = f"Ters MACD/histogram uyumsuzluğu yeni {requested_side} girişini engelledi"

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
                        "Engine": "TREND_PULLBACK",
                        "MarketRegime": str(previous.get("StrategyRegime", "TRANSITION")),
                        "MA Structure": str(previous.get("CombinedDirection", "NONE")),
                        "Response Side": str(previous.get("ResponseSide", "NONE")),
                        "RSI Regime": str(previous.get("RSIRegime", "NEUTRAL")),
                        "RSI Divergence": str(previous.get("RSIDivergence", "NONE")),
                        "RSI Momentum Break": str(previous.get("RSIMomentumBreak", "NONE")),
                        "BB State": str(previous.get("BBState", "NORMAL VOLATİLİTE")),
                        "BB Trend Signal": str(previous.get("BBTrendSignal", "NONE")),
                        "BB Band Walk": str(previous.get("BBBandWalk", "NONE")),
                        "BB Pattern": str(previous.get("BBPattern", "NONE")),
                        "MACD Regime": str(previous.get("MACDRegime", "TRANSITION")),
                        "MACD Momentum": str(previous.get("MACDMomentumState", "MIXED")),
                        "MACD Divergence": str(previous.get("MACDDivergence", "NONE")),
                        "MACD Whipsaw": bool(previous.get("MACDWhipsaw", False)),
                        "MACD ATR": float(previous.get("MACDATR", np.nan)),
                        "MACD Hist ATR": float(previous.get("MACDHistATR", np.nan)),
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
        ["Giriş Modeli", entry_model],
        ["Bollinger Aşırı Volatilite", "Blok" if bb_extreme_volatility_block else "İzinli"],
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


def run_range_mean_reversion_backtest(
    symbol: str,
    tf_name: str,
    period: str,
    initial_balance: float,
    risk_pct: float,
    spread_pips: float,
    pip_value_per_lot: float,
    session_filter: str = "Londra",
    cooldown_bars: int = 16,
    max_holding_bars: int = 16,
    min_reward_r: float = 1.0,
    swing_lookback: int = 8,
) -> BacktestResult:
    """Yalnız RANGE rejiminde Bollinger orta banda dönüş işlemlerini test eder."""
    prm = TIMEFRAMES[tf_name]
    raw = fetch_ohlc(symbol, prm["interval"], period)
    if raw.empty or len(raw) < 160:
        metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["Yeterli veri yok"]})
        return BacktestResult(metrics, pd.DataFrame(), pd.DataFrame())

    df = _utc_index_df(raw)
    model = market_structure_frame(add_indicators(df))
    needed = [
        "ATR14", "BBMidTarget", "BBMeanReversionSide", "StrategyRegime",
        "RSIDivergence", "MACDDivergence", "BBExtremeVolatility",
    ]
    df = model.dropna(subset=["Open", "High", "Low", "Close", "ATR14", "BBMidTarget"]).copy()
    if len(df) < 100 or not set(needed).issubset(df.columns):
        metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["Rejim modeli için veri yetersiz"]})
        return BacktestResult(metrics, pd.DataFrame(), pd.DataFrame())

    pip = get_pip_size(symbol)
    balance = float(initial_balance)
    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_trade: Optional[dict] = None
    last_exit_i = -10**9

    def resolve_exit(trade: dict, bar: pd.Series) -> tuple[Optional[str], Optional[float]]:
        side = trade["Side"]
        stop = float(trade["Stop"])
        target = float(trade["Target"])
        bar_open = float(bar["Open"])
        if side == "LONG":
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
            return "SL", stop
        if hit_stop:
            return "SL", stop
        if hit_target:
            return "TP-MID", target
        return None, None

    def close_position(trade: dict, exit_price: float, exit_time, reason: str) -> None:
        nonlocal balance
        gross_pips = (
            (float(exit_price) - float(trade["Entry"])) / pip
            if trade["Side"] == "LONG"
            else (float(trade["Entry"]) - float(exit_price)) / pip
        )
        pnl_pips = gross_pips - max(float(spread_pips), 0.0)
        pnl = pnl_pips * float(pip_value_per_lot) * float(trade["Lot"])
        balance += pnl
        trades.append({
            "Entry Time": trade["EntryTime"], "Exit Time": exit_time,
            "Side": trade["Side"], "Entry": trade["Entry"], "Exit": float(exit_price),
            "SL": trade["Stop"], "TP": trade["Target"], "Pips": pnl_pips,
            "PnL": pnl, "Balance": balance, "Result": reason, "Lot": trade["Lot"],
            "Risk Amount": trade["RiskAmount"], "Entry Score": np.nan,
            "4H Score": np.nan, "1H Score": np.nan, "15M Score": np.nan,
            "MTF Reason": trade["Reason"], "Engine": "RANGE_MEAN_REVERSION",
            "Market Regime": "RANGE",
        })

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        ts = df.index[i]
        closed_this_bar = False

        if open_trade is not None:
            exit_reason, exit_price = resolve_exit(open_trade, current)
            if exit_reason is None and int(max_holding_bars) > 0 and (i - open_trade["EntryIndex"]) >= int(max_holding_bars):
                exit_reason, exit_price = "TIME", float(current["Open"])
            if exit_reason is not None:
                close_position(open_trade, float(exit_price), ts, exit_reason)
                open_trade = None
                last_exit_i = i
                closed_this_bar = True

        if open_trade is None and not closed_this_bar:
            side = str(previous.get("RangeReversalSignal", previous.get("BBMeanReversionSide", "NONE")))
            if str(previous.get("StrategyRegime", "TRANSITION")) != "RANGE":
                side = "NONE"
            if side != "NONE" and bool(previous.get("BBExtremeVolatility", False)):
                side = "NONE"
            if side == "LONG" and (
                str(previous.get("RSIDivergence", "NONE")) == "BEARISH"
                or str(previous.get("MACDDivergence", "NONE")) == "BEARISH"
            ):
                side = "NONE"
            if side == "SHORT" and (
                str(previous.get("RSIDivergence", "NONE")) == "BULLISH"
                or str(previous.get("MACDDivergence", "NONE")) == "BULLISH"
            ):
                side = "NONE"
            if side != "NONE" and not is_in_trading_session(ts, session_filter):
                side = "NONE"
            if side != "NONE" and (i - last_exit_i) <= int(cooldown_bars):
                side = "NONE"

            if side in {"LONG", "SHORT"}:
                entry = float(current["Open"])
                atr = float(previous["ATR14"])
                recent = df.iloc[max(0, i - int(swing_lookback)):i]
                if side == "LONG":
                    structural_stop = float(recent["Low"].min()) - atr * 0.15
                    stop = min(structural_stop, entry - atr * 0.75)
                    target = float(previous["BBMidTarget"])
                    reward_distance = target - entry
                    stop_distance = entry - stop
                else:
                    structural_stop = float(recent["High"].max()) + atr * 0.15
                    stop = max(structural_stop, entry + atr * 0.75)
                    target = float(previous["BBMidTarget"])
                    reward_distance = entry - target
                    stop_distance = stop - entry

                net_reward_pips = reward_distance / pip - max(float(spread_pips), 0.0)
                risk_pips = stop_distance / pip + max(float(spread_pips), 0.0)
                reward_r = net_reward_pips / risk_pips if risk_pips > 0 else -np.inf
                risk_amount = balance * float(risk_pct) / 100.0
                lot = risk_amount / (risk_pips * float(pip_value_per_lot)) if risk_pips > 0 and pip_value_per_lot > 0 else 0.0
                if reward_distance > 0 and reward_r >= float(min_reward_r) and lot > 0 and np.isfinite(lot):
                    candidate = {
                        "EntryTime": ts, "EntryIndex": i, "Side": side,
                        "Entry": entry, "Stop": stop, "Target": target, "Lot": lot,
                        "RiskAmount": risk_amount,
                        "Reason": f"RANGE rejimi; Bollinger dışından orta banda dönüş, beklenen {reward_r:.2f}R",
                    }
                    immediate_reason, immediate_price = resolve_exit(candidate, current)
                    if immediate_reason is not None:
                        close_position(candidate, float(immediate_price), ts, immediate_reason)
                        last_exit_i = i
                    else:
                        open_trade = candidate

        equity_rows.append({"Time": ts, "Balance": balance})

    if open_trade is not None:
        close_position(open_trade, float(df.iloc[-1]["Close"]), df.index[-1], "EOD")
        if equity_rows:
            equity_rows[-1]["Balance"] = balance

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if trades_df.empty:
        metrics = pd.DataFrame({
            "Metrik": ["Backtest Tipi", "İşlem Sayısı", "Not"],
            "Değer": ["RANGE mean-reversion", "0", "RANGE + orta bant koşullarında işlem oluşmadı"],
        })
        return BacktestResult(metrics, trades_df, equity_df)

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]
    loss_sum = abs(float(losses["PnL"].sum()))
    pf = float(wins["PnL"].sum()) / loss_sum if loss_sum > 0 else np.nan
    r_values = trades_df["PnL"] / trades_df["Risk Amount"].replace(0, np.nan)
    split = max(1, int(len(trades_df) * 0.70))
    oos_pf, oos_avg_r, oos_count = _trade_slice_stats(trades_df.iloc[split:])
    eq = equity_df["Balance"] if not equity_df.empty else pd.Series([initial_balance])
    peak = eq.cummax()
    drawdown = eq - peak
    dd_idx = drawdown.idxmin()
    peak_value = float(peak.loc[dd_idx]) if len(peak) else float(initial_balance)
    dd_pct = 100 * float(drawdown.min()) / peak_value if peak_value else 0.0
    metrics = pd.DataFrame([
        ["Backtest Tipi", "RANGE mean-reversion"],
        ["İşlem Sayısı", len(trades_df)],
        ["Win Rate", f"{100 * len(wins) / len(trades_df):.2f}%"],
        ["Toplam PnL", f"{trades_df['PnL'].sum():.2f}"],
        ["Son Bakiye", f"{balance:.2f}"],
        ["Profit Factor", "-" if pd.isna(pf) else f"{pf:.2f}"],
        ["Maks. Drawdown", f"{drawdown.min():.2f} ({dd_pct:.2f}%)"],
        ["Ortalama Pips", f"{trades_df['Pips'].mean():.2f}"],
        ["Ortalama R", f"{r_values.mean():.3f}R"],
        ["Son %30 İşlem", oos_count],
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


def strategy_evidence_status(matched_quality: Optional[dict]) -> dict:
    """Gösterge kombinasyonunun kanıt durumunu kullanıcıya doğrudan söyler."""
    if not matched_quality:
        return {
            "label": "KANIT BEKLİYOR",
            "css": "warn-box",
            "text": "Bu gösterge modeli henüz test edilmedi. LONG/SHORT üretmesi güvenilir olduğu anlamına gelmez.",
        }
    label = str(matched_quality.get("label", "-"))
    detail = str(matched_quality.get("text", ""))
    if label == "İyi":
        return {
            "label": "DEMO İÇİN DOĞRULANDI",
            "css": "ok-box",
            "text": f"Geçmiş ve ayrı dönem kontrolleri olumlu. Yine de garanti değildir ve önce broker verisinde demo izlenmelidir. {detail}",
        }
    if label == "Orta":
        return {
            "label": "SINIRLI KANIT",
            "css": "warn-box",
            "text": f"Avantaj marjı dar. Bu model ancak demo/çok küçük risk için adaydır. {detail}",
        }
    if label in LOW_SAMPLE_QUALITIES:
        return {
            "label": "DOĞRULANMADI",
            "css": "warn-box",
            "text": f"Yeterli bağımsız işlem örneği yok. Uygulama bu indikatör birleşimine güvenilir LONG/SHORT motoru diyemez. {detail}",
        }
    return {
        "label": "MODEL REDDEDİLDİ",
        "css": "bad-box",
        "text": f"Maliyet sonrası performans/kararlılık yeterli değil. Daha çok sinyal vermek için filtre gevşetilmemeli; başka giriş modeli test edilmeli. {detail}",
    }


def assess_backtest_with_walk_forward(
    bt: BacktestResult,
    min_trades_required: int,
    walk_forward_enabled: bool,
    walk_forward_folds: int,
) -> tuple[str, str, str, pd.DataFrame]:
    """Kalite ile walk-forward sonucunu, yetersiz örneği kötü sonuç saymadan birleştirir."""
    label, css, text = assess_backtest_quality(bt, min_trades_required=int(min_trades_required))
    report = (
        walk_forward_stability_report(bt.trades, int(walk_forward_folds))
        if walk_forward_enabled else pd.DataFrame()
    )
    if not walk_forward_enabled:
        return label, css, text, report

    stable, wf_text = assess_walk_forward_stability(report)
    if report.empty:
        return (
            "Yetersiz Örnek",
            "warn-box",
            f"{text} Walk-forward sonucu üretilemedi: {wf_text}",
            report,
        )
    if not stable:
        return "Zayıf", "bad-box", f"Walk-forward sağlamlık kontrolü başarısız: {wf_text}", report
    return label, css, f"{text} Walk-forward: {wf_text}", report


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


def scan_symbol_live(
    symbol: str,
    change_window_minutes: int,
    selected_tf: Optional[str] = None,
    market_structure_enabled: bool = True,
    entry_model: str = "Düzeltme + Tepki",
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
) -> dict:
    """Tek sembol için canlı çoklu zaman dilimi özetini üretir."""
    summary, _ = analyse_symbol(symbol)
    label, score, note = global_bias(summary, selected_tf)
    price_info = fetch_price_change(symbol, change_window_minutes)
    pct = price_info.get("pct") if price_info else None
    model = build_market_model_status(
        summary,
        selected_tf or "15 Dakika",
        enabled=market_structure_enabled,
        entry_model=entry_model,
        rsi_regime_enabled=rsi_regime_enabled,
        rsi_divergence_filter_enabled=rsi_divergence_filter_enabled,
        bb_extreme_volatility_block=bb_extreme_volatility_block,
        macd_confirmation_enabled=macd_confirmation_enabled,
        macd_divergence_filter_enabled=macd_divergence_filter_enabled,
    )

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
        "Yapı Yönü": model["direction_side"],
        "Hareket Fazı": model["phase"],
        "Tepki Teyidi": model["response_side"],
        "Yapı Giriş Uygun": model["entry_allowed"],
        "Yapı Notu": model["reason"],
        "RSI Rejimi": model["rsi_regime"],
        "RSI Uyumsuzluğu": model["rsi_divergence"],
        "RSI Momentum Kırılımı": model["rsi_momentum_break"],
        "Bollinger Durumu": model["bb_state"],
        "BB Trend Sinyali": model["bb_trend_signal"],
        "Band Walk": model["bb_band_walk"],
        "W/M Formasyonu": model["bb_pattern"],
        "Ortalama Dönüş Adayı": model["bb_mean_reversion_side"],
        "MACD Rejimi": model["macd_regime"],
        "MACD Histogram Durumu": model["macd_momentum_state"],
        "MACD Uyumsuzluğu": model["macd_divergence"],
        "MACD Whipsaw": model["macd_whipsaw"],
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
    market_structure_enabled: bool = True,
    entry_model: str = "Düzeltme + Tepki",
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Pariteler taranıyor...")

    for i, sym in enumerate(symbols, start=1):
        row = scan_symbol_live(
            sym,
            change_window_minutes,
            scanner_tf,
            market_structure_enabled=market_structure_enabled,
            entry_model=entry_model,
            rsi_regime_enabled=rsi_regime_enabled,
            rsi_divergence_filter_enabled=rsi_divergence_filter_enabled,
            bb_extreme_volatility_block=bb_extreme_volatility_block,
            macd_confirmation_enabled=macd_confirmation_enabled,
            macd_divergence_filter_enabled=macd_divergence_filter_enabled,
        )

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
                market_structure_enabled=market_structure_enabled,
                entry_model=entry_model,
                rsi_regime_enabled=rsi_regime_enabled,
                rsi_divergence_filter_enabled=rsi_divergence_filter_enabled,
                bb_extreme_volatility_block=bb_extreme_volatility_block,
                macd_confirmation_enabled=macd_confirmation_enabled,
                macd_divergence_filter_enabled=macd_divergence_filter_enabled,
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
        if (
            market_structure_enabled or rsi_regime_enabled or rsi_divergence_filter_enabled
            or bb_extreme_volatility_block or macd_confirmation_enabled or macd_divergence_filter_enabled
        ) and opportunity != "PAS" and not bool(row.get("Yapı Giriş Uygun", False)):
            opportunity = "PAS"
            opportunity_reason = str(row.get("Yapı Notu", "MA + market yapısı teyidi yok"))
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
    market_structure_enabled: bool = True,
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
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
    filters_enabled = (
        market_structure_enabled or rsi_regime_enabled or rsi_divergence_filter_enabled
        or bb_extreme_volatility_block or macd_confirmation_enabled or macd_divergence_filter_enabled
    )
    if decision in {"LONG", "SHORT"} and filters_enabled and not bool(row.get("Yapı Giriş Uygun", False)):
        return "BEKLE", str(row.get("Yapı Notu", "MA + market yapısı teyidi bekleniyor.")), abs(entry_score) * 0.65
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
    symbols: list[str],
    change_window_minutes: int,
    alert_entry_tf: str,
    signal_threshold: float,
    market_structure_enabled: bool = True,
    entry_model: str = "Düzeltme + Tepki",
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Alarm ekranı hazırlanıyor...")
    for i, sym in enumerate(symbols, start=1):
        row = scan_symbol_live(
            sym,
            change_window_minutes,
            alert_entry_tf,
            market_structure_enabled=market_structure_enabled,
            entry_model=entry_model,
            rsi_regime_enabled=rsi_regime_enabled,
            rsi_divergence_filter_enabled=rsi_divergence_filter_enabled,
            bb_extreme_volatility_block=bb_extreme_volatility_block,
            macd_confirmation_enabled=macd_confirmation_enabled,
            macd_divergence_filter_enabled=macd_divergence_filter_enabled,
        )
        decision, reason, alert_score = alert_decision_from_row(
            row,
            alert_entry_tf,
            signal_threshold,
            market_structure_enabled,
            rsi_regime_enabled,
            rsi_divergence_filter_enabled,
            bb_extreme_volatility_block,
            macd_confirmation_enabled,
            macd_divergence_filter_enabled,
        )
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
    market_structure_enabled: bool = True,
    entry_model: str = "Düzeltme + Tepki",
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
    daily_status: Optional[dict] = None,
) -> None:
    st.header("Parite Alarm Ekranı")
    st.caption("Major ve minör pariteleri tek bakışta LONG / SHORT / BEKLE olarak gösterir. Bu ekran hızlı takip içindir; gerçek işlem için İşlem Asistanı karar kartı ve demo doğrulama kullanılmalı.")
    if daily_status:
        render_daily_trading_desk(daily_status)
        if daily_status.get("blocks_trade"):
            st.info("Günlük plan yeni işlem bildirimlerini durdurdu. Pariteler yalnızca piyasa takibi için gösteriliyor.")

    selected_symbols: list[str] = []
    for group in alert_groups:
        selected_symbols.extend(ALERT_PAIR_GROUPS.get(group, []))
    selected_symbols = list(dict.fromkeys(selected_symbols))

    if not selected_symbols:
        st.warning("En az bir parite grubu seçmelisin.")
        return

    with st.spinner("Major/minör pariteler taranıyor..."):
        board = build_alert_board_rows(
            selected_symbols,
            change_window_minutes,
            alert_entry_tf,
            signal_threshold,
            market_structure_enabled=market_structure_enabled,
            entry_model=entry_model,
            rsi_regime_enabled=rsi_regime_enabled,
            rsi_divergence_filter_enabled=rsi_divergence_filter_enabled,
            bb_extreme_volatility_block=bb_extreme_volatility_block,
            macd_confirmation_enabled=macd_confirmation_enabled,
            macd_divergence_filter_enabled=macd_divergence_filter_enabled,
        )

    if board.empty:
        st.warning("Alarm ekranı için veri alınamadı.")
        return

    for row in board[board["Alarm"].isin(["LONG", "SHORT"])].to_dict("records"):
        floor_rule = {"5 Dakika": "5min", "15 Dakika": "15min", "1 Saat": "1h"}.get(alert_entry_tf, "15min")
        candle_key = f"{alert_entry_tf}|{pd.Timestamp.now(tz='UTC').floor(floor_rule)}"
        payload = {"symbol": row.get("Sembol"), "side": row.get("Alarm"), "reason": row.get("Alarm Nedeni"), "score": row.get("Alarm Skoru")}
        notifications_allowed = not (daily_status and daily_status.get("blocks_trade"))
        if notifications_allowed and record_alert_once(str(row.get("Sembol")), str(row.get("Alarm")), candle_key, payload) and webhook_url.strip():
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
        + (f"MA50/MA200 + swing yapısı ve '{entry_model}' filtresi aktiftir. " if market_structure_enabled else "")
        + f"Değişim %, sol menüdeki '{change_window_label}' seçimine göre Yahoo 1 dakikalık kapanış verisinden hesaplanır."
    )

    major_df = board[board["Sembol"].isin(MAJOR_PAIRS)]
    minor_df = board[board["Sembol"].isin(MINOR_PAIRS)]

    if "Major" in alert_groups:
        render_alert_section("Major Pariteler", major_df)
    if "Minör" in alert_groups:
        render_alert_section("Minör Pariteler", minor_df)

    with st.expander("Tablo görünümü", expanded=False):
        table_cols = [
            "Sembol", "Alarm", "Tek Karar", "Tek Karar Notu", "Alarm Skoru", "Değişim %",
            "Yapı Yönü", "Hareket Fazı", "Tepki Teyidi", "4H", "1H", "15M", "5M", "Alarm Nedeni",
            "RSI Rejimi", "RSI Uyumsuzluğu", "RSI Momentum Kırılımı",
            "Bollinger Durumu", "BB Trend Sinyali", "Band Walk", "W/M Formasyonu", "Ortalama Dönüş Adayı",
            "MACD Rejimi", "MACD Histogram Durumu", "MACD Uyumsuzluğu", "MACD Whipsaw",
        ]
        st.dataframe(board[table_cols], use_container_width=True, height=420)
        st.download_button(
            "Alarm Tablosunu CSV İndir",
            data=board[table_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="forex_alert_board.csv",
            mime="text/csv",
        )


def calculate_manual_pips(symbol: str, side: str, entry: float, exit_price: float) -> Optional[float]:
    if entry is None or exit_price is None or entry <= 0 or exit_price <= 0:
        return None
    pip = get_pip_size(symbol)
    if side == "LONG":
        return (exit_price - entry) / pip
    if side == "SHORT":
        return (entry - exit_price) / pip
    return None


def daily_trading_status(
    journal: pd.DataFrame,
    account_size: float,
    risk_pct: float,
    target_min_usd: float,
    target_max_usd: float,
    max_loss_usd: float,
    max_closed_trades: int,
    stop_after_target: bool = True,
) -> dict:
    """Günlük hedefi bir kazanç vaadi değil, yeni işlem durdurma disiplini olarak uygular."""
    today = pd.Timestamp.now(tz=TR_TZ).date()
    day_rows = pd.DataFrame()
    if journal is not None and not journal.empty:
        journal = journal.copy()
        time_col = "Kayıt Zamanı" if "Kayıt Zamanı" in journal.columns else ("Tarih" if "Tarih" in journal.columns else None)
        if time_col:
            times = pd.to_datetime(journal[time_col], utc=True, errors="coerce")
            local_days = times.dt.tz_convert(TR_TZ).dt.date
            day_rows = journal[local_days == today].copy()

    results = day_rows.get("Sonuç", pd.Series(dtype=str)).astype(str)
    closed_mask = ~results.isin({"Açık", "İptal", "", "nan"})
    closed = day_rows[closed_mask].copy() if not day_rows.empty else pd.DataFrame()

    if closed.empty:
        realized_usd = 0.0
    else:
        direct_pnl = pd.to_numeric(closed.get("PnL USD", pd.Series(np.nan, index=closed.index)), errors="coerce")
        r_values = pd.to_numeric(closed.get("R", pd.Series(np.nan, index=closed.index)), errors="coerce")
        risk_amounts = pd.to_numeric(closed.get("Risk Tutarı", pd.Series(np.nan, index=closed.index)), errors="coerce")
        estimated_pnl = r_values * risk_amounts
        realized_usd = float(direct_pnl.fillna(estimated_pnl).fillna(0.0).sum())

    risk_amount = float(account_size) * float(risk_pct) / 100.0
    target_min = max(float(target_min_usd), 0.0)
    target_max = max(float(target_max_usd), target_min)
    max_loss = max(float(max_loss_usd), 0.0)
    trade_count = int(len(closed))
    progress_pct = 0.0 if target_min <= 0 else float(np.clip(100 * realized_usd / target_min, 0, 100))

    blockers = []
    label = "İŞLEM ARANABİLİR"
    state = "ok"
    if max_loss > 0 and realized_usd <= -max_loss:
        blockers.append(f"günlük zarar limiti -${max_loss:.0f} doldu")
        label, state = "GÜNÜ KAPAT", "bad"
    elif stop_after_target and target_min > 0 and realized_usd >= target_min:
        blockers.append(f"günlük minimum hedef ${target_min:.0f} tamamlandı")
        label, state = "HEDEF TAMAM — GÜNÜ KAPAT", "ok"
    elif max_closed_trades > 0 and trade_count >= int(max_closed_trades):
        blockers.append(f"günlük {int(max_closed_trades)} kapalı işlem limiti doldu")
        label, state = "İŞLEM LİMİTİ DOLDU", "warn"
    elif realized_usd < 0:
        label, state = "SEÇİCİ OL", "warn"

    target_r_low = target_min / risk_amount if risk_amount > 0 else np.nan
    target_r_high = target_max / risk_amount if risk_amount > 0 else np.nan
    return {
        "blocks_trade": bool(blockers),
        "state": state,
        "label": label,
        "text": "; ".join(blockers) if blockers else "Günlük limitler açık; yalnızca onaylı setup değerlendirilebilir.",
        "realized_usd": realized_usd,
        "closed_trades": trade_count,
        "risk_amount": risk_amount,
        "target_min_usd": target_min,
        "target_max_usd": target_max,
        "max_loss_usd": max_loss,
        "progress_pct": progress_pct,
        "target_r_low": target_r_low,
        "target_r_high": target_r_high,
    }


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

    if ema50_slope_atr >= 0.80 and ema_gap_atr >= 0.30:
        return {
            "label": "Trend",
            "state": "ok",
            "text": f"Trend piyasası: EMA eğimi {ema50_slope_atr:.2f} ATR, EMA açıklığı {ema_gap_atr:.2f} ATR.",
            "score_adjust": 8,
        }

    if ema50_slope_atr <= 0.60 and ema_gap_atr <= 0.40 and bb_width_atr <= 4.0:
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


def build_live_range_decision(
    symbol: str,
    selected_tf: str,
    current_price: Optional[float],
    account_size: float,
    risk_pct: float,
    pip_value_per_lot: float,
    total_cost_pips: float,
    engine_quality: Optional[dict],
    market_regime: dict,
    swing_lookback: int = 8,
) -> Optional[dict]:
    """Yatay rejimde teyitli bant dönüşünü ayrı bir canlı karar olarak üretir."""
    if selected_tf != "15 Dakika" or market_regime.get("label") != "Yatay":
        return None
    quality_label = str((engine_quality or {}).get("label", "Test bekliyor"))
    base = {
        "action": "BEKLE",
        "class": "simple-wait",
        "subtitle": "Yatay piyasa motoru aktif; bant dönüş tetiği bekleniyor.",
        "reason": "Trend göstergeleri bu rejimde karar vermiyor. Bollinger dışından orta banda teyitli dönüş gerekli.",
        "steps": ["Trend devamı işlemi açma.", "Bant dönüş mumunun kapanmasını bekle.", "Orta banda en az 1R alan yoksa pas geç."],
        "levels": {},
    }
    if quality_label not in {"İyi", "Orta"}:
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Yatay piyasa motoru doğrulanmadı.",
            "reason": f"RANGE motoru kalite sonucu: {quality_label}.",
        })
        return base
    prm = TIMEFRAMES[selected_tf]
    raw = fetch_ohlc(symbol, prm["interval"], prm["period"])
    if raw.empty or len(raw) < 100 or current_price is None:
        base["reason"] = "Canlı range planı için fiyat veya mum verisi yetersiz."
        return base
    model = market_structure_frame(_utc_index_df(raw).iloc[:-1])
    valid = model.dropna(subset=["ATR14", "BBMidTarget"])
    if valid.empty:
        return base
    row = valid.iloc[-1]
    side = str(row.get("RangeReversalSignal", row.get("BBMeanReversionSide", "NONE")))
    if side not in {"LONG", "SHORT"}:
        return base
    entry = float(current_price)
    atr = float(row["ATR14"])
    recent = valid.tail(max(int(swing_lookback), 4))
    target = float(row["BBMidTarget"])
    pip = get_pip_size(symbol)
    if side == "LONG":
        stop = min(float(recent["Low"].min()) - atr * 0.15, entry - atr * 0.75)
        reward_distance = target - entry
        stop_distance = entry - stop
    else:
        stop = max(float(recent["High"].max()) + atr * 0.15, entry + atr * 0.75)
        reward_distance = entry - target
        stop_distance = stop - entry
    risk_pips = stop_distance / pip + max(float(total_cost_pips), 0.0)
    net_reward_pips = reward_distance / pip - max(float(total_cost_pips), 0.0)
    reward_r = net_reward_pips / risk_pips if risk_pips > 0 else -np.inf
    risk_amount = float(account_size) * float(risk_pct) / 100.0
    lot = risk_amount / (risk_pips * float(pip_value_per_lot)) if risk_pips > 0 and pip_value_per_lot > 0 else 0.0
    if reward_distance <= 0 or reward_r < 1.0 or not np.isfinite(lot) or lot <= 0:
        base["reason"] = f"Bant dönüş adayı var fakat orta banda net alan {reward_r:.2f}R; minimum 1.00R."
        return base
    dec = price_decimals(symbol)
    side_word = "LONG AÇ" if side == "LONG" else "SHORT AÇ"
    return {
        "action": side_word,
        "class": "simple-buy" if side == "LONG" else "simple-sell",
        "subtitle": "Yatay piyasa orta banda dönüş motoru tetiklendi.",
        "reason": f"Teyitli bant dönüşü ve {reward_r:.2f}R net orta bant alanı; motor kalitesi {quality_label}.",
        "steps": [
            "Broker bid/ask fiyatını ve spreadi kontrol et.",
            f"Hedef Bollinger orta bant: {target:.{dec}f}.",
            "Fiyat yeniden bant dışına hızlanırsa stopu büyütme.",
        ],
        "levels": {
            "Giriş": f"{entry:.{dec}f}", "Stop": f"{stop:.{dec}f}",
            "Kâr Al": f"{target:.{dec}f}", "Lot": f"{lot:.2f}",
        },
        "engine": "RANGE",
    }


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


def render_market_model_card(model: dict) -> None:
    if not model.get("filters_enabled"):
        return
    allowed = bool(model.get("entry_allowed"))
    css = "ok-box" if allowed else "warn-box"
    direction = escape(str(model.get("direction_side", "NONE")))
    phase = escape(str(model.get("phase", "-")))
    response = escape(str(model.get("response_side", "NONE")))
    rsi_regime = escape(str(model.get("rsi_regime", "NEUTRAL")))
    divergence = escape(str(model.get("rsi_divergence", "NONE")))
    momentum_break = escape(str(model.get("rsi_momentum_break", "NONE")))
    bb_state = escape(str(model.get("bb_state", "-")))
    bb_signal = escape(str(model.get("bb_trend_signal", "NONE")))
    bb_walk = escape(str(model.get("bb_band_walk", "NONE")))
    bb_pattern = escape(str(model.get("bb_pattern", "NONE")))
    mean_reversion = escape(str(model.get("bb_mean_reversion_side", "NONE")))
    width_pct = model.get("bb_width_percentile")
    width_text = "-" if width_pct is None or pd.isna(width_pct) else f"%{float(width_pct) * 100:.0f}"
    mid_target = model.get("bb_mid_target")
    mid_target_text = "-" if mid_target is None or pd.isna(mid_target) else f"{float(mid_target):.5f}"
    macd_regime = escape(str(model.get("macd_regime", "TRANSITION")))
    macd_state = escape(str(model.get("macd_momentum_state", "MIXED")))
    macd_divergence = escape(str(model.get("macd_divergence", "NONE")))
    macd_whipsaw = "EVET" if model.get("macd_whipsaw") else "HAYIR"
    macd_atr = model.get("macd_atr")
    macd_hist_atr = model.get("macd_hist_atr")
    macd_atr_text = "-" if macd_atr is None or pd.isna(macd_atr) else f"{float(macd_atr):+.3f}"
    macd_hist_atr_text = "-" if macd_hist_atr is None or pd.isna(macd_hist_atr) else f"{float(macd_hist_atr):+.3f}"
    reason = escape(str(model.get("reason", "-")))
    st.markdown(
        f"<div class='{css}'><b>MA + Market Yapısı Modeli</b><br>"
        f"4H: {escape(str(model.get('h4_side', 'NONE')))} | "
        f"1H: {escape(str(model.get('h1_side', 'NONE')))} | Ana yön: {direction}<br>"
        f"Giriş fazı: {phase} | Tepki: {response}<br>"
        f"RSI rejimi: {rsi_regime} | Uyumsuzluk: {divergence} | Erken momentum kırılımı: {momentum_break}<br>"
        f"Bollinger: {bb_state} ({escape(width_text)}) | Trend sinyali: {bb_signal} | Band walk: {bb_walk}<br>"
        f"W/M: {bb_pattern} | Ayrı mean-reversion adayı: {mean_reversion} | Orta bant: {escape(mid_target_text)}<br>"
        f"MACD: {macd_regime} | Histogram: {macd_state} | Uyumsuzluk: {macd_divergence} | Whipsaw: {macd_whipsaw}<br>"
        f"Normalize güç: MACD/ATR {escape(macd_atr_text)} | Histogram/ATR {escape(macd_hist_atr_text)}<br>"
        f"{reason}</div>",
        unsafe_allow_html=True,
    )


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


def _summary_text(summary: pd.DataFrame, tf_name: str, column: str, default: str = "-") -> str:
    if summary is None or summary.empty or column not in summary.columns:
        return default
    row = summary[summary["Zaman Dilimi"] == tf_name]
    if row.empty:
        return default
    return str(row[column].iloc[0])


def build_market_model_status(
    summary: pd.DataFrame,
    selected_tf: str,
    enabled: bool,
    entry_model: str,
    rsi_regime_enabled: bool = True,
    rsi_divergence_filter_enabled: bool = True,
    bb_extreme_volatility_block: bool = True,
    macd_confirmation_enabled: bool = True,
    macd_divergence_filter_enabled: bool = True,
) -> dict:
    """Canlı ekranda MA + yapı yönü ile corrective/response girişini tek statüye indirger."""
    h4_ma = _summary_text(summary, "4 Saat", "MA Yönü", "NEUTRAL")
    h4_structure = _summary_text(summary, "4 Saat", "Market Yapısı", "RANGE")
    h1_ma = _summary_text(summary, "1 Saat", "MA Yönü", "NEUTRAL")
    h1_structure = _summary_text(summary, "1 Saat", "Market Yapısı", "RANGE")
    entry_ma = _summary_text(summary, selected_tf, "MA Yönü", "NEUTRAL")
    entry_structure = _summary_text(summary, selected_tf, "Market Yapısı", "RANGE")
    response_side = _summary_text(summary, selected_tf, "Tepki Teyidi", "NONE")
    phase = _summary_text(summary, selected_tf, "Hareket Fazı", "UYUMSUZ / YATAY")
    rsi_regime = _summary_text(summary, selected_tf, "RSI Rejimi", "NEUTRAL")
    rsi_divergence = _summary_text(summary, selected_tf, "RSI Uyumsuzluğu", "NONE")
    rsi_momentum_break = _summary_text(summary, selected_tf, "RSI Momentum Kırılımı", "NONE")
    bb_state = _summary_text(summary, selected_tf, "Bollinger Durumu", "VERİ YETERSİZ")
    bb_trend_signal = _summary_text(summary, selected_tf, "BB Trend Sinyali", "NONE")
    bb_band_walk = _summary_text(summary, selected_tf, "Band Walk", "NONE")
    bb_pattern = _summary_text(summary, selected_tf, "W/M Formasyonu", "NONE")
    bb_mean_reversion_side = _summary_text(summary, selected_tf, "Ortalama Dönüş Adayı", "NONE")
    bb_mid_target_text = _summary_text(summary, selected_tf, "BB Orta Bant Hedefi", "nan")
    bb_width_percentile_text = _summary_text(summary, selected_tf, "BB Genişlik Yüzdeliği", "nan")
    try:
        bb_mid_target = float(bb_mid_target_text)
    except (TypeError, ValueError):
        bb_mid_target = np.nan
    try:
        bb_width_percentile = float(bb_width_percentile_text)
    except (TypeError, ValueError):
        bb_width_percentile = np.nan
    macd_regime = _summary_text(summary, selected_tf, "MACD Rejimi", "TRANSITION")
    macd_momentum_state = _summary_text(summary, selected_tf, "MACD Histogram Durumu", "MIXED")
    macd_divergence = _summary_text(summary, selected_tf, "MACD Uyumsuzluğu", "NONE")
    macd_whipsaw_text = _summary_text(summary, selected_tf, "MACD Whipsaw", "False")
    macd_whipsaw = str(macd_whipsaw_text).lower() in {"true", "1", "yes", "evet"}
    macd_atr_text = _summary_text(summary, selected_tf, "MACD / ATR", "nan")
    macd_hist_atr_text = _summary_text(summary, selected_tf, "MACD Histogram / ATR", "nan")
    try:
        macd_atr = float(macd_atr_text)
    except (TypeError, ValueError):
        macd_atr = np.nan
    try:
        macd_hist_atr = float(macd_hist_atr_text)
    except (TypeError, ValueError):
        macd_hist_atr = np.nan

    h4_side = "LONG" if h4_ma == h4_structure == "BULLISH" else ("SHORT" if h4_ma == h4_structure == "BEARISH" else "NONE")
    h1_side = "LONG" if h1_ma == h1_structure == "BULLISH" else ("SHORT" if h1_ma == h1_structure == "BEARISH" else "NONE")
    entry_side = "LONG" if entry_ma == entry_structure == "BULLISH" else ("SHORT" if entry_ma == entry_structure == "BEARISH" else "NONE")
    direction_side = h4_side if h4_side == h1_side else "NONE"
    h4_score = _summary_score(summary, "4 Saat")
    h1_score = _summary_score(summary, "1 Saat")
    technical_side = (
        "LONG" if not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score >= 25 and h1_score >= 25
        else ("SHORT" if not pd.isna(h4_score) and not pd.isna(h1_score) and h4_score <= -25 and h1_score <= -25 else "NONE")
    )
    filter_side = direction_side if enabled else technical_side

    if not enabled:
        structure_allowed = True
        structure_reason = "MA + market yapısı filtresi kapalı."
    elif direction_side == "NONE":
        structure_allowed = False
        structure_reason = f"4H/1H MA ve swing yapısı aynı yönde değil (4H {h4_side}, 1H {h1_side})."
    elif entry_model == "Düzeltme + Tepki":
        structure_allowed = response_side == direction_side
        structure_reason = (
            f"{selected_tf} düzeltme sonrası {direction_side} tepki teyitli."
            if structure_allowed
            else f"{selected_tf} fazı {phase}; {direction_side} tepki kapanışı bekleniyor."
        )
    elif entry_model == "Bollinger Trend Devamı":
        structure_allowed = bb_trend_signal == direction_side
        structure_reason = (
            f"{selected_tf} {bb_state}: {direction_side} volatilite açılımı teyitli."
            if structure_allowed
            else f"{selected_tf} {bb_state}; trend yönünde bant dışı kapanış + genişleme + swing kırılımı bekleniyor."
        )
    elif entry_model == "Hibrit (Tepki / Bollinger)":
        structure_allowed = response_side == direction_side or bb_trend_signal == direction_side
        structure_reason = (
            f"{selected_tf} giriş teyidi: response={response_side}, Bollinger={bb_trend_signal}."
            if structure_allowed
            else f"{selected_tf} corrective response veya Bollinger volatilite açılımı bekleniyor."
        )
    else:
        structure_allowed = entry_side == direction_side
        structure_reason = (
            f"{selected_tf} MA ve swing yapısı {direction_side} yönünü doğruluyor."
            if structure_allowed
            else f"{selected_tf} MA ve swing yapısı ana yönle uyuşmuyor."
        )

    expected_rsi = "BULLISH" if filter_side == "LONG" else ("BEARISH" if filter_side == "SHORT" else "NONE")
    rsi_regime_ok = not rsi_regime_enabled or (expected_rsi != "NONE" and rsi_regime == expected_rsi)
    opposing_divergence = "BEARISH" if filter_side == "LONG" else ("BULLISH" if filter_side == "SHORT" else "NONE")
    divergence_ok = not rsi_divergence_filter_enabled or opposing_divergence == "NONE" or rsi_divergence != opposing_divergence
    bb_is_extreme = not pd.isna(bb_width_percentile) and bb_width_percentile >= 0.95
    bb_volatility_ok = not bb_extreme_volatility_block or not bb_is_extreme
    expected_macd = "BULLISH" if filter_side == "LONG" else ("BEARISH" if filter_side == "SHORT" else "NONE")
    macd_regime_ok = not macd_confirmation_enabled or (
        expected_macd != "NONE" and macd_regime == expected_macd and not macd_whipsaw
    )
    opposing_macd_divergence = "BEARISH" if filter_side == "LONG" else ("BULLISH" if filter_side == "SHORT" else "NONE")
    macd_divergence_ok = (
        not macd_divergence_filter_enabled
        or opposing_macd_divergence == "NONE"
        or macd_divergence != opposing_macd_divergence
    )
    entry_allowed = bool(
        structure_allowed and rsi_regime_ok and divergence_ok and bb_volatility_ok
        and macd_regime_ok and macd_divergence_ok
    )
    blockers = []
    if not structure_allowed:
        blockers.append(structure_reason)
    if not rsi_regime_ok:
        blockers.append(f"RSI rejimi {rsi_regime}; {expected_rsi} momentum onayı bekleniyor.")
    if not divergence_ok:
        blockers.append(f"Ters RSI uyumsuzluğu var: {rsi_divergence}; yeni {filter_side} girişi bekletildi.")
    if not bb_volatility_ok:
        blockers.append("Bollinger genişliği tarihsel %95 bölgesinde; haber/aşırı volatilite nedeniyle yeni giriş engellendi.")
    if not macd_regime_ok:
        detail = "whipsaw" if macd_whipsaw else f"rejim {macd_regime}"
        blockers.append(f"MACD {detail}; {expected_macd} sıfır rejimi onayı bekleniyor.")
    if not macd_divergence_ok:
        blockers.append(f"Ters MACD/histogram uyumsuzluğu var: {macd_divergence}; yeni {filter_side} girişi bekletildi.")
    reason = " ".join(blockers) if blockers else structure_reason

    return {
        "enabled": bool(enabled),
        "filters_enabled": bool(
            enabled or rsi_regime_enabled or rsi_divergence_filter_enabled or bb_extreme_volatility_block
            or macd_confirmation_enabled or macd_divergence_filter_enabled
        ),
        "entry_model": entry_model,
        "h4_side": h4_side,
        "h1_side": h1_side,
        "entry_side": entry_side,
        "direction_side": direction_side,
        "response_side": response_side,
        "phase": phase,
        "rsi_regime_enabled": bool(rsi_regime_enabled),
        "rsi_divergence_filter_enabled": bool(rsi_divergence_filter_enabled),
        "rsi_regime": rsi_regime,
        "rsi_regime_ok": bool(rsi_regime_ok),
        "rsi_divergence": rsi_divergence,
        "divergence_ok": bool(divergence_ok),
        "rsi_momentum_break": rsi_momentum_break,
        "bb_extreme_volatility_block": bool(bb_extreme_volatility_block),
        "bb_volatility_ok": bool(bb_volatility_ok),
        "bb_state": bb_state,
        "bb_trend_signal": bb_trend_signal,
        "bb_band_walk": bb_band_walk,
        "bb_pattern": bb_pattern,
        "bb_mean_reversion_side": bb_mean_reversion_side,
        "bb_mid_target": bb_mid_target,
        "bb_width_percentile": bb_width_percentile,
        "macd_confirmation_enabled": bool(macd_confirmation_enabled),
        "macd_divergence_filter_enabled": bool(macd_divergence_filter_enabled),
        "macd_regime": macd_regime,
        "macd_regime_ok": bool(macd_regime_ok),
        "macd_momentum_state": macd_momentum_state,
        "macd_divergence": macd_divergence,
        "macd_divergence_ok": bool(macd_divergence_ok),
        "macd_whipsaw": bool(macd_whipsaw),
        "macd_atr": macd_atr,
        "macd_hist_atr": macd_hist_atr,
        "entry_allowed": bool(entry_allowed),
        "reason": reason,
    }


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
    market_model: Optional[dict] = None,
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
    market_model = market_model or {
        "enabled": False, "filters_enabled": False, "entry_allowed": True,
        "direction_side": "NONE", "reason": "",
    }
    market_direction_ok = (
        not market_model.get("enabled")
        or (setup is not None and market_model.get("direction_side") == setup.side)
    )
    market_entry_ok = not market_model.get("filters_enabled") or bool(market_model.get("entry_allowed"))
    direction_ok = direction_ok and market_direction_ok
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
    candle_ok = bool(
        last_closed_close is not None
        and setup is not None
        and technical_signal == side
        and market_entry_ok
    )

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
        summary = "Ana yön veya MA + market yapısı koşulu giriş için yeterli değil."
        final_step_text = "Ana yön bekleniyor"
        primary_blocker = "Ana yön bekleniyor"
        primary_blocker_text = str(market_model.get("reason")) if not market_direction_ok else "4H + 1H ve seçilen sinyal modu aynı yönde yeterli güç üretmeli."
    elif not market_entry_ok:
        action = "BEKLE"
        status_class = "entry-signal-wait"
        summary = str(market_model.get("reason", "Düzeltme sonrası tepki teyidi bekleniyor."))
        final_step_text = "Tepki teyidi bekleniyor"
        primary_blocker = "Corrective / response bekleniyor"
        primary_blocker_text = summary
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
    elif not market_entry_ok:
        candle_text = str(market_model.get("reason", "Düzeltme/tepki teyidi bekleniyor"))
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
            "label": "4. Teknik Skor",
            "state": "ok" if technical_signal == side else "warn",
            "text": f"{technical_signal}: {technical_reason}",
        },
        {
            "label": "5. Yapı / Tepki",
            "state": "ok" if market_entry_ok else "warn",
            "text": str(market_model.get("reason", candle_text)),
        },
        {
            "label": "6. Sinyal",
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
        "market_model": market_model,
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
            "action": "YÖN YOK",
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
    tracker_market_model = tracker.get("market_model", {})
    m15_ok = tracker.get("technical_signal") == side and bool(tracker_market_model.get("entry_allowed", True))
    market_phase = str(tracker_market_model.get("phase", "UYUMSUZ / YATAY"))
    response_side = str(tracker_market_model.get("response_side", "NONE"))

    if matched_quality is None:
        return {
            "action": "BACKTEST BEKLENİYOR",
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

    if quality_label in LOW_SAMPLE_QUALITIES:
        return {
            "action": "BACKTEST YETERSİZ",
            "class": "simple-wait",
            "subtitle": "Yön okunabiliyor fakat performans örneği karar vermek için az.",
            "reason": f"Mevcut kalite etiketi {quality_label}. Bu, stratejinin kötü olduğu değil henüz yeterince sınanmadığı anlamına gelir.",
            "steps": [
                "Gerçek pozisyon açma; demo/izleme ile yeni örnek biriktir.",
                "15M için 60 günlük testi veya daha yüksek zaman dilimini kullan.",
                "Yeterli işlem oluşunca walk-forward sonucunu yeniden kontrol et.",
            ],
            "levels": levels_from_setup(),
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
        response_started = response_side == side or market_phase == "TEPKİ TEYİTLİ"
        phase_action = "TEPKİ BAŞLADI – İZLE" if response_started else "YÖN VAR – DÜZELTME/TEPKİ BEKLENİYOR"
        phase_subtitle = (
            f"{side_text.capitalize()} yönünde tepki başladı; kapanmış 15M skor teyidi bekleniyor."
            if response_started
            else f"4H + 1H {side_text} yönünde; uygun düzeltme ve tepki kapanışı henüz yok."
        )
        return {
            "action": phase_action,
            "class": "simple-wait",
            "subtitle": phase_subtitle,
            "reason": f"15M fazı: {market_phase}. {tracker.get('technical_reason', '')}",
            "steps": [
                f"{tracker.get('condition', '15M giriş skoru eşiği geçmeden işlem açma.')}",
                "4H ve 1H aynı yönde kalmalı.",
                "Kapanmış mum skoru oluşmadan acele etme.",
            ],
            "levels": levels_from_setup(),
        }

    if tracker.get("signal_now"):
        return {
            "action": f"GİRİŞ TEYİDİ – {side_word}",
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
    tracker_market_model = tracker.get("market_model", {})
    m15_ok = tracker.get("technical_signal") == side and bool(tracker_market_model.get("entry_allowed", True))
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
    momentum_model: Optional[dict] = None,
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
    level_event = position_level_event(side, float(current_price), float(stop), float(target))

    if side == "LONG":
        if level_event == "STOP":
            action, css, reason = "POZİSYONU KAPAT", "simple-sell", f"Fiyat stop seviyesine geldi/altına indi: {stop:.{dec}f}."
        elif level_event == "TARGET":
            action, css, reason = "KÂR AL / POZİSYONU KAPAT", "simple-buy", f"Fiyat hedef seviyeye geldi/üstüne çıktı: {target:.{dec}f}."
        elif to_target_pips is not None and target_distance_pips and 0 <= to_target_pips <= max(target_distance_pips * 0.15, 2):
            action, css, reason = "KÂR AL SEVİYESİNE YAKLAŞTI", "simple-buy", "Fiyat hedefe yaklaştı; plan dışı acele etmeden kâr al/stop takibi yap."
        elif opposite:
            action, css, reason = "KAPATMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."
    else:
        if level_event == "STOP":
            action, css, reason = "POZİSYONU KAPAT", "simple-sell", f"Fiyat stop seviyesine geldi/üstüne çıktı: {stop:.{dec}f}."
        elif level_event == "TARGET":
            action, css, reason = "KÂR AL / POZİSYONU KAPAT", "simple-buy", f"Fiyat hedef seviyeye geldi/altına indi: {target:.{dec}f}."
        elif to_target_pips is not None and target_distance_pips and 0 <= to_target_pips <= max(target_distance_pips * 0.15, 2):
            action, css, reason = "KÂR AL SEVİYESİNE YAKLAŞTI", "simple-buy", "Fiyat hedefe yaklaştı; plan dışı acele etmeden kâr al/stop takibi yap."
        elif opposite:
            action, css, reason = "KAPATMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."

    momentum_model = momentum_model or {}
    macd_regime = str(momentum_model.get("macd_regime", "TRANSITION"))
    macd_state = str(momentum_model.get("macd_momentum_state", "MIXED"))
    macd_divergence = str(momentum_model.get("macd_divergence", "NONE"))
    opposing_regime = (side == "LONG" and macd_regime == "BEARISH") or (side == "SHORT" and macd_regime == "BULLISH")
    weakening_with_divergence = (
        (side == "LONG" and macd_state == "BULLISH_WEAKENING" and macd_divergence == "BEARISH")
        or (side == "SHORT" and macd_state == "BEARISH_WEAKENING" and macd_divergence == "BULLISH")
    )
    if action in {"POZİSYONU TUT", "DİKKAT"}:
        if opposing_regime:
            action, css = "KAPATMAYI DÜŞÜN", "simple-sell"
            reason = f"MACD sıfır rejimi pozisyonun tersine döndü ({macd_regime}); fiyat yapısı ve stopla birlikte çıkışı değerlendir."
        elif weakening_with_divergence and r_multiple is not None and r_multiple >= 1:
            action, css = "KISMİ KÂR / STOP SIKILAŞTIR", "simple-buy"
            reason = "Histogram iki mumdur yavaşlıyor ve ters MACD uyumsuzluğu var; tek başına ters işlem açma, mevcut kârı koru."
        elif weakening_with_divergence:
            action, css = "MOMENTUM ZAYIFLIYOR", "simple-wait"
            reason = "Histogram yavaşlaması ile ters MACD uyumsuzluğu birlikte görüldü; stopu büyütme ve yeni ekleme yapma."

    risk_note = "Stop ve hedef plana göre izleniyor."
    if to_stop_pips is not None and to_stop_pips <= 0:
        risk_note = "Stop seviyesi tetiklendi."
    elif to_stop_pips is not None and stop_distance_pips and to_stop_pips <= max(stop_distance_pips * 0.25, 2):
        risk_note = "Stopa yakın; stopu büyütme."
    elif r_multiple is not None and r_multiple >= 1:
        risk_note = "Pozisyon en az 1R kâr bölgesinde."
    if weakening_with_divergence:
        risk_note += " MACD zayıflama teyidi aktif."

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


def build_intraday_opportunity(
    symbol: str,
    summary: pd.DataFrame,
    current_price: Optional[float],
    account_size: float,
    risk_pct: float,
    pip_value_per_lot: float,
    total_cost_pips: float,
    target_usd: float,
    session_filter: str = "Londra",
) -> dict:
    """Birkaç saatlik adayı ölçer; radar puanını kazanma olasılığı gibi sunmaz."""
    score_weights = {"4 Saat": 0.15, "1 Saat": 0.25, "15 Dakika": 0.35, "5 Dakika": 0.25}
    signed_base = 0.0
    used_weight = 0.0
    for tf_name, weight in score_weights.items():
        value = _summary_score(summary, tf_name)
        if not pd.isna(value):
            signed_base += float(value) * weight
            used_weight += weight
    signed_base = signed_base / used_weight if used_weight else 0.0

    response = _summary_text(summary, "15 Dakika", "Tepki Teyidi", "NONE")
    bb_signal = _summary_text(summary, "15 Dakika", "BB Trend Sinyali", "NONE")
    rsi_break = _summary_text(summary, "15 Dakika", "RSI Momentum Kırılımı", "NONE")
    macd_regime = _summary_text(summary, "15 Dakika", "MACD Rejimi", "TRANSITION")
    macd_state = _summary_text(summary, "15 Dakika", "MACD Histogram Durumu", "MIXED")
    phase = _summary_text(summary, "15 Dakika", "Hareket Fazı", "UYUMSUZ / YATAY")
    ma_direction = _summary_text(summary, "15 Dakika", "MA Yönü", "NEUTRAL")
    market_structure = _summary_text(summary, "15 Dakika", "Market Yapısı", "RANGE")
    h4_score = _summary_score(summary, "4 Saat")
    h1_score = _summary_score(summary, "1 Saat")

    long_points = max(signed_base, 0.0)
    short_points = max(-signed_base, 0.0)
    catalysts = []
    if response == "LONG":
        long_points += 22
        catalysts.append("15M tepki LONG")
    elif response == "SHORT":
        short_points += 22
        catalysts.append("15M tepki SHORT")
    if bb_signal == "LONG":
        long_points += 18
        catalysts.append("Bollinger yukarı açılım")
    elif bb_signal == "SHORT":
        short_points += 18
        catalysts.append("Bollinger aşağı açılım")
    if rsi_break == "BULLISH":
        long_points += 10
        catalysts.append("RSI erken yukarı kırılım")
    elif rsi_break == "BEARISH":
        short_points += 10
        catalysts.append("RSI erken aşağı kırılım")
    if macd_regime == "BULLISH":
        long_points += 8
    elif macd_regime == "BEARISH":
        short_points += 8
    if str(macd_state).startswith("BULLISH"):
        long_points += 7
    elif str(macd_state).startswith("BEARISH"):
        short_points += 7

    # 4H ile 1H ters yöndeyse kısa vade fırsatı tamamen yok sayılmaz; güven düşürülür.
    htf_conflict = (
        not pd.isna(h4_score) and not pd.isna(h1_score)
        and np.sign(float(h4_score)) != 0 and np.sign(float(h1_score)) != 0
        and np.sign(float(h4_score)) != np.sign(float(h1_score))
    )
    if htf_conflict:
        long_points *= 0.72
        short_points *= 0.72
        catalysts.append("4H/1H çelişkisi güveni düşürüyor")

    side = "LONG" if long_points > short_points else ("SHORT" if short_points > long_points else "NONE")
    radar_score = float(np.clip(max(long_points, short_points), 0, 100))

    pip = get_pip_size(symbol)
    lot = 0.0
    required_pips = np.nan
    recent_range_pips = np.nan
    target_price = np.nan
    latest_candle_time = None
    df = fetch_ohlc(symbol, TIMEFRAMES["15 Dakika"]["interval"], TIMEFRAMES["15 Dakika"]["period"])
    if df is not None and not df.empty and len(df) >= 40:
        latest_candle_time = df.index[-2] if len(df) > 1 else df.index[-1]
        ind = add_indicators(df.iloc[:-1])
        row = latest_valid_row(ind)
        if row is not None:
            atr = float(row["ATR14"])
            stop_pips = max(atr * 1.5 / pip, np.finfo(float).eps)
            risk_amount = float(account_size) * float(risk_pct) / 100.0
            cost_adjusted_stop = stop_pips + max(float(total_cost_pips), 0.0)
            if pip_value_per_lot > 0:
                lot = risk_amount / (cost_adjusted_stop * float(pip_value_per_lot))
                if lot > 0 and target_usd > 0:
                    required_pips = float(target_usd) / (float(pip_value_per_lot) * lot)
            rolling_range = ind["High"].rolling(16).max() - ind["Low"].rolling(16).min()
            recent_range = rolling_range.dropna().tail(96)
            if not recent_range.empty:
                recent_range_pips = float(recent_range.median() / pip)
    if current_price is not None and side in {"LONG", "SHORT"} and pd.notna(required_pips):
        direction = 1 if side == "LONG" else -1
        target_price = float(current_price) + direction * float(required_pips) * pip

    capacity_ratio = (
        float(required_pips / recent_range_pips)
        if pd.notna(required_pips) and pd.notna(recent_range_pips) and recent_range_pips > 0
        else np.nan
    )
    if pd.isna(capacity_ratio):
        capacity_text = "Hareket kapasitesi hesaplanamadı."
    elif capacity_ratio <= 0.65:
        capacity_text = "Hedef, son dönem tipik 4 saatlik hareketinin içinde."
    elif capacity_ratio <= 1.0:
        capacity_text = "Hedef mümkün aralıkta ama güçlü hareket gerekiyor."
    else:
        capacity_text = "Hedef tipik 4 saatlik hareketten büyük; birkaç saate sığmayabilir."

    expected_structure = "BULLISH" if side == "LONG" else ("BEARISH" if side == "SHORT" else "NONE")
    catalyst_matches = side in {response, bb_signal}
    structure_matches = (
        side in {"LONG", "SHORT"}
        and ma_direction == expected_structure
        and market_structure == expected_structure
    )
    in_session = bool(latest_candle_time is not None and is_in_trading_session(latest_candle_time, session_filter))
    readiness, readiness_blockers = classify_opportunity_readiness(
        radar_score=radar_score,
        side=side,
        catalyst_matches=catalyst_matches,
        structure_matches=structure_matches,
        htf_conflict=bool(htf_conflict),
        capacity_ratio=capacity_ratio,
        in_session=in_session,
    )
    if readiness == "READY":
        label = f"{side} TETİĞİ HAZIR"
        state = "long" if side == "LONG" else "short"
    elif readiness == "WATCH":
        label = f"{side} İÇİN İZLE"
        state = "long" if side == "LONG" else "short"
    else:
        label = "ŞİMDİLİK NÖTR"
        state = "neutral"

    reasons = catalysts[:3] or [f"15M fazı: {phase}", "Kısa vadeli momentum henüz net değil"]
    return {
        "label": label,
        "state": state,
        "side": side,
        "confidence": radar_score,
        "radar_score": radar_score,
        "readiness": readiness,
        "readiness_blockers": readiness_blockers,
        "catalyst_matches": bool(catalyst_matches),
        "structure_matches": bool(structure_matches),
        "in_session": bool(in_session),
        "session_filter": session_filter,
        "latest_candle_time": latest_candle_time,
        "reason": "; ".join(reasons),
        "phase": phase,
        "lot": lot,
        "required_pips": required_pips,
        "recent_range_pips": recent_range_pips,
        "target_price": target_price,
        "target_usd": float(target_usd),
        "capacity_text": capacity_text,
        "htf_conflict": bool(htf_conflict),
    }


def apply_opportunity_cooldown(
    opportunity: dict,
    symbol: str,
    cooldown_bars: int = 16,
) -> dict:
    """Aynı yöndeki radar tetiğini 15M'de varsayılan dört saat boyunca yeniden kurmaz."""
    out = dict(opportunity)
    if out.get("readiness") != "READY" or int(cooldown_bars) <= 0:
        return out
    candle_time = out.get("latest_candle_time")
    if candle_time is None:
        return out
    current = _to_utc_timestamp(candle_time)
    state_key = "intraday_radar_last_ready"
    saved = dict(st.session_state.get(state_key, {}))
    symbol_key = normalize_symbol(symbol)
    previous = saved.get(symbol_key)
    if previous and previous.get("side") == out.get("side"):
        previous_time = _to_utc_timestamp(previous.get("time"))
        elapsed_bars = int(max((current - previous_time).total_seconds(), 0) // (15 * 60))
        if current != previous_time and elapsed_bars < int(cooldown_bars):
            remaining = int(cooldown_bars) - elapsed_bars
            out["readiness"] = "WATCH"
            out["label"] = f"{out.get('side', '')} COOLDOWN / İZLE"
            out["readiness_blockers"] = list(out.get("readiness_blockers", [])) + [
                f"Aynı yöndeki son tetikten sonra {remaining} adet 15M mum daha bekleniyor"
            ]
            return out
    saved[symbol_key] = {"side": out.get("side"), "time": current.isoformat()}
    st.session_state[state_key] = saved
    return out


def render_intraday_opportunity(opportunity: dict, symbol: str) -> None:
    state = str(opportunity.get("state", "neutral"))
    css = {"long": "opportunity-long", "short": "opportunity-short"}.get(state, "opportunity-neutral")
    required = opportunity.get("required_pips", np.nan)
    recent_range = opportunity.get("recent_range_pips", np.nan)
    target_price = opportunity.get("target_price", np.nan)
    required_text = "-" if pd.isna(required) else f"{float(required):.1f} pip"
    range_text = "-" if pd.isna(recent_range) else f"{float(recent_range):.1f} pip"
    target_text = "-" if pd.isna(target_price) else f"{float(target_price):.{price_decimals(symbol)}f}"
    readiness = str(opportunity.get("readiness", "NEUTRAL"))
    readiness_text = {"READY": "Tetik hazır", "WATCH": "Yalnızca izle", "NEUTRAL": "Nötr"}.get(readiness, readiness)
    blockers = opportunity.get("readiness_blockers", []) or []
    blockers_text = " · ".join(str(item) for item in blockers[:3]) or "Yapı, tetik, seans ve hedef kapasitesi uygun."
    st.markdown(
        f"<div class='opportunity-card {css}'>"
        "<div class='section-kicker'>Önümüzdeki birkaç saat</div>"
        f"<div class='opportunity-title'>{escape(str(opportunity.get('label', '-')))}</div>"
        f"<div class='opportunity-score'>{float(opportunity.get('radar_score', opportunity.get('confidence', 0))):.0f}/100</div>"
        f"<div><b>Durum:</b> {escape(readiness_text)}</div>"
        f"<div>{escape(str(opportunity.get('reason', '-')))}</div>"
        f"<div class='opportunity-line'><b>${float(opportunity.get('target_usd', 0)):.0f} için gereken:</b> {escape(required_text)}</div>"
        f"<div class='opportunity-line'><b>Tahmini hedef fiyat:</b> {escape(target_text)}</div>"
        f"<div class='opportunity-line'><b>Tipik 4 saatlik hareket:</b> {escape(range_text)}</div>"
        f"<div class='opportunity-line'>{escape(str(opportunity.get('capacity_text', '-')))}</div>"
        f"<div class='opportunity-line'><b>Eksik/engel:</b> {escape(blockers_text)}</div>"
        "<div class='opportunity-line'><small>Radar puanı olasılık değildir. Tetik hazır olsa bile aşağıdaki doğrulanmış işlem kararı ayrıca LONG/SHORT demeden emir verilmez.</small></div>"
        "</div>",
        unsafe_allow_html=True,
    )



# =============================================================================
# PLOTS
# =============================================================================

def plot_intraday_change(symbol: str, lookback_minutes: int) -> tuple[go.Figure, Optional[dict]]:
    """Seçilen 1–24 saat penceresinde fiyatın başlangıca göre yüzde değişimini çizer."""
    df = fetch_intraday_history(symbol, "2d")
    fig = go.Figure()
    if df.empty:
        fig.update_layout(height=320, title="Kısa vadeli fiyat verisi alınamadı")
        return fig, None

    close = df["Close"].astype(float).dropna()
    if close.empty:
        fig.update_layout(height=320, title="Kısa vadeli kapanış verisi yok")
        return fig, None

    latest_time = close.index[-1]
    target_time = latest_time - pd.Timedelta(minutes=int(lookback_minutes))
    candidates = close[close.index <= target_time]
    if candidates.empty:
        fig.update_layout(height=320, title="Seçilen pencere için yeterli geçmiş yok")
        return fig, None

    reference_time = candidates.index[-1]
    reference_price = float(candidates.iloc[-1])
    window = close[close.index >= reference_time]
    pct = 100 * (window / reference_price - 1.0)
    local_index = window.index.tz_convert(TR_TZ)
    end_pct = float(pct.iloc[-1])
    line_color = "#198754" if end_pct >= 0 else "#dc3545"

    fig.add_trace(go.Scatter(
        x=local_index,
        y=pct,
        mode="lines",
        name="Değişim %",
        line=dict(color=line_color, width=2),
        customdata=window.to_numpy(),
        hovertemplate="%{x|%d.%m %H:%M}<br>Değişim: %{y:+.3f}%<br>Fiyat: %{customdata:.5f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6c757d")
    fig.add_trace(go.Scatter(
        x=[local_index[-1]], y=[end_pct], mode="markers+text", name="Son",
        marker=dict(color=line_color, size=9), text=[f"{end_pct:+.3f}%"],
        textposition="top center", hoverinfo="skip",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=25, r=20, t=45, b=25),
        title=f"{symbol.replace('=X', '')} | Son {int(lookback_minutes / 60)} Saatlik Değişim",
        xaxis_title="İstanbul saati",
        yaxis_title="Başlangıca göre %",
        showlegend=False,
    )
    return fig, {
        "reference_time": reference_time,
        "reference_price": reference_price,
        "latest_time": latest_time,
        "latest_price": float(window.iloc[-1]),
        "pct": end_pct,
    }

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
    if any(word in a for word in {"BEKLE", "İZLE", "PAS", "KAPAT", "TUT", "YÖN YOK", "YETERSİZ"}):
        return False
    return (
        "LONG AÇ" in a
        or "SHORT AÇ" in a
        or "ONAYLI LONG" in a
        or "ONAYLI SHORT" in a
    )


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
    daily_status: Optional[dict] = None,
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
    if daily_status and daily_status.get("blocks_trade"):
        blockers.append(f"Günlük plan: {daily_status.get('text', '-')}")
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


def apply_engine_evidence_filter(
    decision: dict,
    active_engine: Optional[str],
    engine_quality: Optional[dict],
) -> dict:
    """Çift motor laboratuvarı çalıştıysa yalnız kanıtlanan rejim motoruna izin verir."""
    if not is_new_position_decision(str(decision.get("action", ""))):
        return decision
    if active_engine not in {"TREND", "RANGE"}:
        out = dict(decision)
        out.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Piyasa rejimi geçişte; iki motor da devre dışı.",
            "reason": "Trend veya yatay rejim netleşmeden yeni pozisyon açılmaz.",
        })
        return out
    label = str((engine_quality or {}).get("label", "Test bekliyor"))
    edge_report = (engine_quality or {}).get("edge", {}) or {}
    edge_label = str(edge_report.get("label", "HESAPLANAMADI"))
    if label in {"İyi", "Orta"} and edge_label == "DOĞRULANDI":
        return decision
    out = dict(decision)
    out.update({
        "action": "PAS GEÇ",
        "class": "simple-pass",
        "subtitle": "Aktif piyasa motoru doğrulanmadı.",
        "reason": (
            f"{active_engine} motorunun backtest kalitesi {label}, edge kanıtı {edge_label}. "
            f"{edge_report.get('text', '')} Daha fazla işlem üretmek için başka rejimin motoru kullanılamaz."
        ),
        "steps": [
            "Bu sinyalde yeni pozisyon açma.",
            "Trend + Yatay Motoru ve Edge'i Test Et sonucunu kontrol et.",
            "Broker verisinde Orta/İyi kalite ve DOĞRULANDI edge birlikte oluşmadan gerçek işleme geçme.",
        ],
    })
    return out


def render_daily_trading_desk(status: dict) -> None:
    pnl = float(status.get("realized_usd", 0.0))
    target_min = float(status.get("target_min_usd", 0.0))
    target_max = float(status.get("target_max_usd", 0.0))
    risk_amount = float(status.get("risk_amount", 0.0))
    progress = float(status.get("progress_pct", 0.0))
    state = str(status.get("state", "warn"))
    css = "ok-box" if state == "ok" else ("bad-box" if state == "bad" else "warn-box")

    st.markdown("<div class='section-kicker'>Günlük disiplin planı</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Bugünkü gerçekleşen", f"${pnl:+.2f}")
    cols[1].metric("Günlük hedef bandı", f"${target_min:.0f}–${target_max:.0f}")
    cols[2].metric("Bir işlemde risk", f"${risk_amount:.2f}")
    cols[3].metric("Bugün kapanan işlem", int(status.get("closed_trades", 0)))

    target_r_low = status.get("target_r_low", np.nan)
    target_r_high = status.get("target_r_high", np.nan)
    r_text = "-" if pd.isna(target_r_low) else f"yaklaşık {float(target_r_low):.1f}R–{float(target_r_high):.1f}R"
    st.markdown(
        "<div class='daily-desk'>"
        f"<div class='daily-desk-title'>{escape(str(status.get('label', '-')))}</div>"
        f"<div class='daily-desk-note'>{escape(str(status.get('text', '-')))} "
        f"Hedef bandı mevcut işlem riskinle {escape(r_text)} gerektirir.</div>"
        "<div class='daily-progress-track'>"
        f"<div class='daily-progress-fill' style='width:{progress:.1f}%'></div>"
        "</div>"
        f"<div class='daily-desk-note'>Minimum hedef ilerlemesi: %{progress:.0f}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if status.get("blocks_trade"):
        st.markdown(
            f"<div class='{css}'><b>Yeni işlem kilidi:</b> {escape(str(status.get('text', '-')))}</div>",
            unsafe_allow_html=True,
        )
    elif pd.notna(target_r_low) and float(target_r_low) >= 2.0:
        st.caption(
            "Not: Minimum günlük hedef mevcut riskle en az 2R gerektiriyor. "
            "Bu her gün oluşmayabilir; hedefi tamamlamak için filtresiz veya plansız işlem açma."
        )


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
    data_provider = st.selectbox("Veri kaynağı", ["Yahoo Finance", "MetaTrader 5", "Broker CSV"], index=0)
    st.session_state["data_provider"] = data_provider
    if data_provider == "MetaTrader 5":
        st.caption("MT5 terminali/kitaplığı hazır değilse işlem verisi boş kalır; Yahoo'ya sessiz geçiş yapılmaz.")
    elif data_provider == "Broker CSV":
        broker_csv_symbol = normalize_symbol(st.text_input("CSV sembolü", value=symbol.replace("=X", "")))
        broker_csv_interval = st.selectbox("CSV mum zamanı", ["1m", "5m", "15m"], index=2)
        broker_csv_file = st.file_uploader(
            "MT5 OHLC CSV yükle",
            type=["csv", "txt"],
            help="Beklenen sütunlar: date+time veya time, open, high, low, close; opsiyonel tickvol/volume ve spread.",
        )
        st.session_state["broker_csv_symbol"] = broker_csv_symbol
        st.session_state["broker_csv_interval"] = broker_csv_interval
        if broker_csv_file is not None:
            parsed_broker_csv = parse_broker_csv_bytes(broker_csv_file.getvalue())
            if parsed_broker_csv.empty:
                st.error("Broker CSV okunamadı. Zaman ve OHLC sütunlarını kontrol et.")
                st.session_state.pop("broker_csv_df", None)
            else:
                st.session_state["broker_csv_df"] = parsed_broker_csv
                history_days = (parsed_broker_csv.index[-1] - parsed_broker_csv.index[0]).total_seconds() / 86400
                spread_note = "spread var" if "Spreadpoints" in parsed_broker_csv.columns else "spread yok"
                st.success(f"{len(parsed_broker_csv):,} mum · {history_days:.0f} gün · {spread_note}")
        elif isinstance(st.session_state.get("broker_csv_df"), pd.DataFrame):
            saved_csv = st.session_state["broker_csv_df"]
            st.caption(f"Oturumdaki broker verisi: {len(saved_csv):,} mum")

    chart_tf = st.radio("Grafik zamanı", tf_options, index=1)
    st.caption("Bu seçim grafiği değiştirir. Yeni Başlayan Modu açıksa işlem kararı yine 4H + 1H ana yön ve 15M giriş mantığıyla hesaplanır.")

    st.divider()
    st.subheader("Temel Risk")
    account_size = st.number_input("Hesap büyüklüğü", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("İşlem başına risk %", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    if risk_pct > 1.0:
        st.warning("%1 üzerindeki işlem riski kayıp serilerinde hesabı hızlı küçültebilir.")

    with st.expander("Günlük işlem planım", expanded=True):
        st.caption("Hedef sinyal üretmez; hedefe veya zarar limitine gelince yeni işlemi durdurur.")
        daily_target_min_usd = st.number_input("Minimum günlük hedef ($)", min_value=0.0, value=100.0, step=25.0)
        daily_target_max_usd = st.number_input("Üst günlük hedef ($)", min_value=0.0, value=200.0, step=25.0)
        daily_max_loss_usd = st.number_input("Maksimum günlük zarar ($)", min_value=0.0, value=100.0, step=25.0)
        daily_max_closed_trades = st.number_input("Günlük maksimum kapalı işlem", min_value=1, max_value=20, value=3, step=1)
        stop_after_daily_target = st.checkbox("Minimum hedefe ulaşınca yeni işlemi durdur", value=True)
        if daily_target_max_usd < daily_target_min_usd:
            st.warning("Üst hedef minimum hedeften küçük; uygulama üst hedefi minimum hedefe eşitleyecek.")

    with st.expander("Gelişmiş risk", expanded=False):
        rr = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        atr_mult = st.number_input("ATR Stop Çarpanı", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        stop_mode = st.selectbox("Stop modeli", ["ATR", "Swing + ATR", "Hibrit (uzak olan)"], index=2)
        target_mode = st.selectbox("Hedef modeli", ["Sabit R", "Yapı / minimum 1R"], index=0)
        swing_lookback = st.number_input("Swing bakış mumu", min_value=3, max_value=100, value=10, step=1)
        max_holding_bars = st.number_input(
            "Maksimum işlem süresi (mum, 0=kapalı)",
            min_value=0,
            max_value=500,
            value=24,
            step=4,
            help="Yeni Başlayan Modu/15M için 24 mum yaklaşık 6 saattir; günlük işlemin geceye taşınmasını azaltır.",
        )
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
        market_structure_enabled = st.checkbox(
            "MA50/MA200 + market yapısı filtresi",
            value=True,
            help="4H ve 1H'de MA yönü ile yükselen/düşen swing yapısının aynı yönde olmasını ister.",
        )
        entry_model = st.selectbox(
            "Giriş modeli",
            [
                "Düzeltme + Tepki", "Trend + Yapı", "Bollinger Trend Devamı",
                "Hibrit (Tepki / Bollinger)",
            ],
            index=0,
            disabled=not market_structure_enabled,
            help="Düzeltme + Tepki daha seçicidir; Trend + Yapı düzeltme beklemeden yapısal uyumla girişe izin verir.",
        )
        rsi_regime_enabled = st.checkbox(
            "RSI 50 rejim onayı",
            value=True,
            help="LONG için RSI ≥52, SHORT için RSI ≤48 ister; 48–52 aralığını nötr kabul eder.",
        )
        rsi_divergence_filter_enabled = st.checkbox(
            "Ters RSI uyumsuzluğunda yeni girişi engelle",
            value=True,
            help="Negatif uyumsuzlukta yeni LONG, pozitif uyumsuzlukta yeni SHORT girişini bekletir. Tek başına ters işlem açmaz.",
        )
        bb_extreme_volatility_block = st.checkbox(
            "Aşırı Bollinger genişliğinde yeni girişi engelle",
            value=True,
            help="Bant genişliği son 200 mumun yaklaşık %95 bölgesindeyse haber/aşırı volatilite riski nedeniyle yeni pozisyonu bekletir.",
        )
        macd_confirmation_enabled = st.checkbox(
            "MACD sıfır rejimi ve whipsaw onayı",
            value=True,
            help="LONG için MACD ve sinyal çizgisinin sıfır üstünde, SHORT için sıfır altında olmasını ister; sık kesişen yatay MACD'yi reddeder.",
        )
        macd_divergence_filter_enabled = st.checkbox(
            "Ters MACD uyumsuzluğunda yeni girişi engelle",
            value=True,
            help="Negatif MACD/histogram uyumsuzluğunda yeni LONG, pozitif uyumsuzlukta yeni SHORT girişini bekletir; tek başına ters işlem açmaz.",
        )
        show_position_tracker = st.checkbox("Pozisyon Takip Modu", value=True)
        change_window_label = st.selectbox("Yüzde değişim periyodu", list(PRICE_CHANGE_WINDOWS.keys()), index=1)
        change_window_minutes = PRICE_CHANGE_WINDOWS[change_window_label]
        intraday_chart_minutes = 1440
        st.caption("Ana ekranda son 24 saatin fiyat ve yüzde değişim grafiği gösterilir.")

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
        session_filter = st.selectbox(
            "İşlem seansı",
            list(TRADING_SESSIONS.keys()),
            index=list(TRADING_SESSIONS.keys()).index("Londra"),
            help="Major paritelerde likiditenin daha düzenli olduğu Londra saatleri günlük işlem için varsayılandır.",
        )
        if observed_spread is not None:
            st.caption(f"Broker son mumlarından medyan spread: {observed_spread:.1f} pip.")
        st.caption(session_description(session_filter))
        cooldown_bars = st.number_input(
            "Cooldown (mum)", min_value=0, max_value=200, value=16, step=1,
            help="15M'de 16 mum yaklaşık 4 saattir; aynı hareketi tekrar tekrar yeni fırsat saymayı azaltır.",
        )
        max_same_direction_trades = st.number_input(
            "Aynı yönde maksimum tekrar", min_value=1, max_value=10, value=1, step=1,
            help="Yeni bir trend dalgası oluşmadan aynı yönde ikinci işlem açılmaz.",
        )
        min_trades_required = st.number_input("Minimum backtest işlem sayısı", min_value=20, max_value=500, value=40, step=10)
        walk_forward_enabled = st.checkbox("Walk-forward sağlamlık kontrolü", value=True)
        walk_forward_folds = st.slider("Walk-forward fold", min_value=3, max_value=8, value=4, step=1)
        run_threshold_compare_requested = st.button(
            "45 / 50 / 60 Eşiklerini Karşılaştır",
            use_container_width=True,
            help="Eşiği körlemesine düşürmek yerine aynı ayarlarla üç ayrı backtest sonucu üretir.",
        )
        run_model_compare_requested = st.button(
            "4 Giriş Modelini Karşılaştır",
            use_container_width=True,
            help="Düzeltme, trend yapısı, Bollinger kırılımı ve hibrit modeli aynı maliyet/risk ayarlarında karşılaştırır.",
        )
        strategy_lab_period = st.text_input(
            "Çift motor araştırma periyodu",
            value="365d" if data_provider == "MetaTrader 5" else "60d",
            help="MT5 için 180d–365d önerilir. Yahoo 15M geçmişi pratikte yaklaşık 60 günle sınırlıdır.",
        )
        edge_simulations = st.number_input(
            "Edge testi simülasyonu",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500,
            help="İşlem R bootstrap ve giriş-zamanı kaydırma testlerinin tekrar sayısı.",
        )
        edge_horizon_bars = st.number_input(
            "Zamanlama testi ufku (15M mum)",
            min_value=4,
            max_value=96,
            value=16,
            step=4,
            help="16 mum, sinyalden sonraki yaklaşık 4 saatlik yön avantajını sınar.",
        )
        edge_trial_count = st.number_input(
            "Denenen toplam strateji sayısı",
            min_value=1,
            max_value=200,
            value=24,
            step=1,
            help="Bakılan motor/model/eşik kombinasyonlarının yaklaşık toplamı. Bonferroni düzeltmesinde kullanılır.",
        )
        edge_min_trades = st.number_input(
            "Edge için minimum işlem",
            min_value=20,
            max_value=500,
            value=60,
            step=10,
            help="Bunun altında sonuç olumlu görünse bile edge doğrulanmış sayılmaz.",
        )
        run_dual_engine_lab_requested = st.button(
            "Trend + Yatay Motoru ve Edge'i Test Et",
            use_container_width=True,
            help="İki motoru ayrı test eder; işlem R avantajını ve giriş zamanlamasını rastgele null modellere karşı sınar.",
        )

    run_bt_requested = st.button("Yeniden Hesapla", type="primary", use_container_width=True)

    settings_export = {
        "symbol": symbol, "chart_tf": chart_tf, "risk_pct": risk_pct, "rr": rr,
        "atr_mult": atr_mult, "stop_mode": stop_mode, "target_mode": target_mode,
        "swing_lookback": int(swing_lookback), "signal_threshold": int(signal_threshold),
        "total_cost_pips": spread_pips, "session": session_filter,
        "max_total_risk_pct": max_total_risk_pct, "max_currency_risk_pct": max_currency_risk_pct,
        "daily_stop_r": daily_stop_r, "weekly_stop_r": weekly_stop_r,
        "daily_target_min_usd": daily_target_min_usd,
        "daily_target_max_usd": daily_target_max_usd,
        "daily_max_loss_usd": daily_max_loss_usd,
        "daily_max_closed_trades": int(daily_max_closed_trades),
        "stop_after_daily_target": bool(stop_after_daily_target),
        "strategy_lab_period": strategy_lab_period,
        "edge_simulations": int(edge_simulations),
        "edge_horizon_bars": int(edge_horizon_bars),
        "edge_trial_count": int(edge_trial_count),
        "edge_min_trades": int(edge_min_trades),
        "market_structure_enabled": market_structure_enabled, "entry_model": entry_model,
        "rsi_regime_enabled": rsi_regime_enabled,
        "rsi_divergence_filter_enabled": rsi_divergence_filter_enabled,
        "bb_extreme_volatility_block": bb_extreme_volatility_block,
        "macd_confirmation_enabled": macd_confirmation_enabled,
        "macd_divergence_filter_enabled": macd_divergence_filter_enabled,
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
        _fetch_intraday_history_yahoo.clear()
        fetch_mt5_quote.clear()
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


current_journal = journal_dataframe()
current_daily_status = daily_trading_status(
    journal=current_journal,
    account_size=float(account_size),
    risk_pct=float(risk_pct),
    target_min_usd=float(daily_target_min_usd),
    target_max_usd=float(daily_target_max_usd),
    max_loss_usd=float(daily_max_loss_usd),
    max_closed_trades=int(daily_max_closed_trades),
    stop_after_target=bool(stop_after_daily_target),
)

if screen_mode == "Parite Alarm Ekranı":
    render_pair_alert_screen(
        change_window_minutes=change_window_minutes,
        change_window_label=change_window_label,
        alert_entry_tf=alert_entry_tf,
        alert_groups=alert_groups,
        alert_sort_mode=alert_sort_mode,
        signal_threshold=float(signal_threshold),
        webhook_url=webhook_url,
        market_structure_enabled=bool(market_structure_enabled),
        entry_model=entry_model,
        rsi_regime_enabled=bool(rsi_regime_enabled),
        rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
        bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
        macd_confirmation_enabled=bool(macd_confirmation_enabled),
        macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
        daily_status=current_daily_status,
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
) + (
    bool(walk_forward_enabled), int(walk_forward_folds), stop_mode, target_mode,
    int(swing_lookback), int(max_holding_bars), float(break_even_at_r),
    bool(market_structure_enabled), entry_model,
    bool(rsi_regime_enabled), bool(rsi_divergence_filter_enabled),
    bool(bb_extreme_volatility_block),
    bool(macd_confirmation_enabled), bool(macd_divergence_filter_enabled),
)

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
            market_structure_enabled=bool(market_structure_enabled),
            entry_model=entry_model,
            rsi_regime_enabled=bool(rsi_regime_enabled),
            rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
            bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
            macd_confirmation_enabled=bool(macd_confirmation_enabled),
            macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
        )
        q_label, q_css, q_text, wf_report = assess_backtest_with_walk_forward(
            bt_result,
            min_trades_required=int(min_trades_required),
            walk_forward_enabled=bool(walk_forward_enabled),
            walk_forward_folds=int(walk_forward_folds),
        )
        st.session_state["last_bt_key"] = current_bt_key
        st.session_state["last_bt_result"] = bt_result
        st.session_state["last_wf_report"] = wf_report
        st.session_state["last_bt_quality"] = {
            "label": q_label,
            "css": q_css,
            "text": q_text,
        }


def run_threshold_comparison() -> pd.DataFrame:
    """Aynı stratejiyi 45/50/60 eşiklerinde karşılaştırır; aktif eşiği kendiliğinden değiştirmez."""
    rows = []
    for candidate in [45.0, 50.0, 60.0]:
        candidate_bt = run_backtest(
            symbol=symbol,
            tf_name=bt_tf,
            period=bt_period,
            initial_balance=account_size,
            risk_pct=risk_pct,
            rr=rr,
            atr_mult=atr_mult,
            signal_threshold=candidate,
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
            market_structure_enabled=bool(market_structure_enabled),
            entry_model=entry_model,
            rsi_regime_enabled=bool(rsi_regime_enabled),
            rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
            bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
            macd_confirmation_enabled=bool(macd_confirmation_enabled),
            macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
        )
        label, _, text, _ = assess_backtest_with_walk_forward(
            candidate_bt,
            min_trades_required=int(min_trades_required),
            walk_forward_enabled=bool(walk_forward_enabled),
            walk_forward_folds=int(walk_forward_folds),
        )
        rows.append({
            "Eşik": int(candidate),
            "Kalite": label,
            "İşlem": extract_metric(candidate_bt.metrics, "İşlem Sayısı") or "0",
            "Profit Factor": extract_metric(candidate_bt.metrics, "Profit Factor") or "-",
            "Ortalama R": extract_metric(candidate_bt.metrics, "Ortalama R") or "-",
            "Son %30 PF": extract_metric(candidate_bt.metrics, "Son %30 Profit Factor") or "-",
            "Toplam PnL": extract_metric(candidate_bt.metrics, "Toplam PnL") or "-",
            "Not": text,
        })
    return pd.DataFrame(rows)


def run_entry_model_comparison() -> pd.DataFrame:
    """Gösterge ailesinin dört giriş yorumunu aynı koşullarda karşılaştırır."""
    rows = []
    models = [
        "Düzeltme + Tepki",
        "Trend + Yapı",
        "Bollinger Trend Devamı",
        "Hibrit (Tepki / Bollinger)",
    ]
    for candidate_model in models:
        candidate_bt = run_backtest(
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
            cooldown_bars=max(int(cooldown_bars), 16),
            session_filter=session_filter,
            max_same_direction_trades=1,
            min_trades_required=int(min_trades_required),
            stop_mode=stop_mode,
            target_mode=target_mode,
            swing_lookback=int(swing_lookback),
            max_holding_bars=int(max_holding_bars),
            break_even_at_r=float(break_even_at_r),
            market_structure_enabled=bool(market_structure_enabled),
            entry_model=candidate_model,
            rsi_regime_enabled=bool(rsi_regime_enabled),
            rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
            bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
            macd_confirmation_enabled=bool(macd_confirmation_enabled),
            macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
        )
        label, _, text, _ = assess_backtest_with_walk_forward(
            candidate_bt,
            min_trades_required=int(min_trades_required),
            walk_forward_enabled=bool(walk_forward_enabled),
            walk_forward_folds=int(walk_forward_folds),
        )
        rows.append({
            "Giriş Modeli": candidate_model,
            "Kanıt": label,
            "İşlem": extract_metric(candidate_bt.metrics, "İşlem Sayısı") or "0",
            "Profit Factor": extract_metric(candidate_bt.metrics, "Profit Factor") or "-",
            "Ortalama R": extract_metric(candidate_bt.metrics, "Ortalama R") or "-",
            "Son %30 PF": extract_metric(candidate_bt.metrics, "Son %30 Profit Factor") or "-",
            "Toplam PnL": extract_metric(candidate_bt.metrics, "Toplam PnL") or "-",
            "Karar": text,
        })
    return pd.DataFrame(rows)


def make_dual_engine_lab_key() -> tuple:
    """Laboratuvar sonucunu etkileyen tüm ayarları tek anahtarda toplar."""
    return (
        normalize_symbol(symbol),
        str(strategy_lab_period),
        data_provider,
        session_filter,
        round(float(spread_pips), 4),
        int(edge_simulations),
        int(edge_horizon_bars),
        int(edge_trial_count),
        int(edge_min_trades),
    )


def run_dual_engine_lab() -> tuple[pd.DataFrame, dict[str, BacktestResult], dict[str, dict], pd.DataFrame]:
    """Trend ve range motorlarını karışık toplam yerine ayrı kanıtla raporlar."""
    trend_bt = run_backtest(
        symbol=symbol,
        tf_name="15 Dakika",
        period=strategy_lab_period,
        initial_balance=account_size,
        risk_pct=risk_pct,
        rr=rr,
        atr_mult=atr_mult,
        signal_threshold=float(signal_threshold),
        spread_pips=spread_pips,
        pip_value_per_lot=pip_value_per_lot,
        cooldown_bars=max(int(cooldown_bars), 16),
        session_filter=session_filter,
        max_same_direction_trades=1,
        min_trades_required=int(min_trades_required),
        stop_mode=stop_mode,
        target_mode=target_mode,
        swing_lookback=int(swing_lookback),
        max_holding_bars=int(max_holding_bars),
        break_even_at_r=float(break_even_at_r),
        market_structure_enabled=True,
        entry_model="Düzeltme + Tepki",
        rsi_regime_enabled=True,
        rsi_divergence_filter_enabled=True,
        bb_extreme_volatility_block=True,
        macd_confirmation_enabled=True,
        macd_divergence_filter_enabled=True,
    )
    range_bt = run_range_mean_reversion_backtest(
        symbol=symbol,
        tf_name="15 Dakika",
        period=strategy_lab_period,
        initial_balance=account_size,
        risk_pct=risk_pct,
        spread_pips=spread_pips,
        pip_value_per_lot=pip_value_per_lot,
        session_filter=session_filter,
        cooldown_bars=max(int(cooldown_bars), 16),
        max_holding_bars=min(max(int(max_holding_bars), 8), 24),
        min_reward_r=1.0,
        swing_lookback=int(swing_lookback),
    )
    results = {"TREND": trend_bt, "RANGE": range_bt}
    qualities: dict[str, dict] = {}
    edge_reports: dict[str, dict] = {}
    rows = []
    for engine, result, label_text in [
        ("TREND", trend_bt, "Trend: düzeltme + tepki"),
        ("RANGE", range_bt, "Yatay: Bollinger orta banda dönüş"),
    ]:
        quality, css, note, wf = assess_backtest_with_walk_forward(
            result,
            min_trades_required=int(min_trades_required),
            walk_forward_enabled=bool(walk_forward_enabled),
            walk_forward_folds=int(walk_forward_folds),
        )
        edge_report = build_edge_validation_report(
            symbol=symbol,
            tf_name="15 Dakika",
            period=strategy_lab_period,
            bt=result,
            cost_pips=float(spread_pips),
            fetch_ohlc_fn=fetch_ohlc,
            horizon_bars=int(edge_horizon_bars),
            simulations=int(edge_simulations),
            mean_block_length=5.0,
            trial_count=int(edge_trial_count),
            min_trades=int(edge_min_trades),
        )
        edge_reports[engine] = edge_report
        qualities[engine] = {
            "label": quality,
            "css": css,
            "text": note,
            "walk_forward": wf,
            "edge": edge_report,
        }
        rows.append({
            "Motor": label_text,
            "Backtest Kalitesi": quality,
            "Edge Kanıtı": edge_report.get("label", "HESAPLANAMADI"),
            "İşlem": extract_metric(result.metrics, "İşlem Sayısı") or "0",
            "Profit Factor": extract_metric(result.metrics, "Profit Factor") or "-",
            "Ortalama R": extract_metric(result.metrics, "Ortalama R") or "-",
            "Son %30 PF": extract_metric(result.metrics, "Son %30 Profit Factor") or "-",
            "Toplam PnL": extract_metric(result.metrics, "Toplam PnL") or "-",
            "Karar": note,
        })
    return pd.DataFrame(rows), results, qualities, edge_validation_table(edge_reports)


if auto_plan_control and st.session_state.get("last_bt_key") != current_bt_key:
    run_and_store_backtest()

if run_bt_requested:
    run_and_store_backtest()

if run_threshold_compare_requested:
    with st.spinner("45 / 50 / 60 sinyal eşikleri karşılaştırılıyor..."):
        st.session_state["threshold_comparison_df"] = run_threshold_comparison()
        st.session_state["threshold_comparison_key"] = current_bt_key

if run_model_compare_requested:
    with st.spinner("Dört giriş modeli aynı risk ve maliyet koşullarında karşılaştırılıyor..."):
        st.session_state["entry_model_comparison_df"] = run_entry_model_comparison()
        st.session_state["entry_model_comparison_key"] = current_bt_key

if run_dual_engine_lab_requested:
    with st.spinner("Trend/yatay motorları ve istatistiksel edge testleri çalışıyor..."):
        lab_table, lab_results, lab_qualities, edge_table = run_dual_engine_lab()
        lab_key = make_dual_engine_lab_key()
        st.session_state["dual_engine_lab_key"] = lab_key
        st.session_state["dual_engine_lab_table"] = lab_table
        st.session_state["dual_engine_lab_results"] = lab_results
        st.session_state["dual_engine_lab_qualities"] = lab_qualities
        st.session_state["dual_engine_edge_table"] = edge_table

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
            market_structure_enabled=bool(market_structure_enabled),
            entry_model=entry_model,
            rsi_regime_enabled=bool(rsi_regime_enabled),
            rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
            bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
            macd_confirmation_enabled=bool(macd_confirmation_enabled),
            macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
        )
        st.session_state["scanner_df"] = scanner_df

# Top metrics
price_info = fetch_price_change(symbol, change_window_minutes)
price = price_info["latest"] if price_info and price_info.get("latest") is not None else fetch_last_price(symbol)
broker_quote = fetch_mt5_quote(symbol) if data_provider == "MetaTrader 5" else None
if broker_quote:
    price = float(broker_quote["mid"])

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
    if broker_quote:
        st.metric("Broker Spread", f"{float(broker_quote['spread_pips']):.1f} pip")
    else:
        st.metric("Pip Size", get_pip_size(symbol))

intraday_fig, intraday_info = plot_intraday_change(symbol, 1440)

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
) + (
    bool(walk_forward_enabled), int(walk_forward_folds), stop_mode, target_mode,
    int(swing_lookback), int(max_holding_bars), float(break_even_at_r),
    bool(market_structure_enabled), entry_model,
    bool(rsi_regime_enabled), bool(rsi_divergence_filter_enabled),
    bool(bb_extreme_volatility_block),
    bool(macd_confirmation_enabled), bool(macd_divergence_filter_enabled),
)
matched_quality = get_matching_backtest_quality(plan_bt_key)
allowed_quality_labels = allowed_quality_for_mode(strict_safety_mode, signal_mode)
market_model_status = build_market_model_status(
    summary=summary_df,
    selected_tf=selected_tf,
    enabled=bool(market_structure_enabled),
    entry_model=entry_model,
    rsi_regime_enabled=bool(rsi_regime_enabled),
    rsi_divergence_filter_enabled=bool(rsi_divergence_filter_enabled),
    bb_extreme_volatility_block=bool(bb_extreme_volatility_block),
    macd_confirmation_enabled=bool(macd_confirmation_enabled),
    macd_divergence_filter_enabled=bool(macd_divergence_filter_enabled),
)
preview_setup = build_trade_setup(
    symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult,
    pip_value_per_lot, spread_pips, entry_price=price, stop_mode=stop_mode,
    target_mode=target_mode, swing_lookback=int(swing_lookback),
)
market_regime = classify_market_regime(symbol, selected_tf)
current_strategy_engine = "TREND" if market_regime.get("label") == "Trend" else (
    "RANGE" if market_regime.get("label") == "Yatay" else None
)
live_lab_key = make_dual_engine_lab_key()
live_lab_matches = st.session_state.get("dual_engine_lab_key") == live_lab_key
live_engine_qualities = st.session_state.get("dual_engine_lab_qualities", {}) if live_lab_matches else {}
current_data_health = data_health_status(symbol, selected_tf)
current_news_status = (
    news_blackout_status(symbol, news_events, int(news_before_minutes), int(news_after_minutes))
    if news_filter_enabled
    else {"blocks_trade": False, "state": "ok", "text": "Haber filtresi kapalı."}
)
current_portfolio_status = portfolio_risk_status(
    current_journal,
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
    market_model=market_model_status,
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

if current_strategy_engine == "RANGE":
    range_decision = build_live_range_decision(
        symbol=symbol,
        selected_tf=selected_tf,
        current_price=price,
        account_size=float(account_size),
        risk_pct=float(risk_pct),
        pip_value_per_lot=float(pip_value_per_lot),
        total_cost_pips=float(spread_pips),
        engine_quality=live_engine_qualities.get("RANGE", {}) if live_lab_matches else {},
        market_regime=market_regime,
        swing_lookback=int(swing_lookback),
    )
    if range_decision is not None:
        simple_decision = range_decision

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
    filter_enabled=bool(ml_filter_enabled and current_strategy_engine != "RANGE"),
    threshold_pct=float(ml_threshold_pct),
)
simple_decision = apply_operational_safety_filters(
    decision=simple_decision,
    data_health=current_data_health,
    news_status=current_news_status,
    portfolio_status=current_portfolio_status,
    market_regime=market_regime,
    block_sideways=block_sideways,
    daily_status=current_daily_status,
)
decision_lab_key = make_dual_engine_lab_key()
decision_active_engine = "TREND" if market_regime.get("label") == "Trend" else (
    "RANGE" if market_regime.get("label") == "Yatay" else None
)
decision_lab_matches = st.session_state.get("dual_engine_lab_key") == decision_lab_key
decision_engine_qualities = st.session_state.get("dual_engine_lab_qualities", {}) if decision_lab_matches else {}
simple_decision = apply_engine_evidence_filter(
    simple_decision,
    active_engine=decision_active_engine,
    engine_quality=decision_engine_qualities.get(decision_active_engine, {}) if decision_active_engine else {},
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

intraday_opportunity = build_intraday_opportunity(
    symbol=symbol,
    summary=summary_df,
    current_price=price,
    account_size=float(account_size),
    risk_pct=float(risk_pct),
    pip_value_per_lot=float(pip_value_per_lot),
    total_cost_pips=float(spread_pips),
    target_usd=float(daily_target_min_usd),
    session_filter=session_filter,
)
intraday_opportunity = apply_opportunity_cooldown(
    intraday_opportunity,
    symbol=symbol,
    cooldown_bars=16,
)

st.header("24 Saatlik Fiyat ve Fırsat Radarı")
change_table = intraday_change_snapshot(symbol)
if not change_table.empty:
    summary_cols = st.columns(len(change_table))
    for col, (_, change_row) in zip(summary_cols, change_table.iterrows()):
        pct_value = change_row["Değişim %"]
        col.metric(
            str(change_row["Pencere"]).replace("Son ", ""),
            "-" if pd.isna(pct_value) else f"{float(pct_value):+.3f}%",
        )
radar_chart_col, radar_card_col = st.columns([2.1, 1.0])
with radar_chart_col:
    st.plotly_chart(intraday_fig, use_container_width=True)
    if intraday_info is not None:
        reference_local = _to_istanbul_timestamp(intraday_info["reference_time"]).strftime("%d.%m.%Y %H:%M")
        st.caption(
            f"24 saat referansı: {reference_local} / {intraday_info['reference_price']:.{price_decimals(symbol)}f} · "
            f"Son: {intraday_info['latest_price']:.{price_decimals(symbol)}f} · 1 dakikalık kapanış verisi."
        )
with radar_card_col:
    render_intraday_opportunity(intraday_opportunity, symbol)

st.subheader("Rejim Uyumlu Çift Motor")
active_engine = current_strategy_engine
active_engine_name = {
    "TREND": "Trend motoru — düzeltme + tepki",
    "RANGE": "Yatay motor — Bollinger orta banda dönüş",
}.get(active_engine, "Motor kapalı — piyasa rejimi geçişte")
current_lab_key = make_dual_engine_lab_key()
lab_matches = st.session_state.get("dual_engine_lab_key") == current_lab_key
lab_qualities = st.session_state.get("dual_engine_lab_qualities", {}) if lab_matches else {}
active_lab_quality = lab_qualities.get(active_engine, {}) if active_engine else {}
active_edge_report = active_lab_quality.get("edge", {}) or {}
active_edge_label = active_edge_report.get("label", "Test bekliyor")
engine_css = "ok-box" if (
    active_lab_quality.get("label") in {"İyi", "Orta"} and active_edge_label == "DOĞRULANDI"
) else (
    "bad-box" if active_lab_quality.get("label") == "Zayıf" else "warn-box"
)
engine_evidence = active_lab_quality.get("label", "Test bekliyor")
engine_note = active_lab_quality.get(
    "text",
    "Bu sembol/veri kaynağı/periyot için çift motor testi çalıştırılmadı.",
)
st.markdown(
    f"<div class='{engine_css}'><b>Aktif motor: {escape(active_engine_name)}</b><br>"
    f"Backtest kalitesi: {escape(str(engine_evidence))} · Edge: {escape(str(active_edge_label))}<br>"
    f"{escape(str(engine_note))}<br>{escape(str(active_edge_report.get('text', 'Edge testi bekliyor.')))}</div>",
    unsafe_allow_html=True,
)
if data_provider != "MetaTrader 5":
    st.caption(
        "Yahoo 15M veri geçmişi yaklaşık 60 günle sınırlıdır. 6–12 aylık ciddi doğrulama için "
        "yerel MT5 terminalini seçip araştırma periyodunu 180d–365d yap."
    )
lab_table = st.session_state.get("dual_engine_lab_table") if lab_matches else None
if isinstance(lab_table, pd.DataFrame) and not lab_table.empty:
    st.dataframe(lab_table, use_container_width=True, hide_index=True)
    edge_table = st.session_state.get("dual_engine_edge_table")
    if isinstance(edge_table, pd.DataFrame) and not edge_table.empty:
        st.markdown("**Edge Doğrulama Laboratuvarı**")
        st.dataframe(edge_table, use_container_width=True, hide_index=True)
        st.caption(
            "Bootstrap p: ortalama işlem R'sinin sıfırdan büyük olup olmadığını; zamanlama p: gerçek girişlerin "
            "aynı sinyal dizisinin rastgele kaydırmalarından üstün olup olmadığını sınar. p değerleri denenen strateji "
            "sayısıyla Bonferroni düzeltilmiştir. Radar skoru bir olasılık değildir."
        )
    verified = (
        lab_table["Backtest Kalitesi"].isin({"İyi", "Orta"})
        & lab_table["Edge Kanıtı"].eq("DOĞRULANDI")
    )
    if not bool(verified.any()):
        st.error(
            "İki motorun hiçbirinde hem backtest kalitesi hem istatistiksel edge doğrulanmadı. "
            "Bu sembolde canlı LONG/SHORT için güvenilir stratejik avantaj kanıtı yok."
        )

st.subheader("Doğrulanmış işlem kararı")
health_cols = st.columns(4)
health_cols[0].metric("Veri", current_data_health.get("status", "-"))
health_cols[1].metric("Piyasa", market_regime.get("label", "-"))
health_cols[2].metric("Açık risk", f"%{current_portfolio_status.get('total_risk_pct', 0):.2f}")
health_cols[3].metric("Haber", "ENGEL" if current_news_status.get("blocks_trade") else "TEMİZ")
for title, status in [("Veri", current_data_health), ("Haber", current_news_status), ("Portföy", current_portfolio_status)]:
    if status.get("blocks_trade"):
        st.warning(f"{title}: {status.get('text', '-')}")
evidence_status = strategy_evidence_status(matched_quality)
st.markdown(
    f"<div class='{evidence_status['css']}'><b>Strateji Kanıtı: {escape(evidence_status['label'])}</b><br>"
    f"{escape(evidence_status['text'])}</div>",
    unsafe_allow_html=True,
)
render_top_decision_panel(simple_decision)
if beginner_mode:
    if current_strategy_engine == "RANGE":
        render_simple_decision_card(simple_decision)
        st.caption("Yatay rejimde 4H/1H trend hunisi kullanılmaz; yalnız ayrı orta banda dönüş motoru değerlendirilir.")
    else:
        render_beginner_path(summary_df, matched_quality, entry_signal_tracker, selected_tf)
    with st.expander("Neden böyle dedi?", expanded=False):
        render_market_model_card(market_model_status)
        render_ml_prediction_card(ml_prediction, float(ml_threshold_pct), ml_filter_enabled)
        render_signal_summary_card(simple_decision, entry_signal_tracker, market_regime, signal_mode)
        render_entry_alarm_box(entry_signal_tracker)
        render_entry_signal_tracker(entry_signal_tracker)
else:
    render_market_model_card(market_model_status)
    render_ml_prediction_card(ml_prediction, float(ml_threshold_pct), ml_filter_enabled)
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

threshold_comparison_df = st.session_state.get("threshold_comparison_df")
if (
    st.session_state.get("threshold_comparison_key") == current_bt_key
    and isinstance(threshold_comparison_df, pd.DataFrame)
    and not threshold_comparison_df.empty
):
    with st.expander("45 / 50 / 60 Sinyal Eşiği Karşılaştırması", expanded=True):
        st.dataframe(threshold_comparison_df, use_container_width=True, hide_index=True)
        st.caption("Bu tablo aktif eşiği otomatik değiştirmez. Daha çok işlem tek başına daha iyi strateji anlamına gelmez; son dönem ve walk-forward birlikte değerlendirilmelidir.")

entry_model_comparison_df = st.session_state.get("entry_model_comparison_df")
if (
    st.session_state.get("entry_model_comparison_key") == current_bt_key
    and isinstance(entry_model_comparison_df, pd.DataFrame)
    and not entry_model_comparison_df.empty
):
    with st.expander("Gösterge / Giriş Modeli Karşılaştırması", expanded=True):
        st.dataframe(entry_model_comparison_df, use_container_width=True, hide_index=True)
        accepted = entry_model_comparison_df["Kanıt"].isin({"İyi", "Orta"})
        if not bool(accepted.any()):
            st.error(
                "Bu dört EMA/RSI/MACD/Bollinger giriş yorumundan hiçbiri doğrulanmadı. "
                "Uygulama filtreleri gevşetip zorla işlem üretmemeli; yeni model veya daha iyi broker verisi test edilmeli."
            )
        else:
            names = ", ".join(entry_model_comparison_df.loc[accepted, "Giriş Modeli"].astype(str))
            st.success(f"Testte en az sınırlı kanıt üreten model(ler): {names}. Otomatik seçim yapılmadı.")

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
            "PnL USD": None if manual_pips is None else round(float(manual_pips) * float(pip_value_per_lot) * float(journal_lot), 2),
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
