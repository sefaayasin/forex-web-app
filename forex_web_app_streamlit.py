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
from datetime import datetime
from html import escape
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

TR_TZ = pytz.timezone("Europe/Istanbul")

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
        .decision-buy { background:#d1e7dd; color:#0f5132 !important; }
        .decision-sell { background:#f8d7da; color:#842029 !important; }
        .decision-wait { background:#fff3cd; color:#664d03 !important; }
        .decision-pass { background:#e9ecef; color:#212529 !important; }
        .decision-shell, .decision-shell * { color: inherit !important; }
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
            min-height: 70px;
        }
        .check-item b { display:block; font-size:0.86rem; margin-bottom:4px; }
        .check-item span { display:block; font-size:0.9rem; color:#495057; }
        .check-ok { border-color:#badbcc; background:#f0f8f4; }
        .check-warn { border-color:#ffe69c; background:#fff9e6; }
        .check-bad { border-color:#f5c2c7; background:#fff1f2; }
        .action-row {
            display:flex;
            gap:10px;
            align-items:stretch;
            flex-wrap:wrap;
            margin: 4px 0 14px 0;
        }
        @media (max-width: 900px) {
            .simple-levels, .check-grid {
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }
        }
        @media (max-width: 520px) {
            .simple-levels, .check-grid {
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
    "Asya": (3, 12),
    "Londra": (10, 19),
    "New York": (16, 1),
    "Londra + New York Kesişimi": (16, 19),
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

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
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


def is_in_trading_session(ts, session_name: str) -> bool:
    """Backtest girişlerini seçilen işlem seansına göre filtreler."""
    hours = TRADING_SESSIONS.get(session_name)
    if hours is None:
        return True

    start_hour, end_hour = hours
    local_hour = _to_istanbul_timestamp(ts).hour

    if start_hour < end_hour:
        return start_hour <= local_hour < end_hour

    # Gece yarısını aşan seanslar: örn. New York 16:00-01:00
    return local_hour >= start_hour or local_hour < end_hour


def session_description(session_name: str) -> str:
    hours = TRADING_SESSIONS.get(session_name)
    if hours is None:
        return "Tüm gün aktif"
    start_hour, end_hour = hours
    return f"Europe/Istanbul saatine göre {start_hour:02d}:00–{end_hour:02d}:00"

# =============================================================================
# DATA
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
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

    # Bollinger/volatilite skoru: tek başına AL/SAT değil, pozisyon kalitesi filtresi.
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

    entry = float(row["Close"])
    atr = float(row["ATR14"])
    pip = get_pip_size(symbol)

    if not np.isfinite(atr) or atr <= 0:
        return None

    side = "LONG" if global_label in long_labels else "SHORT"
    stop_distance = atr * atr_mult
    stop_pips = stop_distance / pip
    target_distance = stop_distance * rr
    target_pips = target_distance / pip

    if side == "LONG":
        stop = entry - stop_distance
        target = entry + target_distance
        action = "Breakout / trend devamı bekle"
        activation_rule = f"{entry:.{price_decimals(symbol)}f} üzerinde {selected_tf} mum kapanışı gelirse plan aktif sayılır."
        confirmation_rule = "4H ve 1H alım yönünde kalmalı; 15M ters satışa dönerse bekle; 5M alım yönüne dönerse giriş kalitesi artar."
        invalidation_rule = f"Fiyat {stop:.{price_decimals(symbol)}f} altına iner veya 1H Satış/İşlem Yok'a dönerse plan iptal."
    else:
        stop = entry + stop_distance
        target = entry - target_distance
        action = "Breakdown / trend devamı bekle"
        activation_rule = f"{entry:.{price_decimals(symbol)}f} altında {selected_tf} mum kapanışı gelirse plan aktif sayılır."
        confirmation_rule = "4H ve 1H satış yönünde kalmalı; 15M ters alıma dönerse bekle; 5M satış yönüne dönerse giriş kalitesi artar."
        invalidation_rule = f"Fiyat {stop:.{price_decimals(symbol)}f} üstüne çıkar veya 1H Alım/İşlem Yok'a dönerse plan iptal."

    risk_amount = account_size * (risk_pct / 100)
    if stop_pips <= 0 or pip_value_per_lot <= 0:
        estimated_lot = 0.0
    else:
        estimated_lot = risk_amount / (stop_pips * pip_value_per_lot)

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
        note="Lot hesabı yaklaşık değerdir. Broker, hesap para birimi ve pariteye göre pip değeri değişebilir.",
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
    """Canlı sistem mantığını backtestte uygular: 4H + 1H ana yön, 15M/5M giriş teyidi."""
    if pd.isna(entry_score) or pd.isna(h4_score) or pd.isna(h1_score):
        return "NONE", "Ana zaman dilimi skorları yetersiz"

    htf_long = h4_score >= 25 and h1_score >= 25
    htf_short = h4_score <= -25 and h1_score <= -25

    if tf_name == "5 Dakika":
        # 5M ile giriş aranıyorsa 15M aynı yönü desteklemeli.
        m15_long_ok = not pd.isna(m15_score) and m15_score >= 25
        m15_short_ok = not pd.isna(m15_score) and m15_score <= -25
    else:
        m15_long_ok = True
        m15_short_ok = True

    if htf_long and m15_long_ok and entry_score >= threshold:
        return "LONG", "4H+1H long uyumlu; giriş skoru eşiği geçti"
    if htf_short and m15_short_ok and entry_score <= -threshold:
        return "SHORT", "4H+1H short uyumlu; giriş skoru eşiği geçti"

    return "NONE", "MTF filtre veya giriş skoru uygun değil"


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

        if side == "LONG":
            hit_stop = float(bar["Low"]) <= stop
            hit_target = float(bar["High"]) >= target
        else:
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
            if exit_reason is not None:
                close_trade(open_trade, float(exit_price), ts, exit_reason)
                last_exit_i = i
                open_trade = None
                closed_this_bar = True

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
                stop_distance = atr * atr_mult
                stop_pips = stop_distance / pip
                risk_amount = balance * (risk_pct / 100)
                lot = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 and pip_value_per_lot > 0 else 0.0

                if lot > 0 and np.isfinite(lot):
                    if sig == "LONG":
                        stop = entry - stop_distance
                        target = entry + stop_distance * rr
                    else:
                        stop = entry + stop_distance
                        target = entry - stop_distance * rr

                    candidate_trade = {
                        "EntryTime": ts,
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
        metrics = pd.DataFrame({"Metrik": ["İşlem Sayısı", "Not"], "Değer": [0, "MTF filtrelerle bu periyotta işlem oluşmadı"]})
        return BacktestResult(metrics, trades_df, equity_df)

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]
    total_pnl = trades_df["PnL"].sum()
    win_rate = 100 * len(wins) / len(trades_df)
    loss_sum = abs(losses["PnL"].sum()) if not losses.empty else 0.0
    profit_factor = wins["PnL"].sum() / loss_sum if loss_sum > 0 else np.nan

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
    ], columns=["Metrik", "Değer"])

    return BacktestResult(metrics, trades_df, equity_df)


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

    if pd.notna(pf) and pf >= 1.30 and total_pnl > 0 and avg_pips > 0 and dd_pct > -15:
        return "İyi", "ok-box", f"PF {pf:.2f}, ortalama {avg_pips:.2f} pip ve drawdown {dd_pct:.2f}%. Bu ayar izlemeye değer; yine de demo doğrulama gerekir."

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

        if row["Genel Bias"] == "İşlem Yok":
            decision = "PAS"
        elif row["Backtest Kalitesi"] in {"Zayıf", "Yetersiz", "Yetersiz Örnek"}:
            decision = "PAS"
        elif row["Backtest Kalitesi"] in {"İyi", "Orta"}:
            decision = "İZLE"
        else:
            decision = "ÖN İZLEME"
        row["Karar"] = decision

        rows.append(row)
        progress.progress(i / len(symbols), text=f"{sym} tarandı ({i}/{len(symbols)})")

    progress.empty()
    result = pd.DataFrame(rows)

    # En işe yarar sıralama: önce aksiyon alınabilecekler, sonra skor.
    decision_order = {"İZLE": 0, "ÖN İZLEME": 1, "PAS": 2}
    result["_order"] = result["Karar"].map(decision_order).fillna(9)
    result = result.sort_values(["_order", "Skor"], ascending=[True, False]).drop(columns=["_order"])
    return result


def init_trade_journal() -> None:
    if "trade_journal" not in st.session_state:
        st.session_state["trade_journal"] = []


def add_trade_journal_entry(entry: dict) -> None:
    init_trade_journal()
    st.session_state["trade_journal"].append(entry)


def journal_dataframe() -> pd.DataFrame:
    init_trade_journal()
    return pd.DataFrame(st.session_state["trade_journal"])


def calculate_manual_pips(symbol: str, side: str, entry: float, exit_price: float) -> Optional[float]:
    if entry is None or exit_price is None or entry <= 0 or exit_price <= 0:
        return None
    pip = get_pip_size(symbol)
    if side == "LONG":
        return (exit_price - entry) / pip
    if side == "SHORT":
        return (entry - exit_price) / pip
    return None


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
) -> dict:
    """Teknik ekranı acemi kullanıcı için AL/SAT/BEKLE/PAS GEÇ kararına indirger."""
    dec = price_decimals(symbol)
    base = {
        "action": "BEKLE",
        "class": "simple-wait",
        "subtitle": "Henüz net işlem yok.",
        "reason": filter_note,
        "steps": ["Yeni işlem açma.", "Pariteyi izlemeye devam et.", "Backtest ve ana yön uyumu oluşmadan işlem alma."],
        "levels": {},
    }

    if bt_tf != selected_tf:
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Backtest zaman dilimi ile giriş zaman dilimi eşleşmiyor.",
            "reason": "Basit karar için Backtest zaman dilimi, Grafik/Giriş zaman dilimi ile aynı olmalı.",
            "steps": ["Sidebar'dan backtest zaman dilimini giriş zaman dilimiyle aynı seç.", "Backtest Çalıştır / Planı Onayla butonuna bas.", "Sonra bu karttaki kararı takip et."],
        })
        return base

    if matched_quality is None:
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Önce strateji kontrolü gerekiyor.",
            "reason": "Bu sembol ve zaman dilimi için backtest onayı yok.",
            "steps": ["Sidebar'dan Backtest Çalıştır / Planı Onayla butonuna bas.", "Strateji Kalitesi Orta veya İyi değilse işlem açma.", "Sert Güvenli Mod açıksa sadece İyi kalite kabul edilir."],
        })
        return base

    if strict_safety_mode and matched_quality.get("label") != "İyi":
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Sert Güvenli Mod bu işlemi reddetti.",
            "reason": matched_quality.get("text", "Backtest kalitesi yeterli değil."),
            "steps": ["Bu paritede işlem açma.", "Başka parite tara.", "Sadece Strateji Kalitesi İyi olan fırsatları değerlendir."],
        })
        return base

    if (not strict_safety_mode) and matched_quality.get("label") not in {"İyi", "Orta"}:
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Backtest kalitesi işlem için yeterli değil.",
            "reason": matched_quality.get("text", "Strateji kalitesi zayıf/yetersiz."),
            "steps": ["Bu ayarla işlem açma.", "Başka parite veya daha yüksek zaman dilimi dene.", "Backtest kalitesi düzelmeden gerçek işlem alma."],
        })
        return base

    if strict_safety_mode and final_label not in {"Güçlü Alım Yönlü", "Güçlü Satış Yönlü"}:
        base.update({
            "action": "PAS GEÇ",
            "class": "simple-pass",
            "subtitle": "Ana sinyal yeterince güçlü değil.",
            "reason": f"Sert Güvenli Mod için 'Güçlü Alım' veya 'Güçlü Satış' gerekli. Mevcut: {final_label}.",
            "steps": ["Bu paritede şimdilik işlem açma.", "4H ve 1H güçlü aynı yöne dönene kadar bekle.", "Tarayıcıdan daha net fırsat ara."],
        })
        return base

    if final_label == "İşlem Yok" or setup is None:
        base.update({
            "action": "BEKLE",
            "class": "simple-wait",
            "subtitle": "Sistem işlem planı üretmiyor.",
            "reason": filter_note,
            "steps": ["Yeni işlem açma.", "Ana yön netleşene kadar bekle.", "Risk Planı oluşmadan emir girme."],
        })
        return base

    side = setup.side
    quality_text = matched_quality.get("label", "-")
    levels = {
        "Giriş": f"{setup.entry:.{dec}f}",
        "Stop": f"{setup.stop:.{dec}f}",
        "Kâr Al": f"{setup.target:.{dec}f}",
        "Lot": f"{setup.estimated_lot:.2f}",
    }

    if side == "LONG":
        if price is not None and price >= setup.entry:
            return {
                "action": "AL İÇİN KAPANIŞ BEKLE",
                "class": "simple-wait",
                "subtitle": f"{symbol} alım planı izleme bölgesinde.",
                "reason": f"Anlık fiyat giriş seviyesinde/üstünde; kapanış teyidi olmadan AL sinyali verilmez. Backtest onayı: {quality_text}.",
                "steps": [
                    f"{selected_tf} mum kapanışının {setup.entry:.{dec}f} üstünde kaldığını görmeden alış açma.",
                    f"Alış açarsan stopu {setup.stop:.{dec}f} seviyesine koy.",
                    f"Kâr al seviyesini {setup.target:.{dec}f} yap.",
                    "Stop seviyesine gelirse işlemden çık; stopu büyütme.",
                ],
                "levels": levels,
            }
        return {
            "action": "ALIM İÇİN BEKLE",
            "class": "simple-wait",
            "subtitle": f"{symbol} alım yönünde izlenebilir ama giriş henüz aktif değil.",
            "reason": f"Fiyat giriş seviyesinin altında. Giriş seviyesi: {setup.entry:.{dec}f}.",
            "steps": [
                f"Fiyat {setup.entry:.{dec}f} üstünde {selected_tf} mum kapanışı yaparsa AL düşün.",
                f"Alış açarsan stop {setup.stop:.{dec}f}, kâr al {setup.target:.{dec}f}.",
                "Fiyat girişe gelmeden acele etme.",
            ],
            "levels": levels,
        }

    if side == "SHORT":
        if price is not None and price <= setup.entry:
            return {
                "action": "SAT İÇİN KAPANIŞ BEKLE",
                "class": "simple-wait",
                "subtitle": f"{symbol} satış planı izleme bölgesinde.",
                "reason": f"Anlık fiyat giriş seviyesinde/altında; kapanış teyidi olmadan SAT sinyali verilmez. Backtest onayı: {quality_text}.",
                "steps": [
                    f"{selected_tf} mum kapanışının {setup.entry:.{dec}f} altında kaldığını görmeden satış açma.",
                    f"Satış açarsan stopu {setup.stop:.{dec}f} seviyesine koy.",
                    f"Kâr al seviyesini {setup.target:.{dec}f} yap.",
                    "Stop seviyesine gelirse işlemden çık; stopu büyütme.",
                ],
                "levels": levels,
            }
        return {
            "action": "SATIŞ İÇİN BEKLE",
            "class": "simple-wait",
            "subtitle": f"{symbol} satış yönünde izlenebilir ama giriş henüz aktif değil.",
            "reason": f"Fiyat giriş seviyesinin üstünde. Giriş seviyesi: {setup.entry:.{dec}f}.",
            "steps": [
                f"Fiyat {setup.entry:.{dec}f} altında {selected_tf} mum kapanışı yaparsa SAT düşün.",
                f"Satış açarsan stop {setup.stop:.{dec}f}, kâr al {setup.target:.{dec}f}.",
                "Fiyat girişe gelmeden acele etme.",
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
    elif matched_quality is None:
        bt_state = "warn"
        bt_text = "Plan kontrolü bekliyor"
    elif matched_quality.get("label") in allowed_quality_labels:
        bt_state = "ok"
        bt_text = f"Kalite: {matched_quality.get('label')}"
    else:
        bt_state = "bad"
        bt_text = f"Kalite: {matched_quality.get('label')}"

    risk_state = "ok" if setup is not None and bt_state == "ok" else ("bad" if bt_state == "bad" else "warn")
    risk_text = "Seviyeler hazır" if risk_state == "ok" else "Risk planı kilitli"

    return [
        {"label": "Veri", "state": "ok" if data_ok else "bad", "text": "Fiyat verisi alındı" if data_ok else "Veri bekleniyor"},
        {"label": "Ana Yön", "state": "ok" if htf_ok else "warn", "text": "4H + 1H uyumlu" if htf_ok else "4H + 1H net değil"},
        {"label": "15M Teyit", "state": m15_state, "text": m15_text},
        {"label": "Backtest", "state": bt_state, "text": bt_text},
        {"label": "Risk Planı", "state": risk_state, "text": risk_text},
    ]


def render_readiness_checklist(items: list[dict]) -> None:
    parts = []
    state_symbol = {"ok": "Hazır", "warn": "Bekle", "bad": "Kilitli"}
    for item in items:
        state = str(item.get("state", "warn"))
        parts.append(
            f"<div class='check-item check-{escape(state)}'>"
            f"<b>{escape(str(item.get('label', 'Kontrol')))}: {escape(state_symbol.get(state, 'Bekle'))}</b>"
            f"<span>{escape(str(item.get('text', '')))}</span>"
            f"</div>"
        )
    st.markdown("<div class='check-grid'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


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

    opposite = False
    neutral = final_label == "İşlem Yok"
    if side == "LONG" and "Satış" in final_label:
        opposite = True
    if side == "SHORT" and "Alım" in final_label:
        opposite = True

    action = "TUT"
    css = "simple-buy" if pips is not None and pips >= 0 else "simple-wait"
    reason = "Plan bozulmadı. Stop ve kâr al seviyelerini takip et."

    if side == "LONG":
        if stop > 0 and current_price <= stop:
            action, css, reason = "ÇIK", "simple-sell", f"Fiyat stop seviyesine geldi/altına indi: {stop:.{dec}f}."
        elif target > 0 and current_price >= target:
            action, css, reason = "KÂR AL", "simple-buy", f"Fiyat hedef seviyeye geldi/üstüne çıktı: {target:.{dec}f}."
        elif opposite:
            action, css, reason = "ÇIKMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."
    else:
        if stop > 0 and current_price >= stop:
            action, css, reason = "ÇIK", "simple-sell", f"Fiyat stop seviyesine geldi/üstüne çıktı: {stop:.{dec}f}."
        elif target > 0 and current_price <= target:
            action, css, reason = "KÂR AL", "simple-buy", f"Fiyat hedef seviyeye geldi/altına indi: {target:.{dec}f}."
        elif opposite:
            action, css, reason = "ÇIKMAYI DÜŞÜN", "simple-sell", "Ana yön senin pozisyonunun tersine döndü."
        elif neutral and pips is not None and pips < 0:
            action, css, reason = "DİKKAT", "simple-wait", "Ana yön kararsız ve pozisyon zararda. Stopa sadık kal."

    return {"action": action, "class": css, "text": reason, "pips": pips, "pnl": pnl}


def render_position_tracker_result(result: dict, current_price: Optional[float], symbol: str) -> None:
    dec = price_decimals(symbol)
    price_txt = "-" if current_price is None else f"{current_price:.{dec}f}"
    pips_txt = "-" if result.get("pips") is None else f"{result['pips']:+.1f} pip"
    pnl_txt = "-" if result.get("pnl") is None else f"{result['pnl']:+.2f}"

    card_class = escape(str(result.get("class", "simple-wait")))
    action = escape(str(result.get("action", "TUT")))
    reason = escape(str(result.get("text", "")))
    html = (
        f"<div class='simple-card {card_class}'>"
        f"<div class='simple-action'>{action}</div>"
        f"<div class='simple-subtitle'>Pozisyon takip sonucu</div>"
        f"<div><b>Sebep:</b> {reason}</div>"
        f"<div class='simple-levels'>"
        f"<div class='simple-level'><b>Güncel Fiyat</b><span>{escape(price_txt)}</span></div>"
        f"<div class='simple-level'><b>Pip</b><span>{escape(pips_txt)}</span></div>"
        f"<div class='simple-level'><b>Tahmini PnL</b><span>{escape(pnl_txt)}</span></div>"
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
# UI
# =============================================================================

with st.sidebar:
    st.header("Kontrol Paneli")

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

    selected_tf = st.radio("Giriş zamanı", tf_options, index=1)

    st.divider()
    st.subheader("Temel Risk")
    account_size = st.number_input("Hesap büyüklüğü", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("İşlem başına risk %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    with st.expander("Gelişmiş risk", expanded=False):
        rr = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        atr_mult = st.number_input("ATR Stop Çarpanı", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
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

    with st.expander("Ekran ve güvenlik", expanded=False):
        enable_simple_mode = st.checkbox("Basit İşlem Modu", value=True)
        strict_safety_mode = st.checkbox("Sert Güvenli Mod", value=False)
        show_position_tracker = st.checkbox("Pozisyon Takip Modu", value=True)
        change_window_label = st.selectbox("Yüzde değişim periyodu", list(PRICE_CHANGE_WINDOWS.keys()), index=1)
        change_window_minutes = PRICE_CHANGE_WINDOWS[change_window_label]

    with st.expander("Backtest ayarları", expanded=False):
        if enable_simple_mode:
            bt_tf = selected_tf
            st.caption(f"Backtest zamanı giriş zamanı ile aynı: {bt_tf}")
        else:
            bt_tf = st.selectbox("Backtest zaman dilimi", tf_options, index=tf_options.index(selected_tf))
        default_period = BACKTEST_PERIODS.get(bt_tf, "30d")
        bt_period = st.text_input("Backtest period", value=default_period, key=f"bt_period_{bt_tf}", help="Örn: 5d, 30d, 90d, 120d")
        signal_threshold = st.slider("Sinyal eşiği", min_value=25, max_value=85, value=60, step=5)
        spread_pips = st.number_input("Spread / maliyet (pip)", min_value=0.0, value=1.5, step=0.1)
        session_filter = st.selectbox("İşlem seansı", list(TRADING_SESSIONS.keys()), index=0)
        cooldown_bars = st.number_input("Cooldown (mum)", min_value=0, max_value=200, value=5, step=1)
        max_same_direction_trades = st.number_input("Aynı yönde maksimum tekrar", min_value=1, max_value=10, value=2, step=1)
        min_trades_required = st.number_input("Minimum backtest işlem sayısı", min_value=10, max_value=100, value=20, step=5)

    run_bt_requested = st.button("Planı Kontrol Et", type="primary", use_container_width=True)

    with st.expander("Parite tarayıcı", expanded=False):
        scanner_tf = st.selectbox("Tarayıcı backtest zamanı", tf_options, index=tf_options.index(selected_tf))
        scanner_period = st.text_input("Tarayıcı period", value=BACKTEST_PERIODS.get(scanner_tf, "30d"), key=f"scanner_period_{scanner_tf}")
        scanner_include_backtest = st.checkbox("Backtest kalitesi hesapla", value=False)
        scanner_limit = st.number_input("Maksimum parite", min_value=1, max_value=len(SYMBOL_LIST), value=min(12, len(SYMBOL_LIST)), step=1)
        run_scanner_requested = st.button("Pariteleri Tara", use_container_width=True)

    if st.button("Veriyi Yenile", use_container_width=True):
        fetch_ohlc.clear()
        fetch_last_price.clear()
        fetch_price_change.clear()
        st.rerun()

st.title("Forex Analyzer Pro")
st.caption("Eğitim ve karar destek amaçlıdır; yatırım tavsiyesi değildir. Gerçek işlem öncesi demo test ve broker verisiyle doğrulama yapın.")
if strict_safety_mode:
    st.info("Sert Güvenli Mod aktif: yalnızca güçlü yön + İyi backtest kalitesi olan işlemler için AL/SAT kartı gösterilir.")
else:
    st.info("Pratik Mod aktif: İyi veya Orta backtest kalitesi izlenebilir; yine de gerçek işlemden önce demo/onay önerilir.")

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
        )
        q_label, q_css, q_text = assess_backtest_quality(bt_result, min_trades_required=int(min_trades_required))
        st.session_state["last_bt_key"] = current_bt_key
        st.session_state["last_bt_result"] = bt_result
        st.session_state["last_bt_quality"] = {
            "label": q_label,
            "css": q_css,
            "text": q_text,
        }


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
)
matched_quality = get_matching_backtest_quality(plan_bt_key)
allowed_quality_labels = {"İyi"} if strict_safety_mode else {"İyi", "Orta"}
preview_setup = build_trade_setup(symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult, pip_value_per_lot)

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
)

st.header("Karar Özeti")
render_top_decision_panel(simple_decision)
render_readiness_checklist(
    build_readiness_items(
        summary=summary_df,
        selected_tf=selected_tf,
        bt_tf=bt_tf,
        matched_quality=matched_quality,
        allowed_quality_labels=allowed_quality_labels,
        setup=preview_setup,
    )
)

action_col, quality_col, risk_col = st.columns([1.2, 1.0, 1.0])
with action_col:
    main_run_bt_requested = st.button(
        "Planı Kontrol Et",
        type="primary",
        use_container_width=True,
        disabled=bt_tf != selected_tf,
        key="main_run_bt",
    )
with quality_col:
    st.metric("Strateji Kalitesi", matched_quality.get("label", "Bekliyor") if matched_quality else "Bekliyor")
with risk_col:
    st.metric("İşlem Riski", f"{account_size * (risk_pct / 100):.2f}")

if main_run_bt_requested:
    run_and_store_backtest()
    st.rerun()

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
    fig, chart_df = plot_main_figure(symbol, selected_tf)
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Genel Bias")
    st.plotly_chart(gauge_figure(final_label, final_score), use_container_width=True)

    if final_label == "İşlem Yok":
        st.markdown(f"<div class='warn-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)
    elif "Alım" in final_label:
        st.markdown(f"<div class='ok-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bad-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)

    st.subheader("Risk Planı")

    # Risk Planı, yukarıda hesaplanan aynı sembol + aynı giriş zaman dilimi backtest kalitesine bağlıdır.

    if bt_tf != selected_tf:
        st.markdown(
            "<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
            "Risk planını onaylamak için Backtest zaman dilimi ile Grafik/Giriş zaman dilimi aynı olmalı.</div>",
            unsafe_allow_html=True,
        )
    elif matched_quality is None:
        st.markdown(
            "<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
            "Bu sembol ve giriş zaman dilimi için önce sidebar üzerinden 'Backtest Çalıştır / Planı Onayla' butonuna bas.</div>",
            unsafe_allow_html=True,
        )
    elif matched_quality["label"] not in allowed_quality_labels:
        strict_note = "Sert Güvenli Mod açık olduğu için yalnızca 'İyi' kalite kabul edilir." if strict_safety_mode else "İşlem için en az Orta kalite gerekir."
        st.markdown(
            f"<div class='{matched_quality['css']}'><b>PAS GEÇ — Strateji Kalitesi: {matched_quality['label']}</b><br>"
            f"{matched_quality['text']}<br>{strict_note}</div>",
            unsafe_allow_html=True,
        )
    elif strict_safety_mode and final_label not in {"Güçlü Alım Yönlü", "Güçlü Satış Yönlü"}:
        st.markdown(
            f"<div class='warn-box'><b>Risk Planı Kilitli</b><br>"
            f"Sert Güvenli Mod için yönün Güçlü Alım veya Güçlü Satış olması gerekir. Mevcut yön: {final_label}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='{matched_quality['css']}'><b>Backtest Onayı: {matched_quality['label']}</b><br>"
            f"{matched_quality['text']}</div>",
            unsafe_allow_html=True,
        )
        setup = build_trade_setup(symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult, pip_value_per_lot)
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
    st.plotly_chart(plot_live_trigger(symbol, selected_tf, final_label), use_container_width=True)

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
    st.info("Backtest sonuçlarını görmek ve Risk Planı'nı kalite kontrolüne bağlamak için sidebar'daki 'Backtest Çalıştır / Planı Onayla' butonuna bas.")

st.divider()
st.header("İşlem Günlüğü")
st.caption("Bu günlük Streamlit oturumu içinde tutulur. Kalıcı saklamak için CSV indirip ayrıca kaydetmelisin.")

init_trade_journal()
journal_setup = build_trade_setup(symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult, pip_value_per_lot)
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
            "Pips": None if manual_pips is None else round(float(manual_pips), 2),
            "Not": journal_notes,
        })
        st.success("İşlem günlüğe eklendi.")

journal_df = journal_dataframe()
if journal_df.empty:
    st.info("Henüz işlem günlüğü kaydı yok.")
else:
    st.dataframe(journal_df.tail(100), use_container_width=True, height=300)
    st.download_button(
        "İşlem Günlüğünü CSV İndir",
        data=journal_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="forex_trade_journal.csv",
        mime="text/csv",
    )
    if st.button("İşlem Günlüğünü Temizle"):
        st.session_state["trade_journal"] = []
        st.rerun()

st.divider()
st.markdown(
    """
    **Kullanım Notu:** Bu sistem emir vermek için değil, karar disiplinini korumak için tasarlanmıştır. 
    4H ve 1H yönü çelişiyorsa işlem filtresi devreye girer. Risk Planı, aynı sembol ve giriş zaman dilimi için çalıştırılmış MTF backtest kalitesi uygun değilse kilitli kalır. 
    Sert Güvenli Mod açıksa yalnızca güçlü yön + İyi backtest kalitesi kabul edilir; Pratik Modda Orta kalite de izlenebilir. 15M/5M yalnızca giriş zamanlaması için kullanılmalıdır.
    """
)
