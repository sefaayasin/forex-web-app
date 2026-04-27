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
# - Streamlit Cloud uyumlu: pandas yeni sürümlerde "4h" kullanılır, st.rerun() kullanılır

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


def global_bias(summary: pd.DataFrame) -> tuple[str, float, str]:
    if summary.empty:
        return "İşlem Yok", 0.0, "Veri yok"

    total_weight = 0
    weighted_score = 0.0
    labels_by_tf = {}
    for _, row in summary.iterrows():
        tf = row["Zaman Dilimi"]
        label = row["Bias"]
        score = float(row["Skor"]) if pd.notna(row["Skor"]) else 0.0
        w = TIMEFRAMES.get(tf, {}).get("weight", 1)
        weighted_score += score * w
        total_weight += w
        labels_by_tf[tf] = label

    final_score = weighted_score / total_weight if total_weight else 0.0
    final_label = label_from_score(final_score)

    # Ana yön filtresi: 4H ve 1H ters ise işlem yok.
    h4 = labels_by_tf.get("4 Saat", "İşlem Yok")
    h1 = labels_by_tf.get("1 Saat", "İşlem Yok")
    long_set = {"Alım Yönlü", "Güçlü Alım Yönlü"}
    short_set = {"Satış Yönlü", "Güçlü Satış Yönlü"}

    if (h4 in long_set and h1 in short_set) or (h4 in short_set and h1 in long_set):
        return "İşlem Yok", final_score, "4H ve 1H yönleri çelişiyor. İşlem filtresi devrede."

    if h4 == "İşlem Yok" and h1 == "İşlem Yok":
        return "İşlem Yok", final_score, "Ana zaman dilimleri yön vermiyor."

    return final_label, final_score, "Ana yön filtresi uygun."

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

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        ts = df.index[i]
        prev_ts = df.index[i - 1]

        if open_trade is not None:
            side = open_trade["Side"]
            entry = open_trade["Entry"]
            stop = open_trade["Stop"]
            target = open_trade["Target"]
            lot = open_trade["Lot"]
            risk_amount = open_trade["RiskAmount"]

            if side == "LONG":
                hit_stop = float(current["Low"]) <= stop
                hit_target = float(current["High"]) >= target
            else:
                hit_stop = float(current["High"]) >= stop
                hit_target = float(current["Low"]) <= target

            exit_reason = None
            exit_price = None

            if hit_stop and hit_target:
                exit_reason = "SL"  # muhafazakâr kabul
                exit_price = stop
            elif hit_stop:
                exit_reason = "SL"
                exit_price = stop
            elif hit_target:
                exit_reason = "TP"
                exit_price = target

            if exit_reason is not None:
                if side == "LONG":
                    pnl_pips = (exit_price - entry) / pip
                else:
                    pnl_pips = (entry - exit_price) / pip

                pnl_pips -= spread_pips
                pnl = pnl_pips * pip_value_per_lot * lot
                balance += pnl

                trades.append({
                    "Entry Time": open_trade["EntryTime"],
                    "Exit Time": ts,
                    "Side": side,
                    "Entry": entry,
                    "Exit": exit_price,
                    "SL": stop,
                    "TP": target,
                    "Pips": pnl_pips,
                    "PnL": pnl,
                    "Balance": balance,
                    "Result": exit_reason,
                    "Lot": lot,
                    "Risk Amount": risk_amount,
                    "Entry Score": open_trade["EntryScore"],
                    "4H Score": open_trade["H4Score"],
                    "1H Score": open_trade["H1Score"],
                    "15M Score": open_trade["M15Score"],
                    "MTF Reason": open_trade["Reason"],
                })
                open_trade = None

        if open_trade is None:
            entry_score = float(previous["Score"])

            h4_score = entry_score if tf_name == "4 Saat" else _aligned_value(aligned_scores.get("4 Saat"), prev_ts)
            h1_score = entry_score if tf_name == "1 Saat" else _aligned_value(aligned_scores.get("1 Saat"), prev_ts)
            m15_score = entry_score if tf_name == "15 Dakika" else _aligned_value(aligned_scores.get("15 Dakika"), prev_ts)

            sig, reason = mtf_signal_decision(entry_score, h4_score, h1_score, m15_score, tf_name, signal_threshold)

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

                    open_trade = {
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

        equity_rows.append({"Time": ts, "Balance": balance})

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


def assess_backtest_quality(bt: BacktestResult) -> tuple[str, str, str]:
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

    if trade_count < 30:
        return "Yetersiz Örnek", "warn-box", f"Sadece {trade_count} işlem var. Profit Factor yanıltıcı olabilir; daha uzun periyot veya farklı parite test edilmeli."

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
    )


def get_matching_backtest_quality(current_key: tuple) -> Optional[dict]:
    saved_key = st.session_state.get("last_bt_key")
    saved_quality = st.session_state.get("last_bt_quality")
    if saved_key == current_key and saved_quality:
        return saved_quality
    return None

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
    st.header("Ayarlar")

    default_symbol = st.session_state.get("symbol", "EURUSD=X")
    selected_symbol = st.selectbox(
        "Favori Sembol",
        options=SYMBOL_LIST,
        index=SYMBOL_LIST.index(default_symbol) if default_symbol in SYMBOL_LIST else 0,
    )
    manual_symbol = st.text_input("Elle gir (ör: EURUSD=X veya EURUSD)", value="")
    symbol = normalize_symbol(manual_symbol) if manual_symbol.strip() else selected_symbol
    st.session_state["symbol"] = symbol

    selected_tf = st.radio("Grafik / Giriş Zaman Dilimi", list(TIMEFRAMES.keys()), index=1)

    st.divider()
    st.subheader("Fiyat Değişim Filtresi")
    change_window_label = st.selectbox("Yüzde değişim periyodu", list(PRICE_CHANGE_WINDOWS.keys()), index=1)
    change_window_minutes = PRICE_CHANGE_WINDOWS[change_window_label]

    st.divider()
    st.subheader("Risk Ayarları")
    account_size = st.number_input("Hesap büyüklüğü", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("İşlem başına risk %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    rr = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    atr_mult = st.number_input("ATR Stop Çarpanı", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    pip_value_per_lot = st.number_input("1 lot için yaklaşık pip değeri", min_value=0.1, value=10.0, step=0.5)

    st.divider()
    st.subheader("Backtest Ayarları")
    tf_options = list(TIMEFRAMES.keys())
    bt_tf = st.selectbox("Backtest zaman dilimi", tf_options, index=tf_options.index(selected_tf))
    default_period = BACKTEST_PERIODS.get(bt_tf, "30d")
    bt_period = st.text_input("Backtest period", value=default_period, help="Örn: 5d, 30d, 90d, 120d. Yahoo Finance limitlerine bağlıdır.")
    signal_threshold = st.slider("Sinyal eşiği", min_value=25, max_value=85, value=60, step=5)
    spread_pips = st.number_input("Spread / maliyet (pip)", min_value=0.0, value=1.5, step=0.1)

    run_bt_requested = st.button("Backtest Çalıştır / Planı Onayla")

    if st.button("Veriyi Yenile"):
        fetch_ohlc.clear()
        fetch_last_price.clear()
        fetch_price_change.clear()
        st.rerun()

st.title("Forex Analyzer Pro")
st.caption("Eğitim ve karar destek amaçlıdır; yatırım tavsiyesi değildir. Gerçek işlem öncesi demo test ve broker verisiyle doğrulama yapın.")

current_bt_key = make_backtest_key(
    symbol=symbol,
    tf_name=bt_tf,
    period=bt_period,
    risk_pct=risk_pct,
    rr=rr,
    atr_mult=atr_mult,
    signal_threshold=float(signal_threshold),
    spread_pips=spread_pips,
)

if run_bt_requested:
    with st.spinner("Backtest çalışıyor ve risk planı kalite kontrolüne bağlanıyor..."):
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
        )
        q_label, q_css, q_text = assess_backtest_quality(bt_result)
        st.session_state["last_bt_key"] = current_bt_key
        st.session_state["last_bt_result"] = bt_result
        st.session_state["last_bt_quality"] = {
            "label": q_label,
            "css": q_css,
            "text": q_text,
        }

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
final_label, final_score, filter_note = global_bias(summary_df)

left_col, right_col = st.columns([2.2, 1.0])

with left_col:
    fig, chart_df = plot_main_figure(symbol, selected_tf)
    st.plotly_chart(fig, width="stretch")

with right_col:
    st.subheader("Genel Bias")
    st.plotly_chart(gauge_figure(final_label, final_score), width="stretch")

    if final_label == "İşlem Yok":
        st.markdown(f"<div class='warn-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)
    elif "Alım" in final_label:
        st.markdown(f"<div class='ok-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bad-box'><b>{final_label}</b><br>{filter_note}</div>", unsafe_allow_html=True)

    st.subheader("Risk Planı")

    plan_bt_key = make_backtest_key(
        symbol=symbol,
        tf_name=selected_tf,
        period=bt_period,
        risk_pct=risk_pct,
        rr=rr,
        atr_mult=atr_mult,
        signal_threshold=float(signal_threshold),
        spread_pips=spread_pips,
    )
    matched_quality = get_matching_backtest_quality(plan_bt_key)

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
    elif matched_quality["label"] not in {"İyi", "Orta"}:
        st.markdown(
            f"<div class='{matched_quality['css']}'><b>PAS GEÇ — Strateji Kalitesi: {matched_quality['label']}</b><br>"
            f"{matched_quality['text']}</div>",
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

st.subheader("Çoklu Zaman Dilimi Karar Tablosu")
st.dataframe(summary_df, width="stretch", height=190)

with st.expander("Skor Detayı"):
    st.dataframe(detail_df, width="stretch")

st.subheader("Giriş Tetikleyici Paneli")
st.plotly_chart(plot_live_trigger(symbol, selected_tf, final_label), width="stretch")

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
        st.dataframe(bt.metrics, width="stretch", hide_index=True)
        if saved_quality:
            st.markdown(
                f"<div class='{saved_quality['css']}'><b>Strateji Kalitesi: {saved_quality['label']}</b><br>{saved_quality['text']}</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.plotly_chart(plot_equity_curve(bt.equity), width="stretch")

    st.subheader("İşlem Listesi")
    if bt.trades.empty:
        st.info("Bu ayarlarla işlem oluşmadı veya yeterli veri yok.")
    else:
        view = bt.trades.copy()
        for col in ["Entry", "Exit", "SL", "TP"]:
            view[col] = view[col].astype(float).round(price_decimals(symbol))
        for col in ["Pips", "PnL", "Balance", "Lot", "Risk Amount", "Entry Score", "4H Score", "1H Score", "15M Score"]:
            view[col] = view[col].astype(float).round(2)
        st.dataframe(view.tail(100), width="stretch", height=360)
else:
    st.info("Backtest sonuçlarını görmek ve Risk Planı'nı kalite kontrolüne bağlamak için sidebar'daki 'Backtest Çalıştır / Planı Onayla' butonuna bas.")

st.divider()
st.markdown(
    """
    **Kullanım Notu:** Bu sistem emir vermek için değil, karar disiplinini korumak için tasarlanmıştır. 
    4H ve 1H yönü çelişiyorsa işlem filtresi devreye girer. Risk Planı, aynı sembol ve giriş zaman dilimi için çalıştırılmış MTF backtest kalitesi İyi/Orta değilse kilitli kalır. 
    15M/5M yalnızca giriş zamanlaması için kullanılmalıdır.
    """
)
