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
# - Basit ama muhafazakâr backtest: TP/SL, spread ve işlem maliyeti dikkate alınır
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
        .risk-box { padding: 14px; border-radius: 12px; background: #f8f9fa; border: 1px solid #e9ecef; }
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

.warn-box b, .ok-box b, .bad-box b {
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
            base.resample("4h")
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
    else:
        stop = entry + stop_distance
        target = entry - target_distance

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


def signal_from_score(score: float, threshold: float) -> str:
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
    return score.dropna()


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
    prm = TIMEFRAMES[tf_name]
    df = fetch_ohlc(symbol, prm["interval"], period)
    if df.empty or len(df) < 220:
        empty_metrics = pd.DataFrame({"Metrik": ["Durum"], "Değer": ["Yeterli veri yok"]})
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame())

    df = add_indicators(df)
    score = score_series_for_backtest(df)
    df = df.join(score.rename("Score"), how="left")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "ATR14", "Score"])

    pip = get_pip_size(symbol)
    balance = initial_balance
    equity_rows = []
    trades = []
    open_trade = None

    # İşleme giriş: sinyal barı kapandıktan sonra sonraki barın open fiyatı.
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        ts = df.index[i]

        if open_trade is not None:
            side = open_trade["Side"]
            entry = open_trade["Entry"]
            stop = open_trade["Stop"]
            target = open_trade["Target"]
            lot = open_trade["Lot"]
            risk_amount = open_trade["RiskAmount"]

            hit_stop = False
            hit_target = False

            if side == "LONG":
                hit_stop = float(current["Low"]) <= stop
                hit_target = float(current["High"]) >= target
            else:
                hit_stop = float(current["High"]) >= stop
                hit_target = float(current["Low"]) <= target

            exit_reason = None
            exit_price = None
            pnl = 0.0
            pnl_pips = 0.0

            if hit_stop and hit_target:
                # Muhafazakâr varsayım: aynı mumda TP ve SL görüldüyse önce SL çalıştı kabul edilir.
                exit_reason = "SL"
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
                    "Score": open_trade["Score"],
                })
                open_trade = None

        if open_trade is None:
            sig = signal_from_score(float(previous["Score"]), signal_threshold)
            if sig != "NONE":
                atr = float(previous["ATR14"])
                entry = float(current["Open"])
                stop_distance = atr * atr_mult
                stop_pips = stop_distance / pip
                risk_amount = balance * (risk_pct / 100)
                lot = risk_amount / (stop_pips * pip_value_per_lot) if stop_pips > 0 and pip_value_per_lot > 0 else 0.0

                if lot > 0:
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
                        "Score": float(previous["Score"]),
                    }

        equity_rows.append({"Time": ts, "Balance": balance})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if trades_df.empty:
        metrics = pd.DataFrame({"Metrik": ["İşlem Sayısı"], "Değer": [0]})
        return BacktestResult(metrics, trades_df, equity_df)

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]
    total_pnl = trades_df["PnL"].sum()
    win_rate = 100 * len(wins) / len(trades_df)
    profit_factor = wins["PnL"].sum() / abs(losses["PnL"].sum()) if not losses.empty and losses["PnL"].sum() != 0 else np.nan

    if not equity_df.empty:
        eq = equity_df["Balance"]
        peak = eq.cummax()
        dd = eq - peak
        max_dd = dd.min()
        max_dd_pct = 100 * max_dd / peak.loc[dd.idxmin()] if len(dd) > 0 and peak.loc[dd.idxmin()] else 0
    else:
        max_dd = 0
        max_dd_pct = 0

    metrics = pd.DataFrame([
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
    st.subheader("Risk Ayarları")
    account_size = st.number_input("Hesap büyüklüğü", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.number_input("İşlem başına risk %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    rr = st.number_input("Risk/Reward", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    atr_mult = st.number_input("ATR Stop Çarpanı", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    pip_value_per_lot = st.number_input("1 lot için yaklaşık pip değeri", min_value=0.1, value=10.0, step=0.5)

    st.divider()
    st.subheader("Backtest Ayarları")
    bt_tf = st.selectbox("Backtest zaman dilimi", list(TIMEFRAMES.keys()), index=1)
    default_period = BACKTEST_PERIODS.get(bt_tf, "30d")
    bt_period = st.text_input("Backtest period", value=default_period, help="Örn: 5d, 30d, 90d, 120d. Yahoo Finance limitlerine bağlıdır.")
    signal_threshold = st.slider("Sinyal eşiği", min_value=25, max_value=85, value=60, step=5)
    spread_pips = st.number_input("Spread / maliyet (pip)", min_value=0.0, value=1.5, step=0.1)

    if st.button("Veriyi Yenile"):
        fetch_ohlc.clear()
        fetch_last_price.clear()
        st.rerun()

st.title("Forex Analyzer Pro")
st.caption("Eğitim ve karar destek amaçlıdır; yatırım tavsiyesi değildir. Gerçek işlem öncesi demo test ve broker verisiyle doğrulama yapın.")

# Top metrics
price = fetch_last_price(symbol)
base_key = f"base_price_{symbol}"
if base_key not in st.session_state and price is not None:
    st.session_state[base_key] = price
base_price = st.session_state.get(base_key)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Sembol", symbol)
with m2:
    dec = price_decimals(symbol)
    st.metric("Güncel Fiyat", f"{price:.{dec}f}" if price is not None else "-")
with m3:
    if price is not None and base_price:
        change_pct = 100 * (price - base_price) / base_price
        st.metric("Açılıştan beri", f"{change_pct:+.2f}%")
    else:
        st.metric("Açılıştan beri", "-")
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

    setup = build_trade_setup(symbol, selected_tf, final_label, account_size, risk_pct, rr, atr_mult, pip_value_per_lot)
    st.subheader("Risk Planı")
    if setup is None:
        st.info("Şu anda işlem planı üretmiyorum. Ana yön veya veri koşulları yeterli değil.")
    else:
        dec = price_decimals(symbol)
        st.markdown(
            f"""
            <div class='risk-box'>
            <b>Yön:</b> {setup.side}<br>
            <b>Entry:</b> {setup.entry:.{dec}f}<br>
            <b>SL:</b> {setup.stop:.{dec}f} ({setup.stop_pips:.1f} pip)<br>
            <b>TP:</b> {setup.target:.{dec}f} ({setup.target_pips:.1f} pip)<br>
            <b>RR:</b> {setup.rr:.2f}<br>
            <b>Risk:</b> {setup.risk_amount:.2f}<br>
            <b>Yaklaşık Lot:</b> {setup.estimated_lot:.2f}
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
st.caption("Bu basit backtest, sinyal barı kapandıktan sonra sonraki barın açılışından işleme girer. Aynı mumda hem TP hem SL görülürse muhafazakâr olarak SL kabul edilir.")

run_bt = st.button("Backtest Çalıştır")
if run_bt:
    with st.spinner("Backtest çalışıyor..."):
        bt = run_backtest(
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

    c1, c2 = st.columns([1.0, 2.0])
    with c1:
        st.subheader("Performans")
        st.dataframe(bt.metrics, width="stretch", hide_index=True)
    with c2:
        st.plotly_chart(plot_equity_curve(bt.equity), width="stretch")

    st.subheader("İşlem Listesi")
    if bt.trades.empty:
        st.info("Bu ayarlarla işlem oluşmadı veya yeterli veri yok.")
    else:
        view = bt.trades.copy()
        for col in ["Entry", "Exit", "SL", "TP"]:
            view[col] = view[col].astype(float).round(price_decimals(symbol))
        for col in ["Pips", "PnL", "Balance", "Lot", "Risk Amount", "Score"]:
            view[col] = view[col].astype(float).round(2)
        st.dataframe(view.tail(100), width="stretch", height=360)
else:
    st.info("Backtest sonuçlarını görmek için 'Backtest Çalıştır' butonuna bas.")

st.divider()
st.markdown(
    """
    **Kullanım Notu:** Bu sistem emir vermek için değil, karar disiplinini korumak için tasarlanmıştır. 
    4H ve 1H yönü çelişiyorsa işlem filtresi devreye girer. 15M/5M ise yalnızca giriş zamanlaması için kullanılmalıdır.
    """
)
