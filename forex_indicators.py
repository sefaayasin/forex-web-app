"""Pure technical-indicator calculations for Forex Analyzer Pro.

This module has no Streamlit or market-data dependency.  It can therefore be
tested independently and reused by live analysis and backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_average(values: pd.Series, period: int = 14) -> pd.Series:
    """Seed with the first full SMA, then use Wilder's 1/period recurrence."""
    if not isinstance(period, int) or period < 1:
        raise ValueError("Period must be a positive integer")
    values = values.astype(float)
    seed = values.rolling(period, min_periods=period).mean()
    positions = np.flatnonzero(seed.notna().to_numpy())
    if not len(positions):
        return pd.Series(np.nan, index=values.index)
    first = int(positions[0])
    initialized = values.copy()
    initialized.iloc[:first] = np.nan
    initialized.iloc[first] = seed.iloc[first]
    return initialized.ewm(alpha=1 / period, adjust=False).mean().mask(values.isna())


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI based on exponentially smoothed gains and losses."""
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_average(gain, period)
    avg_loss = wilder_average(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # A run without losses is overbought, not missing data. A flat window
    # carries no directional momentum; retain NaN during the warmup.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def compute_bbands(
    close: pd.Series,
    period: int = 20,
    mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid - mult * std, mid, mid + mult * std


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return wilder_average(true_range, period)


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
    return compute_ichimoku(out)
