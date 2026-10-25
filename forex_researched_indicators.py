"""Causal experimental indicators from author papers/platform formula docs.

Sources and declared trading adaptations: research/new_indicators/sources.json.
Indicators are transforms of past prices, never trading permission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi

from forex_indicators import compute_atr, compute_rsi


def _prices(values):
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Indicators require finite observations")
    return arr


def super_smoother(values, period=10):
    if period < 3:
        raise ValueError("Smoother period must be at least 3")
    x = _prices(values)
    if not len(x):
        return values.copy()
    angle = np.sqrt(2) * np.pi / period
    decay = np.exp(-angle)
    feedback = 2 * decay * np.cos(angle)
    gain = 1 - feedback + decay**2
    numerator, denominator = [gain / 2, gain / 2], [1, -feedback, decay**2]
    filtered, _ = lfilter(numerator, denominator, x, zi=lfilter_zi(numerator, denominator) * x[0])
    return pd.Series(filtered, index=values.index)


def roofing_filter(values, highpass=48, lowpass=10):
    if highpass <= lowpass or lowpass < 3:
        raise ValueError("Roof periods require highpass > lowpass >= 3")
    x = _prices(values)
    if not len(x):
        return values.copy()
    angle = np.sqrt(.5) * 2 * np.pi / highpass
    alpha = (np.cos(angle) + np.sin(angle) - 1) / np.cos(angle)
    gain, decay = (1 - alpha / 2)**2, 1 - alpha
    hp = lfilter([gain, -2 * gain, gain], [1, -2 * decay, decay**2], x - x[0])
    return super_smoother(pd.Series(hp, index=values.index), lowpass)


def fisher_transform(mid, period=10):
    x = _prices(mid)
    low, high = mid.rolling(period).min(), mid.rolling(period).max()
    normalized = (2 * (mid - low) / (high - low).replace(0, np.nan) - 1).fillna(0).to_numpy()
    out = np.full(len(mid), np.nan)
    value, fish = 0., 0.
    for i in range(period - 1, len(x)):
        value = .33 * normalized[i] + .67 * value
        if abs(value) > .99:
            value = np.sign(value) * .999
        fish = np.arctanh(value) + .5 * fish
        out[i] = fish
    return pd.Series(out, index=mid.index)


def kama(close, period=30, fast=2, slow=30):
    if period < 2 or not 1 <= fast < slow:
        raise ValueError("Invalid KAMA periods")
    x = _prices(close)
    er = close.diff(period).abs() / close.diff().abs().rolling(period).sum().replace(0, np.nan)
    alpha = (er.fillna(0) * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1))**2
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        current = x[period-1]
        out[period-1] = current
        a = alpha.to_numpy()
        for i in range(period, len(x)):
            current += a[i] * (x[i] - current)
            out[i] = current
    return pd.Series(out, index=close.index)


def frama(bars, period=16):
    if period < 4 or period % 2:
        raise ValueError("FRAMA needs an even period >= 4")
    half = period // 2
    n1 = (bars.High.rolling(half).max() - bars.Low.rolling(half).min()) / half
    n2 = n1.shift(half)
    n3 = (bars.High.rolling(period).max() - bars.Low.rolling(period).min()) / period
    dimension = (np.log((n1+n2).replace(0, np.nan)) - np.log(n3.replace(0, np.nan))) / np.log(2)
    alpha = np.exp(-4.6 * (dimension - 1)).clip(.01, 1).fillna(1).to_numpy()
    mid = (bars.High + bars.Low) / 2
    x = _prices(mid)
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        current = x[period-1]
        out[period-1] = current
        for i in range(period, len(x)):
            current += alpha[i] * (x[i] - current)
            out[i] = current
    return pd.Series(out, index=bars.index)


def connors_rsi(close, rank_period=100):
    if rank_period < 2:
        raise ValueError("Rank lookback must be >= 2")
    x = _prices(close)
    streak = np.zeros(len(x))
    for i in range(1, len(x)):
        sign = np.sign(x[i] - x[i-1])
        streak[i] = streak[i-1] + sign if sign and np.sign(streak[i-1]) == sign else sign
    ret = close.pct_change().to_numpy()
    rank = np.full(len(x), np.nan)
    if len(x) > rank_period:
        windows = np.lib.stride_tricks.sliding_window_view(ret, rank_period + 1)
        counts = (windows[:, :-1] < windows[:, [-1]]).sum(axis=1)
        valid = np.isfinite(windows).all(axis=1)
        rank[rank_period:] = np.where(valid, counts / rank_period * 100, np.nan)
    return (compute_rsi(close, 3) + compute_rsi(pd.Series(streak, index=close.index), 2)
            + pd.Series(rank, index=close.index)) / 3


def laguerre_rsi(mid, gamma=.5):
    if not 0 < gamma < 1:
        raise ValueError("Laguerre gamma must be between 0 and 1")
    x = _prices(mid)
    out = np.full(len(x), np.nan)
    if not len(x):
        return pd.Series(out, index=mid.index)
    levels = np.full(4, x[0])
    for i, price in enumerate(x):
        previous = levels.copy()
        levels[0] = (1 - gamma) * price + gamma * previous[0]
        for pole in range(1, 4):
            levels[pole] = previous[pole-1] + gamma * (previous[pole] - levels[pole-1])
        differences = levels[:-1] - levels[1:]
        up, down = np.maximum(differences, 0).sum(), np.maximum(-differences, 0).sum()
        out[i] = up / (up+down) if up+down > 1e-15 else .5
    out[:min(16, len(out))] = np.nan
    return pd.Series(out, index=mid.index)


def supertrend(bars, period=10, multiplier=3.):
    if period < 2 or multiplier <= 0:
        raise ValueError("Invalid Supertrend settings")
    atr = compute_atr(bars, period).to_numpy()
    mid = ((bars.High + bars.Low) / 2).to_numpy()
    close = _prices(bars.Close)
    line, side = np.full(len(bars), np.nan), np.zeros(len(bars), dtype=int)
    upper, lower, direction = None, None, -1
    for i in range(len(bars)):
        if not np.isfinite(atr[i]):
            continue
        raw_upper, raw_lower = mid[i] + multiplier * atr[i], mid[i] - multiplier * atr[i]
        if upper is None:
            upper, lower = raw_upper, raw_lower
        else:
            upper = raw_upper if raw_upper < upper or close[i-1] > upper else upper
            lower = raw_lower if raw_lower > lower or close[i-1] < lower else lower
            if direction == -1 and close[i] > upper:
                direction = 1
            elif direction == 1 and close[i] < lower:
                direction = -1
        side[i], line[i] = direction, lower if direction == 1 else upper
    return pd.DataFrame({"line": line, "side": side}, index=bars.index)


def researched_features(bars):
    mid = (bars.High + bars.Low) / 2
    x = pd.DataFrame(index=bars.index)
    x["kama"] = kama(bars.Close)
    x["frama"] = frama(bars)
    x["fisher"] = fisher_transform(mid)
    x["roof"] = roofing_filter(bars.Close)
    span = x.roof.rolling(20).max() - x.roof.rolling(20).min()
    stochastic = ((x.roof - x.roof.rolling(20).min()) / span.replace(0, np.nan)).fillna(.5)
    x["roof_stochastic"] = super_smoother(stochastic)
    x["connors"] = connors_rsi(bars.Close)
    x["laguerre"] = laguerre_rsi(bars.Close)
    x["supertrend_side"] = supertrend(bars).side
    return x


def researched_rule_states(bars):
    x = researched_features(bars)
    close = bars.Close
    def states(long, short):
        return np.select([long, short], [1, -1], default=0).astype(int)
    rules = {}
    rules["supertrend_flip"] = x.supertrend_side.to_numpy()
    rules["supertrend_kama"] = states((x.supertrend_side == 1) & (x.kama.diff(4) > 0),
        (x.supertrend_side == -1) & (x.kama.diff(4) < 0))
    for name in ("kama", "frama"):
        average = x[name]
        rules[f"{name}_cross"] = states((close > average) & (close.shift(1) <= average.shift(1)),
            (close < average) & (close.shift(1) >= average.shift(1)))
        rules[f"{name}_pullback"] = states((average.diff(8) > 0) & (close > average) & (close.shift(1) <= average.shift(1)),
            (average.diff(8) < 0) & (close < average) & (close.shift(1) >= average.shift(1)))
    derivative = x.fisher.diff()
    rules["fisher_turn"] = states((x.fisher < 0) & (derivative > 0) & (derivative.shift(1) <= 0),
        (x.fisher > 0) & (derivative < 0) & (derivative.shift(1) >= 0))
    rules["fisher_zero"] = states((x.fisher > 0) & (x.fisher.shift(1) <= 0),
        (x.fisher < 0) & (x.fisher.shift(1) >= 0))
    oscillator = x.roof_stochastic
    rules["roof_anticipatory"] = states((oscillator < .2) & (oscillator.shift(1) >= .2),
        (oscillator > .8) & (oscillator.shift(1) <= .8))
    rules["roof_confirmed"] = states((oscillator > .2) & (oscillator.shift(1) <= .2),
        (oscillator < .8) & (oscillator.shift(1) >= .8))
    rules["connors_fade"] = states(x.connors < 10, x.connors > 90)
    rules["connors_kama"] = states((x.connors < 10) & (close > x.kama), (x.connors > 90) & (close < x.kama))
    rules["laguerre_reentry"] = states((x.laguerre > .2) & (x.laguerre.shift(1) <= .2),
        (x.laguerre < .8) & (x.laguerre.shift(1) >= .8))
    rules["laguerre_kama"] = states((x.laguerre > .2) & (x.laguerre.shift(1) <= .2) & (close > x.kama),
        (x.laguerre < .8) & (x.laguerre.shift(1) >= .8) & (close < x.kama))
    return rules
