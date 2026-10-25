"""Indicator families never tested in this project, on 15M, 1H and 4H, $30 net / $100 stop.

Earlier rounds covered RSI, MACD, Bollinger, EMA, Ichimoku, Stochastic, Donchian,
Keltner, ADX, Fisher, Roofing, FRAMA, KAMA, Supertrend, ConnorsRSI and Laguerre RSI,
on 15M only. New here: volume (Dukascopy ECN volume), floor/Camarilla pivots,
liquidity sweeps and fair value gaps, candlestick patterns, and other trend and
oscillator formulas; plus 1H and 4H decision bars where cost is a smaller share
of risk. Grid fixed in protocol.json. Periods: development 2009-2017, validation
2018-2022, audit 2023+. Rules are judged pooled over ten pairs so a rule must work
broadly, not on one lucky pair. No broker orders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from forex_adaptive import ROOT, digest, write_json
from forex_config import get_pip_size
from forex_indicators import compute_atr, compute_bbands, wilder_average
from forex_ml_tournament import load_hourly
from research_bracket_core import bracket_trades, cost_pips, day_test, load_pair, summarize

OUT = ROOT / "research/indicator_zoo"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"]
PERIODS = {"development": ("2009-01-01", "2018-01-01"), "validation": ("2018-01-01", "2023-01-01"),
    "audit": ("2023-01-01", "2027-01-01")}
TIMEFRAMES = {"15m": ("historical_15m", pd.Timedelta(minutes=15), pd.Timedelta(hours=4)),
    "1h": ("historical_1h_from_15m", pd.Timedelta(hours=1), pd.Timedelta(hours=24)),
    "4h": ("historical_4h", pd.Timedelta(hours=4), pd.Timedelta(days=5))}
STOPS_ATR = (1.5, 3.)
SKIP_NY_HOURS = (16, 17)  # rollover spread
SWAP_MARKUP = 0.01  # per year, charged on every night held whatever the direction


def state(long, short):
    return np.select([np.asarray(long, bool), np.asarray(short, bool)], [1, -1], default=0).astype(int)


def cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


@njit(cache=True)
def _psar(h, l, step, cap):
    n = len(h)
    side = np.zeros(n, np.int64)
    if n < 3:
        return side
    up, sar, ep, af = True, l[0], h[0], step
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, cap)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, cap)
        side[i] = 1 if up else -1
    return side


@njit(cache=True)
def _fvg_retest(h, l, c, lookback):
    """Long when price dips into the latest unfilled bullish gap (low[t-1] > high[t-3]) and closes above its floor."""
    n = len(h)
    out = np.zeros(n, np.int64)
    bull_top = bull_bot = bear_top = bear_bot = np.nan
    bull_age = bear_age = 10**9
    for i in range(3, n):
        # Gaps are judged on completed bars only: bars i-3 .. i-1.
        if l[i - 1] > h[i - 3]:
            bull_top, bull_bot, bull_age = l[i - 1], h[i - 3], 0
        if h[i - 1] < l[i - 3]:
            bear_top, bear_bot, bear_age = l[i - 3], h[i - 1], 0
        bull_age += 1
        bear_age += 1
        if bull_age <= lookback and l[i] <= bull_top and c[i] > bull_bot:
            out[i] = 1
            bull_age = 10**9
        elif bull_age <= lookback and c[i] <= bull_bot:
            bull_age = 10**9
        if bear_age <= lookback and h[i] >= bear_bot and c[i] < bear_top:
            out[i] = -1 if out[i] == 0 else 0
            bear_age = 10**9
        elif bear_age <= lookback and c[i] >= bear_top:
            bear_age = 10**9
    return out


def previous_day(bars):
    """Previous completed UTC weekday's high/low/close for every bar (Sunday stubs get NaN)."""
    day = bars.index.normalize()
    weekday = bars[bars.index.dayofweek < 5]
    daily = weekday.groupby(weekday.index.normalize()).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    prev = daily.shift(1)
    out = prev.reindex(day)
    out.index = bars.index
    return out


def asian_range(bars):
    hour = bars.index.hour
    asia = bars[hour < 7]
    rng = asia.groupby(asia.index.normalize()).agg(H=("High", "max"), L=("Low", "min"))
    out = rng.reindex(bars.index.normalize())
    out.index = bars.index
    out.loc[np.asarray(hour < 7)] = np.nan  # known only after 07:00 UTC
    return out


def zoo_rules(bars, timeframe):
    o, h, l, c, v = bars.Open, bars.High, bars.Low, bars.Close, bars.Volume
    atr = compute_atr(bars)
    rules = {}
    # Trend and momentum formulas not used before.
    psar = pd.Series(_psar(h.to_numpy(float), l.to_numpy(float), .02, .2), index=bars.index)
    rules["psar_flip"] = np.where(psar != psar.shift(1), psar, 0)
    aroon_up = h.rolling(26).apply(np.argmax, raw=True) / 25 * 100
    aroon_down = l.rolling(26).apply(np.argmin, raw=True) / 25 * 100
    rules["aroon_cross"] = state(cross_up(aroon_up, aroon_down), cross_up(aroon_down, aroon_up))
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vip = (h - l.shift(1)).abs().rolling(14).sum() / tr.rolling(14).sum()
    vim = (l - h.shift(1)).abs().rolling(14).sum() / tr.rolling(14).sum()
    rules["vortex_cross"] = state(cross_up(vip, vim), cross_up(vim, vip))
    ema3 = c.ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean()
    trix = ema3.pct_change() * 1e4
    trix_signal = trix.ewm(span=9, adjust=False).mean()
    rules["trix_cross"] = state(cross_up(trix, trix_signal), cross_up(trix_signal, trix))
    m = c.diff()
    tsi = 100 * m.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean() / \
        m.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi_signal = tsi.ewm(span=7, adjust=False).mean()
    rules["tsi_cross"] = state(cross_up(tsi, tsi_signal), cross_up(tsi_signal, tsi))
    ha_close = (o + h + l + c) / 4
    ha_open = ha_close.copy()
    ha_values, ha_c = ha_open.to_numpy(), ha_close.to_numpy()
    ha_values[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(ha_values)):
        ha_values[i] = (ha_values[i - 1] + ha_c[i - 1]) / 2
    color = pd.Series(np.sign(ha_c - ha_values), index=bars.index)
    run3 = color.shift(1).rolling(3).sum()
    rules["heikin_flip"] = state((color > 0) & (run3 == -3), (color < 0) & (run3 == 3))

    def wma(s, n):
        w = np.arange(1, n + 1)
        return s.rolling(n).apply(lambda x: x @ w / w.sum(), raw=True)

    hma = wma(2 * wma(c, 10) - wma(c, 21), 4)
    slope = np.sign(hma.diff())
    rules["hull_turn"] = state((slope > 0) & (slope.shift(1) < 0), (slope < 0) & (slope.shift(1) > 0))
    ema13 = c.ewm(span=13, adjust=False).mean()
    bull, bear = h - ema13, l - ema13
    rules["elder_ray"] = state((ema13.diff() > 0) & (bear < 0) & (bear > bear.shift(1)) & (bear.shift(1) < bear.shift(2)),
        (ema13.diff() < 0) & (bull > 0) & (bull < bull.shift(1)) & (bull.shift(1) > bull.shift(2)))
    # Oscillators not used before.
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    rules["cci_reentry"] = state(cross_up(cci, pd.Series(-100., index=bars.index)), cross_up(pd.Series(100., index=bars.index), cci))
    willr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    rules["willr_reentry"] = state((willr > -80) & (willr.shift(1) <= -80), (willr < -20) & (willr.shift(1) >= -20))
    low_c = pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    bp = c - low_c
    uo = 100 * (4 * bp.rolling(7).sum() / tr.rolling(7).sum() + 2 * bp.rolling(14).sum() / tr.rolling(14).sum()
        + bp.rolling(28).sum() / tr.rolling(28).sum()) / 7
    rules["ultimate_reentry"] = state((uo > 30) & (uo.shift(1) <= 30), (uo < 70) & (uo.shift(1) >= 70))
    chop = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)) / np.log10(14)
    bb_low, _, bb_up = compute_bbands(c)
    rules["chop_band_fade"] = state((chop > 61.8) & (c < bb_low), (chop > 61.8) & (c > bb_up))
    ret = c.pct_change()
    autocorr = ret.rolling(200).corr(ret.shift(1))
    z20 = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    rules["autocorr_fade"] = state((autocorr < -.05) & (z20 < -2), (autocorr < -.05) & (z20 > 2))
    rules["autocorr_follow"] = state((autocorr > .05) & (z20 > 2), (autocorr > .05) & (z20 < -2))
    # Volume (Dukascopy ECN volume, a different input from price-only formulas).
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    breakout_up = c > h.shift(1).rolling(20).max()
    breakout_down = c < l.shift(1).rolling(20).min()
    rules["obv_confirmed_break"] = state(breakout_up & (obv >= obv.rolling(20).max()),
        breakout_down & (obv <= obv.rolling(20).min()))
    rules["obv_divergence"] = state(breakout_down & (obv > obv.rolling(20).min()), breakout_up & (obv < obv.rolling(20).max()))
    flow = tp * v
    pos_flow = flow.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_flow = flow.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_flow / neg_flow.replace(0, np.nan))
    rules["mfi_reentry"] = state((mfi > 20) & (mfi.shift(1) <= 20), (mfi < 80) & (mfi.shift(1) >= 80))
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    cmf = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    rules["cmf_flip"] = state((cmf > .05) & (cmf.shift(1) <= .05), (cmf < -.05) & (cmf.shift(1) >= -.05))
    day = bars.index.normalize()
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    dev = c - vwap
    dev_z = dev / dev.rolling(100).std().replace(0, np.nan)
    rules["vwap_fade"] = state(dev_z < -2, dev_z > 2)
    climax = (v > 3 * v.shift(1).rolling(20).mean()) & ((h - l) > 2 * atr.shift(1))
    rules["volume_climax_fade"] = state(climax & (c < o), climax & (c > o))
    # Pivots from the previous completed UTC weekday.
    pd_ = previous_day(bars)
    pivot = (pd_.H + pd_.L + pd_.C) / 3
    s1, r1 = 2 * pivot - pd_.H, 2 * pivot - pd_.L
    rules["pivot_bounce"] = state((l <= s1) & (c > s1) & (c > o), (h >= r1) & (c < r1) & (c < o))
    rng = pd_.H - pd_.L
    h3, l3 = pd_.C + rng * 1.1 / 4, pd_.C - rng * 1.1 / 4
    h4, l4 = pd_.C + rng * 1.1 / 2, pd_.C - rng * 1.1 / 2
    rules["camarilla_fade"] = state((l < l3) & (c > l3), (h > h3) & (c < h3))
    rules["camarilla_break"] = state((c > h4) & (c.shift(1) <= h4), (c < l4) & (c.shift(1) >= l4))
    # Liquidity sweeps ("stop hunts") and fair value gaps.
    rules["prev_day_sweep"] = state((l < pd_.L) & (c > pd_.L), (h > pd_.H) & (c < pd_.H))
    if timeframe != "4h":
        asia = asian_range(bars)
        london = (bars.index.hour >= 7) & (bars.index.hour < 10)
        rules["asian_sweep"] = state(london & (l < asia.L) & (c > asia.L), london & (h > asia.H) & (c < asia.H))
    rules["fvg_retest"] = _fvg_retest(h.to_numpy(float), l.to_numpy(float), c.to_numpy(float), 20)
    # Candlestick patterns at 20-bar extremes.
    body = (c - o).abs()
    low20, high20 = l.rolling(20).min(), h.rolling(20).max()
    rules["engulfing_extreme"] = state((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1)) & (l <= low20),
        (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1)) & (h >= high20))
    lower_wick, upper_wick = pd.concat([o, c], axis=1).min(axis=1) - l, h - pd.concat([o, c], axis=1).max(axis=1)
    rules["pinbar_extreme"] = state((lower_wick > 2 * body) & (lower_wick > .6 * (h - l)) & (l <= low20),
        (upper_wick > 2 * body) & (upper_wick > .6 * (h - l)) & (h >= high20))
    inside = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    rules["inside_break"] = state(inside & (c > h.shift(2)), inside & (c < l.shift(2)))
    rng_bar = h - l
    nr7 = rng_bar.shift(1) <= rng_bar.shift(1).rolling(7).min()
    rules["nr7_break"] = state(nr7 & (c > h.shift(1)), nr7 & (c < l.shift(1)))
    return {k: np.asarray(v, dtype=int) for k, v in rules.items()}, atr.to_numpy(float)


def run_pair(symbol, data_dir=ROOT / "data", out=OUT):
    data_dir = Path(data_dir)
    _, fills = load_pair(symbol, data_dir, "15m")
    pip = get_pip_size(symbol)
    rows, frames = [], []
    for tf, (folder, length, hold) in TIMEFRAMES.items():
        bars = load_hourly(data_dir / folder / f"{symbol}.csv", quarantine=True)
        rules, atr = zoo_rules(bars, tf)
        decisions = bars.index + length
        bad = np.zeros(len(bars), bool)
        bad[bars.attrs.get("bad_positions", [])] = True
        usable = (np.arange(len(bars)) >= 300) & ~pd.Series(bad).rolling(301, min_periods=1).max().astype(bool).to_numpy()
        usable &= ~np.isin(decisions.tz_convert("America/New_York").hour, SKIP_NY_HOURS)
        costs = cost_pips(symbol, decisions)
        for name, st in rules.items():
            idx = np.flatnonzero((st != 0) & (st != np.r_[0, st[:-1]]) & usable)
            for k in STOPS_ATR:
                trades = bracket_trades(symbol, fills, decisions[idx], st[idx], atr[idx] * k, decisions[idx] + hold, costs[idx])
                if len(trades):
                    # Nights held: 21:00 UTC crossings; weekend days count, like the triple Wednesday charge.
                    shift = pd.Timedelta(hours=21)
                    nights = ((trades.exit_time - shift).dt.floor("D") - (trades.entry_time - shift).dt.floor("D")).dt.days.to_numpy()
                    price = fills["o"][fills["index"].get_indexer(trades.entry_time)]
                    swap = SWAP_MARKUP * price / 365 * nights / pip
                    risk = trades.stop_pips + trades.cost_pips
                    trades["pnl_usd"] -= swap / risk * 100
                    trades["pnl_stress_usd"] -= 2 * swap / risk * 100
                    trades["net_r"] = trades.pnl_usd / 100
                key = f"{tf}_{name}_{k:g}atr"
                frames.append(trades.assign(config=key))
                for period, (start, end) in PERIODS.items():
                    part = trades[(trades.entry_time >= start) & (trades.entry_time < end)] if len(trades) else trades
                    rows.append({"symbol": symbol, "timeframe": tf, "rule": name, "stop_atr": k, "period": period,
                        **summarize(part)})
        print(f"{symbol} {tf}: done", flush=True)
    folder = Path(out) / symbol
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat([f for f in frames if len(f)], ignore_index=True).to_csv(folder / "trades.csv.gz", index=False, compression="gzip")
    return rows


def pooled(summary):
    """Trade-weighted means across pairs, plus pair breadth, per rule/timeframe/stop and period."""
    s = summary[summary.trades > 0].copy()
    for col in ("mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"):
        s[col + "_w"] = s[col] * s.trades
    g = s.groupby(["timeframe", "rule", "stop_atr", "period"])
    out = g[["trades"]].sum()
    for col in ("mean_usd", "stress_mean_usd", "gross_mean_r", "target_rate"):
        out[col] = g[col + "_w"].sum() / out.trades
    out["pairs_positive"] = g.apply(lambda x: int((x.mean_usd > 0).sum()), include_groups=False)
    out["pairs"] = g.size()
    wide = out.unstack("period")
    dev_ok = (wide[("stress_mean_usd", "development")] > 0) & (wide[("pairs_positive", "development")] >= 6) & \
        (wide[("trades", "development")] >= 500)
    val_ok = dev_ok & (wide[("mean_usd", "validation")] > 0) & (wide[("pairs_positive", "validation")] >= 6)
    return out.reset_index(), wide, dev_ok, val_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-09", "target_net_usd": 30, "planned_stop_usd": 100,
        "pairs": PAIRS, "timeframes": {k: [v[0], str(v[1]), str(v[2])] for k, v in TIMEFRAMES.items()},
        "stops_atr_of_timeframe": STOPS_ATR, "skip_entry_ny_hours": SKIP_NY_HOURS,
        "cost": "measured spread at decision New York hour + 0.7 pip; stress 2x; swap markup 1%/yr per night (2x stress)",
        "periods": PERIODS, "gates": {
            "development": "pooled over pairs: stress mean > 0, >=500 trades, >=6 of 10 pairs positive",
            "validation": "pooled mean > 0 and >=6 of 10 pairs positive", "audit": "report only; daily bootstrap"},
        "script_sha256": digest(Path(__file__))})
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(run_pair, args.symbols):
            rows.extend(part)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    table, wide, dev_ok, val_ok = pooled(summary)
    table.to_csv(OUT / "pooled.csv", index=False)
    show = wide[[("mean_usd", p) for p in PERIODS] + [("gross_mean_r", "development"), ("pairs_positive", "development")]]
    print(show.sort_values(("mean_usd", "development"), ascending=False).head(25).round(3).to_string())
    print(f"configs: {len(wide)}; development passes: {int(dev_ok.sum())}; validation passes: {int(val_ok.sum())}")
    for tf, rule, stop in wide[val_ok].index:
        frames = []
        for symbol in args.symbols:
            t = pd.read_csv(OUT / symbol / "trades.csv.gz", parse_dates=["entry_time", "exit_time"])
            frames.append(t[(t.config == f"{tf}_{rule}_{stop:g}atr") & (t.entry_time >= PERIODS["audit"][0])])
        t = pd.concat(frames)
        print(tf, rule, stop, summarize(t), day_test(t))


if __name__ == "__main__":
    main()
