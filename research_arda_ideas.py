"""Out-of-sample tests of ideas from a friend's MT5 project (arda_forex/forex-bot-main).

Rules were copied, not re-tuned; the fixed protocol is research/arda_ideas/protocol.json.
H1/H1b/H7: USDJPY H4 EMA20 pullback (+28 pairs, partial exit). H3: D1/H4/H1/M30
structure ladder. H4: M15 shock-bar alert. H5: "this release moved the pair X of Y
times" history line. H2/H6 (gold) run through the adapter in gold(). Server time is
New York + 7h, so H4/D1 candles match an XM-style broker. No broker orders.

Run: python -X utf8 research_arda_ideas.py fx|ladder|shock|news|gold
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from forex_adaptive import ROOT, write_json
from forex_config import get_pip_size
from forex_costs import load_spread_profile
from forex_ml_tournament import load_hourly
from research_bracket_core import cost_pips, day_test
from research_night_bracket import PAIRS

OUT = ROOT / "research/arda_ideas"
DATA = ROOT / "data"
SERVER_SHIFT = pd.Timedelta(hours=7)
PERIODS_FX = {"development": ("2009-01-01", "2018-01-01"), "validation": ("2018-01-01", "2023-01-01"),
    "audit": ("2023-01-01", "2025-01-01"), "replication_only": ("2025-01-01", "2027-01-01")}
PERIODS_DISPLAY = {"development": ("2009-01-01", "2018-01-01"), "validation": ("2018-01-01", "2023-01-01"),
    "audit": ("2023-01-01", "2027-01-01")}


# ---------------------------------------------------------------- shared pieces
def server_bars(m15, rule):
    """OHLC on the broker clock; `avail` is the UTC time the bar is complete."""
    srv = m15.index.tz_convert("America/New_York").tz_localize(None) + SERVER_SHIFT
    key = srv.floor(rule) if rule != "D" else srv.normalize()
    g = m15.assign(key=key, utc=m15.index).groupby("key", sort=True)
    bars = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"),
        avail=("utc", "max"))
    bars["avail"] = bars.avail + pd.Timedelta(minutes=15)
    return bars


@njit(cache=True)
def struct_dir(h, l, c, L):
    """Arda's market structure: last close beyond the latest confirmed pivot high/low."""
    n = len(c)
    out = np.zeros(n, np.int64)
    ph = pl = 0.0
    d = 0
    for k in range(n):
        p = k - L
        if p - L >= 0:
            is_h = True
            is_l = True
            for m in range(1, L + 1):
                if not (h[p] > h[p - m] and h[p] >= h[p + m]):
                    is_h = False
                if not (l[p] < l[p - m] and l[p] <= l[p + m]):
                    is_l = False
            if is_h:
                ph = h[p]
            if is_l:
                pl = l[p]
        if ph > 0 and c[k] > ph:
            d = 1
        if pl > 0 and c[k] < pl:
            d = -1
        out[k] = d
    return out


@njit(cache=True)
def atr_wilder(h, l, c):
    n = len(c)
    a = np.zeros(n)
    for i in range(n):
        tr = h[i] - l[i] if i == 0 else max(h[i], c[i - 1]) - min(l[i], c[i - 1])
        a[i] = tr if i == 0 else ((a[i - 1] * i + tr) / (i + 1) if i < 14 else (a[i - 1] * 13 + tr) / 14)
    return a


def asof(avail_times, values, at):
    """Value of the last bar complete at or before each `at` (0 before the first)."""
    k = np.searchsorted(pd.DatetimeIndex(avail_times).asi8, pd.DatetimeIndex(at).asi8, side="right") - 1
    return np.where(k >= 0, np.asarray(values)[np.maximum(k, 0)], 0)


def period_of(times, periods):
    times = pd.DatetimeIndex(times)
    label = np.full(len(times), "", dtype=object)
    for name, (a, b) in periods.items():
        label[(times >= pd.Timestamp(a, tz="UTC")) & (times < pd.Timestamp(b, tz="UTC"))] = name
    return label


def boot(frame, column, seed=73):
    """Day-level stationary bootstrap of the mean of daily sums (research_bracket_core.day_test)."""
    if frame is None or len(frame) < 10:
        return {"ci_low": None, "ci_high": None, "p_value": None}
    test = day_test(frame.rename(columns={"time": "entry_time"}), column, seed)
    return {k: test.get(k) for k in ("ci_low", "ci_high", "p_value")}


# ---------------------------------------------------------------- H1 / H1b / H7
def h4_pullback_signals(h4, d1):
    """USDJPY_Sinyal.mq5 Recompute(): EMA20/50 direction, D1 structure, EMA20 touch-and-close."""
    o, h, l, c = (h4[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    atr = atr_wilder(h, l, c)
    ef = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    es = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    d1dir = struct_dir(d1.High.to_numpy(float), d1.Low.to_numpy(float), d1.Close.to_numpy(float), 3)
    day_end = (d1.index + pd.Timedelta(days=1)).asi8
    q = np.searchsorted(day_end, h4.index.asi8, side="right") - 1  # last server day closed before the H4 open
    d1_at = np.where(q >= 0, d1dir[np.maximum(q, 0)], 0)
    rows, last = [], -1000
    for k in range(50, len(c) - 1):
        d = 1 if ef[k] > es[k] else -1
        if d1_at[k] != d:
            continue
        if not (l[k] <= ef[k] if d > 0 else h[k] >= ef[k]):
            continue
        if (c[k] - ef[k]) * d <= 0 or (c[k] - o[k]) * d <= 0 or k - last < 5:
            continue
        lo = max(0, k - 5)
        ext = l[lo:k + 1].min() if d > 0 else h[lo:k + 1].max()
        sl = ext - d * 0.3 * atr[k]
        risk = (c[k] - sl) * d
        if risk <= 0:
            continue
        last = k
        end = min(k + 100, len(c) - 1)
        rows.append((h4.avail.iloc[k], d, c[k], sl, c[k] + d * 2 * risk, risk, h4.avail.iloc[end]))
    return pd.DataFrame(rows, columns=["time", "side", "close", "sl", "tp", "risk", "deadline"])


@njit(cache=True)
def _paths(o, h, l, c, t, j0, sides, closes, sls, tps, risks, deadlines):
    """Full and partial-exit results per signal in R (before cost); NaN when the path is incomplete."""
    n = len(j0)
    full = np.full(n, np.nan)
    part = np.full(n, np.nan)
    for k in range(n):
        j = j0[k]
        d, e0, sl, tp, r = sides[k], closes[k], sls[k], tps[k], risks[k]
        fill = o[j]
        lvl1 = e0 + d * r
        res = np.nan
        got = 0.0
        half = False
        be = fill
        res_p = np.nan
        i = j
        while i < len(o) and t[i] < deadlines[k]:
            adv = l[i] if d > 0 else h[i]
            fav = h[i] if d > 0 else l[i]
            if np.isnan(res):  # full position
                if i > j and (o[i] - sl) * d <= 0:
                    res = (o[i] - fill) * d / r
                elif (adv - sl) * d <= 0:
                    res = (sl - fill) * d / r
                elif i > j and (o[i] - tp) * d >= 0:
                    res = (o[i] - fill) * d / r
                elif (fav - tp) * d >= 0:
                    res = (tp - fill) * d / r
            if np.isnan(res_p):  # half at +1R, stop to entry
                stop = be if half else sl
                if (adv - stop) * d <= 0:
                    px = o[i] if (i > j and (o[i] - stop) * d <= 0) else stop
                    res_p = got + (0.5 if half else 1.0) * (px - fill) * d / r
                elif half and (fav - tp) * d >= 0:
                    px = o[i] if (o[i] - tp) * d >= 0 else tp
                    res_p = got + 0.5 * (px - fill) * d / r
                elif not half and (fav - lvl1) * d >= 0:
                    half = True
                    got = 0.5 * (lvl1 - fill) * d / r
                    if (fav - tp) * d >= 0:
                        res_p = got + 0.5 * (tp - fill) * d / r
            if not np.isnan(res) and not np.isnan(res_p):
                break
            i += 1
        if i >= len(o) and (np.isnan(res) or np.isnan(res_p)):
            continue  # data ends before the trade does
        last = c[i - 1] if i > j else fill
        if np.isnan(res):
            res = (last - fill) * d / r
        if np.isnan(res_p):
            res_p = got + (0.5 if half else 1.0) * (last - fill) * d / r
        full[k] = res
        part[k] = res_p
    return full, part


def fx_pair(symbol, data_dir=DATA):
    m15 = load_hourly(data_dir / f"historical_15m/{symbol}.csv", quarantine=True)
    fills = load_hourly(data_dir / f"historical_5m/{symbol}.csv", quarantine=True)
    sig = h4_pullback_signals(server_bars(m15, "4h"), server_bars(m15, "D"))
    if sig.empty:
        return pd.DataFrame()
    t = fills.index.asi8
    j = np.searchsorted(t, pd.DatetimeIndex(sig.time).asi8)
    ok = j < len(t)
    sig, j = sig[ok].reset_index(drop=True), j[ok]
    full, part = _paths(fills.Open.to_numpy(float), fills.High.to_numpy(float), fills.Low.to_numpy(float),
        fills.Close.to_numpy(float), t, j.astype(np.int64), sig.side.to_numpy(np.int64), sig.close.to_numpy(float),
        sig.sl.to_numpy(float), sig.tp.to_numpy(float), sig.risk.to_numpy(float), pd.DatetimeIndex(sig.deadline).asi8)
    pip = get_pip_size(symbol)
    cost = cost_pips(symbol, sig.time) * pip / sig.risk.to_numpy()
    sig["symbol"] = symbol
    sig["risk_pips"] = sig.risk / pip
    sig["gross_r"], sig["net_r"], sig["net_r_partial"] = full, full - cost, part - cost
    sig["period"] = period_of(sig.time, PERIODS_FX)
    # H6: 0.01 lot = 1,000 units; USD loss at the stop vs 3 x the planned $15.
    quote_usd = np.where(symbol.endswith("USD"), 1.0, np.nan)
    if symbol.endswith("JPY"):
        quote_usd = 1.0 / sig.close.to_numpy()
    sig["usd_at_stop_001"] = sig.risk.to_numpy() * 1000 * quote_usd
    return sig.dropna(subset=["net_r"])


def worst_month_and_dd(frame, column):
    s = frame.sort_values("time")
    monthly = s.groupby(s.time.dt.strftime("%Y-%m"))[column].sum()
    eq = s[column].cumsum()
    return float(monthly.min()), float((eq.cummax() - eq).max())


def gate(frame, column="net_r", parts=("development", "validation", "audit")):
    unseen = frame[frame.period.isin(parts)]
    by = {p: float(unseen.loc[unseen.period == p, column].mean()) if (unseen.period == p).any() else None for p in parts}
    test = boot(unseen, column)
    ok = (len(unseen) >= 60 and unseen[column].mean() > 0 and (test["ci_low"] or -1) > 0
          and all(v is not None and v > 0 for v in by.values()))
    return {"trades": int(len(unseen)), "mean_net_r": float(unseen[column].mean()) if len(unseen) else None,
        "by_period_mean_net_r": by, **test, "passed": bool(ok)}


def describe(frame, column="net_r"):
    x = frame[column]
    win, loss = x[x > 0].sum(), -x[x < 0].sum()
    return {"trades": int(len(x)), "mean_net_r": float(x.mean()) if len(x) else None,
        "win_rate": float((x > 0).mean()) if len(x) else None, "pf": float(win / loss) if loss > 0 else None,
        "total_r": float(x.sum())}


def fx(workers):
    with ProcessPoolExecutor(workers) as pool:
        trades = pd.concat(list(pool.map(fx_pair, PAIRS)), ignore_index=True)
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT / "rows/h4_pullback_trades.csv.gz", index=False)
    uj = trades[trades.symbol == "USDJPY"]
    result = {"H1_usdjpy": {"gate": gate(uj), "by_period": {p: describe(uj[uj.period == p]) for p in PERIODS_FX},
        "long": describe(uj[(uj.side == 1) & (uj.period != "replication_only")]),
        "short": describe(uj[(uj.side == -1) & (uj.period != "replication_only")]),
        "gross_mean_r_unseen": float(uj.loc[uj.period != "replication_only", "gross_r"].mean()),
        "median_risk_pips": float(uj.risk_pips.median())}}
    unseen = trades[trades.period != "replication_only"]
    pair_means = unseen.groupby("symbol").net_r.mean()
    g = gate(trades)
    g["share_pairs_positive"] = float((pair_means > 0).mean())
    g["passed"] = bool(g["passed"] and g["share_pairs_positive"] >= 0.6)
    result["H1b_28_pairs"] = {"gate": g, "by_period": {p: describe(trades[trades.period == p]) for p in PERIODS_FX},
        "pair_mean_net_r": pair_means.round(4).to_dict()}
    h7 = {}
    for name, frame in (("usdjpy", uj[uj.period != "replication_only"]), ("all_pairs", unseen)):
        base_wm, base_dd = worst_month_and_dd(frame, "net_r")
        part_wm, part_dd = worst_month_and_dd(frame, "net_r_partial")
        h7[name] = {"full": {**describe(frame), "worst_month_r": base_wm, "max_dd_r": base_dd},
            "partial": {**describe(frame, "net_r_partial"), "worst_month_r": part_wm, "max_dd_r": part_dd}}
        h7[name]["offered"] = bool(part_wm >= 0.7 * base_wm and part_dd <= 0.7 * base_dd
                                   and frame.net_r_partial.mean() > 0)  # worst month is negative
    result["H7_partial_exit"] = h7
    result["H6_lot_guard_usdjpy"] = {"trades": int(len(uj)),
        "skipped_at_1000_usd_1p5pct": int((uj.usd_at_stop_001 > 45).sum()),
        "median_usd_at_stop_001": float(uj.usd_at_stop_001.median())}
    write_json(OUT / "fx_results.json", result)
    return result


# ---------------------------------------------------------------- H3 ladder
def ladder_pair(symbol, data_dir=DATA):
    m15 = load_hourly(data_dir / f"historical_15m/{symbol}.csv", quarantine=True)
    pip = get_pip_size(symbol)
    tfs = {name: server_bars(m15, rule) for name, rule in (("D1", "D"), ("H4", "4h"), ("H1", "1h"), ("M30", "30min"))}
    dirs = {name: struct_dir(b.High.to_numpy(float), b.Low.to_numpy(float), b.Close.to_numpy(float), 3)
            for name, b in tfs.items()}
    at = pd.DatetimeIndex(tfs["H1"].avail)
    ladder = np.vstack([asof(tfs[n].avail, dirs[n], at) for n in tfs])
    agree = np.where((ladder == ladder[0]).all(axis=0), ladder[0], 0)
    closes = m15.Close.to_numpy(float)
    end_t = (m15.index + pd.Timedelta(minutes=15)).asi8
    out = []
    for horizon in (1, 4):
        k0 = np.searchsorted(end_t, at.asi8)
        k1 = np.searchsorted(end_t, (at + pd.Timedelta(hours=horizon)).asi8)
        ok = (k1 < len(end_t)) & (k0 < len(end_t))
        ok[ok] &= (end_t[k0[ok]] == at.asi8[ok]) & (end_t[k1[ok]] == (at + pd.Timedelta(hours=horizon)).asi8[ok])
        move = np.full(len(at), np.nan)
        move[ok] = (closes[k1[ok]] - closes[k0[ok]]) / pip
        out.append(move)
    cost = cost_pips(symbol, at)
    frame = pd.DataFrame({"time": at, "agree": agree, "h4": ladder[1], "move1": out[0], "move4": out[1], "cost": cost})
    rows = []
    for variant, col in (("all_four", "agree"), ("h4_only", "h4")):
        x = frame[(frame[col] != 0) & frame.move4.notna() & frame.move1.notna()]
        rows.append(pd.DataFrame({"time": x.time, "variant": variant, "symbol": symbol,
            "gross4": x[col] * x.move4, "net4": x[col] * x.move4 - x.cost, "net1": x[col] * x.move1 - x.cost}))
    rows = pd.concat(rows)
    rows["day"] = rows.time.dt.floor("D")
    share = float((frame.agree != 0).mean())
    daily = rows.groupby(["variant", "day"]).agg(net4=("net4", "sum"), net1=("net1", "sum"),
        gross4=("gross4", "sum"), n=("net4", "size")).reset_index()
    return daily, share


def ladder(workers):
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(ladder_pair, PAIRS))
    daily = pd.concat([p[0] for p in parts]).groupby(["variant", "day"]).sum().reset_index()
    daily["time"] = daily.day
    daily["period"] = period_of(daily.time, PERIODS_DISPLAY)
    result = {"share_hours_all_four_agree": float(np.mean([p[1] for p in parts]))}
    for variant in ("all_four", "h4_only"):
        v = daily[daily.variant == variant]
        per = {}
        for p in PERIODS_DISPLAY:
            x = v[v.period == p]
            n = x.n.sum()
            per[p] = {"signals": int(n), "mean_net4_pips": float(x.net4.sum() / n), "mean_net1_pips": float(x.net1.sum() / n),
                "mean_gross4_pips": float(x.gross4.sum() / n), **boot(x, "net4")}
        result[variant] = per
    a = result["all_four"]
    result["H3_passed"] = all(a[p]["mean_net4_pips"] > 0 and (a[p]["ci_low"] or -1) > 0 for p in PERIODS_DISPLAY)
    write_json(OUT / "ladder_results.json", result)
    return result


# ---------------------------------------------------------------- H4 shock bar
def shock_pair(symbol, data_dir=DATA):
    m = load_hourly(data_dir / f"historical_15m/{symbol}.csv", quarantine=True)
    pip = get_pip_size(symbol)
    o, h, l, c = (m[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    t = m.index.asi8
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h, prev_c) - np.fmin(l, prev_c)
    atr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
    q15 = 15 * 60 * 10**9
    n = len(c)
    contiguous = np.zeros(n, bool)
    contiguous[:n - 16] = (t[16:] - t[:n - 16]) == 16 * q15
    fwd_hi = pd.Series(h[::-1]).rolling(16).max().to_numpy()[::-1]  # bars i..i+15
    fwd_lo = pd.Series(l[::-1]).rolling(16).min().to_numpy()[::-1]
    rng4 = np.full(n, np.nan)
    rng4[:-1] = (fwd_hi[1:] - fwd_lo[1:])  # bars i+1..i+16
    rng4[~contiguous] = np.nan
    slot = m.index.strftime("%H:%M")
    base = (pd.Series(rng4, index=m.index).groupby(slot, group_keys=False)
            .apply(lambda s: s.rolling("60D", closed="left", min_periods=20).median()).reindex(m.index).to_numpy())
    body = c - o
    sig = np.flatnonzero((np.abs(body) >= 3 * atr) & np.isfinite(atr) & (atr > 0) & contiguous & np.isfinite(base))
    side = np.sign(body[sig])
    entry = o[sig + 1]
    cost = cost_pips(symbol, m.index[sig] + pd.Timedelta(minutes=15))
    return pd.DataFrame({"time": m.index[sig] + pd.Timedelta(minutes=15), "symbol": symbol,
        "ratio": rng4[sig] / base[sig], "body_atr": np.abs(body[sig]) / atr[sig],
        "net1": side * (c[sig + 4] - entry) / pip - cost, "net4": side * (c[sig + 16] - entry) / pip - cost,
        "gross4": side * (c[sig + 16] - entry) / pip})


def shock(workers):
    with ProcessPoolExecutor(workers) as pool:
        rows = pd.concat(list(pool.map(shock_pair, PAIRS)), ignore_index=True)
    rows["period"] = period_of(rows.time, PERIODS_DISPLAY)
    result = {}
    for p in PERIODS_DISPLAY:
        x = rows[rows.period == p]
        result[p] = {"signals": int(len(x)), "per_pair_month": float(len(x) / 28 / max(1, x.time.dt.strftime("%Y-%m").nunique())),
            "median_range_ratio": float(x.ratio.median()), "share_ratio_over_1p5": float((x.ratio >= 1.5).mean()),
            "mean_net1_pips": float(x.net1.mean()), "mean_net4_pips": float(x.net4.mean()),
            "mean_gross4_pips": float(x.gross4.mean()), "net4_test": boot(x, "net4"), "net1_test": boot(x, "net1")}
    result["alert_value_passed"] = all(result[p]["median_range_ratio"] >= 1.5 for p in PERIODS_DISPLAY)
    signs = [np.sign(result[p]["net4_test"]["ci_low"] or 0) if (result[p]["net4_test"]["ci_low"] or 0) > 0
             else (-1 if (result[p]["net4_test"]["ci_high"] or 0) < 0 else 0) for p in PERIODS_DISPLAY]
    result["directional"] = bool(len(set(signs)) == 1 and signs[0] != 0)
    write_json(OUT / "shock_results.json", result)
    return result


# ---------------------------------------------------------------- H5 news history line
def news(_workers):
    from research_news_drift import PAIRS as NEWS_PAIRS, load_events
    events = load_events()
    rows = []
    for symbol in sorted({p for v in NEWS_PAIRS.values() for p in v}):
        f = load_hourly(DATA / f"historical_5m/{symbol}.csv", quarantine=True)
        t = f.index.asi8
        for currency, pairs in NEWS_PAIRS.items():
            if symbol not in pairs:
                continue
            e = events[events.currency == currency]
            sign = 1 if symbol.startswith(currency) else -1

            def move(start_min):
                """Close 60 minutes after the release minus the open `start_min` minutes after it."""
                t0 = pd.DatetimeIndex(e.time).asi8 + start_min * 60 * 10**9
                t1 = pd.DatetimeIndex(e.time).asi8 + 55 * 60 * 10**9
                k0, k1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
                ok = (k0 < len(t)) & (k1 < len(t))
                ok[ok] &= (t[k0[ok]] == t0[ok]) & (t[k1[ok]] == t1[ok])
                out = np.full(len(e), np.nan)
                out[ok] = (f.Close.to_numpy()[k1[ok]] - f.Open.to_numpy()[k0[ok]]) * sign
                return np.sign(out)
            rows.append(pd.DataFrame({"time": pd.DatetimeIndex(e.time), "series": e.series.to_numpy(), "name": e.name.to_numpy(),
                "currency": currency, "symbol": symbol, "vote": e.vote.to_numpy(), "outcome": move(0),
                "outcome_after_5min": move(5)}))
    d = pd.concat(rows).dropna(subset=["outcome"])
    d = d[d.outcome != 0].sort_values("time").reset_index(drop=True)
    key = ["series", "symbol", "vote"]
    up = (d.outcome > 0).astype(float)
    d["prior_n"] = d.groupby(key).cumcount()
    d["prior_up"] = up.groupby([d[k] for k in key]).transform(lambda s: s.shift(1).expanding().mean())
    d["prediction"] = np.select([d.prior_up >= 0.65, d.prior_up <= 0.35], [1, -1], 0)
    d.loc[d.prior_n < 8, "prediction"] = 0
    used = d[d.prediction != 0].copy()
    used["hit"] = (used.outcome == used.prediction).astype(float) - 0.5
    used["naive_hit"] = (used.outcome == used.vote).astype(float) - 0.5
    used["period"] = period_of(used.time, PERIODS_DISPLAY)
    d["naive_hit"] = (d.outcome == d.vote).astype(float)
    result = {"events_with_outcome": int(len(d)), "naive_better_up_rate_all": float(d.naive_hit.mean())}
    # Calendar note: how often the currency moved the better/worse way, from the release and from 5 minutes after it.
    d["period"] = period_of(d.time, PERIODS_DISPLAY)
    late = d[d.outcome_after_5min.notna() & (d.outcome_after_5min != 0)]
    result["naive_by_period"] = {p: {"from_release": float(d.loc[d.period == p, "naive_hit"].mean()),
        "from_5min_after": float((late.loc[late.period == p, "outcome_after_5min"]
                                  == late.loc[late.period == p, "vote"]).mean())} for p in PERIODS_DISPLAY}
    for p in PERIODS_DISPLAY:
        x = used[used.period == p]
        result[p] = {"predictions": int(len(x)), "hit_rate": float(x.hit.mean() + 0.5) if len(x) else None,
            "naive_hit_rate_same_events": float(x.naive_hit.mean() + 0.5) if len(x) else None,
            "share_predicting_against_naive": float((x.prediction != x.vote).mean()) if len(x) else None,
            **{f"hit_{k}": v for k, v in boot(x, "hit").items()}}
    result["H5_passed"] = all(result[p]["predictions"] and (result[p]["hit_ci_low"] or -1) > 0
        and result[p]["hit_rate"] > result[p]["naive_hit_rate_same_events"] for p in PERIODS_DISPLAY)
    write_json(OUT / "news_results.json", result)
    return result


# ---------------------------------------------------------------- H2 gold (Arda's own simulator)
GOLD_CONFIGS = {
    "G1_scalp_all_v3000": dict(sigScalp=True, sigTrend=False, sigDonus=False, onlyBest=False, scBrk=True,
        brkBody=0.5, brkBeyond=0.1, exitEma=100, trailStop=True, trailAfterTP=False),
    "G2_strong_trend_v2900": dict(sigScalp=False, sigTrend=True, sigDonus=False, onlyBest=True, trEmaGate=False,
        brkBody=0.5, brkBeyond=0.1, exitEma=100, trailStop=True, trailAfterTP=False),
    "G3_trend_all_v2900": dict(sigScalp=False, sigTrend=True, sigDonus=False, onlyBest=False, trEmaGate=False,
        brkBody=0.5, brkBeyond=0.1, exitEma=100, trailStop=True, trailAfterTP=False),
}


class GoldFrames:
    """Stands in for v29_sim.Sym: server-time bars with a `t` column (seconds) and spread in 0.01 points."""
    name, point = "gold", 0.01

    def __init__(self, m1, spread_by_ny_hour):
        self.cache, self.frames = {}, {}
        srv = m1.index.tz_convert("America/New_York").tz_localize(None) + SERVER_SHIFT
        ny_hour = m1.index.tz_convert("America/New_York").hour
        base = m1.assign(srv=srv, spread=np.asarray(spread_by_ny_hour)[ny_hour] / self.point)
        for tf, rule in (("M5", "5min"), ("M15", "15min"), ("M30", "30min"), ("H1", "1h"), ("H4", "4h")):
            g = base.groupby(base.srv.dt.floor(rule))
            b = g.agg(open=("Open", "first"), high=("High", "max"), low=("Low", "min"), close=("Close", "last"),
                spread=("spread", "first"))
            b["t"] = ((b.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype(np.int64)
            self.frames[tf] = b.reset_index(drop=True)

    def usd_per_unit(self, price):
        return 100.0

    def load(self, tf):
        return self.frames[tf]


def gold(_workers):
    sys.path.insert(0, str(ROOT / "arda_forex/forex-bot-main/research"))
    import v29_sim  # noqa: E402 - Arda's simulator, run unchanged

    m1 = load_hourly(DATA / "historical_gold/XAUUSD_1m.csv")
    spread = pd.read_csv(DATA / "historical_gold/XAUUSD_spread_1m.csv")
    spread.index = pd.to_datetime(spread.pop("timestamp"), unit="ms", utc=True)
    by_hour = spread.spread.groupby(spread.index.tz_convert("America/New_York").hour).median().reindex(range(24))
    by_hour = by_hour.fillna(by_hour.median())
    frames = GoldFrames(m1, by_hour.to_numpy())
    result = {"data": {"first": str(m1.index[0]), "last": str(m1.index[-1]), "bars_1m": int(len(m1)),
        "median_spread_usd_by_ny_hour": by_hour.round(3).tolist()}}
    for name, cfg in GOLD_CONFIGS.items():
        out = v29_sim.run(frames, **cfg)
        out["time"] = pd.to_datetime(out.t, unit="s").dt.tz_localize("UTC") - SERVER_SHIFT  # approx UTC for day buckets
        out["year"] = out.date.dt.year
        unseen = out[out.year <= 2024]
        years = unseen.groupby("year").R.mean()
        test = boot(unseen.assign(net=unseen.R), "net")
        first, second = unseen[unseen.year <= 2022].R.mean(), unseen[unseen.year >= 2023].R.mean()
        ok = (len(unseen) >= 60 and unseen.R.mean() > 0 and (test["ci_low"] or -1) > 0 and first > 0 and second > 0
              and int((years > 0).sum()) >= 4)
        rep = out[out.year >= 2025]
        result[name] = {"unseen": {**describe(unseen.assign(net=unseen.R), "net"), **test,
                "mean_r_2020_2022": float(first), "mean_r_2023_2024": float(second),
                "mean_r_by_year": years.round(4).to_dict(), "passed": bool(ok)},
            "long": describe(unseen[unseen.dir == 1].assign(net=lambda x: x.R), "net"),
            "short": describe(unseen[unseen.dir == -1].assign(net=lambda x: x.R), "net"),
            "replication_2025_2026": describe(rep.assign(net=rep.R), "net") if len(rep) else None}
        out.to_csv(OUT / f"rows/gold_{name}.csv.gz", index=False)
    write_json(OUT / "gold_results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fx", "ladder", "shock", "news", "gold"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    result = {"fx": fx, "ladder": ladder, "shock": shock, "news": news, "gold": gold}[args.step](args.workers)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
