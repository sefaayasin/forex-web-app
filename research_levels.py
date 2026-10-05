"""Lines drawn from candles: automatic support/resistance levels, round numbers and trendlines on 15M.

Rules are in research/levels/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_levels.py build      # trades per pair -> research/levels/trades (git-ignored); counts only
    python -X utf8 research_levels.py evaluate   # outcomes; runs once

Finished pairs are skipped when build is rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_indicators import compute_atr
from research_meta_label import load_bars, simulate

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "levels"
TRADES = OUTPUT / "trades"
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")

SWING_BARS = 12
WINDOW = 480
CLUSTER_ATR, BUFFER_ATR, MIN_STOP_ATR = 0.5, 0.25, 0.5
GRID_PIPS = 50
LINE_SHIFTS_ATR = (-3.0, -1.5, 1.5, 3.0)
GRID_SHIFTS_PIPS = (10.0, 20.0, 30.0, 40.0)
ROLLOVER_NY_HOUR = 17
STRESS_MULTIPLIER = 2.0
FAMILIES = ("levels", "round", "trendline")
KINDS = ("bounce", "break")
HYPOTHESES = [f"{family}_{kind}" for family in FAMILIES for kind in KINDS]
ALPHA = 0.05 / len(HYPOTHESES)
PERIODS = {"2008-2013": ("2008-01-01", "2014-01-01"), "2014-2019": ("2014-01-01", "2020-01-01"),
           "2020-2026": ("2020-01-01", "2100-01-01")}
MIN_POSITIVE_PAIRS, MIN_TRADES = 15, 1000
BOOTSTRAPS, BLOCK_DAYS, SEED = 2000, 5.0, 42
WORKERS = 14
BOTH, SUPPORT, RESISTANCE = 0, 1, 2
FIFTEEN_NS = np.int64(15 * 60 * 10**9)
FIVE_NS = np.int64(5 * 60 * 10**9)
NS_PER_DAY = 86_400 * 10**9


def chart_arrays(bars: pd.DataFrame) -> dict:
    """15M values known at each bar's close; lo/hi bound the prices a line must lie in to trigger a signal."""
    high, low, close = (bars[c].to_numpy(float) for c in ("High", "Low", "Close"))
    atr = compute_atr(bars, 14).to_numpy()
    buffer = BUFFER_ATR * atr
    prev = np.concatenate([[np.nan], close[:-1]])
    return {"high": high, "low": low, "close": close, "prev": prev, "atr": atr, "b": buffer,
            "lo": np.minimum(low, prev) - buffer, "hi": np.maximum(high, prev) + buffer}


def swing_points(high: np.ndarray, low: np.ndarray, k: int = SWING_BARS) -> pd.DataFrame:
    """Swing highs (kind 1) and lows (kind -1): strictly beyond the k bars before, at least the k bars after."""
    n = len(high)
    frames = []
    for kind, values in ((1, high), (-1, low)):
        v = values * kind
        window_max = np.lib.stride_tricks.sliding_window_view(v, k).max(axis=1)
        left, right = np.full(n, np.inf), np.full(n, np.inf)
        left[k:] = window_max[:n - k]
        right[:n - k] = window_max[1:]
        bars = np.flatnonzero((v > left) & (v >= right))
        frames.append(pd.DataFrame({"bar": bars, "price": values[bars], "kind": kind}))
    swings = pd.concat(frames).sort_values(["bar", "kind"]).reset_index(drop=True)
    swings["usable"] = swings.bar + k + 1
    return swings


def cluster_levels(prices: np.ndarray, tolerance: float) -> np.ndarray:
    """Mean price of every group of at least two swing prices joined by neighbour gaps <= tolerance."""
    p = np.sort(prices)
    groups = np.split(p, np.flatnonzero(np.diff(p) > tolerance) + 1)
    return np.array([g.mean() for g in groups if len(g) >= 2])


def level_segments(swings: pd.DataFrame, atr: np.ndarray, n: int) -> list[tuple[int, int, np.ndarray, float]]:
    """(start, end, levels, ATR at start): the levels active on bars start..end-1.

    A swing at bar i counts on bar t when i + SWING_BARS + 1 <= t <= i + WINDOW, so the set only changes at those bars.
    """
    bars, prices = swings.bar.to_numpy(), swings.price.to_numpy()
    events = np.unique(np.concatenate([bars + SWING_BARS + 1, bars + WINDOW + 1, [n]]))
    events = events[events <= n]
    segments = []
    for start, end in zip(events[:-1], events[1:]):
        lo = np.searchsorted(bars, start - WINDOW)
        hi = np.searchsorted(bars, start - SWING_BARS - 1, side="right")
        if hi - lo < 2 or not np.isfinite(atr[start]):
            continue
        levels = cluster_levels(prices[lo:hi], CLUSTER_ATR * atr[start])
        if len(levels):
            segments.append((int(start), int(end), levels, float(atr[start])))
    return segments


def _rows(ts: list, ys: list, yps: list, roles: list) -> tuple[np.ndarray, ...]:
    if not ts:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int)
    return np.concatenate(ts), np.concatenate(ys), np.concatenate(yps), np.concatenate(roles)


def level_rows(segments: list, shift: float, chart: dict) -> tuple[np.ndarray, ...]:
    """(bar, line value, previous line value, role) for every level within reach of a bar."""
    ts, ys = [], []
    for start, end, levels, atr_start in segments:
        t = np.arange(start, end)
        y = levels + shift * atr_start
        near = (y >= chart["lo"][t, None]) & (y <= chart["hi"][t, None])
        ti, li = np.nonzero(near)
        ts.append(t[ti])
        ys.append(y[li])
    return _rows(ts, ys, ys, [np.full(len(t), BOTH) for t in ts])


def round_rows(chart: dict, pip: float, offset_pips: float) -> tuple[np.ndarray, ...]:
    """Grid levels (multiples of GRID_PIPS, shifted by offset_pips) within reach of each bar."""
    grid, offset = GRID_PIPS * pip, offset_pips * pip
    ok = np.isfinite(chart["lo"]) & np.isfinite(chart["hi"])
    bars = np.flatnonzero(ok)
    k_min = np.ceil((chart["lo"][ok] - offset) / grid).astype(np.int64)
    k_max = np.floor((chart["hi"][ok] - offset) / grid).astype(np.int64)
    count = np.clip(k_max - k_min + 1, 0, None)
    t = np.repeat(bars, count)
    step = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    y = (np.repeat(k_min, count) + step) * grid + offset
    return t, y, y, np.full(len(t), BOTH)


def trendlines(swings: pd.DataFrame, chart: dict) -> list[tuple]:
    """(start, end, anchor bar, anchor price, slope per bar, role, ATR at start) for every clean real trendline."""
    close, buffer, atr = chart["close"], chart["b"], chart["atr"]
    n = len(close)
    lines = []
    for kind, role in ((-1, SUPPORT), (1, RESISTANCE)):
        same = swings[swings.kind == kind]
        bars, prices, usable = same.bar.to_numpy(), same.price.to_numpy(), same.usable.to_numpy()
        for j in range(1, len(bars)):
            b1, b2, p1, p2 = bars[j - 1], bars[j], prices[j - 1], prices[j]
            if p2 == p1 or (p2 > p1) != (role == SUPPORT) or b2 - b1 > WINDOW:
                continue
            start = int(usable[j])
            if start >= n or not np.isfinite(atr[start]):
                continue
            end = int(min(usable[j + 1] if j + 1 < len(bars) else n, b2 + WINDOW + 1, n))
            slope = (p2 - p1) / (b2 - b1)
            span = np.arange(b1, start)
            beyond = (close[span] - (p1 + slope * (span - b1))) * (1 if role == SUPPORT else -1)
            if np.any(beyond < -buffer[span]):
                continue
            lines.append((start, end, int(b1), float(p1), float(slope), role, float(atr[start])))
    return lines


def trendline_rows(lines: list, shift: float, chart: dict) -> tuple[np.ndarray, ...]:
    """Rows for each line shifted by shift x ATR; a line ends at its own first break signal."""
    prev, close, buffer = chart["prev"], chart["close"], chart["b"]
    ts, ys, yps, roles = [], [], [], []
    for start, end, b1, p1, slope, role, atr_start in lines:
        t = np.arange(start, end)
        y = p1 + slope * (t - b1) + shift * atr_start
        y_prev = y - slope
        if role == SUPPORT:
            broken = (prev[t] >= y_prev - buffer[t]) & (close[t] < y - buffer[t])
        else:
            broken = (prev[t] <= y_prev + buffer[t]) & (close[t] > y + buffer[t])
        if broken.any():
            last = int(np.argmax(broken)) + 1
            t, y, y_prev = t[:last], y[:last], y_prev[:last]
        near = (y >= chart["lo"][t]) & (y <= chart["hi"][t])
        ts.append(t[near])
        ys.append(y[near])
        yps.append(y_prev[near])
        roles.append(np.full(int(near.sum()), role))
    return _rows(ts, ys, yps, roles)


def classify(t: np.ndarray, y: np.ndarray, y_prev: np.ndarray, role: np.ndarray, chart: dict) -> pd.DataFrame:
    """Bounce and break signals at the close of bar t, with the stop level each implies."""
    p, c, h, l, b = (chart[k][t] for k in ("prev", "close", "high", "low", "b"))
    support, resistance = role != RESISTANCE, role != SUPPORT
    rules = (
        ("bounce", 1, support & (p > y_prev + b) & (l <= y + b) & (c > y), np.minimum(l, y) - b),
        ("bounce", -1, resistance & (p < y_prev - b) & (h >= y - b) & (c < y), np.maximum(h, y) + b),
        ("break", 1, resistance & (p <= y_prev + b) & (c > y + b), y - b),
        ("break", -1, support & (p >= y_prev - b) & (c < y - b), y + b),
    )
    distance = np.abs(c - y)
    return pd.concat([pd.DataFrame({"t": t[m], "kind": kind, "side": side, "stop_level": stop[m], "dist": distance[m]})
                      for kind, side, m, stop in rules], ignore_index=True)


def pick_signals(candidates: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One signal per bar: skip bars with both directions, otherwise use the line closest to the close."""
    rows = candidates[candidates.kind == kind]
    sides = rows.groupby("t").side.agg(["min", "max"])
    rows = rows[~rows.t.isin(sides.index[sides["min"] != sides["max"]])]
    return rows.sort_values(["t", "dist"], kind="stable").drop_duplicates("t").reset_index(drop=True)


def cost_by_ny_hour(pair: str) -> np.ndarray:
    profile = load_spread_profile()
    spreads = [measured_spread_pips(profile, pair, hour) for hour in range(24)]
    if any(s is None for s in spreads):
        raise ValueError(f"{pair}: spread profile incomplete")
    return np.array(spreads) + ECN_COMMISSION_PIPS


def trade_stream(signals: pd.DataFrame, context: dict) -> tuple[list[dict], dict]:
    """Enter each signal at the next 15M open on the 5M path; one position at a time."""
    m5_ns, opens = context["m5_ns"], context["opens"]
    t = signals.t.to_numpy()
    entry_ns = context["open_ns"][t] + FIFTEEN_NS
    pos = np.searchsorted(m5_ns, entry_ns)
    has_bar = (pos < len(m5_ns)) & (m5_ns[np.minimum(pos, len(m5_ns) - 1)] == entry_ns)
    hours = pd.DatetimeIndex(entry_ns.astype("datetime64[ns]"), tz="UTC").tz_convert("America/New_York").hour.to_numpy()
    skipped = {"no_5m_bar": 0, "rollover_hour": 0, "position_open": 0}
    trades, free_from = [], np.int64(np.iinfo(np.int64).min)
    for i, (bar, side, stop_level) in enumerate(zip(t, signals.side.to_numpy(), signals.stop_level.to_numpy())):
        if not has_bar[i]:
            skipped["no_5m_bar"] += 1
            continue
        if hours[i] == ROLLOVER_NY_HOUR:
            skipped["rollover_hour"] += 1
            continue
        entry = int(pos[i])
        if m5_ns[entry] < free_from:
            skipped["position_open"] += 1
            continue
        entry_price = opens[entry]
        distance = max(side * (entry_price - stop_level), MIN_STOP_ATR * context["atr"][bar])
        exit_price, exit_pos, reason = simulate(opens, context["highs"], context["lows"], context["closes"],
                                                context["m5_times"], entry, int(side), distance)
        free_from = m5_ns[exit_pos] + FIVE_NS
        pip = context["pip"]
        gross, stop_pips, cost = side * (exit_price - entry_price) / pip, distance / pip, context["cost"][hours[i]]
        trades.append({
            "entry_ns": int(m5_ns[entry]), "exit_ns": int(free_from), "side": int(side), "stop_pips": stop_pips,
            "cost_pips": cost, "gross_r": gross / stop_pips, "net_r": (gross - cost) / stop_pips,
            "stress_r": (gross - STRESS_MULTIPLIER * cost) / stop_pips, "reason": reason,
        })
    return trades, skipped


def build_pair(pair: str) -> dict:
    counts_path = TRADES / f"{pair}.json"
    if counts_path.exists():
        return json.loads(counts_path.read_text(encoding="utf-8"))
    bars15, repaired15 = load_bars(pair, "historical_15m")
    bars5, repaired5 = load_bars(pair, "historical_5m")
    chart = chart_arrays(bars15)
    n = len(bars15)
    pip = get_pip_size(pair)
    swings = swing_points(chart["high"], chart["low"])
    segments = level_segments(swings, chart["atr"], n)
    lines = trendlines(swings, chart)
    m5_times = bars5.index.values
    context = {
        "open_ns": bars15.index.values.view("int64"), "m5_times": m5_times, "m5_ns": m5_times.view("int64"),
        "opens": bars5.Open.to_numpy(), "highs": bars5.High.to_numpy(), "lows": bars5.Low.to_numpy(),
        "closes": bars5.Close.to_numpy(), "atr": chart["atr"], "pip": pip, "cost": cost_by_ny_hour(pair),
    }
    streams = [("levels", s, level_rows(segments, s, chart)) for s in (0.0, *LINE_SHIFTS_ATR)]
    streams += [("round", s, round_rows(chart, pip, s)) for s in (0.0, *GRID_SHIFTS_PIPS)]
    streams += [("trendline", s, trendline_rows(lines, s, chart)) for s in (0.0, *LINE_SHIFTS_ATR)]
    frames, stream_counts = [], {}
    for family, shift, rows in streams:
        candidates = classify(*rows, chart)
        for kind in KINDS:
            signals = pick_signals(candidates, kind)
            trades, skipped = trade_stream(signals, context)
            name = f"{family}_{kind}"
            stream_counts[f"{name}|{shift:g}"] = {"signals": int(len(signals)), "trades": len(trades), **skipped}
            if trades:
                frames.append(pd.DataFrame(trades).assign(pair=pair, hypothesis=name, shift=shift, placebo=shift != 0))
    pd.concat(frames, ignore_index=True).to_csv(TRADES / f"{pair}.csv.gz", index=False, compression="gzip")
    level_counts = np.array([len(levels) for _, _, levels, _ in segments])
    level_bars = np.array([end - start for start, end, _, _ in segments])
    real_rows = level_rows(segments, 0.0, chart)[0]
    counts = {
        "pair": pair, "bars_15m": int(n), "bars_5m": int(len(bars5)), "ohlc_repaired": {"15m": repaired15, "5m": repaired5},
        "swings": int(len(swings)), "mean_active_levels": float((level_counts * level_bars).sum() / max(n, 1)),
        "share_of_bars_with_level_in_reach": float(len(np.unique(real_rows)) / max(n, 1)),
        "trendlines": {"support": sum(1 for line in lines if line[5] == SUPPORT),
                       "resistance": sum(1 for line in lines if line[5] == RESISTANCE)},
        "streams": stream_counts,
    }
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def safe_build(pair: str) -> dict:
    try:
        return build_pair(pair)
    except Exception as error:  # report and continue with the other pairs
        return {"pair": pair, "error": repr(error)}


def run_build() -> None:
    TRADES.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS) as pool:
        results = pool.map(safe_build, PAIRS)
    errors = {r["pair"]: r["error"] for r in results if "error" in r}
    if errors:
        print(json.dumps(errors, indent=2))
        raise SystemExit("build incomplete; rerun to finish the failed pairs")
    totals = {}
    for result in results:
        for key, value in result["streams"].items():
            name, shift = key.split("|")
            group = totals.setdefault(name, {"real": 0, "placebo": 0})
            group["real" if float(shift) == 0 else "placebo"] += value["trades"]
    summary = {
        "protocol_sha256": hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest(),
        "pairs": len(results), "trades_by_hypothesis": totals,
        "per_pair": {r["pair"]: {k: v for k, v in r.items() if k != "pair"} for r in results},
    }
    (OUTPUT / "build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['pair']}: swings {r['swings']}, active levels {r['mean_active_levels']:.1f}, "
              f"bars near a level {r['share_of_bars_with_level_in_reach']:.1%}, trendlines {r['trendlines']}")
    print(json.dumps(totals, indent=2))


def stationary_indices(n: int, simulations: int = BOOTSTRAPS, block: float = BLOCK_DAYS, seed: int = SEED) -> np.ndarray:
    """Day positions for each resample: blocks of consecutive days (wrapping) with mean length `block`."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(simulations, n))
    restart = rng.random((simulations, n)) < 1.0 / block
    restart[:, 0] = True
    position = np.arange(n)
    last = np.maximum.accumulate(np.where(restart, position, 0), axis=1)
    return ((np.take_along_axis(starts, last, axis=1) + position - last) % n).astype(np.int32)


def day_totals(frame: pd.DataFrame, column: str, day_pos: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.bincount(day_pos, weights=frame[column].to_numpy(), minlength=n)
    return sums, np.bincount(day_pos, minlength=n).astype(float)


def resampled_means(sums: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return sums[indices].sum(axis=1) / counts[indices].sum(axis=1)


def interval(values: np.ndarray) -> dict:
    return {"ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5)),
            "p_at_or_below_zero": float(np.mean(values <= 0))}


def describe(trades: pd.DataFrame) -> dict:
    stops = trades.reason.isin(["stop", "stop_gap"])
    cost_share = trades.cost_pips / (trades.stop_pips + trades.cost_pips)
    return {
        "trades": int(len(trades)), "win_rate": float((trades.net_r > 0).mean()), "stop_rate": float(stops.mean()),
        "mean_gross_r": float(trades.gross_r.mean()), "mean_net_r": float(trades.net_r.mean()),
        "mean_stress_r": float(trades.stress_r.mean()), "median_stop_pips": float(trades.stop_pips.median()),
        "median_cost_share_of_risk": float(cost_share.median()),
        "exit_reasons": {k: int(v) for k, v in trades.reason.value_counts().items()},
    }


def epoch_day(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").value // NS_PER_DAY)


def load_trades() -> pd.DataFrame:
    """All trades with the columns evaluate needs; `day` is the UTC entry day as days since 1970 (saves memory)."""
    columns = ["entry_ns", "stop_pips", "cost_pips", "gross_r", "net_r", "stress_r", "reason", "hypothesis", "shift"]
    frames = []
    for pair in PAIRS:
        frame = pd.read_csv(TRADES / f"{pair}.csv.gz", usecols=columns,
                            dtype={"reason": "category", "hypothesis": "category"})
        frame["day"] = (frame.pop("entry_ns") // NS_PER_DAY).astype(np.int32)
        frame["placebo"] = frame.pop("shift") != 0
        frame["pair"] = pair
        frames.append(frame)
    trades = pd.concat(frames, ignore_index=True)
    for column in ("reason", "hypothesis", "pair"):
        trades[column] = trades[column].astype("category")
    return trades


def evaluate_hypothesis(real: pd.DataFrame, placebo: pd.DataFrame, day_index: np.ndarray, indices: np.ndarray) -> dict:
    n = len(day_index)
    real_pos, placebo_pos = np.searchsorted(day_index, real.day), np.searchsorted(day_index, placebo.day)
    boot = {}
    for column in ("net_r", "gross_r"):
        r_sum, r_cnt = day_totals(real, column, real_pos, n)
        p_sum, p_cnt = day_totals(placebo, column, placebo_pos, n)
        boot[column] = (resampled_means(r_sum, r_cnt, indices), resampled_means(p_sum, p_cnt, indices))
    periods = {name: float(real[(real.day >= epoch_day(a)) & (real.day < epoch_day(b))].net_r.mean())
               for name, (a, b) in PERIODS.items()}
    by_pair = real.groupby("pair", observed=True).net_r.mean()
    summary = {
        "real": describe(real), "placebo": describe(placebo),
        "net_r_test": {"mean": float(real.net_r.mean()), **interval(boot["net_r"][0])},
        "net_r_minus_placebo": {"difference": float(real.net_r.mean() - placebo.net_r.mean()),
                                **interval(boot["net_r"][0] - boot["net_r"][1])},
        "gross_r_minus_placebo_info": {"difference": float(real.gross_r.mean() - placebo.gross_r.mean()),
                                       **interval(boot["gross_r"][0] - boot["gross_r"][1])},
        "net_r_by_period": periods,
        "pairs_positive": int((by_pair > 0).sum()), "pairs": int(len(by_pair)),
    }
    summary["checks"] = {
        "mean_positive_significant": bool(summary["net_r_test"]["mean"] > 0 and summary["net_r_test"]["p_at_or_below_zero"] <= ALPHA),
        "beats_placebo_significant": bool(summary["net_r_minus_placebo"]["difference"] > 0
                                          and summary["net_r_minus_placebo"]["p_at_or_below_zero"] <= ALPHA),
        "positive_at_stress_cost": bool(summary["real"]["mean_stress_r"] > 0),
        "positive_in_every_period": bool(all(v > 0 for v in periods.values())),
        "positive_in_15_of_28_pairs": bool(summary["pairs_positive"] >= MIN_POSITIVE_PAIRS),
        "at_least_1000_trades": bool(len(real) >= MIN_TRADES),
    }
    summary["passes"] = all(summary["checks"].values())
    return summary


def run_evaluate() -> None:
    if (OUTPUT / "results.json").exists():
        raise ValueError("Evaluation already ran; results are not recomputed or tuned")
    build = json.loads((OUTPUT / "build.json").read_text(encoding="utf-8"))
    protocol_hash = hashlib.sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()
    if build["protocol_sha256"] != protocol_hash:
        raise ValueError("protocol.json changed after build")
    trades = load_trades()
    day_index = np.unique(trades.day.to_numpy())
    indices = stationary_indices(len(day_index))
    results = {}
    for name in HYPOTHESES:
        group = trades[trades.hypothesis == name]
        results[name] = evaluate_hypothesis(group[~group.placebo], group[group.placebo], day_index, indices)
    real = trades[~trades.placebo]
    year = pd.to_datetime(real.day.astype(np.int64), unit="D").dt.year.rename("year")
    real.groupby([real.hypothesis, year], observed=True).net_r.agg(["count", "mean"]).unstack(0).to_csv(OUTPUT / "by_year.csv")
    real.groupby(["hypothesis", "pair"], observed=True).net_r.agg(["count", "mean"]).unstack(0).to_csv(OUTPUT / "by_pair.csv")
    output = {"protocol_sha256": protocol_hash, "alpha_per_hypothesis": ALPHA, "days": int(len(day_index)),
              "hypotheses": results, "any_passes": any(r["passes"] for r in results.values())}
    (OUTPUT / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, r in results.items():
        print(f"{name:18s} trades {r['real']['trades']:7d}  net {r['real']['mean_net_r']:+.3f}R  "
              f"placebo {r['placebo']['mean_net_r']:+.3f}R  gross {r['real']['mean_gross_r']:+.3f}R vs "
              f"{r['placebo']['mean_gross_r']:+.3f}R  passes {r['passes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "evaluate"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    {"build": run_build, "evaluate": run_evaluate}[args.stage]()
