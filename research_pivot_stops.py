"""Daily pivot levels: does price turn back at them more often than at fake levels?

Rules are in research/pivot_stops/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_pivot_stops.py build      # per pair: check the level lookup, then count outcomes per day
    python -X utf8 research_pivot_stops.py evaluate

Per-pair daily counts go to research/pivot_stops/rows (git-ignored); finished pairs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from forex_pivot_plan import atr14
from forex_pivots import LEVEL_NAMES, levels_for_day, pivot_levels
from research_meta_label import load_bars

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "pivot_stops"
ROWS = OUTPUT / "rows"
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")
SHIFTS = (0.0, -0.5, -0.25, 0.25, 0.5)  # x previous day's range; 0 = real level
HORIZON_BARS = 16
MAX_SPAN = pd.Timedelta(hours=4) - pd.Timedelta(minutes=15)
WARMUP_BARS = 20
SUB_PERIODS = {"2008-2015": (2008, 2015), "2016-2023": (2016, 2023), "2024-2026": (2024, 2026)}
CHECK_SAMPLES = 200
BOOTSTRAPS = 2000
WORKERS = 4
BACK, PASS, UNRESOLVED = 1, 0, -1


def day_levels(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Levels valid on each 15M bar's UTC day (previous weekday's daily bar), vectorised."""
    levels = pivot_levels(daily)
    days = bars.index.tz_convert(None).normalize()
    pos = levels.index.searchsorted(days, side="left") - 1
    out = levels.iloc[np.clip(pos, 0, None)].set_axis(bars.index)
    out.loc[pos < 0] = np.nan
    return out


def outcomes(high: np.ndarray, low: np.ndarray, times: np.ndarray, touch: np.ndarray, level: np.ndarray,
             step: np.ndarray, sign: int) -> np.ndarray:
    """BACK / PASS / UNRESOLVED for each touch position; sign +1 = upward touch, -1 = downward."""
    n = len(high)
    offsets = np.arange(HORIZON_BARS)
    idx = touch[:, None] + offsets[None, :]
    valid = idx < n
    idx = np.minimum(idx, n - 1)
    valid &= (times[idx] - times[touch][:, None]) <= MAX_SPAN.to_timedelta64()
    far = level + sign * step
    near = level - sign * step
    if sign > 0:
        passed, back = high[idx] >= far[:, None], low[idx] <= near[:, None]
    else:
        passed, back = low[idx] <= far[:, None], high[idx] >= near[:, None]
    passed &= valid
    back &= valid
    back[:, 0] = False  # on the touch bar the other side may have been reached before the touch
    first_pass = np.where(passed.any(1), passed.argmax(1), HORIZON_BARS)
    first_back = np.where(back.any(1), back.argmax(1), HORIZON_BARS)
    result = np.full(len(touch), UNRESOLVED)
    result[first_back < first_pass] = BACK
    result[first_pass < first_back] = PASS
    return result


def pair_touches(pair: str) -> pd.DataFrame:
    bars, _ = load_bars(pair, "historical_15m")
    daily, _ = load_bars(pair, "historical_1d")
    levels = day_levels(bars, daily)
    check_lookup(pair, bars, daily, levels)

    high, low, close = (bars[c].to_numpy(float) for c in ("High", "Low", "Close"))
    times = bars.index.tz_convert(None).to_numpy()
    prev_close = np.r_[np.nan, close[:-1]]
    gap_ok = np.r_[False, np.diff(times) == np.timedelta64(15, "m")]
    step = np.r_[np.nan, atr14(bars)[:-1]]
    usable = gap_ok & np.isfinite(step) & (step > 0) & (np.arange(len(bars)) >= WARMUP_BARS)
    day = bars.index.tz_convert(None).normalize()
    span = levels["range"].to_numpy(float)

    parts = []
    for name in LEVEL_NAMES:
        base = levels[name].to_numpy(float)
        for shift in SHIFTS:
            level = base + shift * span
            for sign, hit in ((1, (prev_close < level) & (high >= level)), (-1, (prev_close > level) & (low <= level))):
                hit = hit & usable & np.isfinite(level)
                first = hit & (pd.Series(hit).groupby(day).cumsum().to_numpy() == 1)
                touch = np.flatnonzero(first)
                if not len(touch):
                    continue
                result = outcomes(high, low, times, touch, level[touch], step[touch], sign)
                parts.append(pd.DataFrame({"day": day[touch], "level": name, "fake": shift != 0.0,
                                           "side": "up" if sign > 0 else "down", "result": result}))
    touches = pd.concat(parts, ignore_index=True)
    touches.insert(0, "pair", pair)
    return touches


def check_lookup(pair: str, bars: pd.DataFrame, daily: pd.DataFrame, levels: pd.DataFrame) -> None:
    """Vectorised levels must equal forex_pivots.levels_for_day on sampled bars."""
    rng = np.random.default_rng(7)
    for pos in rng.choice(np.arange(WARMUP_BARS, len(bars)), CHECK_SAMPLES, replace=False):
        when = bars.index[pos]
        want = levels_for_day(daily, when.tz_convert(None))
        got = levels.iloc[pos]
        if want is None:
            if got.notna().any():
                raise AssertionError(f"{pair} {when}: vectorised levels where module has none")
            continue
        for name in (*LEVEL_NAMES, "range"):
            if not np.isclose(got[name], want[name], rtol=0, atol=1e-12):
                raise AssertionError(f"{pair} {when} {name}: vectorised {got[name]} != module {want[name]}")


def build_pair(pair: str) -> str:
    target = ROWS / f"{pair}.csv.gz"
    if target.exists():
        return f"{pair}: exists"
    touches = pair_touches(pair)
    touches["back"] = (touches.result == BACK).astype(int)
    touches["passed"] = (touches.result == PASS).astype(int)
    touches["unresolved"] = (touches.result == UNRESOLVED).astype(int)
    daily = touches.groupby(["pair", "day", "level", "fake", "side"], as_index=False)[["back", "passed", "unresolved"]].sum()
    daily.to_csv(target, index=False, compression="gzip")
    return f"{pair}: {len(touches):,} touches, lookup check passed on {CHECK_SAMPLES}"


def safe_build(pair: str) -> str:
    try:
        return build_pair(pair)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the others; a failed check is reported
        return f"{pair}: FAILED {type(exc).__name__}: {exc}"


def rate(frame: pd.DataFrame) -> dict:
    back, passed, unresolved = (int(frame[c].sum()) for c in ("back", "passed", "unresolved"))
    resolved = back + passed
    return {"touches": back + passed + unresolved, "back": back, "passed": passed,
            "reversal_rate": back / resolved if resolved else np.nan,
            "unresolved_share": unresolved / (resolved + unresolved) if resolved + unresolved else np.nan}


def bootstrap_difference(frame: pd.DataFrame, seed: int = 42) -> tuple[float, list[float]]:
    """Reversal rate real - fake, with a day-block bootstrap 95% interval."""
    per_day = frame.pivot_table(index="day", columns="fake", values=["back", "passed"], aggfunc="sum", fill_value=0)
    back_real, back_fake = per_day[("back", False)].to_numpy(float), per_day[("back", True)].to_numpy(float)
    pass_real, pass_fake = per_day[("passed", False)].to_numpy(float), per_day[("passed", True)].to_numpy(float)
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(per_day), len(per_day)), minlength=len(per_day))
                        for _ in range(BOOTSTRAPS)])
    real = (weights @ back_real) / (weights @ (back_real + pass_real))
    fake = (weights @ back_fake) / (weights @ (back_fake + pass_fake))
    point = back_real.sum() / (back_real + pass_real).sum() - back_fake.sum() / (back_fake + pass_fake).sum()
    return float(point), [float(np.quantile(real - fake, 0.025)), float(np.quantile(real - fake, 0.975))]


def evaluate() -> None:
    data = pd.concat([pd.read_csv(f, parse_dates=["day"]) for f in sorted(ROWS.glob("*.csv.gz"))], ignore_index=True)
    year = data.day.dt.year
    diff, ci = bootstrap_difference(data)
    results = {"pairs": int(data.pair.nunique()), "period": [str(data.day.min().date()), str(data.day.max().date())],
               "real": rate(data[~data.fake]), "fake": rate(data[data.fake]),
               "H1": {"difference": diff, "ci95": ci}}
    periods = {}
    for label, (y0, y1) in SUB_PERIODS.items():
        part = data[(year >= y0) & (year <= y1)]
        periods[label] = {"real": rate(part[~part.fake]), "fake": rate(part[part.fake]),
                          "difference": bootstrap_difference(part)[0]}
    results["sub_periods"] = periods
    results["H1"]["passes"] = bool(ci[0] > 0 and all(p["difference"] > 0 for p in periods.values()))
    results["by_level"] = {f"{name} {side}": {"real": rate(g[~g.fake]), "fake": rate(g[g.fake])}
                           for (name, side), g in data.groupby(["level", "side"])}
    results["by_pair"] = {pair: {"real": rate(g[~g.fake])["reversal_rate"], "fake": rate(g[g.fake])["reversal_rate"]}
                          for pair, g in data.groupby("pair")}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def line(label: str, r: dict) -> str:
        return (f"{label:14s} touches {r['touches']:>10,}  döndü %{100 * r['reversal_rate']:.2f}  "
                f"çözülmedi %{100 * r['unresolved_share']:.1f}")
    print(f"pairs {results['pairs']} | {results['period']}")
    print(line("gerçek", results["real"]))
    print(line("sahte", results["fake"]))
    print(f"H1 fark {100 * diff:+.2f} puan, %95 GA [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] -> passes={results['H1']['passes']}")
    for label, p in periods.items():
        print(f"  {label}: gerçek %{100 * p['real']['reversal_rate']:.2f}  sahte %{100 * p['fake']['reversal_rate']:.2f}  "
              f"fark {100 * p['difference']:+.2f}")
    for key, block in results["by_level"].items():
        print(f"  {key:8s} gerçek %{100 * block['real']['reversal_rate']:.2f} (n={block['real']['touches']:,})  "
              f"sahte %{100 * block['fake']['reversal_rate']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "evaluate"])
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.stage == "build":
        ROWS.mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            for message in pool.imap_unordered(safe_build, args.pairs):
                print(message, flush=True)
    else:
        evaluate()
