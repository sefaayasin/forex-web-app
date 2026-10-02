"""Do the Özet card's freshness and reversal lines predict the next move?

Rules are in research/freshness/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_freshness.py build      # per pair: check the vectorised rules, then aggregate outcomes
    python -X utf8 research_freshness.py evaluate

Per-pair daily aggregates go to research/freshness/rows (git-ignored); finished pairs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import ast
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from forex_analysis import market_structure_frame
from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_freshness import (
    FRESH_MAX_RATIO, LATE_MIN_RATIO, RSI_EXTREME, SIDE_THRESHOLD, STRETCH_LATE_ATR, WARNING_LOOKBACK_BARS,
    reversal_level, reversal_warnings, signal_freshness,
)
from forex_indicators import add_indicators
from research_meta_label import load_bars

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "freshness"
ROWS = OUTPUT / "rows"
APP = ROOT / "forex_web_app_streamlit_v14_alert_decision.py"
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")
HORIZONS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
SUB_PERIODS = {"2008-2015": (2008, 2015), "2016-2023": (2016, 2023), "2024-2026": (2024, 2026)}
WARMUP_15M_BARS = 500  # indicators (EMA200, 96-bar typical range) settle before the first decision
CHUNK_WARMUP_5M_BARS = 3000  # 5M frames are built year by year with this much history in front
CHECK_SAMPLES = 300
BOOTSTRAPS = 2000
CI_LEVEL = 0.9875  # two-sided, Bonferroni 0.05 / 4
WORKERS = 4
STATES = {0: "Taze", 1: "Olgun", 2: "Geç"}
LEVELS = {0: "Yok", 1: "Tek tük", 2: "Var"}


def load_score_function():
    """score_series_for_backtest compiled from the app, exactly as the live card uses it."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    namespace = {"np": np, "pd": pd, "add_indicators": add_indicators}
    nodes = [n for n in tree.body if getattr(n, "name", None) in {"score_series_for_backtest", "_utc_index_series"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace["score_series_for_backtest"]


SCORE = load_score_function()


def structure_with_score(bars: pd.DataFrame) -> pd.DataFrame:
    frame = market_structure_frame(bars)
    frame["Score"] = SCORE(bars).reindex(frame.index)
    return frame


def warning_counts(frame: pd.DataFrame, sign: int) -> pd.Series:
    """Number of forex_freshness.reversal_warnings for `sign` at every bar, vectorised."""
    against = "BEARISH" if sign > 0 else "BULLISH"
    pattern = "M_CONFIRMED" if sign > 0 else "W_CONFIRMED"
    weakening = "BULLISH_WEAKENING" if sign > 0 else "BEARISH_WEAKENING"

    def recent(mask: pd.Series) -> pd.Series:
        return mask.astype(float).rolling(WARNING_LOOKBACK_BARS, min_periods=1).max().astype(int)

    rsi, score = frame["RSI14"].astype(float), frame["Score"].astype(float)
    return (
        recent(frame["RSIDivergence"].astype(str) == against)
        + recent(frame["MACDDivergence"].astype(str) == against)
        + recent(frame["BBPattern"].astype(str) == pattern)
        + (frame["MACDMomentumState"].astype(str) == weakening).astype(int)
        + (rsi.notna() & (sign * (rsi - 50.0) >= RSI_EXTREME - 50.0)).astype(int)
        + (score.notna() & (sign * score <= -SIDE_THRESHOLD)).astype(int)
    )


def freshness_states(frame: pd.DataFrame, pip: float) -> pd.DataFrame:
    """Side, freshness state and its inputs at every 15M bar, vectorised from forex_freshness.signal_freshness."""
    score = frame["Score"].dropna()
    bars = frame.loc[score.index]
    close = bars["Close"].to_numpy(float)
    rng = bars["High"].rolling(16).max() - bars["Low"].rolling(16).min()
    typical = (rng.rolling(96).median() / pip).to_numpy(float)
    atr, ema20 = bars["ATR14"].to_numpy(float), bars["EMA20"].to_numpy(float)
    s = score.to_numpy(float)
    side = np.where(s >= SIDE_THRESHOLD, 1, np.where(s <= -SIDE_THRESHOLD, -1, 0))
    positions = np.arange(len(s))
    start = np.zeros(len(s), dtype=int)
    for sign in (1, -1):
        off = (sign * s) < SIDE_THRESHOLD
        last_off = np.maximum.accumulate(np.where(off, positions, -1))
        start = np.where(side == sign, last_off + 1, start)
    moved = side * (close - close[start]) / pip
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(typical > 0, moved / typical, np.nan)
        stretch = np.where(atr > 0, side * (close - ema20) / atr, np.nan)
    late = (ratio >= LATE_MIN_RATIO) | (stretch >= STRETCH_LATE_ATR)
    state = np.where(late, 2, np.where(ratio >= FRESH_MAX_RATIO, 1, 0))
    return pd.DataFrame({"side": side, "state": state, "start": start, "moved": moved, "typical": typical,
                         "score": s}, index=score.index)


def five_minute_counts(bars_5m: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Warning counts on 5M for both sides, built year by year; index = bar close time. Also the yearly frames."""
    parts, frames = [], {}
    years = bars_5m.index.year
    for year in np.unique(years):
        first = int(np.searchsorted(years, year))
        last = int(np.searchsorted(years, year, side="right"))
        chunk = bars_5m.iloc[max(first - CHUNK_WARMUP_5M_BARS, 0):last]
        frame = structure_with_score(chunk)
        keep = frame.index.year == year
        parts.append(pd.DataFrame({"long": warning_counts(frame, 1)[keep], "short": warning_counts(frame, -1)[keep]}))
        frames[int(year)] = frame
    counts = pd.concat(parts)
    counts.index = counts.index + pd.Timedelta(minutes=5)
    return counts, frames


def check_against_module(pair: str, frame15: pd.DataFrame, states: pd.DataFrame, c15: pd.DataFrame,
                         frames5: dict[int, pd.DataFrame], moments: pd.DataFrame, pip: float) -> None:
    """Every sampled moment must give the same state, warning counts and level as forex_freshness itself."""
    rng = np.random.default_rng(7)
    sample = moments.iloc[np.sort(rng.choice(len(moments), min(CHECK_SAMPLES, len(moments)), replace=False))]
    for when, row in sample.iterrows():
        side = "LONG" if row.side > 0 else "SHORT"
        upto = frame15.loc[:when]
        app_range = upto["High"].rolling(16).max() - upto["Low"].rolling(16).min()
        app_typical = float(app_range.dropna().tail(96).median() / pip)  # build_intraday_opportunity's formula
        if not np.isclose(app_typical, row.typical, rtol=1e-9):
            raise AssertionError(f"{pair} {when}: typical move {row.typical} != app {app_typical}")
        fresh = signal_freshness(upto, upto["Score"], side, pip, row.typical)
        expected_state = {"FRESH": 0, "MATURE": 1, "LATE": 2}[fresh["state"]]
        w15 = reversal_warnings(upto, side)
        frame5 = frames5[(when + pd.Timedelta(minutes=10)).year]
        w5 = reversal_warnings(frame5.loc[:when + pd.Timedelta(minutes=10)], side)
        expected_level = {"NONE": 0, "WEAK": 1, "STRONG": 2}[reversal_level(w15, w5)]
        got = (int(row.state), int(row.c15), int(row.c5), int(row.level))
        want = (expected_state, len(w15), len(w5), expected_level)
        if got != want:
            raise AssertionError(f"{pair} {when}: vectorised {got} != module {want}")


def build_pair(pair: str) -> str:
    target = ROWS / f"{pair}.csv.gz"
    if target.exists():
        return f"{pair}: exists"
    pip = get_pip_size(pair)
    bars15, _ = load_bars(pair, "historical_15m")
    bars5, _ = load_bars(pair, "historical_5m")
    frame15 = structure_with_score(bars15)
    states = freshness_states(frame15, pip).iloc[WARMUP_15M_BARS:]
    c15 = pd.DataFrame({"long": warning_counts(frame15, 1), "short": warning_counts(frame15, -1)})
    c5, frames5 = five_minute_counts(bars5)

    moments = states[states.side != 0].copy()
    decision = moments.index + pd.Timedelta(minutes=15)
    moments["c15"] = np.where(moments.side > 0, c15["long"].reindex(moments.index), c15["short"].reindex(moments.index))
    c5_at = c5.reindex(decision, method="ffill")
    moments["c5"] = np.where(moments.side > 0, c5_at["long"].to_numpy(), c5_at["short"].to_numpy())
    moments = moments[pd.notna(moments.c5)]
    moments["level"] = np.where((moments.c15 >= 2) & (moments.c5 >= 1), 2, np.where(moments.c15 + moments.c5 >= 1, 1, 0))
    check_against_module(pair, frame15, states, c15, frames5, moments, pip)

    opens = bars5["Open"]
    decision = moments.index + pd.Timedelta(minutes=15)
    entry_pos = opens.index.get_indexer(decision)
    has_entry = entry_pos >= 0
    moments, decision, entry_pos = moments[has_entry], decision[has_entry], entry_pos[has_entry]
    entry = opens.to_numpy()[entry_pos]
    profile = load_spread_profile()
    ny_hour = decision.tz_convert("America/New_York").hour
    spread = pd.Series([measured_spread_pips(profile, pair, h) for h in ny_hour], dtype=float).fillna(1.5).to_numpy()
    out = pd.DataFrame({"day": decision.floor("D"), "state": moments.state.to_numpy(), "level": moments.level.to_numpy(),
                        "high_score": (moments.score.abs() >= 90).to_numpy()})
    cost = spread + ECN_COMMISSION_PIPS
    for name, horizon in HORIZONS.items():
        exit_pos = opens.index.searchsorted(decision + horizon, side="right") - 1
        move = moments.side.to_numpy() * (opens.to_numpy()[exit_pos] - entry) / pip
        out[f"move_{name}"] = move
        out[f"net_{name}"] = move - cost
        out[f"hit_{name}"] = (move > 0).astype(int)
    out["n"] = 1
    daily = out.groupby(["day", "state", "level", "high_score"], as_index=False).sum()
    daily.insert(0, "pair", pair)
    daily.to_csv(target, index=False, compression="gzip")
    return f"{pair}: {len(out):,} moments, check passed on {min(CHECK_SAMPLES, len(moments))}"


def safe_build(pair: str) -> str:
    try:
        return build_pair(pair)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the others; a failed check is reported
        return f"{pair}: FAILED {type(exc).__name__}: {exc}"


def bootstrap_weights(days: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(BOOTSTRAPS)])


def group_stats(daily: pd.DataFrame, mask: pd.Series, columns: list[str], weights: np.ndarray, days: pd.Index) -> dict:
    """Group means and their bootstrap draws (rows = resamples), from per-day sums."""
    part = daily[mask].groupby("day")[columns + ["n"]].sum().reindex(days, fill_value=0)
    sums, n = part[columns].to_numpy(float), part["n"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (weights @ sums) / (weights @ n)[:, None]
    return {"n": int(n.sum()), "mean": sums.sum(0) / n.sum() if n.sum() else np.full(len(columns), np.nan), "draws": draws}


def evaluate() -> None:
    files = sorted(ROWS.glob("*.csv.gz"))
    daily = pd.concat([pd.read_csv(f, parse_dates=["day"]) for f in files], ignore_index=True)
    columns = [f"{kind}_{h}" for kind in ("move", "net", "hit") for h in HORIZONS]
    tail = (1 - CI_LEVEL) / 2
    results = {"pairs": int(daily.pair.nunique()), "period": [str(daily.day.min().date()), str(daily.day.max().date())],
               "moments": int(daily.n.sum())}

    def compare(subset: pd.Series, a: pd.Series, b: pd.Series) -> dict:
        days = pd.Index(np.sort(daily.loc[subset, "day"].unique()))
        weights = bootstrap_weights(days.to_numpy())
        ga = group_stats(daily, subset & a, columns, weights, days)
        gb = group_stats(daily, subset & b, columns, weights, days)
        out = {"n_a": ga["n"], "n_b": gb["n"]}
        for i, col in enumerate(columns):
            diff = ga["draws"][:, i] - gb["draws"][:, i]
            out[col] = {"a": float(ga["mean"][i]), "b": float(gb["mean"][i]), "diff": float(ga["mean"][i] - gb["mean"][i]),
                        "ci": [float(np.nanquantile(diff, tail)), float(np.nanquantile(diff, 1 - tail))]}
        return out

    every = pd.Series(True, index=daily.index)
    year = daily.day.dt.year
    tests = {"H1_fresh_beats_late": (daily.state == 0, daily.state == 2),
             "H2_reversal_warning_precedes_worse_move": (daily.level == 0, daily.level == 2)}
    for name, (a, b) in tests.items():
        block = {"all": compare(every, a, b), "high_score_only": compare(daily.high_score.astype(bool), a, b)}
        for label, (y0, y1) in SUB_PERIODS.items():
            block[label] = compare((year >= y0) & (year <= y1), a, b)
        full = block["all"]
        block["passes"] = bool(
            all(full[f"move_{h}"]["ci"][0] > 0 for h in HORIZONS)
            and all(block[label]["move_4h"]["diff"] > 0 for label in SUB_PERIODS)
        )
        results[name] = block

    def group_table(column: str, names: dict) -> dict:
        table = {}
        for code, label in names.items():
            part = daily[daily[column] == code]
            n = part.n.sum()
            table[label] = {"n": int(n), **{col: float(part[col].sum() / n) for col in columns}}
        return table

    results["info_by_state"] = group_table("state", STATES)
    results["info_by_level"] = group_table("level", LEVELS)
    results["median_cost_note"] = "net = move - (measured spread at NY hour + 0.7 pip)"
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"pairs {results['pairs']} | {results['period']} | moments {results['moments']:,}")
    for title, table in (("freshness", results["info_by_state"]), ("reversal", results["info_by_level"])):
        print(f"--- {title}")
        for label, row in table.items():
            print(f"  {label:8s} n={row['n']:>10,}  move 1h {row['move_1h']:+.2f}  4h {row['move_4h']:+.2f}  "
                  f"net 1h {row['net_1h']:+.2f}  4h {row['net_4h']:+.2f}  hit 1h {row['hit_1h']:.3f}  4h {row['hit_4h']:.3f}")
    for name in tests:
        block = results[name]
        print(f"=== {name}: passes={block['passes']}")
        for label in ["all", "high_score_only", *SUB_PERIODS]:
            parts = [f"{h} {block[label][f'move_{h}']['diff']:+.3f} [{block[label][f'move_{h}']['ci'][0]:+.3f}, "
                     f"{block[label][f'move_{h}']['ci'][1]:+.3f}]" for h in HORIZONS]
            print(f"  {label:16s} " + "  ".join(parts))


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
