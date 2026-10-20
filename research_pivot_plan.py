"""Pivot trader plan (forex_pivot_plan): if the user followed every plan the Özet box shows, what happened?

Rules are in research/pivot_plan/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_pivot_plan.py build      # per pair: every plan, real and fake levels -> rows/<PAIR>.csv.gz
    python -X utf8 research_pivot_plan.py evaluate

Rows go to research/pivot_plan/rows (git-ignored); finished pairs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_pivot_plan import atr14, pivot_plan
from forex_pivots import LEVEL_NAMES
from research_meta_label import load_bars
from research_pivot_stops import day_levels

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "pivot_plan"
ROWS = OUTPUT / "rows"
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")
DECISION_HOURS = (0, 4, 8, 12, 16, 20)
VARIANTS = {"real": 0.0, "fake_down": -0.25, "fake_up": 0.25}
FILL_WINDOW = pd.Timedelta(hours=4)
SUB_PERIODS = {"2008-2015": (2008, 2015), "2016-2023": (2016, 2023), "2024-2026": (2024, 2026)}
BOOTSTRAPS = 2000
CI_LEVEL = 0.975
WORKERS = 4


def simulate(high: np.ndarray, low: np.ndarray, close: np.ndarray, times: np.ndarray, last_of_day: np.ndarray,
             decision: int, plan: dict) -> dict | None:
    """Fill within 4 hours, then stop / target / end of the fill's UTC day. None if never filled."""
    sign = 1 if plan["side"] == "LONG" else -1
    entry, stop, target = plan["entry"], plan["stop"], plan["target"]
    n = len(high)
    fill = None
    j = decision + 1
    while j < n and times[j] - times[decision] <= FILL_WINDOW.to_timedelta64():
        if (sign > 0 and low[j] <= entry) or (sign < 0 and high[j] >= entry):
            fill = j
            break
        j += 1
    if fill is None:
        return None
    for j in range(fill, last_of_day[fill] + 1):
        stop_hit = low[j] <= stop if sign > 0 else high[j] >= stop
        target_hit = j > fill and (high[j] >= target if sign > 0 else low[j] <= target)
        if stop_hit:
            return {"fill": fill, "exit": j, "kind": "stop", "price": stop}
        if target_hit:
            return {"fill": fill, "exit": j, "kind": "target", "price": target}
    end = last_of_day[fill]
    return {"fill": fill, "exit": end, "kind": "day_end", "price": close[end]}


def build_pair(pair: str) -> str:
    target_file = ROWS / f"{pair}.csv.gz"
    if target_file.exists():
        return f"{pair}: exists"
    pip = get_pip_size(pair)
    bars, _ = load_bars(pair, "historical_15m")
    daily, _ = load_bars(pair, "historical_1d")
    close_times = bars.index + pd.Timedelta(minutes=15)
    levels = day_levels(bars.set_axis(close_times), daily)  # levels of the decision's (close time's) UTC day
    high, low, close = (bars[c].to_numpy(float) for c in ("High", "Low", "Close"))
    times = bars.index.tz_convert(None).to_numpy()
    day = bars.index.tz_convert(None).normalize().to_numpy()
    last_of_day = np.r_[np.flatnonzero(day[1:] != day[:-1]), len(day) - 1]
    last_of_day = last_of_day[np.searchsorted(last_of_day, np.arange(len(day)))]
    atr = atr14(bars)
    decisions = np.flatnonzero(close_times.hour.isin(DECISION_HOURS) & (close_times.minute == 0)
                               & (close_times.dayofweek < 5) & np.isfinite(atr) & levels["P"].notna().to_numpy())
    profile = load_spread_profile()

    level_values = {name: levels[name].to_numpy(float) for name in (*LEVEL_NAMES, "range")}
    rows = []
    for variant, shift in VARIANTS.items():
        busy_until = -1
        for pos in decisions:
            if pos <= busy_until:
                continue
            shifted = {name: level_values[name][pos] + shift * level_values["range"][pos] for name in LEVEL_NAMES}
            plan = pivot_plan(float(close[pos]), shifted, float(atr[pos]))
            record = {"day": close_times[pos].tz_convert(None).normalize(), "variant": variant, "side": plan["side"],
                      "filled": 0, "kind": "", "gross_pips": np.nan, "cost_pips": np.nan, "risk_pips": np.nan}
            if plan["side"] != "WAIT":
                result = simulate(high, low, close, times, last_of_day, pos, plan)
                sign = 1 if plan["side"] == "LONG" else -1
                if result is None:
                    busy_until = pos + 16
                else:
                    busy_until = result["exit"]
                    ny_hour = bars.index[result["fill"]].tz_convert("America/New_York").hour
                    spread = measured_spread_pips(profile, pair, ny_hour)
                    record.update(filled=1, kind=result["kind"],
                                  gross_pips=sign * (result["price"] - plan["entry"]) / pip,
                                  cost_pips=(1.5 if spread is None or not np.isfinite(spread) else spread) + ECN_COMMISSION_PIPS,
                                  risk_pips=abs(plan["entry"] - plan["stop"]) / pip)
            rows.append(record)
    out = pd.DataFrame(rows)
    out.insert(0, "pair", pair)
    out.to_csv(target_file, index=False, compression="gzip")
    real = out[(out.variant == "real") & (out.filled == 1)]
    return f"{pair}: {len(out[out.variant == 'real']):,} real plans, {len(real):,} filled"


def safe_build(pair: str) -> str:
    try:
        return build_pair(pair)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the others
        return f"{pair}: FAILED {type(exc).__name__}: {exc}"


def day_bootstrap(frame: pd.DataFrame, column: str, by: str | None = None, seed: int = 42) -> tuple[float, list[float]]:
    """Mean of `column` per filled plan (or real - fake difference when by='variant'), day-block bootstrap."""
    tail = (1 - CI_LEVEL) / 2
    rng = np.random.default_rng(seed)
    if by is None:
        per_day = frame.groupby("day").agg(s=(column, "sum"), n=(column, "size"))
        idx = rng.integers(0, len(per_day), (BOOTSTRAPS, len(per_day)))
        s, n = per_day.s.to_numpy(), per_day.n.to_numpy()
        draws = s[idx].sum(1) / n[idx].sum(1)
        return float(s.sum() / n.sum()), [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]
    real = frame[frame.variant == "real"].groupby("day")[column].agg(["sum", "size"])
    fake = frame[frame.variant != "real"].groupby("day")[column].agg(["sum", "size"])
    both = real.join(fake, how="outer", lsuffix="_r", rsuffix="_f").fillna(0)
    idx = rng.integers(0, len(both), (BOOTSTRAPS, len(both)))
    arr = {c: both[c].to_numpy() for c in both.columns}
    draws = arr["sum_r"][idx].sum(1) / arr["size_r"][idx].sum(1) - arr["sum_f"][idx].sum(1) / arr["size_f"][idx].sum(1)
    point = arr["sum_r"].sum() / arr["size_r"].sum() - arr["sum_f"].sum() / arr["size_f"].sum()
    return float(point), [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]


def summary(frame: pd.DataFrame) -> dict:
    filled = frame[frame.filled == 1]
    kinds = filled.kind.value_counts(normalize=True)
    return {"plans": int(len(frame)), "wait_share": float((frame.side == "WAIT").mean()),
            "fill_rate": float(len(filled) / max((frame.side != "WAIT").sum(), 1)), "filled": int(len(filled)),
            "target_share": float(kinds.get("target", 0)), "stop_share": float(kinds.get("stop", 0)),
            "day_end_share": float(kinds.get("day_end", 0)),
            "gross_pips": float(filled.gross_pips.mean()), "net_pips": float((filled.gross_pips - filled.cost_pips).mean()),
            "gross_R": float(filled.gross_R.mean()), "net_R": float(filled.net_R.mean()),
            "median_risk_pips": float(filled.risk_pips.median()), "median_cost_pips": float(filled.cost_pips.median())}


def evaluate() -> None:
    data = pd.concat([pd.read_csv(f, parse_dates=["day"]) for f in sorted(ROWS.glob("*.csv.gz"))], ignore_index=True)
    data["gross_R"] = data.gross_pips / data.risk_pips
    data["net_R"] = (data.gross_pips - data.cost_pips) / data.risk_pips
    filled = data[data.filled == 1]
    real = filled[filled.variant == "real"]
    year = filled.day.dt.year
    h1, h1_ci = day_bootstrap(real, "net_R")
    h2, h2_ci = day_bootstrap(filled, "gross_R", by="variant")
    periods = {}
    for label, (y0, y1) in SUB_PERIODS.items():
        part = filled[(year >= y0) & (year <= y1)]
        part_real = part[part.variant == "real"]
        periods[label] = {"real": summary(data[(data.variant == "real") & data.day.dt.year.between(y0, y1)]),
                          "H1_net_R": float(part_real.net_R.mean()),
                          "H2_gross_R_diff": float(part_real.gross_R.mean() - part[part.variant != "real"].gross_R.mean())}
    results = {
        "pairs": int(data.pair.nunique()), "period": [str(data.day.min().date()), str(data.day.max().date())],
        "real": summary(data[data.variant == "real"]), "fake": summary(data[data.variant != "real"]),
        "by_side": {side: summary(data[(data.variant == "real") & (data.side == side)]) for side in ("LONG", "SHORT")},
        "H1_plan_pays_after_costs": {"net_R": h1, "ci": h1_ci,
                                     "passes": bool(h1_ci[0] > 0 and all(p["H1_net_R"] > 0 for p in periods.values()))},
        "H2_real_levels_beat_fake": {"gross_R_diff": h2, "ci": h2_ci,
                                     "passes": bool(h2_ci[0] > 0 and all(p["H2_gross_R_diff"] > 0 for p in periods.values()))},
        "sub_periods": periods,
        "by_pair": {pair: summary(g) for pair, g in data[data.variant == "real"].groupby("pair")},
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def line(label: str, s: dict) -> str:
        return (f"{label:10s} plans {s['plans']:>8,} bekle %{100 * s['wait_share']:.0f} dolma %{100 * s['fill_rate']:.0f} "
                f"| hedef %{100 * s['target_share']:.1f} stop %{100 * s['stop_share']:.1f} gün sonu %{100 * s['day_end_share']:.1f} "
                f"| brüt {s['gross_pips']:+.2f} pip net {s['net_pips']:+.2f} pip | brüt R {s['gross_R']:+.3f} net R {s['net_R']:+.3f} "
                f"| risk {s['median_risk_pips']:.1f} maliyet {s['median_cost_pips']:.1f}")
    print(f"pairs {results['pairs']} | {results['period']}")
    print(line("gerçek", results["real"]))
    print(line("sahte", results["fake"]))
    for side, s in results["by_side"].items():
        print(line(side, s))
    for name in ("H1_plan_pays_after_costs", "H2_real_levels_beat_fake"):
        block = results[name]
        value = block.get("net_R", block.get("gross_R_diff"))
        print(f"{name}: {value:+.4f} R, %97,5 GA [{block['ci'][0]:+.4f}, {block['ci'][1]:+.4f}] passes={block['passes']}")
    for label, p in periods.items():
        print(f"  {label}: H1 net R {p['H1_net_R']:+.3f}  H2 fark {p['H2_gross_R_diff']:+.4f}  hedef %{100 * p['real']['target_share']:.1f}")


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
