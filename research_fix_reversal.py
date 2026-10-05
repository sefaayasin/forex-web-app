"""Dollar reversals around the Tokyo, ECB and London fixes (Krohn, Mueller & Whelan): do they pay after costs?

Rules are in research/fix_reversal/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app.

    python -X utf8 research_fix_reversal.py build      # one trade per pair, window and day -> trades.csv.gz
    python -X utf8 research_fix_reversal.py evaluate   # implementation check, hypotheses, information tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from research_meta_label import load_bars

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "fix_reversal"
TRADES = OUTPUT / "trades.csv.gz"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
NY, TOKYO = "America/New_York", "Asia/Tokyo"
FIRST_DAY, LAST_DAY = pd.Timestamp("2008-01-08"), pd.Timestamp("2026-09-11")
MAX_GAP = pd.Timedelta(minutes=10)
# Window -> USD side of the trade (+1 long USD, -1 short USD) and the paper's sign of the foreign-vs-USD move.
WINDOWS = {"pre-T": 1, "post-T": -1, "pre-E": 1, "post-L": -1}
PRIMARY = ("2020-01-01", "2026-09-11")
SUB_PERIODS = {"2020-2022": ("2020-01-01", "2022-12-31"), "2023-2026": ("2023-01-01", "2026-09-11")}
REPLICATION = ("2008-01-08", "2019-12-31")
BOOTSTRAPS = 2000
CI_LEVEL = 0.9875  # two-sided, Bonferroni 0.05 / 4


def window_bounds(day: pd.Timestamp, pre_t_start_hour: int = 17) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """UTC start/end of the four windows that belong to New York trading date `day` (DST aware)."""
    def ny(date: pd.Timestamp, hour: int, minute: int = 0) -> pd.Timestamp:
        return pd.Timestamp(date.year, date.month, date.day, hour, minute).tz_localize(NY).tz_convert("UTC")

    previous = day - pd.Timedelta(days=1)
    tokyo_fix = pd.Timestamp(day.year, day.month, day.day, 9, 55).tz_localize(TOKYO).tz_convert("UTC")
    return {
        "pre-T": (ny(previous, pre_t_start_hour), tokyo_fix),
        "post-T": (tokyo_fix, ny(day, 2)),
        "pre-E": (ny(day, 2), ny(day, 8, 15)),
        "post-L": (ny(day, 11), ny(day, 17)),
    }


def bar_at(index: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    """Position of the first bar at or after `when`, or -1 if none within MAX_GAP."""
    pos = index.searchsorted(when)
    return pos if pos < len(index) and index[pos] - when <= MAX_GAP else -1


def foreign_sign(pair: str) -> int:
    """+1 if the pair rises when the foreign currency appreciates against USD (XXXUSD), -1 for USDXXX."""
    return 1 if pair.endswith("USD") else -1


def pair_trades(pair: str, bars: pd.DataFrame, profile: dict, pre_t_start_hour: int = 17) -> pd.DataFrame:
    pip, index, opens = get_pip_size(pair), bars.index, bars["Open"].to_numpy()
    records = []
    for day in pd.bdate_range(FIRST_DAY, LAST_DAY):
        for window, bounds in window_bounds(day, pre_t_start_hour).items():
            entry, exit_ = bar_at(index, bounds[0]), bar_at(index, bounds[1])
            if entry < 0 or exit_ < 0 or exit_ <= entry:
                continue
            foreign_move = foreign_sign(pair) * (opens[exit_] - opens[entry]) / pip
            hours = [index[entry].tz_convert(NY).hour, index[exit_].tz_convert(NY).hour]
            spreads = [measured_spread_pips(profile, pair, h) for h in hours]
            cost = sum(1.5 if s is None else s for s in spreads) / 2 + ECN_COMMISSION_PIPS
            gross = -WINDOWS[window] * foreign_move  # long USD profits when the foreign currency falls
            records.append({"day": day, "pair": pair, "window": window, "entry": index[entry], "exit": index[exit_],
                            "foreign_move": foreign_move, "gross": gross, "cost": cost, "net": gross - cost})
    return pd.DataFrame(records)


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = load_spread_profile()
    frames = []
    for pair in PAIRS:
        bars, _ = load_bars(pair, "historical_5m")
        trades = pair_trades(pair, bars, profile)
        late = pair_trades(pair, bars, profile, pre_t_start_hour=18)
        late = late[late.window == "pre-T"].assign(window="pre-T@18")
        frames += [trades, late]
        print(f"{pair}: {len(trades):,} trades", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(TRADES, index=False, compression="gzip")


def bootstrap_mean(values: pd.Series, days: pd.Series, seed: int = 42) -> dict:
    per_day = pd.DataFrame({"day": days.to_numpy(), "v": values.to_numpy()}).groupby("day")["v"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(per_day), len(per_day)), minlength=len(per_day))
                        for _ in range(BOOTSTRAPS)])
    draws = (weights @ per_day["sum"].to_numpy()) / (weights @ per_day["count"].to_numpy())
    tail = (1 - CI_LEVEL) / 2
    return {"n": int(per_day["count"].sum()), "mean": float(values.mean()),
            "ci": [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]}


def between(trades: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    return trades[(trades.day >= period[0]) & (trades.day <= period[1])]


def stats(part: pd.DataFrame) -> dict:
    return {"n": len(part), "gross": float(part.gross.mean()), "net": float(part.net.mean()),
            "hit": float((part.gross > 0).mean()), "cost": float(part.cost.mean())}


def evaluate() -> None:
    trades = pd.read_csv(TRADES, parse_dates=["day"])
    main = trades[trades.window.isin(list(WINDOWS))]
    results = {"pairs": PAIRS, "trades": len(main)}

    replication = between(main, REPLICATION)
    check = {w: float(replication[replication.window == w].groupby("pair").foreign_move.mean().mean()) for w in WINDOWS}
    paper_sign = {"pre-T": -1, "post-T": 1, "pre-E": -1, "post-L": 1}
    results["implementation_check"] = {"mean_foreign_move_2008_2019": check,
                                       "passes": all(np.sign(check[w]) == paper_sign[w] for w in WINDOWS)}
    print("implementation check (2008-2019 foreign move, pips):", {w: round(v, 2) for w, v in check.items()},
          "passes" if results["implementation_check"]["passes"] else "FAILS")
    if not results["implementation_check"]["passes"]:
        (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        raise SystemExit("Implementation check failed; windows or time zones need fixing before reading 2020+.")

    primary = between(main, PRIMARY)
    hypotheses = {}
    for window in WINDOWS:
        part = primary[primary.window == window]
        block = {"all": bootstrap_mean(part.net, part.day), "gross": float(part.gross.mean()), "hit": float((part.gross > 0).mean())}
        for label, period in SUB_PERIODS.items():
            block[label] = stats(between(part, period))
        block["passes"] = bool(block["all"]["ci"][0] > 0 and all(block[label]["net"] > 0 for label in SUB_PERIODS))
        hypotheses[f"H_{window.replace('-', '')}"] = block
    results["hypotheses"] = hypotheses

    info = {"per_pair": {}, "per_year": {}, "combined": {}}
    for period_name, period in (("2020-2026", PRIMARY), ("2008-2019", REPLICATION)):
        part = between(trades, period)
        info["per_pair"][period_name] = {
            f"{w} {p}": stats(g) for (w, p), g in part.groupby(["window", "pair"])}
    for (window, year), g in main.groupby(["window", main.day.dt.year]):
        info["per_year"][f"{window} {year}"] = stats(g)
    for name, (a, b) in {"Tokyo (pre-T + post-T)": ("pre-T", "post-T"), "Europe (pre-E + post-L)": ("pre-E", "post-L")}.items():
        both = primary[primary.window.isin([a, b])].groupby(["day", "pair"])[["gross", "net"]].sum()
        info["combined"][name] = {"n_days": len(both), "gross": float(both.gross.mean()), "net": float(both.net.mean())}
    p90 = load_spread_profile()
    info["cost_at_p90"] = {}
    for window in WINDOWS:
        part = primary[primary.window == window]
        p90_cost = [
            sum(measured_spread_pips(p90, pair, ts.tz_convert(NY).hour, "p90") or 1.5 for ts in (entry, exit_)) / 2
            + ECN_COMMISSION_PIPS
            for pair, entry, exit_ in zip(part.pair, pd.to_datetime(part.entry, utc=True), pd.to_datetime(part.exit, utc=True))]
        info["cost_at_p90"][window] = float((part.gross - np.array(p90_cost)).mean())
    results["information_only"] = info
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"trades {len(main):,} | primary 2020-2026: {len(primary):,}")
    for name, block in hypotheses.items():
        a = block["all"]
        print(f"=== {name}: passes={block['passes']}  net {a['mean']:+.2f} [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}] "
              f"gross {block['gross']:+.2f} hit {block['hit']:.3f} n={a['n']:,} | "
              + " | ".join(f"{label} net {block[label]['net']:+.2f}" for label in SUB_PERIODS))
    print("--- per pair 2020-2026 (gross / net / hit)")
    for key, row in info["per_pair"]["2020-2026"].items():
        print(f"  {key:16s} n={row['n']:>5} gross {row['gross']:+6.2f} net {row['net']:+6.2f} hit {row['hit']:.3f} cost {row['cost']:.2f}")
    print("--- combined", info["combined"])
    print("--- net at p90 spreads", {w: round(v, 2) for w, v in info["cost_at_p90"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "evaluate"])
    args = parser.parse_args()
    build() if args.stage == "build" else evaluate()
