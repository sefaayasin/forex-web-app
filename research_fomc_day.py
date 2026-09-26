"""Short-USD returns on scheduled FOMC days: out-of-sample check of Mueller, Tahbaz-Salehi & Vedolin (2017).

Rules are in research/fomc_day/protocol.json and were written before any return was computed.
Offline research; never changes the live app.

Run: python -X utf8 research_fomc_day.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from forex_config import get_pip_size
from forex_ml_tournament import load_hourly

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "fomc_day"
HOURLY = ROOT / "data" / "historical_1h_from_15m"
NEW_YORK = pytz.timezone("America/New_York")
# +1: holding the currency against USD gains when the pair rises (XXXUSD); -1 for USDXXX.
PAIRS = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1, "USDJPY": -1, "USDCAD": -1, "USDCHF": -1}
EXTRA_SCHEDULED = ["2008-12-16"]
PRIMARY_START = "2014-01-01"
SUBPERIODS = {"2014-2019": ("2014-01-01", "2020-01-01"), "2020-2026": ("2020-01-01", "2027-01-01")}
COSTS_PIPS = (1.5, 3.0)
PERMUTATIONS = 20000


def new_york_utc(day: date, hour: int) -> pd.Timestamp:
    return pd.Timestamp(NEW_YORK.localize(datetime(day.year, day.month, day.day, hour))).tz_convert("UTC")


def price_at(closes: pd.Series, day: date, hour: int) -> float:
    """Price at `hour`:00 New York time = close of the hourly bar that opened one hour earlier."""
    bar_open = new_york_utc(day, hour) - pd.Timedelta(hours=1)
    return float(closes.get(bar_open, np.nan))


def previous_weekday(day: date) -> date:
    step = 3 if day.weekday() == 0 else 1
    return day - timedelta(days=step)


def scheduled_fomc_days(protocol: dict) -> tuple[set, set]:
    statements = pd.read_csv(ROOT / "data" / "news" / "fomc_statements.csv", parse_dates=["date"]).date.dt.date
    excluded = {date.fromisoformat(d) for d in protocol["fomc_days"]["unscheduled_excluded_from_both_groups"]}
    non_meeting = {date(2010, 5, 9), date(2020, 3, 15)} | excluded
    scheduled = {d for d in statements if d not in non_meeting} | {date.fromisoformat(d) for d in EXTRA_SCHEDULED}
    return scheduled, excluded


def daily_returns(start: date, end: date) -> pd.DataFrame:
    """One row per weekday with each currency's 4pm-to-4pm ET return (bps), pre/post 2pm split and entry prices."""
    closes = {pair: load_hourly(HOURLY / f"{pair}.csv", quarantine=True).Close for pair in PAIRS}
    rows = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            prev = previous_weekday(day)
            row = {"date": day}
            for pair, sign in PAIRS.items():
                p0, p14, p16 = price_at(closes[pair], prev, 16), price_at(closes[pair], day, 14), price_at(closes[pair], day, 16)
                row[pair] = sign * np.log(p16 / p0) * 1e4
                row[f"{pair}_pre"] = sign * np.log(p14 / p0) * 1e4
                row[f"{pair}_post"] = sign * np.log(p16 / p14) * 1e4
                row[f"{pair}_entry"] = p0
            rows.append(row)
        day += timedelta(days=1)
    frame = pd.DataFrame(rows).set_index("date")
    return frame.dropna(subset=list(PAIRS))


def cost_bps(frame: pd.DataFrame, pips: float) -> pd.Series:
    return pd.concat([pips * get_pip_size(p) / frame[f"{p}_entry"] * 1e4 for p in PAIRS], axis=1).mean(axis=1)


def difference_test(values: pd.Series, is_fomc: pd.Series, seed: int = 42) -> dict:
    fomc, other = values[is_fomc], values[~is_fomc]
    diff = float(fomc.mean() - other.mean())
    welch_t = diff / np.sqrt(fomc.var(ddof=1) / len(fomc) + other.var(ddof=1) / len(other))
    rng = np.random.default_rng(seed)
    data, n_fomc = values.to_numpy(), int(is_fomc.sum())
    greater = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(data)
        greater += (shuffled[:n_fomc].mean() - shuffled[n_fomc:].mean()) >= diff
    return {"fomc_days": n_fomc, "other_days": int(len(other)), "fomc_mean_bps": float(fomc.mean()),
            "other_mean_bps": float(other.mean()), "difference_bps": diff, "welch_t": float(welch_t),
            "permutation_p_one_sided": (greater + 1) / (PERMUTATIONS + 1)}


def period_report(frame: pd.DataFrame, scheduled: set, start: str, end: str) -> dict:
    part = frame[(frame.index >= date.fromisoformat(start)) & (frame.index < date.fromisoformat(end))]
    is_fomc = pd.Series(part.index.isin(list(scheduled)), index=part.index)
    basket = part[list(PAIRS)].mean(axis=1)
    report = {"basket": difference_test(basket, is_fomc),
              "per_currency": {p: difference_test(part[p], is_fomc) for p in PAIRS}}
    for pips in COSTS_PIPS:
        report[f"fomc_mean_net_{pips}pips_bps"] = float((basket - cost_bps(part, pips))[is_fomc].mean())
    for window in ("pre", "post"):
        report[f"basket_{window}"] = difference_test(part[[f"{p}_{window}" for p in PAIRS]].mean(axis=1), is_fomc)
    report["fomc_dates_used"] = [str(d) for d in part.index[is_fomc]]
    return report


def main() -> None:
    protocol_bytes = (OUTPUT / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    scheduled, excluded = scheduled_fomc_days(protocol)
    frame = daily_returns(date(2008, 1, 2), date(2026, 9, 11))
    frame = frame[~frame.index.isin(list(excluded))]
    end = "2027-01-01"
    results = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "days_with_prices": int(len(frame)),
        "primary": period_report(frame, scheduled, PRIMARY_START, end),
        "secondary_2008_2013": period_report(frame, scheduled, "2008-01-01", PRIMARY_START),
        "subperiods": {name: period_report(frame, scheduled, a, b)["basket"] for name, (a, b) in SUBPERIODS.items()},
    }
    primary = results["primary"]
    checks = {
        "primary_difference_significant": bool(primary["basket"]["difference_bps"] > 0 and primary["basket"]["permutation_p_one_sided"] <= .05),
        "primary_positive_after_costs": bool(primary["fomc_mean_net_1.5pips_bps"] > 0),
        "positive_in_both_subperiods": all(v["difference_bps"] > 0 for v in results["subperiods"].values()),
    }
    results["checks"], results["passes"] = checks, all(checks.values())
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fomc_by_year = frame[frame.index.isin(list(scheduled))][list(PAIRS)].mean(axis=1).groupby(lambda d: d.year).agg(["count", "mean"])
    fomc_by_year.to_csv(OUTPUT / "fomc_basket_by_year.csv")
    summary = {k: v for k, v in results.items() if k not in ("primary", "secondary_2008_2013")}
    summary["primary_basket"] = primary["basket"]
    summary["primary_pre"], summary["primary_post"] = primary["basket_pre"], primary["basket_post"]
    summary["primary_net"] = {k: v for k, v in primary.items() if k.startswith("fomc_mean_net")}
    summary["secondary_basket"] = results["secondary_2008_2013"]["basket"]
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame({p: primary["per_currency"][p] for p in PAIRS}).T[["fomc_mean_bps", "other_mean_bps", "difference_bps", "welch_t"]].round(2).to_string())
    print(fomc_by_year.round(2).to_string())


if __name__ == "__main__":
    main()
