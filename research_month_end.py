"""Month-end equity hedging flows at the London 4pm fix (Melvin & Prins 2015): do they pay after costs?

Rules are in research/month_end/protocol.json and were written before any outcome was computed.
Equity closes were saved to research/month_end/equity_closes.csv before the run.

    python -X utf8 research_month_end.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from research_fix_reversal import bar_at, foreign_sign
from research_meta_label import load_bars

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "month_end"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
INDEX = {"USD": "^GSPC", "EUR": "^STOXX50E", "GBP": "^FTSE", "AUD": "^AXJO", "NZD": "^NZ50",
         "JPY": "^N225", "CAD": "^GSPTSE", "CHF": "^SSMI"}
LONDON, NY = "Europe/London", "America/New_York"
FIRST_MONTH, LAST_MONTH = pd.Period("2008-01", "M"), pd.Period("2026-08", "M")
PRIMARY = ("2014-01-01", "2026-08-31")
SUB_PERIODS = {"2014-2019": ("2014-01-01", "2019-12-31"), "2020-2026": ("2020-01-01", "2026-08-31")}
IN_PAPER = ("2008-01-01", "2012-12-31")
BOOTSTRAPS = 2000


def foreign_currency(pair: str) -> str:
    return pair[:3] if pair.endswith("USD") else pair[3:]


def close_on_or_before(closes: pd.Series, day: pd.Timestamp) -> float:
    part = closes.loc[:day].dropna()
    return float(part.iloc[-1]) if len(part) else np.nan


def month_return(closes: pd.Series, previous_end: pd.Timestamp, second_last: pd.Timestamp) -> float:
    start, end = close_on_or_before(closes, previous_end), close_on_or_before(closes, second_last)
    return end / start - 1 if start > 0 and end > 0 else np.nan


def hour_trade(bars: pd.DataFrame, day: pd.Timestamp, pair: str, profile: dict, pip: float) -> dict | None:
    """15:00 -> 16:00 London on `day`: foreign-vs-USD move in pips and the round-trip cost, or None."""
    start = pd.Timestamp(day.year, day.month, day.day, 15).tz_localize(LONDON).tz_convert("UTC")
    end = pd.Timestamp(day.year, day.month, day.day, 16).tz_localize(LONDON).tz_convert("UTC")
    entry, exit_ = bar_at(bars.index, start), bar_at(bars.index, end)
    if entry < 0 or exit_ < 0 or exit_ <= entry:
        return None
    opens = bars["Open"].to_numpy()
    move = foreign_sign(pair) * (opens[exit_] - opens[entry]) / pip
    hours = [bars.index[i].tz_convert(NY).hour for i in (entry, exit_)]
    median = sum(measured_spread_pips(profile, pair, h) or 1.5 for h in hours) / 2 + ECN_COMMISSION_PIPS
    p90 = sum(measured_spread_pips(profile, pair, h, "p90") or 1.5 for h in hours) / 2 + ECN_COMMISSION_PIPS
    return {"foreign_move": move, "cost": median, "cost_p90": p90}


def build_trades() -> pd.DataFrame:
    closes = pd.read_csv(OUTPUT / "equity_closes.csv", index_col=0, parse_dates=True)
    profile = load_spread_profile()
    records = []
    for pair in PAIRS:
        bars, _ = load_bars(pair, "historical_5m")
        pip, ccy = get_pip_size(pair), foreign_currency(pair)
        for month in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"):
            days = pd.bdate_range(month.start_time, month.end_time.normalize())
            previous_end = pd.bdate_range((month - 1).start_time, (month - 1).end_time.normalize())[-1]
            # Last business day with bars at both times; walk back over holidays.
            last = next(((i, t) for i in range(len(days) - 1, 0, -1)
                         if (t := hour_trade(bars, days[i], pair, profile, pip)) is not None), None)
            if last is None:
                continue
            last_i, last_trade = last
            second_last = days[last_i - 1]  # equity returns stop the business day before the trade day
            r_foreign = month_return(closes[INDEX[ccy]], previous_end, second_last)
            r_us = month_return(closes[INDEX["USD"]], previous_end, second_last)
            base = {"pair": pair, "month": str(month), "relative": r_foreign - r_us, "own": r_foreign}
            for i in range(last_i):  # control: the same hour on the month's earlier business days
                trade = hour_trade(bars, days[i], pair, profile, pip)
                if trade is not None:
                    records.append({**base, "day": days[i], "month_end": False, **trade})
            records.append({**base, "day": days[last_i], "month_end": True, **last_trade})
    return pd.DataFrame(records)


def signed(trades: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Gross/net pips of trading the predicted side: predictor > 0 -> foreign currency falls."""
    side = -np.sign(trades[predictor])
    out = trades[side != 0].copy()
    out["gross"] = side[side != 0] * out["foreign_move"]
    out["net"] = out["gross"] - out["cost"]
    out["net_p90"] = out["gross"] - out["cost_p90"]
    return out


def bootstrap(part: pd.DataFrame, column: str = "net", seed: int = 42) -> dict:
    per_day = part.groupby("day")[column].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(per_day), len(per_day)), minlength=len(per_day))
                        for _ in range(BOOTSTRAPS)])
    draws = (weights @ per_day["sum"].to_numpy()) / (weights @ per_day["count"].to_numpy())
    return {"n": len(part), "month_ends": len(per_day), "mean": float(part[column].mean()),
            "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def between(part: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    return part[(part.day >= period[0]) & (part.day <= period[1])]


def stats(part: pd.DataFrame) -> dict:
    return {"n": len(part), "gross": float(part.gross.mean()), "net": float(part.net.mean()),
            "hit": float((part.gross > 0).mean()), "net_p90": float(part.net_p90.mean())}


def main() -> None:
    trades = build_trades()
    trades.to_csv(OUTPUT / "trades.csv.gz", index=False, compression="gzip")
    month_end = signed(trades[trades.month_end], "relative")
    primary = between(month_end, PRIMARY)
    h1 = {"all": bootstrap(primary), **{label: stats(between(primary, p)) for label, p in SUB_PERIODS.items()}}
    h1["passes"] = bool(h1["all"]["ci"][0] > 0 and all(h1[label]["net"] > 0 for label in SUB_PERIODS))

    info = {
        "primary_stats": stats(primary),
        "per_pair": {pair: stats(g) for pair, g in primary.groupby("pair")},
        "in_paper_2008_2012": stats(between(month_end, IN_PAPER)),
        "own_index_predictor": stats(between(signed(trades[trades.month_end], "own"), PRIMARY)),
        "control_other_days": stats(between(signed(trades[~trades.month_end], "relative"), PRIMARY)),
    }
    cut = primary.relative.abs().quantile(2 / 3)
    info["top_third_abs_R"] = stats(primary[primary.relative.abs() >= cut])
    info["gross_ci"] = bootstrap(primary, "gross")
    results = {"H1": h1, "information_only": info}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    a = h1["all"]
    print(f"H1 passes={h1['passes']}  net {a['mean']:+.2f} [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}]  n={a['n']} "
          f"month-ends={a['month_ends']} | " + " | ".join(f"{k} net {h1[k]['net']:+.2f}" for k in SUB_PERIODS))
    g = info["gross_ci"]
    print(f"gross {g['mean']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}]")
    for name, row in info.items():
        if name in {"per_pair", "gross_ci"}:
            continue
        print(f"  {name:22s} n={row['n']:>6} gross {row['gross']:+.2f} net {row['net']:+.2f} hit {row['hit']:.3f} net@p90 {row['net_p90']:+.2f}")
    for pair, row in info["per_pair"].items():
        print(f"  {pair}: n={row['n']} gross {row['gross']:+.2f} net {row['net']:+.2f} hit {row['hit']:.3f}")


if __name__ == "__main__":
    main()
