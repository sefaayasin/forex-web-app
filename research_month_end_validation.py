"""Validation of the large-flow month-end rule (research/month_end/validation_protocol.json).

    python -X utf8 research_month_end_validation.py            # A: unseen 2003-2007 history
    python -X utf8 research_month_end_validation.py --forward  # B: month-ends from 2026-09 on (Yahoo data)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yfinance as yf

from forex_config import get_pip_size
from forex_costs import load_spread_profile
from research_month_end import INDEX, OUTPUT, PAIRS, foreign_currency, hour_trade, month_return

THRESHOLD = 0.0291
VALIDATION_INDEX = {**INDEX, "EUR": "^GDAXI"}  # ^STOXX50E starts in 2007-03 on Yahoo
BOOTSTRAPS = 2000


def equity_closes(path, start: str, end: str) -> pd.DataFrame:
    if not path.exists():
        df = yf.download(sorted(set(VALIDATION_INDEX.values())), start=start, end=end, interval="1d",
                         auto_adjust=False, progress=False, threads=False)["Close"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def read_bars(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.columns = [c.capitalize() for c in df.columns]
    return df.sort_index()[["Open", "High", "Low", "Close"]]


def month_end_trades(bars_by_pair: dict, closes: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    profile = load_spread_profile()
    records = []
    for pair, bars in bars_by_pair.items():
        pip, ccy = get_pip_size(pair), foreign_currency(pair)
        for month in months:
            days = pd.bdate_range(month.start_time, month.end_time.normalize())
            previous_end = pd.bdate_range((month - 1).start_time, (month - 1).end_time.normalize())[-1]
            last = next(((i, t) for i in range(len(days) - 1, 0, -1)
                         if (t := hour_trade(bars, days[i], pair, profile, pip)) is not None), None)
            if last is None:
                continue
            last_i, trade = last
            second_last = days[last_i - 1]
            r = (month_return(closes[VALIDATION_INDEX[ccy]], previous_end, second_last)
                 - month_return(closes[VALIDATION_INDEX["USD"]], previous_end, second_last))
            if not np.isfinite(r) or abs(r) < THRESHOLD:
                continue
            gross = -np.sign(r) * trade["foreign_move"]
            records.append({"pair": pair, "day": days[last_i], "relative": r, "gross": gross,
                            "net": gross - trade["cost"], "net_p90": gross - trade["cost_p90"]})
    return pd.DataFrame(records)


def summarise(trades: pd.DataFrame, name: str) -> dict:
    per_day = trades.groupby("day")["net"].agg(["sum", "count"])
    rng = np.random.default_rng(42)
    weights = np.stack([np.bincount(rng.integers(0, len(per_day), len(per_day)), minlength=len(per_day))
                        for _ in range(BOOTSTRAPS)])
    draws = (weights @ per_day["sum"].to_numpy()) / (weights @ per_day["count"].to_numpy())
    out = {"name": name, "n": len(trades), "month_ends": len(per_day), "gross": float(trades.gross.mean()),
           "net": float(trades.net.mean()), "net_p90": float(trades.net_p90.mean()), "hit": float((trades.gross > 0).mean()),
           "ci90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
           "per_pair": {p: {"n": len(g), "net": float(g.net.mean())} for p, g in trades.groupby("pair")},
           "per_year": {int(y): {"n": len(g), "net": float(g.net.mean())} for y, g in trades.groupby(trades.day.dt.year)}}
    out["passes"] = bool(out["ci90"][0] > 0)
    return out


def main(forward: bool) -> None:
    if forward:
        months = pd.period_range("2026-09", pd.Timestamp.now().to_period("M") - 1, freq="M")
        closes = equity_closes(OUTPUT / "equity_closes_forward.csv", "2026-07-01", str(pd.Timestamp.now().date()))
        bars = {}
        for pair in PAIRS:
            df = yf.download(f"{pair}=X", period="60d", interval="5m", progress=False, auto_adjust=False, threads=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            bars[pair] = df[["Open", "High", "Low", "Close"]].tz_convert("UTC")
        result = summarise(month_end_trades(bars, closes, months), "B_forward")
    else:
        months = pd.period_range("2003-06", "2007-12", freq="M")
        closes = equity_closes(OUTPUT / "equity_closes_2002_2008.csv", "2002-01-01", "2008-03-01")
        bars = {pair: read_bars(OUTPUT / "m5_2003_2007" / f"{pair}.csv") for pair in PAIRS}
        result = summarise(month_end_trades(bars, closes, months), "A_unseen_2003_2007")
    (OUTPUT / f"validation_{result['name']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"per_pair", "per_year"}}, indent=1))
    print("per pair:", {p: (v["n"], round(v["net"], 2)) for p, v in result["per_pair"].items()})
    print("per year:", {y: (v["n"], round(v["net"], 2)) for y, v in result["per_year"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true")
    main(parser.parse_args().forward)
