"""Monthly currency factor strategies (carry, 12-1 cross-sectional momentum, 12-month time-series momentum).

Rules are in research/fx_factors/protocol.json and were written before any return was computed.
Offline research; never changes the live app. Needs data/historical_1h_from_15m
(python rebuild_hourly_from_15m.py); FRED rates are downloaded once into research/fx_factors/rates.csv.

Run: python -X utf8 research_fx_factors.py
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from forex_config import get_pip_size
from forex_decision_core import stationary_bootstrap_mean_test
from forex_ml_tournament import load_hourly

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "fx_factors"
RATES_CSV = OUTPUT / "rates.csv"
HOURLY = ROOT / "data" / "historical_1h_from_15m"
NEW_YORK = pytz.timezone("America/New_York")
# currency -> (archive pair, True when the pair is USDXXX and must be inverted to USD per unit)
CURRENCIES = {"EUR": ("EURUSD", False), "GBP": ("GBPUSD", False), "AUD": ("AUDUSD", False), "NZD": ("NZDUSD", False),
              "JPY": ("USDJPY", True), "CHF": ("USDCHF", True), "CAD": ("USDCAD", True)}
RATE_SERIES = {"USD": "US", "EUR": "EZ", "GBP": "GB", "JPY": "JP", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
FIRST_HOLDING_MONTH = pd.Timestamp("2009-01-31")
COSTS = {"interbank": {"round_trip_pips": 1.5, "swap_markup_per_year": 0.0},
         "retail": {"round_trip_pips": 3.0, "swap_markup_per_year": 0.01}}
SUBPERIODS = {"2009-2013": ("2009-01-01", "2014-01-01"), "2014-2019": ("2014-01-01", "2020-01-01"), "2020-2026": ("2020-01-01", "2027-01-01")}
BONFERRONI_P = 0.05 / 4


def load_rates() -> pd.DataFrame:
    """Monthly 3-month interbank rates in percent, indexed by month end; downloaded once and cached."""
    if not RATES_CSV.exists():
        frames = {}
        for currency, code in RATE_SERIES.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01{code}M156N"
            raw = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode()))
            raw.columns = ["date", "value"]
            frames[currency] = pd.Series(pd.to_numeric(raw.value, errors="coerce").to_numpy(),
                                         index=pd.to_datetime(raw.date)).dropna()
        pd.DataFrame(frames).to_csv(RATES_CSV, index_label="month")
    rates = pd.read_csv(RATES_CSV, index_col="month", parse_dates=True)
    rates.index = rates.index + pd.offsets.MonthEnd(0)
    return rates


def last_weekday(month_end: pd.Timestamp) -> pd.Timestamp:
    day = month_end
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def price_before(closes: pd.Series, five_pm: pd.Timestamp) -> float:
    window = closes[(closes.index <= five_pm - pd.Timedelta(hours=1)) & (closes.index >= five_pm - pd.Timedelta(hours=12))]
    return float(window.iloc[-1]) if len(window) else np.nan


def month_end_prices(months: pd.DatetimeIndex, fallback: bool = True) -> pd.DataFrame:
    """USD per unit of each currency at 5 pm New York on the last weekday of each month.

    With `fallback` (data_amendment.json), a month-end missing from the 15M-rebuilt archive is
    taken from the native hourly archive by the same rule; the two agree within a tick on 99.8%
    of shared hours, and each has a few gaps the other does not.
    """
    prices, pair_prices = {}, {}
    for currency, (pair, inverted) in CURRENCIES.items():
        closes = load_hourly(HOURLY / f"{pair}.csv", quarantine=True).Close
        native = load_hourly(ROOT / "data" / "historical_1h" / f"{pair}.csv", quarantine=True).Close if fallback else None
        values = []
        for month_end in months:
            day = last_weekday(month_end)
            five_pm = pd.Timestamp(NEW_YORK.localize(pd.Timestamp(day.date()).to_pydatetime().replace(hour=17))).tz_convert("UTC")
            value = price_before(closes, five_pm)
            if np.isnan(value) and native is not None:
                value = price_before(native, five_pm)
            values.append(value)
        pair_prices[currency] = pd.Series(values, index=months)
        prices[currency] = 1 / pair_prices[currency] if inverted else pair_prices[currency]
    return pd.DataFrame(prices), pd.DataFrame(pair_prices)


def excess_returns(prices: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Monthly excess return of holding each currency against USD, indexed by the month-end it is realized."""
    spot = np.log(prices / prices.shift(1))
    days = prices.index.to_series().diff().dt.days
    held = rates.reindex(prices.index)
    carry = held[list(CURRENCIES)].sub(held["USD"], axis=0).div(100).mul(days / 365, axis=0)
    return spot + carry


def rank_weights(signal: pd.Series, n: int = 2) -> pd.Series:
    """+0.5 on the n highest and -0.5 on the n lowest values (dollar neutral); zeros if data is missing."""
    weights = pd.Series(0.0, index=signal.index)
    if signal.isna().any():
        return weights
    order = signal.sort_values()
    weights[order.index[-n:]] = 1.0 / n
    weights[order.index[:n]] = -1.0 / n
    return weights


def strategy_weights(xr: pd.DataFrame, rates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Weights decided at each month-end t, to be held over month t+1."""
    lagged_rates = rates[list(CURRENCIES)].shift(1).reindex(xr.index)
    momentum_12_1 = xr.shift(1).rolling(11).sum()
    momentum_12 = xr.rolling(12).sum()
    carry = pd.DataFrame([rank_weights(lagged_rates.loc[t]) for t in xr.index], index=xr.index)
    xs = pd.DataFrame([rank_weights(momentum_12_1.loc[t]) for t in xr.index], index=xr.index)
    ts = np.sign(momentum_12).fillna(0.0) / len(CURRENCIES)
    return {"carry": carry, "xs_momentum": xs, "ts_momentum": ts}


def net_returns(weights: pd.DataFrame, xr: pd.DataFrame, pair_prices: pd.DataFrame, cost: dict) -> pd.Series:
    """Next-month portfolio return minus trading cost at the rebalance and any swap markup."""
    held = weights.shift(1)
    gross = (held * xr).sum(axis=1)
    one_way = pd.DataFrame({c: cost["round_trip_pips"] / 2 * get_pip_size(CURRENCIES[c][0]) / pair_prices[c] for c in CURRENCIES})
    turnover = weights.diff().abs().fillna(weights.abs())
    trading = (turnover * one_way).sum(axis=1).shift(1)
    days = xr.index.to_series().diff().dt.days
    swap = held.abs().sum(axis=1) * cost["swap_markup_per_year"] * days / 365
    return (gross - trading - swap)[xr.index >= FIRST_HOLDING_MONTH]


def describe(monthly: pd.Series) -> dict:
    equity = monthly.cumsum()
    test = stationary_bootstrap_mean_test(monthly.tolist(), simulations=5000, mean_block_length=3.0, seed=42)
    return {
        "months": int(len(monthly)),
        "annual_mean_pct": float(monthly.mean() * 12 * 100),
        "annual_vol_pct": float(monthly.std(ddof=1) * np.sqrt(12) * 100),
        "sharpe": float(monthly.mean() / monthly.std(ddof=1) * np.sqrt(12)),
        "max_drawdown_pct": float((equity - equity.cummax()).min() * 100),
        "worst_month_pct": float(monthly.min() * 100),
        "skewness": float(monthly.skew()),
        "mean_monthly_ci_pct": [float(test["ci_low"] * 100), float(test["ci_high"] * 100)],
        "p_value_one_sided": float(test["p_value"]),
        "subperiod_annual_mean_pct": {
            name: float(monthly[(monthly.index >= a) & (monthly.index < b)].mean() * 12 * 100) for name, (a, b) in SUBPERIODS.items()
        },
    }


def main() -> None:
    protocol_bytes = (OUTPUT / "protocol.json").read_bytes()
    rates = load_rates()
    last = pd.Timestamp("2026-08-31")
    months = pd.date_range("2007-12-31", last, freq="ME")
    rates = rates.reindex(rates.index.union(months)).ffill().reindex(months)
    prices, pair_prices = month_end_prices(months)
    xr = excess_returns(prices, rates)
    weights = strategy_weights(xr, rates)

    results = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
               "missing_month_end_prices": int(prices.isna().sum().sum()),
               "rates_last_observation": {c: str(load_rates()[c].dropna().index.max().date()) for c in RATE_SERIES}}
    monthly = {}
    for cost_name, cost in COSTS.items():
        series = {name: net_returns(w, xr, pair_prices, cost) for name, w in weights.items()}
        series["combo"] = pd.concat(series.values(), axis=1).mean(axis=1)
        monthly[cost_name] = series
        results[cost_name] = {name: describe(s) for name, s in series.items()}
    results["gross_no_costs"] = {name: describe(net_returns(w, xr, pair_prices, {"round_trip_pips": 0.0, "swap_markup_per_year": 0.0}))
                                 for name, w in weights.items()}
    verdicts = {}
    for name in ("carry", "xs_momentum", "ts_momentum", "combo"):
        inter = results["interbank"][name]
        present = inter["annual_mean_pct"] > 0 and inter["p_value_one_sided"] <= BONFERRONI_P and \
            sum(v > 0 for v in inter["subperiod_annual_mean_pct"].values()) >= 2
        verdicts[name] = {"premium_present": bool(present),
                          "usable_at_retail": bool(present and results["retail"][name]["annual_mean_pct"] > 0)}
    results["verdicts"] = verdicts
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame({f"{c}_{n}": s for c, d in monthly.items() for n, s in d.items()}).to_csv(OUTPUT / "monthly_returns.csv")
    pd.set_option("display.width", 220)
    for cost_name in ("interbank", "retail"):
        table = pd.DataFrame(results[cost_name]).T[["annual_mean_pct", "annual_vol_pct", "sharpe", "max_drawdown_pct",
                                                     "worst_month_pct", "skewness", "p_value_one_sided"]]
        print(f"== {cost_name}\n{table.astype(float).round(3).to_string()}")
        print(pd.DataFrame({k: v["subperiod_annual_mean_pct"] for k, v in results[cost_name].items()}).round(2).to_string())
    print(json.dumps(verdicts, indent=2))
    print("missing month-end prices:", results["missing_month_end_prices"], "| rates last obs:", results["rates_last_observation"])


if __name__ == "__main__":
    main()
