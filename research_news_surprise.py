"""Price-independent growth/labor news-surprise experiment, retrospective only.

Initial-release vintages/feed latency are unverified; no deployment permission.
Fixed budget $20 <= user's $100 maximum, target $30, 3 ATR stop, four-hour limit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from research_calendar_sources import load_forexfactory
from research_cash_target import day_test, describe, prepare_paths, simulate
from research_new_indicators import SYMBOLS

OUT = ROOT / "research/news_surprise"
SPLIT = "2025-10-01"
POLARITY = {
    "Non-Farm Employment Change": 1, "ADP Non-Farm Employment Change": 1,
    "Employment Change": 1, "Unemployment Rate": -1, "Unemployment Claims": -1,
    "ISM Manufacturing PMI": 1, "ISM Services PMI": 1,
    "Flash Manufacturing PMI": 1, "Flash Services PMI": 1,
    "Retail Sales m/m": 1, "Core Retail Sales m/m": 1,
    "GDP m/m": 1, "GDP q/q": 1, "GDP y/y": 1,
    "Advance GDP q/q": 1, "Prelim GDP q/q": 1,
}


def surprise_signals(events):
    selected = events[(events.impact == "high") & events.name.isin(POLARITY)
                      & events.actual.notna() & events.forecast.notna()].copy()
    selected = selected.sort_values("time")
    selected["surprise"] = selected.actual - selected.forecast
    # Each scale uses earlier published observations only, never future rows.
    selected["past_error_scale"] = selected.groupby("series", sort=False).surprise.transform(
        lambda s: s.abs().shift(1).rolling(12, min_periods=3).median())
    take = selected.past_error_scale.notna() & selected.surprise.ne(0) & (
        selected.surprise.abs() >= selected.past_error_scale)
    selected = selected[take].copy()
    selected["vote"] = np.sign(selected.surprise) * selected.name.map(POLARITY)
    # At least five minutes' assumed dissemination delay, then a completed 15M
    # decision boundary. Most 00/30-minute releases therefore wait 15 minutes.
    selected["decision_at"] = (selected.time + pd.Timedelta(minutes=5)).dt.ceil("15min")
    return selected


def pair_states(signals, symbol, index):
    relevant = signals[signals.currency.isin([symbol[:3], symbol[3:]])].copy()
    relevant["pair_vote"] = relevant.vote * np.where(relevant.currency == symbol[:3], 1, -1)
    votes = relevant.groupby("decision_at").pair_vote.sum()
    positions = index.get_indexer(votes.index - pd.Timedelta(minutes=15))
    states = np.zeros(len(index), dtype=int)
    good = positions >= 0
    states[positions[good]] = np.sign(votes.to_numpy()[good]).astype(int)
    return states, np.flatnonzero(states)


def run_pair(symbol, signals):
    paths = [ROOT / f"data/historical_{tf}/{symbol}.csv" for tf in ("15m", "5m")]
    bars, fills = [load_hourly(p, quarantine=True) for p in paths]
    bars.attrs["symbol"] = symbol
    state, entries = pair_states(signals, symbol, bars.index)
    profile = load_spread_profile()
    hours = (bars.index + pd.Timedelta(minutes=15)).tz_convert("America/New_York").hour
    costs = np.array([(measured_spread_pips(profile, symbol, h) or 1.5) + ECN_COMMISSION_PIPS for h in hours])
    prepared = prepare_paths(bars, fills)
    atr = compute_atr(bars, 14)
    end = str((fills.index[-1] + pd.Timedelta(days=1)).date())
    samples = {}
    for name, start, stop in (("development", "2024-10-01", SPLIT), ("audit", SPLIT, end)):
        samples[name] = simulate(bars, fills, state, atr, costs, start, stop, 3., 4,
            30., 20., entries=entries, prepared=prepared)
    audit = samples["audit"]
    report = {"symbol": symbol, "source": "ForexFactory local historical calendar",
        "status": "EXPLORATORY_SCENARIO", "trade_permission": False,
        "target_usd": 30., "planned_stop_loss_usd": 20., "user_maximum_loss_usd": 100.,
        "development": describe(samples["development"], loss_usd=20.),
        "audit": describe(audit, loss_usd=20.),
        "day_test": day_test(audit, "pnl_usd"), "stress_day_test": day_test(audit, "pnl_stress_usd"),
        "price_end": str(fills.index[-1]+pd.Timedelta(minutes=5)),
        "source_hashes": {str(p): digest(p) for p in paths},
        "limitations": ["Feed availability assumed by next 15M boundary after release+5m; not historically timestamped.",
            "Archived actual/consensus numbers not verified initial vintages; no live authorization even if positive.",
            "2026 regular-hour spread scenario; actual news spreads/slippage can be much higher.",
            "Fixed $20 pip risk scale, not a broker-margin/lot simulation; goal $30 and four-hour limit.",
            "Calendar sample Oct2024-Sep2026; not comparable to older indicator-development sample.",
            "Scheduled UTC07-17 decisions only; CPI/rates/wages excluded rather than assuming uniform inflation-policy signs."]}
    folder = OUT / symbol
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "report.json", report)
    if len(audit):
        audit.to_csv(folder / "audit_trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: news surprise audit {report['audit']}", flush=True)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-08", "symbols": SYMBOLS,
        "research_source": "https://www.nber.org/papers/w8959", "target_usd": 30., "risk_usd": 20.,
        "stop_atr": 3., "hours": 4, "polarity": POLARITY, "split": SPLIT,
        "vintages_unverified": True, "selection": "none; one predeclared rule per pair",
        "code_sha256": digest(Path(__file__))})
    raw = load_forexfactory()
    signals = surprise_signals(raw)
    signals.to_csv(OUT / "signals.csv", index=False)
    print(f"Calendar rows={len(raw)}; eligible surprise releases={len(signals)}", flush=True)
    results = [run_pair(s, signals) for s in SYMBOLS]
    pd.DataFrame([{"symbol": r["symbol"], **r["audit"]} for r in results]).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
