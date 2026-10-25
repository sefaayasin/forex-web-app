"""Internet-researched indicator tournament; fixed rules, chronological selection.

New formulas and sources in forex_researched_indicators.py / sources.json.
No strategy or broker permission is silently promoted by a retrospective run.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forex_adaptive import MAJORS, ROOT, digest, write_json
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from forex_researched_indicators import researched_rule_states
from research_cash_target import day_test, describe, prepare_paths, simulate

OUT = ROOT / "research/new_indicators"
SYMBOLS = MAJORS + ["EURCHF"]
DEVELOPMENT = (("2016-01-01", "2019-01-01"), ("2019-01-01", "2023-01-01"))
STOPS, HOLDS, BUDGETS = (1.5, 3.), (1, 4), (20., 50., 100.)


def choose_candidate(rows):
    eligible = [r for r in rows if all(p["trades"] >= 100 for p in r["subperiods"])]
    if not eligible:
        raise ValueError("No candidate with 100 trades in both development subperiods")
    return max(eligible, key=lambda r: min(p["stress_mean_usd"] for p in r["subperiods"]))


def audit_checks(selected, summary, normal, stress, yearly, hypothesis_count):
    return {
        "development_net_and_stress_positive": all(p["mean_usd"] > 0 and p["stress_mean_usd"] > 0 for p in selected["subperiods"]),
        "at_least_300_audit_trades": summary["trades"] >= 300,
        "normal_ci_positive": normal["ci_low"] is not None and normal["ci_low"] > 0,
        "stress_ci_positive": stress["ci_low"] is not None and stress["ci_low"] > 0,
        "adjusted_p_passes": normal["p_value"] is not None and normal["p_value"] * hypothesis_count <= .05,
        "every_audit_year_positive": bool(yearly) and all(r["mean_usd"] > 0 for r in yearly.values()),
    }


def run_pair(symbol, hypothesis_count=8):
    paths = [ROOT / f"data/historical_{tf}/{symbol}.csv" for tf in ("15m", "5m")]
    print(f"{symbol}: loading archive and new researched indicators", flush=True)
    bars, fills = [load_hourly(p, quarantine=True) for p in paths]
    bars.attrs["symbol"] = symbol
    rules = researched_rule_states(bars)
    atr = compute_atr(bars, 14)
    prepared = prepare_paths(bars, fills)
    corrupt = pd.Series(False, index=bars.index)
    corrupt.iloc[bars.attrs.get("bad_positions", [])] = True
    prepared["history_bad"] = corrupt.rolling(4001, min_periods=1).max().astype(bool).to_numpy()
    profile = load_spread_profile()
    ny_hours = (bars.index + pd.Timedelta(minutes=15)).tz_convert("America/New_York").hour
    costs = np.array([(measured_spread_pips(profile, symbol, h) or 1.5) + ECN_COMMISSION_PIPS for h in ny_hours])
    trials = []
    for name, states in rules.items():
        for stop, hold, budget in itertools.product(STOPS, HOLDS, BUDGETS):
            periods = [describe(simulate(bars, fills, states, atr, costs, start, end,
                stop, hold, 30., budget, prepared=prepared), loss_usd=budget) for start, end in DEVELOPMENT]
            trials.append({"rule": name, "stop_atr": stop, "hours": hold, "loss_usd": budget,
                "subperiods": periods})
        print(f"{symbol}: {name}, 12 settings tested on development only", flush=True)
    selected = choose_candidate(trials)
    # Freeze the exact selected configuration before opening the final-period results.
    folder = OUT / symbol
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "selected.json", selected)
    end = str((fills.index[-1] + pd.Timedelta(days=1)).date())
    test = simulate(bars, fills, rules[selected["rule"]], atr, costs, "2023-01-01", end,
        selected["stop_atr"], selected["hours"], 30., selected["loss_usd"], prepared=prepared)
    summary = describe(test, loss_usd=selected["loss_usd"])
    normal, stress = day_test(test, "pnl_usd"), day_test(test, "pnl_stress_usd")
    yearly = {str(year): describe(part, loss_usd=selected["loss_usd"]) for year, part in
              test.groupby(test.entry_time.dt.year)} if len(test) else {}
    checks = audit_checks(selected, summary, normal, stress, yearly, hypothesis_count)
    report = {"symbol": symbol, "target_usd": 30., "max_allowed_loss_usd": 100.,
        "planned_stop_loss_usd": selected["loss_usd"], "initial_account_usd": 1000.,
        "status": "RETROSPECTIVE_RESEARCH", "trade_permission": False,
        "screen_passed": all(checks.values()), "selected": selected, "audit": summary,
        "normal_day_test": normal, "stress_day_test": stress, "checks": checks, "yearly": yearly,
        "hypothesis_count": hypothesis_count, "development_trials": trials,
        "source_hashes": {str(p): digest(p) for p in paths},
        "code_hashes": {name: digest(ROOT / name) for name in ("research_new_indicators.py", "forex_researched_indicators.py", "research_cash_target.py")},
        "formula_source_hash": digest(OUT / "sources.json"),
        "price_end": str(fills.index[-1] + pd.Timedelta(minutes=5)),
        "notes": ["2023+ history already inspected in prior studies; not a new blind/forward period.",
            "14 declared rules x 2 stops x 2 durations x 3 risk budgets; selection uses 2016-2022 only.",
            "Retrospective 2026 measured spread scenario, not historical broker execution.",
            "Dollars are per-trade pip risk scale; lot step, leverage, exit FX conversion and broker execution omitted.",
            "Fixed-budget account replay stops at insufficient funds; average trade statistics include all hypothetical opportunities.",
            "Gap/slippage can exceed the planned loss. Positive retrospective screens need independent forward confirmation."]}
    write_json(folder / "report.json", report)
    test.to_csv(folder / "audit_trades.csv.gz", index=False, compression="gzip")
    print(f"{symbol}: {selected['rule']} risk=${selected['loss_usd']:g}, audit mean=${summary['mean_usd']:.3f}, "
          f"stress=${summary['stress_mean_usd']:.3f}, screen={report['screen_passed']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = parser.parse_args()
    if not args.symbols or any(len(s) != 6 or not s.isalpha() for s in args.symbols):
        parser.error("Expected six-letter symbols")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"date": "2026-10-08", "symbols": args.symbols,
        "target_usd": 30., "maximum_planned_loss_usd": 100., "risk_budgets": BUDGETS,
        "stop_atr": STOPS, "holding_hours": HOLDS, "development": DEVELOPMENT,
        "audit_start": "2023-01-01", "audit_already_inspected": True,
        "selection": "At least 100 trades in each development period; maximize worst-period stress mean USD",
        "sources": "sources.json", "script_sha256": digest(Path(__file__))})
    rows = []
    for symbol in args.symbols:
        report = run_pair(symbol, hypothesis_count=len(args.symbols))
        rows.append({"symbol": symbol, **report["selected"], **report["audit"], "screen_passed": report["screen_passed"]})
        pd.DataFrame([{k: v for k, v in r.items() if k != "subperiods"} for r in rows]).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame([{k: v for k, v in r.items() if k != "subperiods"} for r in rows]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
