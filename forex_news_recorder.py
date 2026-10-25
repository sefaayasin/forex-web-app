"""Read-only public-calendar recorder for prospective research; no orders.

python -X utf8 forex_news_recorder.py --watch 600
Paper intents require a forecast actually captured BEFORE release and an
actual first seen AFTER release. They are future research records, not fills.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import os

import numpy as np
import pandas as pd

from forex_adaptive import ROOT, write_json
from forex_calendar import fetch_calendar
from research_calendar_sources import load_forexfactory, parse_value
from research_news_surprise import POLARITY
from research_new_indicators import SYMBOLS
from forex_ml_tournament import load_hourly
from forex_indicators import compute_atr
from forex_costs import load_spread_profile, measured_spread_pips, ECN_COMMISSION_PIPS
from research_cash_target import simulate

OUT = ROOT / "research/news_forward"


def initial_state(now):
    history = {}
    archived = load_forexfactory().sort_values("time")
    archived = archived[(archived.time < now) & archived.name.isin(POLARITY)
                        & archived.actual.notna() & archived.forecast.notna()]
    for (currency, name), group in archived.groupby(["currency", "name"]):
        history[f"{currency}|{name}"] = (group.actual-group.forecast).abs().tail(12).tolist()
    return {"events": {}, "history": history, "paper_intents": [], "trade_permission": False}


def record_snapshot(state, calendar, now, with_actuals, cost_profile=None):
    """Mutate chronological observations, preserving first actual and prior consensus."""
    now = pd.Timestamp(now)
    pending_votes = {}
    for row in calendar.itertuples():
        name = str(row.event).strip()
        if name not in POLARITY or getattr(row,"impact",0) != 3 or str(row.time_label or "").strip():
            continue
        released = pd.Timestamp(row.time)
        key = f"{row.currency}|{name}|{released.isoformat()}"
        event = state["events"].setdefault(key, {"currency": row.currency, "name": name,
            "released_at": released.isoformat(), "forecast_before_release": None,
            "forecast_seen_at": None, "actual_first": None, "actual_seen_at": None})
        forecast, actual = parse_value(row.forecast), parse_value(row.actual)
        if now < released:
            if forecast is not None and actual is None:
                event["forecast_before_release"], event["forecast_seen_at"] = forecast, now.isoformat()
            continue
        if not with_actuals or actual is None or event["actual_seen_at"] is not None:
            continue
        event["actual_first"], event["actual_seen_at"] = actual, now.isoformat()
        if event["forecast_before_release"] is None:
            event["research_status"] = "no_pre_release_forecast"
            continue
        surprise = actual - event["forecast_before_release"]
        hist_key = f"{row.currency}|{name}"
        previous = state["history"].setdefault(hist_key, [])
        scale = float(np.median(previous[-12:])) if len(previous) >= 3 else None
        previous.append(abs(surprise))
        del previous[:-12]
        if now - released > pd.Timedelta(minutes=60):
            event["research_status"] = "observed_too_late"
            continue
        if scale is None or surprise == 0 or abs(surprise) < scale:
            event["research_status"] = "below_predeclared_surprise_threshold"
            continue
        # Choose a FUTURE bar opening. The price and ATR are deliberately unknown
        # now; they may later be evaluated from newly downloaded price archives.
        earliest = max(now, released+pd.Timedelta(minutes=5))
        planned = earliest.ceil("15min")
        if planned <= now+pd.Timedelta(minutes=1):
            planned += pd.Timedelta(minutes=15)
        if not 7 <= planned.hour < 17:
            event["research_status"] = "outside_declared_session"
            continue
        event["research_status"] = "recorded_for_paper_research"
        event["planned_entry_at"] = planned.isoformat()
        event["currency_vote"] = int(np.sign(surprise)*POLARITY[name])
    # Related releases can arrive in separate polls. Combine everything known
    # for the same still-future entry; never rewrite a decision after its entry.
    for event in state["events"].values():
        planned = event.get("planned_entry_at")
        if not planned or pd.Timestamp(planned) <= now+pd.Timedelta(minutes=1):
            continue
        vote_key=(planned,event["currency"])
        pending_votes[vote_key]=pending_votes.get(vote_key,0)+event["currency_vote"]
    planned_times = sorted({key[0] for key in pending_votes})
    for planned in planned_times:
        for symbol in SYMBOLS:
            vote = pending_votes.get((planned,symbol[:3]),0)-pending_votes.get((planned,symbol[3:]),0)
            intent_key = f"{symbol}|{planned}"
            existing=next((i for i in state["paper_intents"] if i["id"]==intent_key),None)
            if existing is not None:
                new_side="LONG" if vote>0 else "SHORT" if vote<0 else None
                desired_status="UNEXECUTED_RESEARCH_INTENT" if vote else "CANCELLED_BEFORE_ENTRY"
                if existing["side"] != new_side or existing["status"] != desired_status:
                    existing.setdefault("revisions",[]).append({"observed_at":now.isoformat(),
                        "previous_side":existing["side"],"reason":"newly observed related release before entry"})
                    existing["side"],existing["status"]=new_side,desired_status
                continue
            if not vote:
                continue
            state["paper_intents"].append({"id": intent_key, "symbol": symbol,
                "side": "LONG" if vote > 0 else "SHORT", "recorded_at": now.isoformat(),
                "planned_entry_at": planned, "target_usd": 30., "planned_loss_usd": 20.,
                "stop_atr": 3., "holding_hours": 4, "status": "UNEXECUTED_RESEARCH_INTENT",
                "cost_pips": (measured_spread_pips(cost_profile or {},symbol,
                    pd.Timestamp(planned).tz_convert("America/New_York").hour) or 1.5)+ECN_COMMISSION_PIPS,
                "fee_scenario": "forecast-time known hourly median spread + 0.7 pip commission",
                "trade_permission": False, "price": None, "lot": None})
    return state


def evaluate_paper_outcomes(state, now):
    """Label pre-recorded future intents only after complete price paths arrive."""
    signatures = state.setdefault("price_signatures", {})
    errors = {}
    for symbol in SYMBOLS:
        pending = [i for i in state["paper_intents"] if i["symbol"] == symbol
                   and i["status"] not in {"SIMULATED_FORWARD_OUTCOME","CANCELLED_BEFORE_ENTRY"}
                   and pd.Timestamp(i["planned_entry_at"])+pd.Timedelta(hours=4) < now]
        if not pending:
            continue
        paths = [ROOT/f"data/forward_prices/historical_{tf}/{symbol}.csv" for tf in ("15m","5m")]
        if not all(p.exists() for p in paths):
            continue
        signature = [(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        if signatures.get(symbol) == [list(pair) for pair in signature]:
            continue
        signatures[symbol] = [list(pair) for pair in signature]
        try:
            bars, fills = [load_hourly(p,quarantine=True) for p in paths]
            bars.attrs["symbol"] = symbol
            atr = compute_atr(bars,14)
            for intent in pending:
                entry = pd.Timestamp(intent["planned_entry_at"])
                position = bars.index.get_indexer([entry-pd.Timedelta(minutes=15)])[0]
                if position < 128 or entry+pd.Timedelta(hours=4) > fills.index[-1]+pd.Timedelta(minutes=5):
                    intent["status"] = "AWAITING_COMPLETE_PRICE_HISTORY"
                    continue
                # An unexplained multi-day hole must not contaminate the ATR warmup.
                if (bars.index[max(0,position-128):position+1].to_series().diff()>pd.Timedelta(days=4)).any():
                    intent["status"] = "AWAITING_CONTINUOUS_FEATURE_HISTORY"
                    continue
                states = np.zeros(len(bars),dtype=int)
                states[position] = 1 if intent["side"] == "LONG" else -1
                costs = np.full(len(bars),float(intent.get("cost_pips",2.2)))
                result = simulate(bars,fills,states,atr,costs,str(entry.date()),
                    str((entry+pd.Timedelta(days=1)).date()),3.,4,30.,20.,entries=[position])
                if result.empty:
                    intent["status"] = "AWAITING_COMPLETE_PRICE_HISTORY"
                    continue
                row=result.iloc[0]
                intent["status"] = "SIMULATED_FORWARD_OUTCOME"
                intent["outcome"] = {"evaluated_at":now.isoformat(),"pnl_usd":float(row.pnl_usd),
                    "stress_pnl_usd":float(row.pnl_stress_usd),"target_reached":row.reason in {"target","target_gap"},
                    "exit_reason":str(row.reason),"exit_time":str(row.exit_time),
                    "price_source":"Dukascopy BID archive + stated fee scenario; no broker execution"}
                # Preserve the actual prices behind this result before rolling
                # archives or provider revisions can replace them.
                evidence=ROOT/"research/news_forward/outcomes"
                evidence.mkdir(parents=True,exist_ok=True)
                key=intent["id"].replace("|","_").replace(":","-")
                fills.loc[entry:entry+pd.Timedelta(hours=4)].to_csv(evidence/f"{key}_5m.csv")
                bars.iloc[max(0,position-128):position+1].to_csv(evidence/f"{key}_context15m.csv")
        except (OSError,ValueError) as exc:
            errors[symbol]=type(exc).__name__
    state["price_errors"] = errors
    return state


def poll_once():
    OUT.mkdir(parents=True, exist_ok=True)
    requested_at = pd.Timestamp.now(tz="UTC")
    calendar, with_actuals = fetch_calendar(today=requested_at.date())
    # Availability is when the response has arrived, not when the request began.
    now = pd.Timestamp.now(tz="UTC")
    ledger = OUT / "ledger.json"
    state = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else initial_state(now)
    write_json(OUT / f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.json", {
        "observed_at": now.isoformat(), "with_actuals": with_actuals,
        "rows": [{c: str(v) for c, v in row.items()} for row in calendar.to_dict("records")]})
    state = record_snapshot(state, calendar, now, with_actuals, load_spread_profile())
    # Persist the future decision before any potentially slow historical replay.
    write_json(ledger, state)
    state = evaluate_paper_outcomes(state, now)
    write_json(ledger, state)
    result = {"checked_at": now.isoformat(), "with_actuals": with_actuals,
        "status": "recording" if with_actuals else "forecast_only_no_actuals",
        "tracked_events": len(state["events"]), "paper_intents": len(state["paper_intents"]),
        "evaluated_paper_intents": sum(i["status"] == "SIMULATED_FORWARD_OUTCOME" for i in state["paper_intents"]),
        "broker_orders": 0, "trade_permission": False, "error": None}
    write_json(OUT / "monitor.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    if args.watch and args.watch < 60:
        parser.error("Watch interval must be at least 60 seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"monitor.pid").write_text(str(os.getpid()),encoding="utf-8")
    while True:
        try:
            result = poll_once()
            print(json.dumps(result), flush=True)
        except Exception as exc:
            OUT.mkdir(parents=True, exist_ok=True)
            # Avoid exposing network/proxy details; keep last valid ledger intact.
            write_json(OUT / "monitor.json", {"checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "fetch_failed", "error": type(exc).__name__, "trade_permission": False})
            print(f"Calendar recorder: {type(exc).__name__}", flush=True)
            if not args.watch:
                raise SystemExit(1)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
