"""Fast bounded news-rule sensitivity test; each price archive loaded once.

All periods were previously inspected: this is retrospective development,
not a fresh blind test. Selection uses two early subperiods only.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from forex_adaptive import ROOT, digest, write_json
from forex_costs import load_spread_profile, measured_spread_pips, ECN_COMMISSION_PIPS
from forex_indicators import compute_atr
from forex_ml_tournament import load_hourly
from research_calendar_sources import load_forexfactory
from research_news_surprise import surprise_signals, pair_states
from research_cash_target import simulate, describe, prepare_paths, day_test
from research_new_indicators import SYMBOLS

OUT = ROOT / "research/news_fast"
DEVELOPMENT = (("2024-10-01","2025-04-01"),("2025-04-01","2025-10-01"))

def choose(rows):
    eligible = [r for r in rows if all(p["trades"] >= 10 for p in r["development"])]
    return max(eligible,key=lambda r:min(p["stress_mean_usd"] for p in r["development"])) if eligible else None

def run(symbol, base_signals):
    bars,fills=[load_hourly(ROOT/f"data/historical_{tf}/{symbol}.csv",quarantine=True) for tf in ("15m","5m")]
    bars.attrs["symbol"]=symbol
    atr=compute_atr(bars,14)
    prepared=prepare_paths(bars,fills)
    profile=load_spread_profile()
    hours=(bars.index+pd.Timedelta(minutes=15)).tz_convert("America/New_York").hour
    costs=np.array([(measured_spread_pips(profile,symbol,h) or 1.5)+ECN_COMMISSION_PIPS for h in hours])
    schedules={}
    for delay, strength, direction in itertools.product((5,30,60),(1.,2.),(1,-1)):
        signals=base_signals[base_signals.surprise.abs()>=strength*base_signals.past_error_scale].copy()
        signals["decision_at"]=(signals.time+pd.Timedelta(minutes=delay)).dt.ceil("15min")
        states,entries=pair_states(signals,symbol,bars.index)
        schedules[delay,strength,direction]=(states*direction,entries)
    trials=[]
    for delay,strength,direction,budget,stop,hold in itertools.product((5,30,60),(1.,2.),(1,-1),(20.,50.,100.),(1.5,3.),(1,4)):
        states,entries=schedules[delay,strength,direction]
        periods=[describe(simulate(bars,fills,states,atr,costs,start,end,stop,hold,30.,budget,
            entries=entries,prepared=prepared),loss_usd=budget) for start,end in DEVELOPMENT]
        trials.append({"delay_minutes":delay,"surprise_multiplier":strength,"direction":direction,
            "risk_usd":budget,"stop_atr":stop,"hours":hold,"development":periods})
    selected=choose(trials)
    folder=OUT/symbol
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/"selection.json",{"selected":selected,"trials":trials})
    if selected is None:
        print(f"{symbol}: {len(trials)} settings, insufficient samples",flush=True)
        return {"symbol":symbol,"status":"INSUFFICIENT_SAMPLES","trade_permission":False,"trials":len(trials)}
    states,entries=schedules[selected["delay_minutes"],selected["surprise_multiplier"],selected["direction"]]
    end=str((fills.index[-1]+pd.Timedelta(days=1)).date())
    audit=simulate(bars,fills,states,atr,costs,"2025-10-01",end,selected["stop_atr"],selected["hours"],
        30.,selected["risk_usd"],entries=entries,prepared=prepared)
    metrics=describe(audit,loss_usd=selected["risk_usd"])
    test=day_test(audit,"pnl_usd")
    report={"symbol":symbol,"status":"RETROSPECTIVE_ONLY","trade_permission":False,
        "selected":selected,"audit":metrics,"day_test":test,"trials":len(trials),
        "development_both_positive":all(p["mean_usd"]>0 and p["stress_mean_usd"]>0 for p in selected["development"]),
        "limitations":["Audit already inspected, no fresh holdout.","Low sample count; no live authorization.",
            "Calendar vintages/feed timing and actual news spreads unverified."]}
    write_json(folder/"report.json",report)
    if len(audit):audit.to_csv(folder/"audit.csv.gz",index=False,compression="gzip")
    print(f"{symbol}: {len(trials)} settings; delay={selected['delay_minutes']}m, side={selected['direction']}, "
          f"risk=${selected['risk_usd']:g}; audit n={len(audit)}, net={metrics['mean_usd']}, stress={metrics['stress_mean_usd']}",flush=True)
    return report

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"protocol.json",{"date":"2026-10-08","development":DEVELOPMENT,"audit_start":"2025-10-01",
        "audit_already_inspected":True,"minimum_trades_per_development_half":10,
        "selection":"highest worst-half stress mean, selection on early periods only",
        "delay_minutes":[5,30,60],"surprise_multiplier":[1,2],"directions":[1,-1],
        "risk_usd":[20,50,100],"stop_atr":[1.5,3],"hours":[1,4],"target_usd":30,
        "code_sha256":digest(Path(__file__))})
    signals=surprise_signals(load_forexfactory())
    results=[run(symbol,signals) for symbol in SYMBOLS]
    rows=[{"symbol":r["symbol"],"trials":r["trials"],"status":r["status"],
        **r.get("audit",{}),"both_development_halves_positive":r.get("development_both_positive",False)} for r in results]
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=="__main__":main()
