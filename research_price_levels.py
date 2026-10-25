"""Fixed round-number events motivated by Osler, not a paper replication."""
import itertools
import numpy as np
import pandas as pd
from forex_adaptive import ROOT, digest, write_json
from forex_ml_tournament import load_hourly
from forex_indicators import compute_atr
from forex_config import get_pip_size
from forex_costs import load_spread_profile, measured_spread_pips, ECN_COMMISSION_PIPS
from research_cash_target import simulate, describe, prepare_paths, day_test

OUT=ROOT/'research/price_levels'
PERIODS=[('2016-01-01','2019-01-01'),('2019-01-01','2023-01-01')]
SYMBOLS=['EURUSD','GBPUSD','USDJPY']

def level_states(bars, pip):
    atr=compute_atr(bars,14)
    previous=bars.Close.shift(1)
    valid=(bars.index.to_series().diff()==pd.Timedelta(minutes=15))
    rules={}
    for step in [50,100]:
        spacing=step*pip
        lower=np.floor(previous/spacing)*spacing
        upper=lower+spacing
        buffer=.1*atr
        for name,long,short in [
            ('break',(bars.Close>upper+buffer),(bars.Close<lower-buffer)),
            ('reject',(bars.Low<lower)&(bars.Close>lower+buffer)&(bars.Close>bars.Open),
                (bars.High>upper)&(bars.Close<upper-buffer)&(bars.Close<bars.Open))]:
            rules[f'{name}_{step}pip']=np.select([valid&long&~short,valid&short&~long],[1,-1],default=0).astype(int)
    return rules,atr

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',{'source':'https://www.newyorkfed.org/research/staff_reports/sr125.html',
        'replication':False,'rules':'50/100 pip round levels fixed from previous close; 0.1 ATR close confirmation; ambiguous events ignored',
        'development':PERIODS,'audit_start':'2023-01-01','audit_previously_inspected':True,
        'stop_atr':3,'hold_hours':[1,4],'risk_usd':[20,50,100],'target_usd':30,
        'selection':'Minimum 100 trades per development period, maximize worst-period stress net',
        'code_hash':digest(ROOT/'research_price_levels.py'),'trade_permission':False})
    summaries=[]
    profile=load_spread_profile()
    for symbol in SYMBOLS:
        bars,fills=[load_hourly(ROOT/f'data/historical_{tf}/{symbol}.csv',quarantine=True) for tf in ['15m','5m']]
        bars.attrs['symbol']=symbol
        states,atr=level_states(bars,get_pip_size(symbol))
        prefix,_=level_states(bars.iloc[:-1000],get_pip_size(symbol))
        for name in states:np.testing.assert_array_equal(states[name][:-1000],prefix[name])
        prepared=prepare_paths(bars,fills)
        ny=(bars.index+pd.Timedelta(minutes=15)).tz_convert('America/New_York').hour
        costs=np.array([(measured_spread_pips(profile,symbol,h) or 1.5)+ECN_COMMISSION_PIPS for h in ny])
        trials=[]
        for name,hold,risk in itertools.product(states,[1,4],[20.,50.,100.]):
            metrics=[describe(simulate(bars,fills,states[name],atr,costs,a,b,3.,hold,30.,risk,prepared=prepared),loss_usd=risk) for a,b in PERIODS]
            trials.append({'rule':name,'hours':hold,'risk':risk,'development':metrics})
        eligible=[t for t in trials if all(p['trades']>=100 for p in t['development'])]
        selected=max(eligible,key=lambda t:min(p['stress_mean_usd'] for p in t['development'])) if eligible else None
        folder=OUT/symbol
        folder.mkdir(exist_ok=True)
        write_json(folder/'selection.json',{'selected':selected,'trials':trials})
        if selected is None:
            print(symbol,'insufficient development samples',flush=True)
            continue
        end=str((fills.index[-1]+pd.Timedelta(days=1)).date())
        def audit(fee):
            return simulate(bars,fills,states[selected['rule']],atr,fee,'2023-01-01',end,3.,selected['hours'],30.,selected['risk'],prepared=prepared)
        trades=audit(costs)
        normal,stress=day_test(trades,'pnl_usd'),day_test(trades,'pnl_stress_usd')
        metrics=describe(trades,loss_usd=selected['risk'])
        sensitivity={str(fee):describe(audit(np.full(len(bars),fee)),loss_usd=selected['risk']) for fee in [0.,.5,1.,2.]}
        years={str(y):describe(g,loss_usd=selected['risk']) for y,g in trades.groupby(trades.entry_time.dt.year)} if len(trades) else {}
        checks={'development_positive':all(p['stress_mean_usd']>0 for p in selected['development']),
            'sample_size':len(trades)>=300,'stress_lower_ci_positive':stress['ci_low'] is not None and stress['ci_low']>0,
            'adjusted_p':stress.get('p_value') is not None and stress['p_value']*len(SYMBOLS)<.05,
            'all_years_positive':bool(years) and all(v['mean_usd']>0 for v in years.values())}
        write_json(folder/'report.json',{'symbol':symbol,'selected':selected,'audit':metrics,'normal_test':normal,'stress_test':stress,
            'fee_scenarios_pips':sensitivity,'years':years,'checks':checks,'screen_passed':all(checks.values()),'trade_permission':False,
            'limitations':['No actual stop-order locations available; round prices are a proxy.','Previously inspected historical audit.','Generic fee model; no real brokerage lot/margin execution.','Zero-fee case is diagnostic only.'],
            'price_hashes':{tf:digest(ROOT/f'data/historical_{tf}/{symbol}.csv') for tf in ['15m','5m']}})
        trades.to_csv(folder/'audit.csv.gz',index=False,compression='gzip')
        row={'symbol':symbol,'rule':selected['rule'],'risk':selected['risk'],'hours':selected['hours'],**metrics,'screen_passed':all(checks.values())}
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
        print(row,flush=True)

if __name__=='__main__':main()
