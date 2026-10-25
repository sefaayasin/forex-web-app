"""Paper-motivated London post-fix screen; fixed rules, no parameter selection.

Adaptation, not replication: EURUSD/GBPUSD long at 16:15 London, 1/4 hours.
Compare timed price drift with $30 net-target/$20 planned-stop adaptation.
"""
import numpy as np
import pandas as pd
from forex_adaptive import ROOT, write_json, digest
from forex_ml_tournament import load_hourly
from forex_indicators import compute_atr
from forex_config import get_pip_size
from forex_costs import load_spread_profile, measured_spread_pips, ECN_COMMISSION_PIPS
from research_cash_target import simulate, describe, prepare_paths, day_test

OUT=ROOT/'research/fixing'
SOURCE='https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',{'source':SOURCE,'adaptation_not_replication':True,
        'entry':'16:15 Europe/London, buy EURUSD and GBPUSD separately',
        'holding_hours':[1,4],'target_usd':30,'planned_risk_usd':20,'stop_atr':3,
        'periods':[['2016-01-01','2020-01-01'],['2020-01-01','2023-01-01'],['2023-01-01','archive_end']],
        'all_results_reported':True,'previously_inspected_history':True,
        'trade_permission':False,'code_hash':digest(ROOT/'research_fixing.py')})
    rows=[]
    profile=load_spread_profile()
    for symbol in ['EURUSD','GBPUSD']:
        bars,fills=[load_hourly(ROOT/f'data/historical_{tf}/{symbol}.csv',quarantine=True) for tf in ('15m','5m')]
        bars.attrs['symbol']=symbol
        prepared=prepare_paths(bars,fills)
        atr=compute_atr(bars,14)
        decisions=bars.index+pd.Timedelta(minutes=15)
        local=decisions.tz_convert('Europe/London')
        state=((local.hour==16)&(local.minute==15)).astype(int)
        entries=np.flatnonzero(state)
        ny=decisions.tz_convert('America/New_York').hour
        costs=np.array([(measured_spread_pips(profile,symbol,h) or 1.5)+ECN_COMMISSION_PIPS for h in ny])
        end=str((fills.index[-1]+pd.Timedelta(days=1)).date())
        for hold in [1,4]:
            for start,finish in [('2016-01-01','2020-01-01'),('2020-01-01','2023-01-01'),('2023-01-01',end)]:
                trades=simulate(bars,fills,state,atr,costs,start,finish,3.,hold,30.,20.,entries=entries,prepared=prepared)
                timed=[]
                for i in entries:
                    when=decisions[i]
                    if not pd.Timestamp(start,tz='UTC')<=when<pd.Timestamp(finish,tz='UTC') or prepared['history_bad'][i]:continue
                    j=fills.index.searchsorted(when); n=hold*12
                    if j+n>len(fills) or fills.index[j]!=when or fills.index[j+n-1]!=when+pd.Timedelta(hours=hold,minutes=-5):continue
                    if when+pd.Timedelta(hours=hold)>=pd.Timestamp(finish,tz='UTC'):continue
                    if prepared['bad_sum'][j+n]!=prepared['bad_sum'][j]:continue
                    gross=(prepared['values'][j+n-1,3]-prepared['values'][j,0])/get_pip_size(symbol)
                    timed.append({'entry_time':when,'gross_pips':gross,'net_pips':gross-costs[i],'stress_pips':gross-2*costs[i],'cost_pips':costs[i]})
                timed=pd.DataFrame(timed)
                row={'symbol':symbol,'hours':hold,'start':start,'end':finish,**describe(trades,loss_usd=20.),
                    'timed_trades':len(timed),'timed_gross_pips':float(timed.gross_pips.mean()),
                    'timed_net_pips':float(timed.net_pips.mean()),'timed_stress_pips':float(timed.stress_pips.mean())}
                stem=f'{symbol}_{hold}h_{start}'
                write_json(OUT/f'{stem}.json',{'summary':row,'bracket_test':day_test(trades,'pnl_usd'),
                    'timed_test':day_test(timed,'net_pips'),'trade_permission':False})
                trades.to_csv(OUT/f'{stem}_bracket.csv.gz',index=False,compression='gzip')
                timed.to_csv(OUT/f'{stem}_timed.csv.gz',index=False,compression='gzip')
                rows.append(row)
                print(row,flush=True)
        write_json(OUT/f'{symbol}_sources.json',{tf:digest(ROOT/f'data/historical_{tf}/{symbol}.csv') for tf in ('15m','5m')})
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)

if __name__=='__main__':main()
