"""Bounded causal session / cross-market research. No live promotion."""
import itertools
import numpy as np
import pandas as pd
from forex_adaptive import ROOT, MAJORS, write_json, digest
from forex_ml_tournament import load_hourly
from forex_indicators import compute_atr
from forex_costs import load_spread_profile, measured_spread_pips, ECN_COMMISSION_PIPS
from research_cash_target import simulate, describe, prepare_paths, day_test

OUT = ROOT / 'research/session_relative'
PERIODS = [('2016-01-01','2019-01-01'),('2019-01-01','2023-01-01')]

def rules(bars, peers, symbol):
    local = bars.index.tz_convert('Europe/London')
    day = local.normalize()
    before = local.hour < 8
    hi = bars.High.where(before).groupby(day).cummax().groupby(day).ffill()
    lo = bars.Low.where(before).groupby(day).cummin().groupby(day).ffill()
    count = pd.Series(before.astype(int), index=bars.index).groupby(day).cumsum()
    window = (local.hour >= 8) & (local.hour < 10) & (count >= 24)
    state = lambda long, short: np.select([long,short],[1,-1],default=0).astype(int)
    breakout = state(window & (bars.Close > hi),window & (bars.Close < lo))
    rejection = state(window & (bars.Low < lo) & (bars.Close > lo),window & (bars.High > hi) & (bars.Close < hi))
    atr = compute_atr(bars,14)
    anchor = bars.Close.groupby(day).expanding().mean().reset_index(level=0,drop=True).reindex(bars.index)
    revert = state(window & (bars.Close < anchor-2*atr),window & (bars.Close > anchor+2*atr))
    sign = 1 if symbol.endswith('USD') else -1
    own = np.log(bars.Close).diff(16) * sign
    other = peers.drop(columns=symbol).reindex(bars.index)
    basket = other.mean(axis=1).where(other.notna().all(axis=1))
    scale = own.rolling(480,min_periods=400).std().shift(1)
    residual = own-basket
    lead = state((basket>scale) & (own<basket*.5), (basket < -scale) & (own>basket*.5))*sign
    fade = state(residual < -2*scale,residual > 2*scale)*sign
    follow = -fade
    return {'london_break':breakout,'london_reject':rejection,'session_mean_revert':revert,
            'peer_catchup':lead,'peer_residual_revert':fade,'peer_relative_momentum':follow}, atr

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',{'development':PERIODS,'audit_start':'2023-01-01',
        'audit_previously_inspected':True,'target_usd':30,'risk_usd':[20,50,100],
        'stop_atr':3,'hold_hours':[1,4],'minimum_development_trades':100,
        'sources':['https://arxiv.org/abs/1411.2153','https://arxiv.org/abs/1103.5664'],
        'source_use':'Motivation only; original research rules, not paper replications.',
        'trade_permission':False,'code_hash':digest(ROOT/'research_session_relative.py')})
    archives={s:load_hourly(ROOT/f'data/historical_15m/{s}.csv',quarantine=True) for s in MAJORS}
    peer_data={}
    for s,b in archives.items():
        bad=pd.Series(False,index=b.index)
        bad.iloc[b.attrs.get('bad_positions',[])]=True
        valid=~bad.rolling(501,min_periods=1).max().astype(bool)
        peer_data[s]=(np.log(b.Close).diff(16)*(1 if s.endswith('USD') else -1)).where(valid)
    peers=pd.DataFrame(peer_data)
    profile=load_spread_profile()
    summaries=[]
    for symbol,bars in archives.items():
        bars.attrs['symbol']=symbol
        fills=load_hourly(ROOT/f'data/historical_5m/{symbol}.csv',quarantine=True)
        states,atr=rules(bars,peers,symbol)
        # Prefix invariance verifies that future candles cannot change past signals.
        cut=len(bars)-1000
        truncated,_=rules(bars.iloc[:cut],peers.iloc[:0].reindex(bars.index[:cut]).combine_first(peers.reindex(bars.index[:cut])),symbol)
        for name in states: np.testing.assert_array_equal(states[name][:cut],truncated[name])
        prepared=prepare_paths(bars,fills)
        ny=(bars.index+pd.Timedelta(minutes=15)).tz_convert('America/New_York').hour
        costs=np.array([(measured_spread_pips(profile,symbol,h) or 1.5)+ECN_COMMISSION_PIPS for h in ny])
        trials=[]
        for name,hold,risk in itertools.product(states,(1,4),(20.,50.,100.)):
            metrics=[describe(simulate(bars,fills,states[name],atr,costs,a,z,3.,hold,30.,risk,prepared=prepared),loss_usd=risk) for a,z in PERIODS]
            trials.append({'rule':name,'hours':hold,'risk':risk,'development':metrics})
        eligible=[t for t in trials if all(m['trades']>=100 for m in t['development'])]
        selected=max(eligible,key=lambda t:min(m['stress_mean_usd'] for m in t['development'])) if eligible else None
        folder=OUT/symbol
        folder.mkdir(exist_ok=True)
        write_json(folder/'selection.json',{'selected':selected,'trials':trials})
        if selected is None: continue
        end=str((fills.index[-1]+pd.Timedelta(days=1)).date())
        sample=simulate(bars,fills,states[selected['rule']],atr,costs,'2023-01-01',end,3.,selected['hours'],30.,selected['risk'],prepared=prepared)
        metrics=describe(sample,loss_usd=selected['risk'])
        normal,stress=day_test(sample,'pnl_usd'),day_test(sample,'pnl_stress_usd')
        positive=all(m['stress_mean_usd']>0 for m in selected['development'])
        passed=positive and len(sample)>=300 and stress['ci_low'] is not None and stress['ci_low']>0 and stress['p_value']*len(MAJORS)<=.05
        write_json(folder/'report.json',{'selected':selected,'audit':metrics,'normal_test':normal,'stress_test':stress,'screen_passed':passed,'trade_permission':False,
            'limitations':['Previously inspected historical audit, no fresh holdout.','Estimated spreads, no broker lot/margin simulation.','Cross-market alignment uses exact timestamps, no forward fill.'],
            'source_hashes':{tf:digest(ROOT/f'data/historical_{tf}/{symbol}.csv') for tf in ('15m','5m')}})
        sample.to_csv(folder/'audit.csv.gz',index=False,compression='gzip')
        row={'symbol':symbol,'rule':selected['rule'],'risk':selected['risk'],'hours':selected['hours'],**metrics,'screen_passed':passed}
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
        print(f'{symbol}: {len(trials)} configs; {row}',flush=True)

if __name__=='__main__':main()
