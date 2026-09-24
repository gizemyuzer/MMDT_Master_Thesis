"""Reproduce baseline and portfolio evaluation from reduced inputs and NPZ archives.
Run: python evaluate_portfolios.py INPUTS.zip NPZ.zip OUTPUT_DIRECTORY
No training. Fixed 20-session schedule, 20% exclusion, 10bp per traded dollar.
"""
import sys,io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef

def rebalance(h,cash,chosen,cost):
    nav=h.sum()+cash;w=np.zeros(len(h));w[chosen]=1/len(chosen)
    lo,hi=0.,nav
    for _ in range(60):
        v=(lo+hi)/2
        if v+cost*np.abs(v*w-h).sum()>nav:hi=v
        else:lo=v
    new=lo*w;fee=cost*np.abs(new-h).sum();remaining=nav-fee-new.sum()
    assert remaining>=-1e-12 and abs(new.sum()+remaining+fee-nav)<1e-10
    return new,max(0.,remaining),fee

def tests():
    h,c,f=rebalance(np.zeros(10),1.,np.arange(10),.01)
    assert abs(h.sum()-1/1.01)<1e-12 and c<1e-12
    h=np.ones(10)/10;h[0]*=2;h[0]*=.5;assert abs(h.sum()-1)<1e-12
    eq=np.array([1.,.9,.95]);assert np.isclose((eq/np.maximum.accumulate(eq)-1).min(),-.1)
    # Terminal return settles to cash exactly once.
    h=np.array([.5,.5]);h*=np.array([.8,1.]);cash=h[0];h[0]=0
    assert np.isclose(h.sum()+cash,.9)

def threshold(y,s):
    order=np.argsort(-s,kind='stable');v=s[order];yy=y[order];ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1]
    tp=np.cumsum(yy)[ends];fp=ends+1-tp;fn=y.sum()-tp;tn=len(y)-y.sum()-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));m=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(tp,dtype=float),where=den>0)
    return float(v[ends[np.argmax(m)]])

def main(inp,preds,out):
    tests();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(inp) as z:
        p=pd.read_csv(z.open('technical_panel_subset.csv'),parse_dates=['date']);l=pd.read_csv(z.open('return_ledger_subset.csv'),parse_dates=['date'])
    for d in (p,l):assert not d.duplicated(['date','permno']).any()
    assert not l.UnresolvedExit.any() and np.isfinite(l.TotalReturn).all() and (l.TotalReturn>=-1).all()
    pi=p.set_index(['date','permno']);forecasts={};base={}
    with zipfile.ZipFile(preds) as z:
        for g in ['I','I+II','I+II+IV']:
            for seed in range(42,53):
                for split in ['val','test']:
                    with np.load(io.BytesIO(z.read(f'npz/{g}/{seed}/{split}_predictions.npz')),allow_pickle=False) as a:
                        df=pd.DataFrame({'date':a['dates'],'permno':a['permnos'],'y':a['y_true'],'score':a['y_prob']}).set_index(['date','permno']).sort_index()
                    assert df.index.is_unique
                    if split not in base:base[split]=df
                    assert df.index.equals(base[split].index) and np.array_equal(df.y,base[split].y)
                    assert np.array_equal(pi.loc[df.index,'Target'],df.y)
                    if split=='test':forecasts[(g,seed)]=df.score
    baselines={}
    for name,sign in [('Inverse volatility',-1),('Volatility',1)]:
        v=sign*pi.loc[base['val'].index,'Vol_20d'].to_numpy();t=sign*pi.loc[base['test'].index,'Vol_20d'].to_numpy();assert np.isfinite(v).all() and np.isfinite(t).all()
        cut=threshold(base['val'].y.to_numpy(),v);y=base['test'].y
        baselines[name]={'threshold':cut,'roc_auc':roc_auc_score(y,t),'pr_auc':average_precision_score(y,t),'mcc':matthews_corrcoef(y,t>=cut)}
    calendar=np.sort(p.date.unique());start=base['test'].index.get_level_values('date').min();end=base['test'].index.get_level_values('date').max();dates=calendar[(calendar>=start)&(calendar<=end)]
    ids=np.sort(p.permno.unique());col={v:i for i,v in enumerate(ids)}
    rr=l.pivot(index='date',columns='permno',values='TotalReturn').reindex(index=dates,columns=ids).to_numpy()
    exits=l.pivot(index='date',columns='permno',values='ExitAfterReturn').reindex(index=dates,columns=ids).fillna(False).to_numpy(dtype=bool)
    prices=p.pivot(index='date',columns='permno',values='Close').reindex(index=dates,columns=ids).to_numpy()
    decisions={};eligible_counts=[]
    for k in range(0,len(dates)-1,20):
        day=pd.Timestamp(dates[k]);df=base['test'].loc[day];ready=p.loc[(p.date==day)&p.Sequence20Ready,'permno']
        assert set(ready)==set(df.index),f'Missing inference on {day}'
        # Execution eligibility uses observations available at next close, not future returns.
        eligible=[v for v in df.index if np.isfinite(prices[k+1,col[v]]) and prices[k+1,col[v]]>0 and not exits[k+1,col[v]]]
        decisions[k+1]=(day,np.array(eligible,dtype=int));eligible_counts.append(len(eligible))
    rows=[];curves={}
    strategies=[('Equal weight',None),('Buy and hold',None),('Inverse volatility',None),('Oracle binary',None)]+list(forecasts)
    for name,seed in strategies:
        h=np.zeros(len(ids));cash=1.;equity=[1.];fees=0.;settlements=0
        for k in range(len(dates)):
            held=h>0
            if held.any():
                assert np.isfinite(rr[k,held]).all(),f'Missing held return {name} {dates[k]}'
                h[held]*=1+rr[k,held]
                ex=held&exits[k];settlements+=int(ex.sum());cash+=h[ex].sum();h[ex]=0
            if k in decisions and (name!='Buy and hold' or k==1):
                day,eligible=decisions[k];chosen=eligible
                if name not in ['Equal weight','Buy and hold']:
                    if name=='Inverse volatility':scores=-pi.loc[[(day,v) for v in eligible],'Vol_20d'].to_numpy()
                    elif name=='Oracle binary':scores=base['test'].loc[day].loc[eligible,'y'].to_numpy()
                    else:scores=forecasts[(name,seed)].loc[day].loc[eligible].to_numpy()
                    # Descending risk, ascending PERMNO for ties; fixed floor(20% n).
                    ranked=np.lexsort((eligible,-scores));chosen=eligible[ranked[int(np.floor(.2*len(eligible))):]]
                h,cash,fee=rebalance(h,cash,np.array([col[v] for v in chosen]),.001);fees+=fee
            nav=h.sum()+cash;assert nav>0 and cash>=-1e-12;equity.append(nav)
        eq=np.array(equity);dd=float((eq/np.maximum.accumulate(eq)-1).min());years=(pd.Timestamp(dates[-1])-pd.Timestamp(dates[1])).days/365.25;cagr=eq[-1]**(1/years)-1
        rows.append({'strategy':name,'seed':seed,'cagr':cagr,'max_drawdown':dd,'calmar':cagr/abs(dd),'terminal_nav':eq[-1],'fees_per_initial_capital':fees,'settlements':settlements});curves[f'{name}_{seed}']=eq[1:].tolist()
    result={'baseline_classification':baselines,'portfolios':rows,'start_signal':str(start),'first_execution':str(pd.Timestamp(dates[1])),'end':str(end),'rebalance_count':len(decisions),'eligible_min':min(eligible_counts),'eligible_max':max(eligible_counts),'test_prevalence':float(base['test'].y.mean()),'sha256':{str(x):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in [inp,preds]},'self_tests':'PASS'}
    (out/'portfolio_results.json').write_text(json.dumps(result,indent=2));(out/'equity_curves.json').write_text(json.dumps({'dates':[str(pd.Timestamp(d).date()) for d in dates],'curves':curves}))
    print(json.dumps({k:v for k,v in result.items() if k!='portfolios'},indent=2));df=pd.DataFrame(rows);print(df.groupby('strategy')[['cagr','max_drawdown','calmar']].agg(['mean','std']).to_string());print('beat EW',df[df.seed.notna()].groupby('strategy').calmar.apply(lambda s:int((s>df.iloc[0].calmar).sum())).to_dict())
if __name__=='__main__':main(*sys.argv[1:])
