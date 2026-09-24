
"""Build a versioned PERMNO technical panel; no network, training, or old-cache writes.

Requires numpy, pandas and TA-Lib. No approximate indicator fallback is allowed.
Default inputs are the existing exported daily_prices.csv, delistings.csv and
datasets/universe.csv. The original feature_engineering module is NOT imported.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd

FEATURES = [
 'RSI','MACD_Hist_Rel','MACD_Hist_Slope_Rel','BB_Pct','BB_Width',
 'Returns','Vol_20d','Vol_5d','Current_Drawdown_5d','Current_Drawdown_20d',
 'Current_Return_5d','ADX_14','Price_vs_SMA50','Price_vs_SMA200','ROC_10',
 'Dist_to_Max252d','Rel_Volume','Vol_Trend','Price_vs_Max20d','RSI_vs_Max20d',
 'RSI_Divergence','Vol_Dist_Ratio','Rel_Returns','Rel_RSI','Rel_Vol_20d',
 'Rel_MACD_Ratio','Rel_ROC_10','Rel_ATR_Ratio','CCI_14','MFI_14','WILLR_14','OBV_Z20']
SECTOR_BASES = {'Returns':'Rel_Returns','RSI':'Rel_RSI','Vol_20d':'Rel_Vol_20d',
                '_MACD_Ratio':'Rel_MACD_Ratio','ROC_10':'Rel_ROC_10','_ATR_Ratio':'Rel_ATR_Ratio'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def ratio(a,b):
    # No scale-dependent epsilon. Undefined ratios stay missing.
    return a/b.where(b.ne(0))

def indicators(g,ta):
    c,h,l,v=(g[x].astype(float) for x in ['Close','High','Low','VolumeAdjusted'])
    ca,ha,la,va=(np.ascontiguousarray(x.values,dtype=np.float64) for x in [c,h,l,v])
    f=pd.DataFrame(index=g.index)
    f['RSI']=ta.RSI(ca,timeperiod=14)
    macd,_,hist=ta.MACD(ca,fastperiod=12,slowperiod=26,signalperiod=9)
    f['_MACD_Ratio']=pd.Series(macd,index=g.index)/c
    f['MACD_Hist_Rel']=pd.Series(hist,index=g.index)/c
    f['MACD_Hist_Slope_Rel']=pd.Series(hist,index=g.index).diff()/c
    upper,mid,lower=ta.BBANDS(ca,timeperiod=20,nbdevup=2,nbdevdn=2,matype=0)
    mid=pd.Series(mid,index=g.index);width=pd.Series(upper-lower,index=g.index)
    f['BB_Pct']=ratio(c-pd.Series(lower,index=g.index),width)
    f['BB_Width']=ratio(width,mid)
    f['Returns']=c.pct_change(fill_method=None)
    f['Vol_20d']=f.Returns.rolling(20).std(ddof=1)*np.sqrt(252)
    f['Vol_5d']=f.Returns.rolling(5).std(ddof=1)*np.sqrt(252)
    for n in [5,20]:
        peak=h.rolling(n).max()
        f[f'Current_Drawdown_{n}d']=ratio(c-peak,peak)
    f['Current_Return_5d']=c.pct_change(5,fill_method=None)
    f['ADX_14']=ta.ADX(ha,la,ca,timeperiod=14)
    f['_ATR_Ratio']=pd.Series(ta.ATR(ha,la,ca,timeperiod=14),index=g.index)/c
    for n in [50,200]:
        sma=pd.Series(ta.SMA(ca,timeperiod=n),index=g.index)
        f[f'Price_vs_SMA{n}']=ratio(c-sma,sma)
    f['ROC_10']=ta.ROC(ca,timeperiod=10)/100
    peak=c.rolling(252).max();f['Dist_to_Max252d']=ratio(c-peak,peak)
    meanvol=v.rolling(20).mean()
    f['Rel_Volume']=ratio(v,meanvol)
    f['Vol_Trend']=ratio(v.rolling(5).mean()-meanvol,meanvol)
    f['Price_vs_Max20d']=ratio(c,c.rolling(20).max())
    f['RSI_vs_Max20d']=ratio(f.RSI,f.RSI.rolling(20).max())
    f['RSI_Divergence']=f.Price_vs_Max20d-f.RSI_vs_Max20d
    up=(v*c.gt(c.shift()).astype(float)).rolling(20).sum()
    down=(v*c.lt(c.shift()).astype(float)).rolling(20).sum()
    f['Vol_Dist_Ratio']=ratio(down,up+down)
    f['CCI_14']=ta.CCI(ha,la,ca,timeperiod=14)
    f['MFI_14']=ta.MFI(ha,la,ca,va,timeperiod=14)
    f['WILLR_14']=ta.WILLR(ha,la,ca,timeperiod=14)
    obv=pd.Series(ta.OBV(ca,va),index=g.index)
    f['OBV_Z20']=ratio(obv-obv.rolling(20).mean(),obv.rolling(20).std(ddof=1))
    return f.replace([np.inf,-np.inf],np.nan)

def forward_target(close,vol,calendar,exit_date=None,dlret=np.nan,horizon=20):
    """Adjusted-price target, terminal cash-equivalent settlement, no dividend reinvestment.
    Every market session must be observed until the exit. Post-exit value is cash.
    Unknown exit censors every window that reaches it, even an earlier breach.
    """
    price=close.reindex(calendar).copy()
    planned=pd.Series(calendar,index=calendar).shift(-horizon)
    endpoint=planned.copy()
    reaches=pd.Series(False,index=calendar)
    known=False
    if exit_date is not None:
        reaches=(price.index<exit_date)&planned.ge(exit_date)
        last=price.get(exit_date,np.nan)
        known=bool(np.isfinite(last) and last>0 and np.isfinite(dlret) and dlret>=-1)
        if known:
            settlement=last*(1+dlret)
            # Observe last traded close AND settlement at exit, then hold cash.
            price.loc[exit_date]=min(last,settlement)
            price.loc[price.index>exit_date]=settlement
            endpoint.loc[reaches]=exit_date
    futures=pd.concat([price.shift(-k) for k in range(1,horizon+1)],axis=1)
    minimum=futures.min(axis=1,skipna=False)
    observed=close.reindex(calendar)
    threshold=(1.5*vol.reindex(calendar)*np.sqrt(horizon/252)).clip(.05,.25)
    valid=observed.gt(0)&minimum.notna()&threshold.notna()&planned.notna()
    if exit_date is not None:
        valid &= observed.index<exit_date
        if not known:valid &= ~reaches
    drawdown=1-minimum/observed
    return pd.DataFrame({'Target':drawdown.ge(threshold).astype(float).where(valid),
        'ForwardDrawdown':drawdown.where(valid),'LabelThreshold':threshold,
        'LabelHorizonEnd':planned,'LabelEndDate':endpoint.where(valid),
        'LabelUsesTerminal':reaches&valid,'UnknownExitWindow':reaches & (not known)},index=calendar)

def sequence_mask(frame,feature_columns=FEATURES,length=20):
    """Must be recomputed after any subsequent filtering or feature join.
    Sequence rows need consecutive SessionID, same PERMNO AND SegmentID.
    """
    finite=np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all(axis=1)
    result=pd.Series(False,index=frame.index)
    for _,g in frame.groupby(['permno','SegmentID'],sort=False):
        good=pd.Series(finite[g.index],index=g.index)
        continuous=g.SessionID.diff().eq(1)
        # A run of 20 rows has 19 consecutive transitions.
        result.loc[g.index]=(good.rolling(length).sum().eq(length)&
            continuous.astype(int).rolling(length-1).sum().eq(length-1)).values
    return result

def self_tests(ta):
    idx=pd.bdate_range('2018-01-01',periods=360)
    x=100*np.exp(.0003*np.arange(360)+.025*np.sin(np.arange(360)/4))
    g=pd.DataFrame({'Close':x,'High':x*1.01,'Low':x*.99,
                    'VolumeAdjusted':1000+100*np.cos(np.arange(360))},index=idx)
    f=indicators(g,ta)
    scaled=g.copy();scaled[['Close','High','Low']]*=7;scaled.VolumeAdjusted*=3
    np.testing.assert_allclose(f.values,indicators(scaled,ta).values,rtol=1e-7,atol=1e-7,equal_nan=True)
    truncated=indicators(g.iloc[:300],ta)
    np.testing.assert_allclose(f.iloc[:300].values,truncated.values,equal_nan=True)
    c=pd.Series(100.,index=idx);v=pd.Series(.2,index=idx)
    end=idx[300]
    for ret,target in [(-1.,1.),(-.5,1.),(.1,0.)]:
        t=forward_target(c.loc[:end],v,idx,end,ret)
        assert t.loc[idx[290],'Target']==target
        assert t.loc[idx[290],'LabelEndDate']==end
        assert pd.isna(t.loc[end,'Target'])
    missing=forward_target(c.loc[:end],v,idx,end,np.nan)
    assert missing.loc[idx[280]:idx[299],'Target'].isna().all()
    gap=c.copy();gap.iloc[295]=np.nan
    assert pd.isna(forward_target(gap,v,idx).loc[idx[290],'Target'])
    assert forward_target(c,v,idx).Target.iloc[-20:].isna().all()
    seq=pd.DataFrame({'permno':1,'SegmentID':1,'SessionID':np.r_[np.arange(25),np.arange(26,51)],'f':1.})
    mask=sequence_mask(seq,['f'])
    assert mask.iloc[19] and not mask.iloc[25:44].any() and mask.iloc[44]
    seq.loc[25:,'SegmentID']=2
    assert not sequence_mask(seq,['f']).iloc[25:44].any()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='datasets/permno_sources_20260917T191937Z_f78c6a09')
    ap.add_argument('--universe',default='datasets/universe.csv')
    ap.add_argument('--delistings',default=None,help='Optional delisting_review.csv or raw delistings.csv')
    ap.add_argument('--output-parent',default='datasets')
    ap.add_argument('--self-test-only',action='store_true')
    args=ap.parse_args()
    try:import talib as ta
    except ImportError:
        raise SystemExit('TA-Lib is required. Run this script in your thesis venv where TA-Lib is installed. No fallback was used.')
    self_tests(ta)
    print('PASS | indicator causality/scale invariance, terminal labels, missing windows, sequence gaps',flush=True)
    if args.self_test_only:return
    src=Path(args.source)
    inputs={'prices':src/'daily_prices.csv','universe':Path(args.universe),
            'delistings':Path(args.delistings) if args.delistings else src/'delistings.csv'}
    hashes={k:sha(p) for k,p in inputs.items()}
    u=pd.read_csv(inputs['universe']);d=pd.read_csv(inputs['prices'],parse_dates=['date'])
    exits=pd.read_csv(inputs['delistings'],parse_dates=['dlstdt'])
    for f in [u,d,exits]:
        ids=pd.to_numeric(f.permno,errors='raise')
        if ids.isna().any() or ids.mod(1).ne(0).any():raise ValueError('Invalid PERMNO')
        f['permno']=ids.astype('int64')
    if u.permno.duplicated().any() or u[['ticker','sector']].isna().any().any():raise ValueError('Invalid universe')
    if d.date.isna().any() or d.duplicated(['permno','date']).any():raise ValueError('Invalid price keys')
    if set(u.permno)!=set(d.permno):raise ValueError('Price/universe PERMNO mismatch')
    codes=pd.to_numeric(exits.dlstcd,errors='raise')
    if codes.isna().any():raise ValueError('Missing exit code')
    active=int(codes.eq(100).sum());exits=exits.loc[~codes.eq(100)].copy()
    if exits.permno.duplicated().any():raise ValueError('Multiple exits require review')
    calendar=pd.DatetimeIndex(sorted(d.date.unique()))
    last_dates=d.groupby('permno').date.max()
    if not exits.dlstdt.eq(exits.permno.map(last_dates)).all():raise ValueError('Exit is not final observed date')
    for col in ['prc','askhi','bidlo','vol','cfacpr','cfacshr','ret']:
        d[col]=pd.to_numeric(d[col],errors='coerce')
    for col in ['cfacpr','cfacshr']:
        if (~np.isfinite(d[col])|d[col].le(0)).any():raise ValueError('Invalid '+col)
    out=Path(args.output_parent)/('corrected_technical_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    meta=u.set_index('permno');parts=[];ledger=[];counts={'segments':0,'invalid_ohlcv_rows':0}
    for number,(permno,raw) in enumerate(d.groupby('permno',sort=True),1):
        raw=raw.set_index('date').sort_index()
        idx=calendar[(calendar>=raw.index.min())&(calendar<=raw.index.max())]
        raw=raw.reindex(idx)
        g=pd.DataFrame(index=idx)
        for dest,col in [('Close','prc'),('High','askhi'),('Low','bidlo')]:
            g[dest]=(raw[col].abs()/raw.cfacpr).where(raw[col].abs().gt(0))
        g['VolumeAdjusted']=raw.vol*raw.cfacshr
        good=np.isfinite(g).all(axis=1)&g.Close.gt(0)&g.Low.gt(0)&g.High.ge(g.Low)&g.VolumeAdjusted.ge(0)
        # All indicator recurrences restart after any invalid OHLCV/session.
        g['SegmentID']=(good&~good.shift(fill_value=False)).cumsum().astype('int64')
        g['InputValid']=good
        local=[]
        for _,seg in g.loc[good].groupby('SegmentID',sort=False):
            local.append(indicators(seg,ta));counts['segments']+=1
        if local:g=g.join(pd.concat(local))
        else:
            for c in FEATURES+list(SECTOR_BASES):
                if c not in g:g[c]=np.nan
        counts['invalid_ohlcv_rows']+=int((~good).sum())
        e=exits.loc[exits.permno.eq(permno)]
        exit_date=None;dlret=np.nan
        if len(e):
            exit_date=e.iloc[0].dlstdt
            dlret=pd.to_numeric(pd.Series([e.iloc[0].dlret]),errors='coerce').iloc[0]
        lab=forward_target(g.Close,g.Vol_20d,calendar,exit_date,dlret)
        g=g.join(lab)
        g['permno']=permno;g['FormationTicker']=meta.loc[permno,'ticker'];g['Sector']=meta.loc[permno,'sector']
        g['SessionID']=calendar.get_indexer(idx)
        g['date']=idx
        g['TradableAtDecision']=good & (idx!=exit_date if exit_date is not None else True)
        r=raw.ret.where(np.isfinite(raw.ret)&raw.ret.ge(-1))
        total=r.copy();isexit=idx==exit_date if exit_date is not None else np.zeros(len(idx),bool)
        known_dl=bool(np.isfinite(dlret) and dlret>=-1)
        if exit_date is not None:total.loc[exit_date]=(1+r.loc[exit_date])*(1+dlret)-1 if known_dl else np.nan
        ledger.append(pd.DataFrame({'date':idx,'permno':permno,'RegularReturn':r.values,
            'TotalReturn':total.values,'ExitAfterReturn':isexit,
            'DelistingReturn':np.where(isexit,dlret,np.nan),
            'UnresolvedExit':isexit & total.isna().values}))
        parts.append(g.reset_index(drop=True))
        if number%50==0:print(f'Processed {number}/{len(u)} PERMNOs',flush=True)
    panel=pd.concat(parts,ignore_index=True).sort_values(['permno','date']).reset_index(drop=True)
    for c,dest in SECTOR_BASES.items():
        median=panel.groupby(['date','Sector'])[c].transform('median')
        panel[dest]=ratio(panel[c],median) if c=='Vol_20d' else panel[c]-median
    panel[FEATURES]=panel[FEATURES].replace([np.inf,-np.inf],np.nan)
    panel['FeaturesComplete']=np.isfinite(panel[FEATURES]).all(axis=1)
    panel['Sequence20Ready']=sequence_mask(panel)&panel.TradableAtDecision
    panel['EligibleEndpoint']=panel.Sequence20Ready & panel.Target.notna()
    split_stats={}
    for name,lo,hi in [('Train','2010-01-01','2019-12-31'),('Validation','2020-01-01','2021-12-31'),('Test','2022-01-01','2024-12-31')]:
        inside=panel.date.between(lo,hi)
        selected=inside&panel.EligibleEndpoint&panel.LabelEndDate.le(hi)
        panel[name+'Endpoint']=selected
        split_stats[name]={'rows':int(inside.sum()),'eligible_before_purge':int((inside&panel.EligibleEndpoint).sum()),
            'purged':int((inside&panel.EligibleEndpoint&panel.LabelEndDate.gt(hi)).sum()),
            'retained':int(selected.sum()),'event_rate':float(panel.loc[selected,'Target'].mean())}
    if panel.loc[panel.UnknownExitWindow,'Target'].notna().any():raise AssertionError('Unknown exit label')
    if panel.loc[panel.EligibleEndpoint,FEATURES].isna().any().any():raise AssertionError('Incomplete eligible feature')
    # Row count and source hashes are checked; no input is overwritten.
    if hashes!={k:sha(p) for k,p in inputs.items()}:raise RuntimeError('Source changed during build')
    public=['date','permno','FormationTicker','Sector','SessionID','SegmentID','Close','High','Low','VolumeAdjusted']+FEATURES+[
      'InputValid','TradableAtDecision','FeaturesComplete','Sequence20Ready','Target','ForwardDrawdown',
      'LabelThreshold','LabelHorizonEnd','LabelEndDate','LabelUsesTerminal','UnknownExitWindow',
      'EligibleEndpoint','TrainEndpoint','ValidationEndpoint','TestEndpoint']
    panel[public].to_csv(out/'technical_panel.csv',index=False)
    pd.concat(ledger,ignore_index=True).to_csv(out/'return_ledger.csv',index=False)
    pd.DataFrame(split_stats).T.to_csv(out/'split_summary.csv')
    pd.DataFrame({'feature':FEATURES}).to_csv(out/'feature_columns.csv',index=False)
    report={'status':'TECHNICAL_PANEL_BUILT_NOT_INTEGRATED_WITH_TRAINER','output_folder':str(out),
      'source_hashes':hashes,'versions':{'pandas':pd.__version__,'numpy':np.__version__,'talib':ta.__version__},
      'universe':len(u),'source_rows':len(d),'output_rows_including_2009_context':len(panel),
      'feature_count':len(FEATURES),'counts':counts,'splits':split_stats,
      'terminal_label_count':int(panel.LabelUsesTerminal.sum()),
      'unknown_exit_windows':int(panel.UnknownExitWindow.sum()),'active_code100_ignored':active,
      'protocol':{'calendar':'Union of source trading dates; horizon=20 market sessions, sequence length=20.',
        'price':'abs(PRC/ASKHI/BIDLO) divided by CFACPR; volume=VOL*CFACSHR.',
        'target':'Adjusted-price decline from decision close, volatility threshold clip(1.5*Vol20*sqrt(20/252),.05,.25).',
        'terminal':'At exit observe final close and Close*(1+DLRET); thereafter cash-equivalent value stays constant. No reinvested regular dividends in target. Same-day cash settlement is a modeling convention, including stock mergers.',
        'endpoints':'LabelHorizonEnd is scheduled day20; LabelEndDate is last required outcome date (exit date if earlier). Unknown outcomes are never imputed.',
        'features':'TA-Lib only; each valid OHLCV segment restarts. MACD histogram/slope and sector MACD/ATR use price-relative definitions, with renamed columns.',
        'sequence':'20 consecutive sessions within PERMNO and SegmentID; all 32 features finite in every row. Carry past context across split boundaries; purge endpoints by LabelEndDate.',
        'scaling':'No fitted scaler or imputation here. Future trainer must fit them only on training data.',
        'identity':'PERMNO is key; FormationTicker is display-only.',
        'feature_allowlist':'Only feature_columns.csv may be used as technical inputs. No automatic numeric-column selection.'},
      'remaining':['PERMNO-based fundamental and textual linkage; trainer integration and smoke test.',
                   'Universe formation eligibility/name-history check is separate.',
                   'Backtests must stop on unresolved held returns; no missing-return zero fill.'],
      'self_tests':'PASS'}
    (out/'build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('PASS | technical panel built; no training; historical outputs unchanged',flush=True)

if __name__=='__main__':main()
