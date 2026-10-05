"""Build nine daily macro features once; no model training or legacy cache changes."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

MACRO = ['VIX_Close_L1','Market_Trend_50_L1','Market_Trend_200_L1',
         'VIX_5d_Delta_L1','VIX_10d_Delta_L1','Market_Trend_Accel_L1',
         'Yield_Spread_10Y2Y_L1','Is_Yield_Inverted_L1','Yield_Spread_Delta_20d_L1']
TECH_PATH='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def indexed(df, columns):
    df=df.copy(); df['date']=pd.to_datetime(df['date'])
    if df.date.isna().any() or df.date.duplicated().any(): raise ValueError('Invalid/duplicate source dates')
    df=df.set_index('date').sort_index()
    return df[columns].apply(pd.to_numeric,errors='raise').astype(float)
def features(market,vix,rates):
    m=indexed(market,['vwretd']); v=indexed(vix,['vix']); r=indexed(rates,['dgs2','dgs10'])
    if not np.isfinite(m.vwretd).all() or (m.vwretd<=-1).any(): raise ValueError('Invalid market returns')
    # CRSP sessions define rolling windows. No backward fill or invented zero values.
    v=v.reindex(m.index).ffill(limit=5)
    r=r.reindex(r.index.union(m.index)).sort_index().ffill(limit=5).reindex(m.index)
    if (v.vix.dropna()<=0).any(): raise ValueError('Invalid VIX')
    level=(1+m.vwretd).cumprod()
    t50=level/level.rolling(50,min_periods=50).mean()-1
    t200=level/level.rolling(200,min_periods=200).mean()-1
    spread=r.dgs10-r.dgs2
    inv=(spread<0).astype(float).where(spread.notna())
    out=pd.concat([v.vix,t50,t200,v.vix.diff(5),v.vix.diff(10),t50.diff(5),spread,inv,spread.diff(20)],axis=1)
    out.columns=MACRO
    # Uniform one-session lag: inputs at decision t use at most source session t-1.
    out=out.shift(1)
    out['MacroSourceSession']=pd.Series(m.index,index=m.index).shift(1)
    return out.rename_axis('date').reset_index()
def self_test():
    d=pd.bdate_range('2008-01-01',periods=350)
    m=pd.DataFrame({'date':d,'vwretd':.001})
    v=pd.DataFrame({'date':d,'vix':20.+np.arange(350)/100})
    r=pd.DataFrame({'date':d,'dgs2':3.,'dgs10':2.})
    a=features(m,v,r)
    assert a.loc[250,'Is_Yield_Inverted_L1']==1
    assert a.loc[250,'MacroSourceSession']==d[249]
    v.loc[250:,'vix']=90
    b=features(m,v,r)
    pd.testing.assert_frame_equal(a.iloc[:251],b.iloc[:251])
    r.loc[230:250,['dgs2','dgs10']]=np.nan
    b=features(m,v,r)
    assert pd.isna(b.loc[245,'Is_Yield_Inverted_L1'])
    print('PASS | macro lag, future perturbation, missing-rate handling')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--technical',default=TECH_PATH)
    p.add_argument('--output',default='datasets/corrected_macro_v1')
    p.add_argument('--username',default='gizemyuzer')
    p.add_argument('--raw-dir',help='Offline directory: market_raw.csv, vix_raw.csv, rates_raw.csv')
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args(); self_test()
    if a.self_test_only:return
    tech=Path(a.technical).resolve()
    if not tech.is_file():raise FileNotFoundError(tech)
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    report={'status':'BUILDING','output_folder':str(out),'created_utc':datetime.now(timezone.utc).isoformat()}
    try:
        if a.raw_dir:
            raw=Path(a.raw_dir)
            frames=[pd.read_csv(raw/n) for n in ['market_raw.csv','vix_raw.csv','rates_raw.csv']]
        else:
            import wrds
            db=wrds.Connection(wrds_username=a.username)
            try:
                frames=[]
                for table,cols in [('crsp.dsi','vwretd'),('cboe.cboe','vix'),('frb.rates_daily','dgs2, dgs10')]:
                    print('Fetching',table,flush=True)
                    frames.append(db.raw_sql(f"SELECT date, {cols} FROM {table} WHERE date >= '2008-01-01' AND date <= '2024-12-31' ORDER BY date",date_cols=['date']))
            finally:db.close()
        names=['market_raw.csv','vix_raw.csv','rates_raw.csv']
        for n,f in zip(names,frames):
            if f.empty:raise ValueError('Empty source: '+n)
            f.to_csv(out/n,index=False)
        daily=features(*frames)
        panel_dates=pd.read_csv(tech,usecols=['date'],parse_dates=['date']).date.drop_duplicates().sort_values()
        if not panel_dates.isin(daily.date).all():raise ValueError('Panel dates absent from CRSP calendar')
        daily=daily.loc[daily.date.isin(panel_dates)].copy()
        # Check context too; retain precisely the pre-existing technical endpoint set.
        required=daily.date.ge('2009-12-01')
        if not np.isfinite(daily.loc[required,MACRO].to_numpy()).all():
            daily.loc[required & daily[MACRO].isna().any(axis=1)].to_csv(out/'missing_macro_dates.csv',index=False)
            raise ValueError('Missing macro history; inspect missing_macro_dates.csv. No zero filling allowed.')
        daily.to_csv(out/'macro_daily.csv',index=False)
        pd.DataFrame({'feature':MACRO}).to_csv(out/'macro_feature_columns.csv',index=False)
        report.update(status='PASS',technical_sha256=sha(tech),macro_sha256=sha(out/'macro_daily.csv'),
            source_sha256={n:sha(out/n) for n in names},rows=len(daily),columns=MACRO,
            protocol={'market':'CRSP value-weighted total-return index, not SPY',
            'rates':'WRDS FRB DGS10 minus DGS2, percentage points',
            'availability':'All nine features lagged one CRSP session; no backfill; maximum five source-calendar rows forward fill',
            'vintage':'Historical downloaded series, not a vintage database; one-session lag does not resolve subsequent revisions',
            'normalization':'No within-date normalization. Train-only RobustScaler in loader.',
            'design':'I+II+III: 32 technical, 6 fundamental, 9 macro; no handcrafted interactions or text'})
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:(out/'macro_build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS | macro panel built:',out/'macro_daily.csv')
if __name__=='__main__':main()
