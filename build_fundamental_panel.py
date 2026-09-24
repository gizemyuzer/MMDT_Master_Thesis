"""Offline, date-aware CCM/fundq sidecar. No training, imputation or old-file edits.
Ordinary Compustat is not a historical-vintage database. RDQ alignment alone
does not prove that all accounting fields were publicly available on that date.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
import hashlib,json
import numpy as np
import pandas as pd

FEATURES=['Liabilities_to_Assets','Net_Profit_Margin_TTM','Current_Ratio',
          'Altman_Z_TTM_Proxy','Retained_Earnings_TA','Market_Value_to_Liab_QuarterEnd']
NUM=['atq','ltq','actq','lctq','req','niq','revtq','oiadpq','cshoq','prccq']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def divide(a,b):return a/b.where(b.gt(0))

def quarters(raw,calendar):
    q=raw.copy();q.gvkey=q.gvkey.astype(str).str.zfill(6)
    q.datadate=pd.to_datetime(q.datadate,errors='raise')
    q.rdq=pd.to_datetime(q.rdq,errors='coerce')
    for c in NUM+['fyearq','fqtr']:q[c]=pd.to_numeric(q[c],errors='coerce')
    duplicate=q.duplicated(['gvkey','datadate'],keep=False)
    quarantined=q.loc[duplicate].copy()
    # Preserve a missing-value event so an ambiguous NEW quarter replaces old values.
    placeholders=[]
    for _,g in q.loc[duplicate].groupby(['gvkey','datadate']):
        r=g.iloc[0].copy()
        r[NUM+['fyearq','fqtr']]=np.nan
        dates=g.loc[g.rdq.ge(g.datadate),'rdq'].dropna()
        r['rdq']=dates.min() if len(dates) else pd.NaT
        r['QuarantinedQuarter']=True;placeholders.append(r)
    q=q.loc[~duplicate].copy();q['QuarantinedQuarter']=False
    if placeholders:q=pd.concat([q,pd.DataFrame(placeholders)],ignore_index=True)
    q=q.sort_values(['gvkey','datadate']).reset_index(drop=True)
    q['ValidRDQ']=q.rdq.notna()&q.rdq.ge(q.datadate)
    q.loc[q.curcdq.ne('USD'),NUM]=np.nan
    q['Liabilities_to_Assets']=divide(q.ltq,q.atq)
    q['Current_Ratio']=divide(q.actq,q.lctq)
    q['Retained_Earnings_TA']=divide(q.req,q.atq)
    q['Market_Value_to_Liab_QuarterEnd']=divide((q.cshoq*q.prccq).where(q.cshoq.ge(0)&q.prccq.gt(0)),q.ltq)
    q['Net_Profit_Margin_TTM']=np.nan;q['Altman_Z_TTM_Proxy']=np.nan
    q['TTMComplete']=False
    for _,g in q.groupby('gvkey',sort=False):
        fiscal=g.fyearq*4+g.fqtr
        steps=fiscal.diff().eq(1)&g.datadate.diff().dt.days.between(60,120)
        continuous=steps.astype(int).rolling(3).sum().eq(3)
        known=g.ValidRDQ.astype(int).rolling(4).sum().eq(4)
        # Previous quarter values must already have been announced at current RDQ.
        rdq_days=g.rdq.map(lambda x:x.toordinal() if pd.notna(x) else np.nan)
        known &= rdq_days.rolling(4).max().le(rdq_days)
        currency=g.curcdq.eq('USD').astype(int).rolling(4).sum().eq(4)
        base=continuous&known&currency
        sales=g.revtq.rolling(4,min_periods=4).sum().where(base)
        income=g.niq.rolling(4,min_periods=4).sum().where(base)
        operating=g.oiadpq.rolling(4,min_periods=4).sum().where(base)
        q.loc[g.index,'Net_Profit_Margin_TTM']=divide(income,sales)
        # OIADPQ is an operating-income proxy, not an exact EBIT reconstruction.
        z=(1.2*divide(g.actq-g.lctq,g.atq)+1.4*divide(g.req,g.atq)+
           3.3*divide(operating,g.atq)+.6*g.Market_Value_to_Liab_QuarterEnd+divide(sales,g.atq))
        q.loc[g.index,'Altman_Z_TTM_Proxy']=z
        q.loc[g.index,'TTMComplete']=sales.notna()&income.notna()&operating.notna()
    q[FEATURES]=q[FEATURES].replace([np.inf,-np.inf],np.nan)
    q.loc[q.QuarantinedQuarter,FEATURES]=np.nan
    q['AvailableDate']=pd.NaT
    usable=q.ValidRDQ
    positions=calendar.searchsorted(pd.DatetimeIndex(q.loc[usable,'rdq']),side='right')
    rows=q.index[usable][positions<len(calendar)]
    q.loc[rows,'AvailableDate']=calendar[positions[positions<len(calendar)]]
    return q,quarantined

def align(keys,links,q,max_age=365):
    """Map each daily PERMNO only inside the corresponding CCM interval."""
    daily=keys[['date','permno']].copy().reset_index(drop=True)
    daily['gvkey']=pd.Series(pd.NA,index=daily.index,dtype='string')
    daily['LinkStart']=pd.NaT;daily['LinkEnd']=pd.NaT
    for e in links.itertuples():
        start=pd.Timestamp(e.linkdt) if pd.notna(e.linkdt) else daily.date.min()
        end=pd.Timestamp(e.linkenddt) if pd.notna(e.linkenddt) else daily.date.max()
        mask=daily.permno.eq(int(e.lpermno))&daily.date.between(start,end)
        conflict=mask&daily.gvkey.notna()&daily.gvkey.ne(str(e.gvkey).zfill(6))
        if conflict.any():raise ValueError('Conflicting GVKEY mapping on a daily observation')
        daily.loc[mask,'gvkey']=str(e.gvkey).zfill(6)
        daily.loc[mask,'LinkStart']=start;daily.loc[mask,'LinkEnd']=end
    events={}
    for gvkey,g in q[q.AvailableDate.notna()].groupby('gvkey',sort=False):
        g=g.sort_values(['AvailableDate','datadate'])
        # A late release of an older fiscal period must not replace a newer period.
        g=g[g.datadate.eq(g.datadate.cummax())]
        g=g.drop_duplicates('AvailableDate',keep='last')
        events[gvkey]=g
    for c in FEATURES:daily[c]=np.nan
    for c in ['FundPeriodEnd','FundRDQ','FundAvailableDate']:daily[c]=pd.NaT
    daily['QuarantinedQuarter']=False;daily['StaleFundamentals']=False
    for gvkey,idx in daily[daily.gvkey.notna()].groupby('gvkey').groups.items():
        g=events.get(gvkey)
        if g is None:continue
        left=daily.loc[idx,['date']].sort_values('date').reset_index().rename(columns={'index':'_row'})
        right=g[['AvailableDate','datadate','rdq','QuarantinedQuarter']+FEATURES]
        merged=pd.merge_asof(left,right,left_on='date',right_on='AvailableDate',direction='backward')
        rows=merged._row.to_numpy()
        stale=(merged.date-merged.datadate).dt.days.gt(max_age)
        merged.loc[stale,FEATURES]=np.nan
        daily.loc[rows,FEATURES]=merged[FEATURES].to_numpy()
        for source,dest in [('datadate','FundPeriodEnd'),('rdq','FundRDQ'),('AvailableDate','FundAvailableDate')]:
            daily.loc[rows,dest]=merged[source].to_numpy()
        daily.loc[rows,'QuarantinedQuarter']=merged.QuarantinedQuarter.eq(True).to_numpy()
        daily.loc[rows,'StaleFundamentals']=stale.to_numpy()
    daily['FundAny']=daily[FEATURES].notna().any(axis=1)
    daily['FundAllSix']=daily[FEATURES].notna().all(axis=1)
    known=daily.FundAvailableDate.notna()
    assert daily.loc[known,'FundAvailableDate'].le(daily.loc[known,'date']).all()
    assert daily.loc[known,'FundRDQ'].lt(daily.loc[known,'date']).all()
    assert daily.loc[daily.QuarantinedQuarter,FEATURES].isna().all().all()
    return daily

def self_tests():
    cal=pd.bdate_range('2008-01-01','2012-12-31')
    q=pd.DataFrame({'gvkey':['000001']*5,'datadate':pd.to_datetime(['2009-03-31','2009-06-30','2009-09-30','2009-12-31','2010-03-31']),
       'rdq':pd.to_datetime(['2009-05-01','2009-08-03','2009-11-02','2010-02-01','2010-05-03']),
       'fyearq':[2009]*4+[2010],'fqtr':[1,2,3,4,1],'curcdq':['USD']*5})
    for c in NUM:q[c]=10.
    q.atq=100.;q.revtq=20.
    built,_=quarters(q,cal)
    assert built.Net_Profit_Margin_TTM.iloc[3]==.5
    assert built.Net_Profit_Margin_TTM.iloc[:3].isna().all()
    links=pd.DataFrame({'gvkey':['000001','000002'],'lpermno':[1,1],
        'linkdt':['2008-01-01','2010-04-01'],'linkenddt':['2010-03-31',None]})
    keys=pd.DataFrame({'date':pd.to_datetime(['2010-02-01','2010-02-02','2010-04-01']),'permno':[1]*3})
    d=align(keys,links,built)
    assert pd.isna(d.Net_Profit_Margin_TTM.iloc[0]) and d.Net_Profit_Margin_TTM.iloc[1]==.5
    assert not d.FundAny.iloc[2] # no old-GVKEY carryover
    duplicate=pd.concat([q,q.iloc[[3]]],ignore_index=True)
    dup,_=quarters(duplicate,cal)
    assert align(keys,links,dup).loc[1,FEATURES].isna().all()
    altered=q.copy();altered.loc[2,'rdq']=pd.Timestamp('2010-03-01')
    assert pd.isna(quarters(altered,cal)[0].Net_Profit_Margin_TTM.iloc[3])
    bad=q.copy();bad.loc[3,'req']=np.nan
    assert pd.isna(quarters(bad,cal)[0].Altman_Z_TTM_Proxy.iloc[3])
    pd.testing.assert_frame_equal(built.loc[:3,FEATURES],quarters(q.iloc[:4],cal)[0][FEATURES])
    missing=q.copy();missing.loc[2,'rdq']=pd.NaT
    assert pd.isna(quarters(missing,cal)[0].Net_Profit_Margin_TTM.iloc[3])
    open_link=links.iloc[[0]].copy();open_link['linkenddt']=None
    stale_keys=pd.DataFrame({'date':[pd.Timestamp('2012-01-03')],'permno':[1]})
    assert not align(stale_keys,open_link,built).FundAny.any()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',default='datasets/fundamental_sources_20260919T164626Z_e10608aa')
    ap.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd')
    ap.add_argument('--output-parent',default='datasets')
    ap.add_argument('--max-age-days',type=int,default=365)
    ap.add_argument('--self-test-only',action='store_true')
    a=ap.parse_args();self_tests();print('PASS | announcement timing, GVKEY changes, duplicate quarantine, TTM and missing components',flush=True)
    if a.self_test_only:return
    if a.max_age_days<1:raise ValueError('Invalid maximum age')
    src=Path(a.sources);technical=Path(a.technical)
    files={'fundq':src/'fundq_raw.csv','links':src/'ccm_links.csv','technical':technical/'technical_panel.csv',
           'technical_report':technical/'build_report.json'}
    hashes={k:sha(p) for k,p in files.items()}
    keys=pd.read_csv(files['technical'],usecols=['date','permno','TrainEndpoint','ValidationEndpoint','TestEndpoint'],parse_dates=['date'],low_memory=False)
    technical_report=json.loads(files['technical_report'].read_text(encoding='utf-8'))
    if len(keys)!=technical_report['output_rows_including_2009_context']:
        raise ValueError('Technical CSV row count differs from its build report: incomplete file or wrong folder')
    for flag in ['TrainEndpoint','ValidationEndpoint','TestEndpoint']:
        if keys[flag].isna().any() or keys[flag].dtype!=bool:raise ValueError('Invalid endpoint flags: '+flag)
    if keys.duplicated(['date','permno']).any():raise ValueError('Duplicate technical panel keys')
    raw=pd.read_csv(files['fundq'],dtype={'gvkey':str});links=pd.read_csv(files['links'],dtype={'gvkey':str})
    if not links.linktype.isin(['LC','LU']).all() or not links.linkprim.isin(['P','C']).all():raise ValueError('Unexpected link filters')
    if not raw.curcdq.eq('USD').all():raise ValueError('Non-USD observations require currency review')
    cal=pd.DatetimeIndex(sorted(keys.date.unique()))
    q,quarantine=quarters(raw,cal)
    print('Quarterly ratios constructed; aligning by PERMNO/GVKEY and date...',flush=True)
    daily=align(keys,links,q,a.max_age_days)
    coverage={}
    for split in ['Train','Validation','Test']:
        mask=keys[split+'Endpoint']
        if mask.dtype!=bool:raise ValueError('Endpoint flags must be boolean')
        part=daily.loc[mask]
        coverage[split]={'endpoints':len(part),'any_feature_rows':int(part.FundAny.sum()),
            'all_six_rows':int(part.FundAllSix.sum()),'missing_by_feature':{c:int(part[c].isna().sum()) for c in FEATURES}}
    out=Path(a.output_parent)/('fundamental_panel_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    daily.to_csv(out/'fundamental_daily.csv',index=False)
    q.to_csv(out/'quarterly_features.csv',index=False)
    quarantine.to_csv(out/'quarantined_duplicate_quarters.csv',index=False)
    pd.DataFrame({'feature':FEATURES}).to_csv(out/'fund_feature_columns.csv',index=False)
    if hashes!={k:sha(p) for k,p in files.items()}:raise RuntimeError('Sources changed during build')
    report={'status':'FUNDAMENTAL_SIDECAR_BUILT_NOT_TRAINING_READY','output_folder':str(out),
      'source_hashes':hashes,'daily_rows':len(daily),'permnos':int(daily.permno.nunique()),
      'unlinked_permnos':sorted(set(daily.permno)-set(daily.loc[daily.gvkey.notna(),'permno'])),
      'duplicate_rows_quarantined':len(quarantine),
      'duplicate_periods_quarantined':len(quarantine[['gvkey','datadate']].drop_duplicates()),
      'quarterly_rows_after_quarantine':len(q),'valid_rdq_rows':int(q.ValidRDQ.sum()),
      'coverage':coverage,'features':FEATURES,'max_age_from_fiscal_period_end_days':a.max_age_days,
      'protocol':['Next observed trading session strictly AFTER valid RDQ; no assumed 60-day replacement.',
        'RDQ-aligned standard Compustat, NOT a historical-vintage point-in-time dataset. Revisions and announcement-field availability remain limitations.',
        'LC/LU, P/C CCM intervals applied at observation date. Different GVKEYs never share forward-filled data.',
        'All duplicate GVKEY-period groups quarantined, even when numeric fields agree. A dated missing-value event blocks previous-period carryover.',
        'No zero fill, train-median imputation, scaling or clipping in this step. Future trainer must use a feature allowlist and training-only fitted preprocessing.',
        'TTM requires four consecutive fiscal quarters, 60-120 days between adjacent period ends, all required announcements known by current RDQ.',
        'Altman_Z_TTM_Proxy uses OIADPQ as an EBIT proxy; sales and operating income are trailing four-quarter sums. Every component required.',
        'Market equity is CSHOQ*PRCCQ at fiscal quarter end, NOT daily market equity.',
        'Liabilities_to_Assets=LTQ/ATQ. It is not debt/equity.',
        'Quarterly data expire 365 days after fiscal period end by default. Missing observations stay in the panel.',
        'FundAny, FundAllSix, GVKEY, dates, and audit flags are metadata, not authorized model features.'],
      'self_tests':'PASS','remaining':'Text identity linkage; safe sidecar join; training-only imputation/scaling and trainer smoke test.'}
    (out/'fundamental_build_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('PASS | fundamental sidecar built; no training or historical-file changes')

if __name__=='__main__':main()
