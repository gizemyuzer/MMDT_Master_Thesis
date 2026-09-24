"""Export CCM link history and raw Compustat quarters for the frozen PERMNO universe.
No feature merge, model training, date imputation, or overwrite of historical data.
Run in your WRDS-enabled thesis environment. --self-test-only needs no WRDS login.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4
import numpy as np
import pandas as pd

LINK_SQL = """SELECT gvkey, lpermno, lpermco, liid, linktype, linkprim, linkdt, linkenddt
FROM crsp.ccmxpf_lnkhist
WHERE lpermno IN %(permnos)s
  AND linktype IN ('LC','LU') AND linkprim IN ('P','C')
  AND (linkdt IS NULL OR linkdt <= %(end)s)
  AND (linkenddt IS NULL OR linkenddt >= %(start)s)
ORDER BY lpermno, linkdt, gvkey
"""
QUARTER_SQL = """SELECT gvkey, datadate, rdq, fyearq, fqtr, curcdq,
       atq, ltq, actq, lctq, ceqq, dlttq, dlcq,
       req, niq, revtq, oiadpq, cshoq, prccq,
       indfmt, datafmt, popsrc, consol
FROM comp.fundq
WHERE gvkey IN %(gvkeys)s
  AND datadate >= %(start)s AND datadate <= %(end)s
  AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
ORDER BY gvkey, datadate
"""
NUMERIC=['atq','ltq','actq','lctq','ceqq','dlttq','dlcq','req','niq','revtq','oiadpq','cshoq','prccq']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gvkeys(values):
    s=pd.to_numeric(values,errors='raise')
    if s.isna().any() or s.mod(1).ne(0).any() or s.le(0).any():
        raise ValueError('Invalid GVKEY')
    return s.astype('int64').astype(str).str.zfill(6)

def bounded_date(value,lo,hi,missing_value):
    if pd.isna(value):return missing_value
    # The database uses date-typed endpoints, possibly beyond pandas' range.
    text=str(value)[:10]
    try:day=datetime.strptime(text,'%Y-%m-%d').date()
    except ValueError as exc:raise ValueError('Invalid CCM date: '+str(value)) from exc
    if day<lo.date():return lo
    if day>hi.date():return hi
    return pd.Timestamp(day)

def audit_links(links,universe,start,end):
    l=links.copy()
    l['gvkey']=gvkeys(l.gvkey)
    ids=pd.to_numeric(l.lpermno,errors='raise')
    if ids.isna().any() or ids.mod(1).ne(0).any():raise ValueError('Invalid CCM PERMNO')
    l['lpermno']=ids.astype('int64')
    if not set(l.lpermno).issubset(set(universe.permno)):raise ValueError('Unexpected PERMNO in CCM result')
    if not l.linktype.isin(['LC','LU']).all() or not l.linkprim.isin(['P','C']).all():
        raise ValueError('Unexpected link filter result')
    lo,hi=pd.Timestamp(start),pd.Timestamp(end)
    l['_start']=[bounded_date(x,lo,hi,lo) for x in l.linkdt]
    l['_end']=[bounded_date(x,lo,hi,hi) for x in l.linkenddt]
    if l._start.gt(l._end).any():raise ValueError('Reversed link interval')
    overlaps=[]
    for permno,g in l.groupby('lpermno'):
        records=g.to_dict('records')
        for i,a in enumerate(records):
            for b in records[i+1:]:
                first=max(a['_start'],b['_start']);last=min(a['_end'],b['_end'])
                if a['gvkey']!=b['gvkey'] and first<=last:
                    overlaps.append({'permno':int(permno),'gvkey_a':a['gvkey'],'gvkey_b':b['gvkey'],
                        'overlap_start':str(first.date()),'overlap_end':str(last.date())})
    conflicts=pd.DataFrame(overlaps,columns=['permno','gvkey_a','gvkey_b','overlap_start','overlap_end'])
    report={'rows':len(l),'linked_permnos':int(l.lpermno.nunique()),'gvkeys':int(l.gvkey.nunique()),
        'permnos_without_selected_link':sorted(set(map(int,universe.permno))-set(map(int,l.lpermno))),
        'permnos_with_multiple_gvkeys_over_history':int(l.groupby('lpermno').gvkey.nunique().gt(1).sum()),
        'overlapping_different_gvkey_pairs':len(conflicts),
        'missing_link_start_rows':int(l.linkdt.isna().sum()),
        'missing_link_end_rows':int(l.linkenddt.isna().sum()),
        'exact_duplicate_link_rows':int(links.duplicated().sum())}
    return l.drop(columns=['_start','_end']),conflicts,report

def audit_quarters(quarters,selected_gvkeys,end):
    q=quarters.copy();q['gvkey']=gvkeys(q.gvkey)
    if not set(q.gvkey).issubset(set(selected_gvkeys)):raise ValueError('Unexpected Compustat GVKEY')
    q['datadate']=pd.to_datetime(q.datadate,errors='raise')
    if q.datadate.isna().any():raise ValueError('Missing fiscal-period date')
    original_rdq=q.rdq.copy();q['rdq']=pd.to_datetime(q.rdq,errors='coerce')
    for c in NUMERIC:q[c]=pd.to_numeric(q[c],errors='coerce')
    duplicate=q.duplicated(['gvkey','datadate'],keep=False)
    missing=q.rdq.isna();early=q.rdq.lt(q.datadate)
    flags=q.loc[missing|early,['gvkey','datadate','rdq']].copy()
    flags['reason']=np.where(flags.rdq.isna(),'missing_or_unparseable_rdq','rdq_before_period_end')
    report={'rows':len(q),'gvkeys':int(q.gvkey.nunique()),
      'gvkeys_without_quarters':sorted(set(selected_gvkeys)-set(q.gvkey)),
      'duplicate_gvkey_period_rows':int(duplicate.sum()),
      'missing_rdq_rows':int(missing.sum()),
      'unparseable_nonmissing_rdq_rows':int((original_rdq.notna()&missing).sum()),
      'rdq_before_period_end_rows':int(early.sum()),
      'rdq_after_sample_end_rows':int(q.rdq.gt(pd.Timestamp(end)).sum()),
      'currencies':{str(k):int(v) for k,v in q.curcdq.fillna('MISSING').value_counts().items()},
      'missing_fields':{c:int(q[c].isna().sum()) for c in NUMERIC}}
    return q,flags,q.loc[duplicate].copy(),report

def self_tests():
    u=pd.DataFrame({'permno':[1,2,3]})
    links=pd.DataFrame([
       ['1001',1,'LC','P','2007-01-01','2012-12-31'],
       ['1002',1,'LC','P','2013-01-01',None],
       ['1003',2,'LU','C','2010-01-01','9999-12-31'],
       ['1004',2,'LC','P','2011-01-01','2011-12-31']],
       columns=['gvkey','lpermno','linktype','linkprim','linkdt','linkenddt'])
    _,conflicts,r=audit_links(links,u,'2007-01-01','2024-12-31')
    assert len(conflicts)==1 and conflicts.iloc[0].permno==2
    assert r['permnos_without_selected_link']==[3]
    q=pd.DataFrame({'gvkey':['1001']*3,'datadate':['2010-03-31','2010-06-30','2010-06-30'],
        'rdq':[None,'2010-06-01','2010-08-01'],'curcdq':['USD']*3})
    for c in NUMERIC:q[c]=np.nan
    clean,flags,duplicates,r=audit_quarters(q,['001001'],'2024-12-31')
    assert len(flags)==2 and len(duplicates)==2
    assert clean.rdq.isna().sum()==1 and clean[NUMERIC].isna().all().all()
    assert 'tic =' not in QUARTER_SQL and 'gvkey IN %(gvkeys)s' in QUARTER_SQL
    assert LINK_SQL.lstrip().startswith('SELECT ') and QUARTER_SQL.lstrip().startswith('SELECT ')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--universe',default='datasets/universe.csv')
    ap.add_argument('--output-parent',default='datasets')
    ap.add_argument('--username',default='gizemyuzer')
    ap.add_argument('--start',default='2007-01-01',help='Includes pre-2009 history for later trailing-four-quarter ratios')
    ap.add_argument('--end',default='2024-12-31')
    ap.add_argument('--self-test-only',action='store_true')
    args=ap.parse_args();self_tests()
    print('PASS | offline tests: link intervals, ambiguous links, missing RDQ, duplicate quarters',flush=True)
    if args.self_test_only:return
    start=pd.Timestamp(args.start);end=pd.Timestamp(args.end)
    if start>=end:raise ValueError('Invalid date range')
    source=Path(args.universe);before=sha(source)
    u=pd.read_csv(source)
    ids=pd.to_numeric(u.permno,errors='raise')
    if ids.isna().any() or ids.mod(1).ne(0).any() or ids.duplicated().any():raise ValueError('Invalid universe PERMNOs')
    u['permno']=ids.astype('int64')
    if u.empty:raise ValueError('Empty universe')
    out=Path(args.output_parent)/('fundamental_sources_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,out/'universe_snapshot.csv')
    report={'status':'STARTED','output_folder':str(out),'universe_size':len(u),
        'universe_sha256':before,'start':args.start,'end':args.end,
        'source_tables':['crsp.ccmxpf_lnkhist','comp.fundq'],
        'link_filters':{'linktype':['LC','LU'],'linkprim':['P','C']},
        'protocol_notes':[
           'This export is not a daily feature panel. No ambiguous mapping is resolved automatically.',
           'A missing selected CCM link means missing coverage, not absence of a company.',
           'No RDQ is replaced with fiscal date plus an assumed lag.',
           'RDQ is an earnings announcement date, not proof that every balance-sheet field was available then.',
           'Ordinary Compustat fundq may contain later revisions. RDQ alignment does not make it a vintage point-in-time database.',
           'Current source versions and hashes are archived; no claim of historical data vintages.',
           'Daily matching must enforce link validity on the observation date and avoid carrying values across a GVKEY change.',
           'Use next observed trading session after a valid announcement date unless verified timestamps support same-day use.',
           'Currency, duplicate-quarter, fiscal-quarter continuity and missing components need review before constructing ratios.',
           'The old Debt_to_Equity formula was LTQ/ATQ. Renaming/redefining it and rebuilding Altman inputs occurs in the next stage.']}
    db=None
    try:
        import wrds
        print('[1/2] Connecting to WRDS and exporting CCM history...',flush=True)
        db=wrds.Connection(wrds_username=args.username)
        params={'permnos':tuple(map(int,u.permno)),'start':args.start,'end':args.end}
        raw_links=db.raw_sql(LINK_SQL,params=params)
        raw_links.to_csv(out/'ccm_links_raw.csv',index=False)
        if raw_links.empty:raise ValueError('No selected CCM links returned; inspect subscription and filters')
        links,conflicts,link_report=audit_links(raw_links,u,args.start,args.end)
        links.to_csv(out/'ccm_links.csv',index=False)
        conflicts.to_csv(out/'link_conflicts.csv',index=False)
        selected=sorted(links.gvkey.unique().tolist())
        report['links']=link_report
        print(f'[2/2] Exporting quarterly records for {len(selected)} GVKEYs...',flush=True)
        raw_q=db.raw_sql(QUARTER_SQL,params={'gvkeys':tuple(selected),'start':args.start,'end':args.end})
        raw_q.to_csv(out/'fundq_raw.csv',index=False)
        if raw_q.empty:raise ValueError('No quarterly data returned')
        q,flags,duplicates,qreport=audit_quarters(raw_q,selected,args.end)
        q.to_csv(out/'fundq_review.csv',index=False)
        flags.to_csv(out/'report_date_issues.csv',index=False)
        duplicates.to_csv(out/'duplicate_quarters.csv',index=False)
        report['quarters']=qreport
        if sha(source)!=before:raise RuntimeError('Universe source changed during export')
        report['file_hashes']={p.name:sha(p) for p in sorted(out.glob('*.csv'))}
        report['status']='EXPORTED_REVIEW_REQUIRED_NOT_TRAINING_READY'
    except Exception as exc:
        report.update(status='FAILED_PARTIAL_EXPORT',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        (out/'fundamental_source_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if db is not None:db.close()
    print(json.dumps(report,indent=2))
    print('PASS | sources exported; no imputation, merge, training or historical-file changes',flush=True)

if __name__=='__main__':main()
