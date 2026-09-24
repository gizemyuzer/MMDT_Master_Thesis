"""Build a DAILY TEXT SIDECAR using dated CCM + current-company CIK candidates.
This operational linkage is NOT a historically verified CIK bridge. Missing
candidates/archives remain missing, never fabricated or filled from old tickers.
No network, training, source overwrites, imputation, or fitted normalization.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4
import numpy as np
import pandas as pd

LANG = ['LM_Negative','LM_Positive','LM_Uncertainty','LM_Litigious','LM_Constraining',
        'LM_WordCount','TXT_SimPrev','TXT_LenChange']
EVENTS = {'EK_Bankruptcy':('1.03',252),'EK_DebtTrigger':('2.04',252),
          'EK_Impairment':('2.06',252),'EK_ListingWarn':('3.01',252),
          'EK_AuditorChange':('4.01',252),'EK_Restatement':('4.02',252),
          'EK_MgmtChange':('5.02',126)}
FEATURES = list(EVENTS)+['EK_Activity_60d','EK_DaysSince']+LANG+['TXT_DaysSinceFiling','TXT_HasCoverage']
FLAGS = ['TrainEndpoint','ValidationEndpoint','TestEndpoint']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def ident(s, allow_missing=False):
    n=pd.to_numeric(s,errors='raise')
    if ((n.dropna()%1)!=0).any() or (n.dropna()<=0).any(): raise ValueError('Invalid identifier')
    if not allow_missing and n.isna().any(): raise ValueError('Missing identifier')
    return n.astype('Int64')

def boolean(s):
    if s.dtype==bool:return s
    t=s.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if t.isna().any():raise ValueError('Invalid endpoint flag')
    return t.astype(bool)

def available(frame, calendar):
    frame=frame.copy()
    pos=calendar.searchsorted(frame.filing_date,side='right')
    frame=frame.loc[pos<len(calendar)].copy(); pos=pos[pos<len(calendar)]
    frame['AvailableDate']=calendar.take(pos).to_numpy()
    frame['AvailableSession']=pos
    if not (frame.AvailableDate>frame.filing_date).all():raise AssertionError('Publication leakage')
    return frame

def read_inputs(panel, links, filings, index):
    b=pd.read_csv(panel,usecols=['date','permno']+FLAGS,parse_dates=['date'])
    b['permno']=ident(b.permno)
    if b.date.isna().any() or b.duplicated(['date','permno']).any():raise ValueError('Invalid panel keys')
    for c in FLAGS:b[c]=boolean(b[c])
    if (b[FLAGS].sum(axis=1)>1).any():raise ValueError('Overlapping partitions')
    calendar=pd.DatetimeIndex(sorted(b.date.unique()))
    l=pd.read_csv(links,dtype={'gvkey':str})
    l['permno']=ident(l.lpermno);l['cik']=ident(l.cik,True)
    l['linkdt']=pd.to_datetime(l.linkdt,errors='raise')
    l['linkenddt']=pd.to_datetime(l.linkenddt,errors='raise').fillna(pd.Timestamp('2099-12-31'))
    if l.linkdt.isna().any() or (l.linkdt>l.linkenddt).any():raise ValueError('Invalid link dates')
    l=l.drop_duplicates(['permno','gvkey','cik','linkdt','linkenddt']).sort_values(['permno','linkdt'])
    for _,g in l.groupby('permno'):
        if (g.linkdt.to_numpy()[1:]<=g.linkenddt.cummax().to_numpy()[:-1]).any():
            raise ValueError('Overlapping CCM intervals need explicit resolution')
    f=pd.read_csv(filings,dtype={'accession':str,'previous_accession':str},parse_dates=['filing_date'])
    i=pd.read_csv(index,dtype={'accession':str,'items':str},parse_dates=['filing_date'])
    for x in (f,i):
        x['cik']=ident(x.cik)
        if x.filing_date.isna().any() or x.accession.isna().any():raise ValueError('Missing filing metadata')
    if f.duplicated(['cik','accession']).any() or not f.status.eq('OK').all():raise ValueError('Duplicate or failed filing features')
    ratios=LANG[:5]
    if (f[ratios].isna() | f[ratios].lt(0) | f[ratios].gt(1)).any().any():raise ValueError('Invalid LM ratios')
    # Shared-class filings are one SEC event, not two events.
    if i.groupby(['cik','accession'])[['form','filing_date','items']].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting index entries')
    i=i.drop_duplicates(['cik','accession'])
    chk=f.merge(i[['cik','accession','form','filing_date']],on=['cik','accession'],how='left',suffixes=('','_index'),validate='one_to_one')
    if not (chk.form.eq(chk.form_index)&chk.filing_date.eq(chk.filing_date_index)).all():raise ValueError('Features/index mismatch')
    return b,l,available(f,calendar),available(i,calendar),calendar

def rolling_count(event_sessions, decision_sessions, width):
    e=np.sort(np.asarray(event_sessions,dtype=int))
    return np.searchsorted(e,decision_sessions,side='right')-np.searchsorted(e,decision_sessions-width,side='right')

def build(base, links, filings, index, calendar):
    out=base[['date','permno']].copy()
    for c in FEATURES:out[c]=0.0 if c=='TXT_HasCoverage' else np.nan
    out['TextLinkCandidate']=False
    out['TextCIK']=pd.Series(pd.NA,index=out.index,dtype='Int64')
    out['TextSourceFilingDate']=pd.NaT
    out['TextSourceAvailableDate']=pd.NaT
    out['TextEventHistorySessions']=np.nan
    issues=[]
    fg={int(k):v for k,v in filings.groupby('cik')};ig={int(k):v for k,v in index.groupby('cik')}
    bg={int(k):g.index.to_numpy() for k,g in base.groupby('permno')}
    assigned=np.zeros(len(base),dtype=bool)
    for number,r in enumerate(links.itertuples(),1):
        positions=bg.get(int(r.permno),np.array([],dtype=int))
        dates=base.loc[positions,'date']
        positions=positions[(dates>=r.linkdt)&(dates<=r.linkenddt)]
        if not len(positions):continue
        positions=base.loc[positions].sort_values('date').index.to_numpy()
        if assigned[positions].any():raise ValueError('Multiple mappings for one panel row')
        assigned[positions]=True
        issue={'permno':int(r.permno),'gvkey':r.gvkey,'linkdt':str(r.linkdt.date()),
               'linkenddt':str(r.linkenddt.date()),'cik':None if pd.isna(r.cik) else int(r.cik),'panel_rows':len(positions)}
        if pd.isna(r.cik):issues.append(dict(issue,reason='NO_CIK_CANDIDATE'));continue
        cik=int(r.cik);out.loc[positions,'TextLinkCandidate']=True;out.loc[positions,'TextCIK']=cik
        archive=ig.get(cik)
        if archive is None:issues.append(dict(issue,reason='NO_ARCHIVE_FOR_CIK'));continue
        archive=archive.loc[archive.filing_date.between(r.linkdt,r.linkenddt)&(archive.AvailableDate<=r.linkenddt)]
        if archive.empty:issues.append(dict(issue,reason='NO_FILINGS_WITHIN_LINK'));continue
        decisions=base.loc[positions,'date'];session=calendar.get_indexer(decisions)
        # Observed recent filing is an availability proxy, not certified archive completeness.
        a=archive.sort_values(['AvailableDate','accession'])
        recent=pd.merge_asof(pd.DataFrame({'date':decisions.to_numpy(),'position':positions}).sort_values('date'),
                            a[['AvailableDate','filing_date']].sort_values('AvailableDate'),
                            left_on='date',right_on='AvailableDate',direction='backward')
        fresh=recent.filing_date.notna() & ((recent.date-recent.filing_date).dt.days<=365)
        out.loc[recent.position,'TXT_HasCoverage']=fresh.astype(float).to_numpy()
        first=int(a.AvailableSession.min());history=session-first+1
        out.loc[positions,'TextEventHistorySessions']=np.maximum(history,0)
        ev=a.loc[a.form.eq('8-K')].copy()
        itemsets=ev['items'].fillna('').map(lambda x:set(re.findall(r'\b\d\.\d{2}\b',x)))
        for name,(code,width) in EVENTS.items():
            counts=rolling_count(ev.loc[itemsets.map(lambda s:code in s),'AvailableSession'],session,width).astype(float)
            counts[(history<width)|~fresh.to_numpy()]=np.nan
            out.loc[positions,name]=counts
        activity=rolling_count(ev.AvailableSession,session,60).astype(float)
        activity[(history<60)|~fresh.to_numpy()]=np.nan
        out.loc[positions,'EK_Activity_60d']=activity
        if not ev.empty:
            es=pd.merge_asof(pd.DataFrame({'date':decisions.to_numpy(),'position':positions}).sort_values('date'),
                             ev[['AvailableDate','filing_date']].sort_values('AvailableDate'),
                             left_on='date',right_on='AvailableDate',direction='backward')
            age=(es.date-es.filing_date).dt.days.astype(float);age[~fresh.to_numpy()]=np.nan
            out.loc[es.position,'EK_DaysSince']=age.to_numpy()
        fs=fg.get(cik)
        if fs is None:issues.append(dict(issue,reason='NO_LANGUAGE_FILINGS'));continue
        fs=fs.loc[fs.filing_date.between(r.linkdt,r.linkenddt)&(fs.AvailableDate<=r.linkenddt)].copy()
        if fs.empty:issues.append(dict(issue,reason='NO_LANGUAGE_WITHIN_LINK'));continue
        # Do not carry predecessor comparisons across a CCM interval boundary.
        allowed=set(fs.accession)
        fs.loc[~fs.previous_accession.isin(allowed),['TXT_SimPrev','TXT_LenChange']]=np.nan
        # Same-day documents are all public by next session; accession order is a deterministic tie rule.
        fs=fs.sort_values(['AvailableDate','filing_date','accession']).drop_duplicates('AvailableDate',keep='last')
        j=pd.merge_asof(pd.DataFrame({'date':decisions.to_numpy(),'position':positions}).sort_values('date'),
                        fs[['AvailableDate','filing_date']+LANG].sort_values('AvailableDate'),
                        left_on='date',right_on='AvailableDate',direction='backward')
        age=(j.date-j.filing_date).dt.days
        usable=j.filing_date.notna()&age.le(365)
        pos=j.loc[usable,'position']
        out.loc[pos,LANG]=j.loc[usable,LANG].to_numpy()
        out.loc[pos,'TXT_DaysSinceFiling']=age.loc[usable].to_numpy()
        out.loc[pos,'TextSourceFilingDate']=j.loc[usable,'filing_date'].to_numpy()
        out.loc[pos,'TextSourceAvailableDate']=j.loc[usable,'AvailableDate'].to_numpy()
        if number%50==0:print(f'Linked {number}/{len(links)} CCM intervals',flush=True)
    for p in sorted(set(base.permno.astype(int))-set(links.permno.astype(int))):
        issues.append({'permno':p,'reason':'NO_CCM_LINK'})
    used=out.TextSourceFilingDate.notna()
    assert (out.loc[used,'TextSourceFilingDate']<out.loc[used,'date']).all()
    assert (out.loc[used,'TextSourceAvailableDate']<=out.loc[used,'date']).all()
    assert len(out)==len(base) and not out.duplicated(['date','permno']).any()
    return out,pd.DataFrame(issues)

def self_test():
    cal=pd.bdate_range('2020-01-01',periods=270)
    b=pd.DataFrame({'date':cal,'permno':1})
    for flag in FLAGS:b[flag]=False
    l=pd.DataFrame({'permno':[1],'gvkey':['1'],'cik':[1],'linkdt':[cal[0]],'linkenddt':[cal[-2]]})
    f=pd.DataFrame([dict(cik=1,accession='a',previous_accession='',form='10-K',filing_date=cal[1],**{x:.1 for x in LANG})])
    i=pd.DataFrame([dict(cik=1,accession='a',form='10-K',filing_date=cal[1],items=''),
                    dict(cik=1,accession='b',form='8-K',filing_date=cal[252],items='1.03,4.02')])
    out,_=build(b,l,available(f,cal),available(i,cal),cal)
    assert out.loc[1,'TXT_HasCoverage']==0 and pd.isna(out.loc[1,'LM_Negative'])
    assert out.loc[2,'LM_Negative']==.1
    assert pd.isna(out.loc[252,'EK_Bankruptcy'])
    assert out.loc[253,'EK_Bankruptcy']==1
    assert out.loc[269,'TXT_HasCoverage']==0 and pd.isna(out.loc[269,'LM_Negative'])
    assert rolling_count([2,3],np.array([3,4]),2).tolist()==[2,1]
    print('PASS | next-session availability, event windows, missing history, mapping expiry')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
    p.add_argument('--links',default='datasets/text_reuse_20260920T180213Z_8cc71088/dated_link_cik_candidates.csv')
    p.add_argument('--filings',default='datasets/filing_features_20260920T212519Z_e7bbdcc5/filing_features.csv')
    p.add_argument('--index',default='datasets/edgar_index.csv')
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args();self_test()
    if a.self_test_only:return
    paths={k:Path(v) for k,v in vars(a).items() if k!='self_test_only'}
    hashes={k:sha(v) for k,v in paths.items()}
    b,l,f,i,cal=read_inputs(paths['technical'],paths['links'],paths['filings'],paths['index'])
    technical_report=json.loads(paths['technical'].with_name('build_report.json').read_text(encoding='utf-8'))
    if len(b)!=technical_report['output_rows_including_2009_context']:raise ValueError('Technical panel row count mismatch')
    outdir=Path('datasets')/('daily_text_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    outdir.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','output_folder':str(outdir),'source_hashes':hashes}
    try:
        out,issues=build(b,l,f,i,cal)
        out.to_csv(outdir/'text_daily.csv',index=False)
        issues.to_csv(outdir/'text_linkage_issues.csv',index=False)
        pd.DataFrame({'feature':FEATURES}).to_csv(outdir/'text_feature_columns.csv',index=False)
        split={}
        for flag in FLAGS:
            x=out.loc[b[flag]]
            split[flag]={'endpoints':len(x),'recent_filing_coverage':int(x.TXT_HasCoverage.sum()),
                         'language_available':int(x.LM_Negative.notna().sum()),
                         'all_event_counts_available':int(x[list(EVENTS)].notna().all(axis=1).sum())}
        report.update(status='DAILY_TEXT_BUILT_CANDIDATE_LINKAGE_REQUIRES_REVIEW',rows=len(out),permnos=int(out.permno.nunique()),
                      features=len(FEATURES),splits=split,linkage_issue_rows=len(issues),
                      identity_policy='Dated CCM plus current comp.company CIK candidates. NOT historically verified; no manual overrides.',
                      limitations=['Current CIK candidates can differ from historical filers; unknown or absent-archive intervals remain missing.',
                                   'Archive completeness is not established by file existence. Counts describe the supplied index.',
                                   'Coverage means a mapped filing observed in preceding 365 calendar days; it is not proof of complete coverage.'],
                      protocol={'availability':'Strictly next observed market session after filing date.',
                                'history':'No filing/predecessor transfer across CCM interval boundaries.',
                                'language':'Latest available filing; 365 calendar-day expiry; accession tie-break on same availability date.',
                                'events':'252/126 market-session windows, activity 60 sessions. Counts missing until full window since first mapped filing.',
                                'days_since':'Calendar days, not trading sessions; missing if no prior event.',
                                'missing':'Retain NaNs. No imputation/scaling. Model inputs restricted to text_feature_columns.csv.'})
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:(outdir/'text_daily_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('DONE | daily sidecar created; no training or changes to existing panels')

if __name__=='__main__':main()
