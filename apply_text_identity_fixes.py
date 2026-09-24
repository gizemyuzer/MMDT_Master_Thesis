"""Two evidence-backed text-link repairs; rebuild and compare without training.
Requires the existing build_daily_text_panel.py in the project root.
Retains provisional status: these repairs do not verify the entire CIK bridge.
"""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd

LINK_HASH='9441621b604d739e5624d54a0c38fadee4fc364a9b6c578a8265202f71073523'
SOURCES={
 'comstock':['Archived 10-K: CIK 23194, accession 0000950123-10-018130, filed 2010-02-26, cover COMSTOCK RESOURCES INC, EIN 94-1667468',
 'https://investors.comstockresources.com/node/15216'],
 'genon':['https://www.sec.gov/Archives/edgar/data/1126294/000095012310111604/c09394e8vk.htm',
 'https://www.sec.gov/Archives/edgar/data/1010775/000119312510275719/d8k.htm']}

def patch(l):
    l=l.copy(); records=[]
    specs=[(11644,23002,'1987-08-28','2018-08-13',None,23194,'comstock'),
           (88992,140033,'2010-12-06','2012-12-31',1010775,1126294,'genon')]
    for permno,gvkey,start,end,old,new,source in specs:
        mask=(pd.to_numeric(l.lpermno).eq(permno)&pd.to_numeric(l.gvkey).eq(gvkey)&l.linkdt.eq(start)&l.linkenddt.eq(end))
        if int(mask.sum())!=1:raise ValueError(f'Expected exactly one original interval for {permno}')
        v=pd.to_numeric(l.loc[mask,'cik'],errors='raise').iloc[0]
        if not (pd.isna(v) if old is None else v==old):raise ValueError(f'Unexpected original CIK for {permno}')
        l.loc[mask,'cik']=str(new)
        l.loc[mask,'identity_status']='TARGETED_DOCUMENTED_REPAIR_NOT_GLOBAL_VERIFICATION'
        records.append(dict(permno=permno,gvkey=gvkey,start=start,end=end,old_cik=old,new_cik=new,evidence=SOURCES[source]))
    return l,records

def compare(old,new,base,features):
    keys=['date','permno']
    if old.duplicated(keys).any() or new.duplicated(keys).any():raise ValueError('Duplicate sidecar keys')
    old=old.set_index(keys);new=new.set_index(keys)
    if len(old)!=len(new) or not old.index.isin(new.index).all():raise ValueError('Old/new key sets differ')
    old=old.reindex(new.index)
    a=old[features].to_numpy(dtype=float);b=new[features].to_numpy(dtype=float)
    changed=~np.isclose(a,b,rtol=1e-12,atol=1e-14,equal_nan=True)
    flag=base.set_index(keys).reindex(new.index)
    result={}
    for split in ['TrainEndpoint','ValidationEndpoint','TestEndpoint']:
        take=flag[split].to_numpy(dtype=bool)
        result[split]=dict(endpoints=int(take.sum()),changed_endpoint_feature_rows=int(changed[take].any(axis=1).sum()),
                          changed_cells=int(changed[take].sum()))
    # Endpoint counts alone omit altered rows inside the preceding 20-session sequences.
    all_changed=changed.any(axis=1)
    ids=new.index.get_level_values('permno').to_numpy()
    if not set(ids[all_changed]).issubset({11644,88992}):raise AssertionError('Unrelated PERMNO features changed')
    counts=pd.DataFrame({'permno':ids,'changed':all_changed}).groupby('permno').changed.sum()
    return result,{str(k):int(v) for k,v in counts[counts>0].items()},dict(zip(features,changed.sum(axis=0).astype(int).tolist()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
    p.add_argument('--links',default='datasets/text_reuse_20260920T180213Z_8cc71088/dated_link_cik_candidates.csv')
    p.add_argument('--filings',default='datasets/filing_features_20260920T212519Z_e7bbdcc5/filing_features.csv')
    p.add_argument('--index',default='datasets/edgar_index.csv')
    p.add_argument('--old-text',default='datasets/daily_text_20260920T220718Z_8b07a12d/text_daily.csv')
    p.add_argument('--builder',default='build_daily_text_panel.py')
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('daily_builder',a.builder)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.self_test()
    if mod.sha(a.links)!=LINK_HASH:raise ValueError('Original candidate file hash differs; review before applying')
    oldreport=json.loads(Path(a.old_text).with_name('text_daily_report.json').read_text(encoding='utf-8'))
    paths={k:Path(getattr(a,k)) for k in ['technical','links','filings','index']}
    for k,path in paths.items():
        if oldreport['source_hashes'].get(k)!=mod.sha(path):raise ValueError(f'Original text source mismatch: {k}')
    original=pd.read_csv(a.links,dtype=str)
    repaired,changes=patch(original)
    out=Path('datasets')/('text_identity_repair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]);out.mkdir(parents=True)
    repaired.to_csv(out/'dated_links_repaired.csv',index=False)
    report=dict(status='RUNNING',output_folder=str(out),repairs=changes,
                entire_bridge_verified=False,training_ready=False)
    try:
        b,l,f,i,cal=mod.read_inputs(a.technical,out/'dated_links_repaired.csv',a.filings,a.index)
        new,issues=mod.build(b,l,f,i,cal)
        old=pd.read_csv(a.old_text,usecols=['date','permno']+mod.FEATURES,parse_dates=['date'])
        old['permno']=mod.ident(old.permno)
        stats,by_id,by_feature=compare(old,new,b,mod.FEATURES)
        new.to_csv(out/'text_daily.csv',index=False)
        issues.to_csv(out/'text_linkage_issues.csv',index=False)
        pd.DataFrame({'feature':mod.FEATURES}).to_csv(out/'text_feature_columns.csv',index=False)
        daily=dict(oldreport)
        daily.update(status='TARGETED_REPAIRS_BUILT_REVIEW_REQUIRED',output_folder=str(out),
                     rows=len(new),permnos=int(new.permno.nunique()),linkage_issue_rows=len(issues),
                     identity_policy='Two documented manual repairs; other links retain candidate status.',
                     repairs=changes)
        daily['source_hashes']=dict(oldreport['source_hashes'],links=mod.sha(out/'dated_links_repaired.csv'))
        daily['splits']={}
        for flag in mod.FLAGS:
            x=new.loc[b[flag]]
            daily['splits'][flag]=dict(endpoints=len(x),recent_filing_coverage=int(x.TXT_HasCoverage.sum()),
              language_available=int(x.LM_Negative.notna().sum()),all_event_counts_available=int(x[list(mod.EVENTS)].notna().all(axis=1).sum()))
        (out/'text_daily_report.json').write_text(json.dumps(daily,indent=2),encoding='utf-8')
        report.update(status='TARGETED_REPAIRS_BUILT_AND_COMPARED',endpoint_changes=stats,
                      changed_rows_by_permno=by_id,changed_cells_by_feature=by_feature,
                      old_text_sha256=mod.sha(a.old_text),new_text_sha256=mod.sha(out/'text_daily.csv'),
                      note='Changes count direct endpoint features, not all affected sequence windows. No training performed.')
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:(out/'identity_repair_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('DONE | two repairs + comparison; entire bridge still under review; no training',flush=True)

if __name__=='__main__':main()
