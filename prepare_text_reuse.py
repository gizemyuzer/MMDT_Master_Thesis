"""Check existing EDGAR archive paths and export CCM-GVKEY CIK candidates.
No SEC downloads, training, deletion, or automatic approval of historical CIKs.
"""
import argparse,hashlib,json,tempfile
from pathlib import Path,PureWindowsPath
from datetime import datetime,timezone
from uuid import uuid4
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def local_path(root,value):
    if pd.isna(value):raise ValueError('Missing archive path')
    w=PureWindowsPath(str(value))
    if w.is_absolute() or w.drive or w.root or '..' in w.parts:
        raise ValueError('Archive path must be relative and inside the project: '+str(value))
    p=(root/Path(*w.parts)).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Archive path escapes project')
    return p

def validate_manifest(m,i):
    key=['ticker','cik','accession']
    if m.duplicated(key).any() or i.duplicated(key).any():raise ValueError('Duplicate filing keys')
    x=m.merge(i,on=key,how='outer',suffixes=('_manifest','_index'),indicator=True,validate='one_to_one')
    if not x._merge.eq('both').all():raise ValueError('Manifest and index keys do not match')
    for c in ['form','filing_date','period_of_report']:
        if not x[c+'_manifest'].fillna('').eq(x[c+'_index'].fillna('')).all():raise ValueError('Manifest/index mismatch: '+c)
    return {'rows':len(m),'status_counts':{str(k):int(v) for k,v in m.status.value_counts(dropna=False).items()},
        'unique_paths':int(m.path.nunique()),'securities':int(i.permno.nunique()),
        'forms':{str(k):int(v) for k,v in m.form.value_counts().items()}}

def candidate_table(links,company):
    l=links.copy();c=company.copy()
    for f in [l,c]:f['gvkey']=pd.to_numeric(f.gvkey,errors='raise').astype('int64').astype(str).str.zfill(6)
    if c.gvkey.duplicated().any():raise ValueError('Duplicate company GVKEYs require review')
    c['cik']=pd.to_numeric(c.cik,errors='coerce').astype('Float64')
    invalid=c.cik.notna() & (c.cik.le(0)|c.cik.mod(1).ne(0))
    if invalid.any():raise ValueError('Invalid CIK candidates')
    c['cik']=c.cik.astype('Int64')
    merged=l.merge(c,on='gvkey',how='left',validate='many_to_one')
    merged['identity_status']='CURRENT_COMPANY_CIK_CANDIDATE_NOT_HISTORICALLY_VERIFIED'
    return merged

def self_tests():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        assert local_path(root,r'data\edgar\1\a.txt.gz')==root/'data'/'edgar'/'1'/'a.txt.gz'
        for bad in ['../secret',r'C:\secret',r'\secret']:
            try:local_path(root,bad)
            except ValueError:pass
            else:raise AssertionError('Unsafe path accepted')
    l=pd.DataFrame({'gvkey':[1,2],'lpermno':[10,10],'linkdt':['2000-01-01','2010-01-01']})
    c=pd.DataFrame({'gvkey':[1,2],'cik':[11,22],'conm':['A','B']})
    out=candidate_table(l,c)
    assert len(out)==2 and out.cik.tolist()==[11,22]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',default='.')
    ap.add_argument('--links',default='datasets/fundamental_sources_20260919T164626Z_e10608aa/ccm_links.csv')
    ap.add_argument('--username',default='gizemyuzer')
    ap.add_argument('--self-test-only',action='store_true')
    a=ap.parse_args();self_tests();print('PASS | path checks and preservation of multiple GVKEY histories',flush=True)
    if a.self_test_only:return
    root=Path(a.project).resolve()
    paths={'manifest':root/'datasets/edgar_manifest.csv','index':root/'datasets/edgar_index.csv','links':root/a.links}
    hashes={k:sha(v) for k,v in paths.items()}
    m=pd.read_csv(paths['manifest'],dtype={'accession':str});i=pd.read_csv(paths['index'],dtype={'accession':str})
    l=pd.read_csv(paths['links'],dtype={'gvkey':str})
    info=validate_manifest(m,i)
    out=root/'datasets'/('text_reuse_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'STARTED','output_folder':str(out),'source_hashes':hashes,'manifest':info}
    db=None
    try:
        exists={}
        for number,p in enumerate(m.path.drop_duplicates(),1):
            exists[p]=local_path(root,p).is_file()
            if number%5000==0:print(f'Archive paths checked: {number}/{info["unique_paths"]}',flush=True)
        m['LocalFileExists']=m.path.map(exists)
        m.to_csv(out/'archive_path_audit.csv',index=False)
        missing=m.loc[~m.LocalFileExists].copy();missing.to_csv(out/'missing_archive_files.csv',index=False)
        report['archive']={'existing_unique_paths':sum(exists.values()),'missing_unique_paths':sum(not v for v in exists.values()),
            'missing_manifest_rows':len(missing),'missing_10k_10q_rows':int(missing.form.isin(['10-K','10-Q']).sum()),
            'gzip_contents_verified':False}
        print('Fetching CIK candidates from WRDS for all linked GVKEYs (no SEC requests)...',flush=True)
        import wrds
        db=wrds.Connection(wrds_username=a.username)
        keys=tuple(sorted(pd.to_numeric(l.gvkey).astype('int64').astype(str).str.zfill(6).unique()))
        c=db.raw_sql('SELECT gvkey, cik, conm FROM comp.company WHERE gvkey IN %(gvkeys)s',params={'gvkeys':keys})
        c.to_csv(out/'company_cik_candidates.csv',index=False)
        candidates=candidate_table(l,c)
        candidates.to_csv(out/'dated_link_cik_candidates.csv',index=False)
        report['candidates']={'link_rows':len(candidates),'gvkeys':int(candidates.gvkey.nunique()),
           'gvkeys_without_cik':sorted(candidates.loc[candidates.cik.isna(),'gvkey'].unique().tolist())}
        report['notes']=['File existence only; gzip integrity and text content are not checked in this step.',
            'comp.company supplies current CIK candidates, not a dated historical CIK bridge. CCM dates do not establish CIK validity.',
            'All dated CCM links retained; no longest-link selection. Candidates need review before changing the text panel.',
            'No SEC document downloads, archive modifications or training.']
        if hashes!={k:sha(v) for k,v in paths.items()}:raise RuntimeError('Source changed during run')
        report['status']='ARCHIVE_CHECKED_CIK_CANDIDATES_EXPORTED_REVIEW_REQUIRED'
    except Exception as e:
        report.update(status='FAILED_PARTIAL_OUTPUT',error_type=type(e).__name__,error=str(e));raise
    finally:
        (out/'text_reuse_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if db is not None:db.close()
    print(json.dumps(report,indent=2));print('PASS | review outputs saved; no downloads or training')

if __name__=='__main__':main()
