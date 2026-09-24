"""Offline evidence extraction, not automatic PERMNO-CIK verification.
Run from project root. Reads existing cleaned gzip filings; never edits inputs.
Dependencies: pandas. No WRDS, network, torch, or training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
from uuid import uuid4
import pandas as pd

LABEL = re.compile(r'exact\s+name\s+of\s+(?:the\s+)?registrant', re.I)
CIK = re.compile(r'(?:central\s+index\s+key|\bCIK)\s*[:#=]?\s*(\d{1,10})\b', re.I)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def ident(v):
    s = str(v).strip()
    if not re.fullmatch(r'\d+(?:\.0+)?', s): raise ValueError('Invalid identifier: '+s)
    return str(int(s.split('.')[0]))

def safe_path(root, value):
    p = PureWindowsPath(str(value))
    if p.drive or p.root or '..' in p.parts: raise ValueError('Unsafe archive path')
    out = root.joinpath(*p.parts).resolve(); out.relative_to(root)
    return out

def evidence(text):
    # Limit identity searches to cover/header region to avoid subsidiary mentions.
    cover = re.sub(r'\s+', ' ', text[:25000]).strip()
    hits = list(LABEL.finditer(cover))
    contexts = [cover[max(0, m.start()-250):m.end()+160] for m in hits[:3]]
    ciks = sorted({ident(m.group(1)) for m in CIK.finditer(cover)}, key=int)
    return cover[:1400], ' || '.join(contexts), '|'.join(ciks)

def tests():
    assert evidence('ACME INC. (Exact name of registrant as specified) CIK: 00000123')[2]=='123'
    assert 'ACME' in evidence('ACME INC. (Exact name of registrant as specified)')[1]
    assert evidence('No identifying header here')[1:]==('', '')
    assert evidence('x'*25000+' CIK: 999')[2]==''
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root=Path(d).resolve()
        for v in ['../bad.gz', 'C:\\bad.gz']:
            try: safe_path(root,v)
            except ValueError: pass
            else: raise AssertionError('Unsafe path accepted')
        p=root/'bad.gz';p.write_bytes(b'corrupt')
        try:
            with gzip.open(p,'rt',encoding='utf-8',errors='strict') as f: f.read()
        except OSError: pass
        else: raise AssertionError('Bad gzip accepted')
    print('PASS | cover evidence, missing identity, path containment, corrupt archive',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--manifest',default='datasets/edgar_manifest.csv')
    p.add_argument('--links',default='datasets/text_reuse_20260920T180213Z_8cc71088/dated_link_cik_candidates.csv')
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args();tests()
    if a.self_test_only:return
    root=Path(a.project).resolve();mp=root/a.manifest;lp=root/a.links
    m=pd.read_csv(mp,dtype=str,keep_default_na=False)
    links=pd.read_csv(lp,dtype=str,keep_default_na=False)
    for cols,frame in [({'cik','accession','form','filing_date','path','status'},m),({'cik','gvkey','lpermno','linkdt','linkenddt'},links)]:
        if not cols.issubset(frame):raise ValueError('Missing columns: '+str(cols-set(frame)))
    m=m.loc[m.form.isin(['10-K','10-Q'])].copy()
    if m.empty:raise ValueError('No 10-K/10-Q filings in manifest')
    m['cik']=m.cik.map(ident)
    if m.groupby(['cik','accession'])[['path','form','filing_date','status']].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting metadata for duplicate filing')
    m=m.drop_duplicates(['cik','accession']).sort_values(['cik','filing_date','accession'])
    m['parsed_date']=pd.to_datetime(m.filing_date,errors='raise')
    out=root/'results'/('archive_identity_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]);out.mkdir(parents=True)
    rows=[]
    for i,r in enumerate(m.to_dict('records'),1):
        rec={k:r[k] for k in ['cik','accession','form','filing_date','path']}
        rec.update(status='READ_OK',error='',text_sha256='',cover_excerpt='',registrant_context='',explicit_cover_ciks='')
        try:
            if r['status']!='ok':raise ValueError('Manifest status is not ok: '+r['status'])
            with gzip.open(safe_path(root,r['path']),'rt',encoding='utf-8',errors='strict') as f:text=f.read()
            if not text.strip():raise ValueError('Empty archive text')
            rec['text_sha256']=hashlib.sha256(text.encode('utf-8')).hexdigest()
            rec['cover_excerpt'],rec['registrant_context'],rec['explicit_cover_ciks']=evidence(text)
        except (OSError,EOFError,UnicodeError,ValueError) as e:
            rec.update(status='READ_FAILED',error=f'{type(e).__name__}: {e}')
        # A match authenticates neither ownership of the security nor historical mapping.
        cs=rec['explicit_cover_ciks'].split('|') if rec['explicit_cover_ciks'] else []
        rec['cik_evidence_status']=('NO_EXPLICIT_CIK' if not cs else 'SINGLE_CIK_MATCH' if cs==[r['cik']] else 'REVIEW_CIK_MISMATCH_OR_MULTIPLE')
        rows.append(rec)
        if i%500==0:print(f'Read {i}/{len(m)} filings',flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'filing_identity_evidence.csv',index=False)
    good=d.loc[d.status.eq('READ_OK')].copy();good['parsed_date']=pd.to_datetime(good.filing_date)
    review=[]
    for r in links.to_dict('records'):
        c=ident(r['cik']) if r['cik'].strip() else ''
        start=pd.to_datetime(r['linkdt'],errors='raise') if r['linkdt'] else pd.Timestamp.min
        end=pd.to_datetime(r['linkenddt'],errors='raise') if r['linkenddt'] else pd.Timestamp.max
        g=good.loc[good.cik.eq(c)&good.parsed_date.ge(start)&good.parsed_date.le(end)]
        rec=dict(r);rec.update(readable_language_filings_in_interval=len(g),
            first_archived_filing=g.filing_date.min() if len(g) else '',last_archived_filing=g.filing_date.max() if len(g) else '',
            cover_name_contexts=int(g.registrant_context.ne('').sum()),
            explicit_cik_review_count=int(g.cik_evidence_status.eq('REVIEW_CIK_MISMATCH_OR_MULTIPLE').sum()),
            audit_status='EVIDENCE_AVAILABLE_NOT_VERIFIED' if len(g) else 'NO_READABLE_LANGUAGE_EVIDENCE')
        review.append(rec)
    pd.DataFrame(review).to_csv(out/'candidate_interval_evidence.csv',index=False)
    report=dict(status='EVIDENCE_EXTRACTED_NOT_IDENTITY_VERIFIED',output_folder=str(out),
        source_hashes={'manifest':sha(mp),'links':sha(lp)},unique_language_filings=len(d),
        read_status_counts=dict(Counter(d.status)),cik_evidence_counts=dict(Counter(d.cik_evidence_status)),
        filings_with_registrant_context=int(d.registrant_context.ne('').sum()),candidate_intervals=len(review),
        limitations=['Cover excerpts require review against independent historical security/company records.',
        'CIK from manifest or accession is not independent evidence of PERMNO ownership.',
        'Matching explicit cover CIK alone does not verify historical PERMNO-CIK linkage.',
        'Only existing 10-K/10-Q archive inspected; absent historical filers cannot be recovered by this script.',
        'No datasets, mappings, predictions or checkpoints changed.'])
    (out/'archive_identity_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('DONE | offline evidence only; no mapping approved; no training',flush=True)

if __name__=='__main__':main()
