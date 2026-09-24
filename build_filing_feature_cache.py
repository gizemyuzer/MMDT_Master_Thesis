"""Extract reusable filing-level features. No WRDS/SEC/training or panel joins.
Run from project root. Requires pandas and datasets/LM_MasterDictionary.csv.
CIK identifies filer, NOT a verified historical security mapping.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
from uuid import uuid4
import pandas as pd

CATEGORIES = ('negative', 'positive', 'uncertainty', 'litigious', 'constraining')
WORD = re.compile(r"[A-Za-z']+")

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def archive_path(root, value):
    p = PureWindowsPath(str(value))
    if p.drive or p.root or '..' in p.parts:
        raise ValueError('Archive path must be relative and remain inside project')
    out = root.joinpath(*p.parts).resolve()
    out.relative_to(root)
    return out

def dictionary(path):
    d = pd.read_csv(path, keep_default_na=False)
    d.columns = d.columns.str.strip().str.lower()
    required = {'word', *CATEGORIES}
    if not required.issubset(d.columns):
        raise ValueError(f'LM dictionary missing columns: {required - set(d.columns)}')
    if d.word.isna().any():
        raise ValueError('Missing dictionary word')
    cats = {c: set(d.loc[pd.to_numeric(d[c], errors='raise').fillna(0) > 0,
                            'word'].str.lower()) for c in CATEGORIES}
    if any(not v for v in cats.values()):
        raise ValueError('An LM category is empty')
    return cats

def scores(counts, previous, cats):
    n = sum(counts.values())
    if not n:
        raise ValueError('No words in filing')
    out = {'LM_WordCount': n}
    out.update({'LM_' + c.capitalize(): sum(v for w, v in counts.items() if w in vocab) / n
                for c, vocab in cats.items()})
    out.update(TXT_SimPrev=float('nan'), TXT_LenChange=float('nan'))
    if previous:
        dot = sum(v * previous.get(w, 0) for w, v in counts.items())
        out['TXT_SimPrev'] = dot / math.sqrt(sum(v*v for v in counts.values()) *
                                            sum(v*v for v in previous.values()))
        out['TXT_LenChange'] = math.log(n / sum(previous.values()))
    return out

def prepare_manifest(path):
    m = pd.read_csv(path, dtype={'cik':str, 'accession':str, 'path':str})
    m = m.loc[m.form.isin(['10-K', '10-Q'])].copy()
    if m.empty or m[['cik', 'accession', 'path', 'filing_date']].isna().any().any():
        raise ValueError('Empty or incomplete filing manifest')
    m['cik'] = pd.to_numeric(m.cik, errors='raise').astype('int64').astype(str)
    m['filing_date'] = pd.to_datetime(m.filing_date, errors='raise')
    # Multiple share classes may refer to the same original SEC document.
    check = m.groupby(['cik', 'accession'])[['form','filing_date','path','status']].nunique(dropna=False)
    if check.gt(1).any().any():
        raise ValueError('Conflicting metadata for the same CIK/accession')
    return m.drop_duplicates(['cik','accession']).sort_values(['cik','form','filing_date','accession'])

def extract(m, root, cats, output):
    features = ['LM_WordCount'] + ['LM_' + c.capitalize() for c in CATEGORIES] + ['TXT_SimPrev','TXT_LenChange']
    cols = ['cik','accession','form','filing_date','period_of_report','path','text_sha256',
            'previous_accession','same_day_group_size','status','error'] + features
    failures = []; total = 0; good = 0
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for (_, _), group in m.groupby(['cik','form'], sort=False):
            previous = None; previous_accession = ''
            for date, day in group.groupby('filing_date', sort=True):
                day_counts = []; day_accessions = []
                for r in day.to_dict('records'):
                    rec = {k:r.get(k, '') for k in cols[:6]}
                    rec.update(filing_date=date.strftime('%Y-%m-%d'), previous_accession=previous_accession,
                               same_day_group_size=len(day), status='OK', error='')
                    counts = None
                    try:
                        if r['status'] != 'ok':
                            raise ValueError('Manifest status is not ok')
                        with gzip.open(archive_path(root, r['path']), 'rt', encoding='utf-8', errors='strict') as g:
                            txt = g.read()  # Reading to EOF checks gzip CRC.
                        rec['text_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
                        counts = Counter(WORD.findall(txt.lower()))
                        rec.update(scores(counts, previous, cats))
                        good += 1
                    except (OSError, EOFError, UnicodeError, ValueError) as e:
                        counts = None
                        rec.update(status='FAILED', error=f'{type(e).__name__}: {e}')
                        failures.append({k:rec.get(k, '') for k in ('cik','accession','path','error')})
                    writer.writerow(rec); total += 1
                    day_counts.append(counts); day_accessions.append(r['accession'])
                    if total % 250 == 0:
                        f.flush(); print(f'Processed {total}/{len(m)} filings | failed={len(failures)}', flush=True)
                # With no acceptance timestamps, same-day order is unknown.
                # Clear predecessor on ambiguity/read failure; never silently skip it.
                if len(day_counts) == 1 and day_counts[0]:
                    previous, previous_accession = day_counts[0], day_accessions[0]
                else:
                    previous, previous_accession = None, ''
    return {'unique_filings':total, 'success':good, 'failed':len(failures)}, failures

def self_test():
    import tempfile
    cats = {c:{'risk'} for c in CATEGORIES}
    s = scores(Counter({'risk':2,'safe':2}), Counter({'risk':1,'safe':1}), cats)
    assert s['LM_Negative'] == .5 and abs(s['TXT_SimPrev']-1) < 1e-12
    assert abs(s['TXT_LenChange']-math.log(2)) < 1e-12
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve(); rows = []
        for j, date in enumerate(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-04','2020-01-05']):
            p = root / f'{j}.gz'
            if j == 1: p.write_bytes(b'bad gzip')
            else:
                with gzip.open(p, 'wt', encoding='utf-8') as f: f.write('risk safe')
            rows.append(dict(cik='1', accession=str(j), form='10-K', filing_date=pd.Timestamp(date), path=p.name, status='ok'))
        stats, _ = extract(pd.DataFrame(rows), root, cats, root/'out.csv')
        result = pd.read_csv(root/'out.csv')
        assert stats['failed'] == 1
        assert pd.isna(result.loc[2,'TXT_SimPrev'])
        assert result.loc[3,'TXT_SimPrev'] == result.loc[4,'TXT_SimPrev'] == 1
        assert pd.isna(result.loc[5,'TXT_SimPrev'])
    print('PASS | ratios, similarity, corrupt archive, predecessor gaps, same-day ambiguity')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='.')
    p.add_argument('--dictionary', default='datasets/LM_MasterDictionary.csv')
    p.add_argument('--self-test-only', action='store_true')
    a = p.parse_args(); self_test()
    if a.self_test_only: return
    root = Path(a.project).resolve()
    manifest = root/'datasets/edgar_manifest.csv'; lm = root/a.dictionary
    cats = dictionary(lm); m = prepare_manifest(manifest)
    hashes = {'manifest':sha(manifest), 'dictionary':sha(lm)}
    out = root/'datasets'/('filing_features_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = {'status':'RUNNING', 'output_folder':str(out), 'source_hashes':hashes,
              'dictionary_categories':{k:len(v) for k,v in cats.items()},
              'protocol':{'key':'CIK + accession; no security identity approval or daily alignment',
                          'text':'Existing cleaned text; strict UTF-8 and gzip CRC. Cleaning quality not certified.',
                          'dictionary':'Frozen supplied dictionary; vocabulary is not a historical-vintage dictionary.',
                          'similarity':'Same CIK and form; strictly earlier filing date. Clear predecessor after failed or multiple same-day filings.',
                          'length_change':'Natural log of word-count ratio, not percentage change',
                          'availability':'Filing date retained only. Daily join must use next trading session.',
                          'coverage':'10-K/10-Q only; no 8-K content processing or coverage assumptions'}}
    try:
        print(f'Extracting {len(m)} unique 10-K/10-Q filings (CPU, no network)...', flush=True)
        stats, failures = extract(m, root, cats, out/'filing_features.csv')
        report.update(stats)
        pd.DataFrame(failures, columns=['cik','accession','path','error']).to_csv(out/'filing_failures.csv', index=False)
        if hashes != {'manifest':sha(manifest), 'dictionary':sha(lm)}:
            raise RuntimeError('Source changed while running')
        report['status'] = 'FEATURE_CACHE_READY_IDENTITY_JOIN_PENDING' if not failures else 'FEATURE_CACHE_HAS_FAILURES_REVIEW_REQUIRED'
        report['output_sha256'] = sha(out/'filing_features.csv')
    except Exception as e:
        report.update(status='FAILED_PARTIAL_OUTPUT', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        (out/'filing_feature_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('DONE | filing cache only; existing datasets unchanged; no training')

if __name__ == '__main__':
    main()
