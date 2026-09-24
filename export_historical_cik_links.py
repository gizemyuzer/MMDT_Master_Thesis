"""Read-only export of discovered WRDS CIK sources for thesis GVKEYs.
Does not choose historical mappings, modify datasets or train models.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import pandas as pd

DEFAULT = 'datasets/text_reuse_20260920T180213Z_8cc71088/dated_link_cik_candidates.csv'

def normalize(values):
    n = pd.to_numeric(values, errors='raise').dropna()
    if ((n % 1 != 0) | (n <= 0)).any():
        raise ValueError('Invalid GVKEY')
    return sorted(set(n.astype(int)))

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--username', default='gizemyuzer')
    a.add_argument('--links', default=DEFAULT)
    a.add_argument('--self-test', action='store_true')
    args = a.parse_args()
    if args.self_test:
        assert normalize(pd.Series(['005496', '5496', '13431'])) == [5496, 13431]
        print('PASS | GVKEY normalization; no WRDS connection')
        return
    source = Path(args.links)
    if not source.is_file():
        raise FileNotFoundError(f'Missing input: {source.resolve()}; use --links PATH')
    old = pd.read_csv(source, dtype={'gvkey': 'string'})
    if not {'gvkey','lpermno'}.issubset(old.columns):
        raise ValueError('Input needs gvkey and lpermno')
    keys = normalize(old.gvkey)
    if not keys:
        raise ValueError('No GVKEYs')
    tokens = sorted({s for k in keys for s in (str(k), str(k).zfill(6))})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path('results') / f'historical_cik_links_{stamp}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    report = {'status':'RUNNING', 'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'requested_gvkeys':len(keys), 'requested_permnos':int(old.lpermno.nunique()),
              'tables':{}, 'limitations':['Export is evidence for review, not an approved mapping.',
              'Link dates are retained as supplied; no assumptions about their semantics or automatic overrides.']}
    db = None
    try:
        import wrds
        db = wrds.Connection(wrds_username=args.username)
        # Known metadata-confirmed tables only. No broad scans or old/sample schema fallback.
        jobs = [
            ('wrdssec.wciklink_gvkey', 'historical_cik_links.csv',
             'SELECT * FROM wrdssec.wciklink_gvkey WHERE gvkey IN %(keys)s ORDER BY gvkey, cik, link_start_date',
             {'keys':tuple(tokens)}),
            ('comp.filings', 'filing_cik_evidence.csv',
             '''SELECT gvkey, date, name, form, edgar, cik, datechange, reportdate
                FROM comp.filings WHERE gvkey IN %(keys)s
                AND date >= %(start)s AND date <= %(end)s
                ORDER BY gvkey, date, cik''',
             {'keys':tuple(keys), 'start':'2009-01-01', 'end':'2024-12-31'})]
        for table, filename, sql, params in jobs:
            print('Reading',table,flush=True)
            try:
                frame = db.raw_sql(sql, params=params)
            except Exception as exc:
                code = getattr(getattr(exc,'orig',exc),'pgcode',None)
                report['tables'][table] = {'status':'FAILED','error_type':type(exc).__name__,'sqlstate':code}
                if code in ('42501','42P01'):
                    db.connection.rollback()
                    print(f'{table}: unavailable (SQLSTATE {code}); recorded.',flush=True)
                    continue
                raise
            frame.to_csv(out/filename,index=False)
            found = normalize(frame.gvkey)
            report['tables'][table] = {'status':'EXPORTED','rows':len(frame),'gvkeys':len(found),
                                       'missing_gvkeys':sorted(set(keys)-set(found))}
            print(f'{filename}: {len(frame):,} rows, {len(found)} GVKEYs',flush=True)
        old.to_csv(out/'input_candidates.csv',index=False)
        primary = report['tables'].get('wrdssec.wciklink_gvkey',{})
        report['status'] = 'EXPORTED_NOT_YET_VALIDATED' if primary.get('rows',0)>0 else 'PRIMARY_SOURCE_UNAVAILABLE_OR_EMPTY'
        print('\n'+report['status'],flush=True)
        print('Output:',out.resolve(),flush=True)
        print('No dataset changes; no training.',flush=True)
    except Exception as exc:
        report.update(status='FAILED',error_type=type(exc).__name__)
        raise
    finally:
        (out/'historical_cik_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if db is not None:
            db.close()

if __name__ == '__main__':
    main()
