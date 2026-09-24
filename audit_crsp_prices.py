"""Read-only price audit. Reads an existing cache and WRDS; never trains or replaces it.

Run from the project root: python audit_crsp_prices.py
Only new files under results/price_audit_<timestamp>_<id>/ are written.
This diagnoses adjustment discrepancies; it does NOT rebuild labels or portfolios.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_prices(raw):
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date']).dt.normalize()
    raw['ticker'] = raw['ticker'].astype(str).str.strip()
    for col in ['prc', 'cfacpr', 'ret', 'retx']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    # Exact duplicate name-history joins are harmless; conflicting ones are not.
    raw = raw.drop_duplicates()
    if raw.duplicated(['permno', 'date']).any():
        raise ValueError('Conflicting PERMNO/date rows from name-history join; inspect raw export.')
    raw = raw.sort_values(['permno', 'date']).reset_index(drop=True)
    raw['raw_close'] = raw['prc'].abs().where(raw['prc'].abs() > 0)
    raw['adjusted_close'] = raw['raw_close'] / raw['cfacpr'].where(raw['cfacpr'] > 0)
    g = raw.groupby('permno', sort=False)
    raw['raw_price_return'] = raw['raw_close'] / g['raw_close'].shift() - 1
    raw['adjusted_price_return'] = raw['adjusted_close'] / g['adjusted_close'].shift() - 1
    prev = g['cfacpr'].shift()
    raw['factor_changed'] = (prev.gt(0) & raw['cfacpr'].gt(0)
                             & ~np.isclose(raw['cfacpr'], prev, equal_nan=True))
    raw['return_difference'] = raw['raw_price_return'] - raw['adjusted_price_return']
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', default='datasets/final_dataset.csv')
    ap.add_argument('--username', default='gizemyuzer')
    ap.add_argument('--raw-csv', help='Reuse this audit\'s crsp_daily_raw.csv; no WRDS call.')
    args = ap.parse_args()
    cache_path = Path(args.cache).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f'Existing cache not found: {cache_path}. Specify --cache PATH.')
    before = sha256(cache_path)
    header = pd.read_csv(cache_path, nrows=0)
    names = {str(c).lower(): c for c in header.columns}
    required = ['date', 'ticker', 'close']
    if not all(c in names for c in required):
        raise ValueError(f'Cache must contain Date, Ticker, Close; found {list(header.columns)}')
    panel = pd.read_csv(cache_path, usecols=[names[c] for c in required])
    panel = panel.rename(columns={names[c]: c for c in required})
    panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
    if panel[required].isna().any().any():
        raise ValueError('Missing date/ticker/close in cache; resolve before audit.')
    panel['ticker'] = panel['ticker'].astype(str).str.strip()
    panel['close'] = pd.to_numeric(panel['close'], errors='raise')
    if panel.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate cache date/ticker keys.')
    out = Path('results') / ('price_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                              + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', cache_sha256=before, cache_rows=len(panel),
                  cache_tickers=int(panel.ticker.nunique()),
                  scope='Price adjustment diagnostic only; no label reconstruction or training.')
    try:
        if args.raw_csv:
            raw = pd.read_csv(args.raw_csv)
        else:
            import wrds
            tickers = sorted(panel.ticker.unique().tolist())
            # Match the original ticker/name-date selection to audit what it fetched.
            # This is NOT an endorsement of ticker-based security identity.
            sql = '''SELECT a.date, a.permno, b.ticker, a.prc, a.cfacpr, a.ret, a.retx
                     FROM crsp.dsf a JOIN crsp.stocknames b ON a.permno=b.permno
                     WHERE b.ticker IN %(tickers)s
                     AND a.date BETWEEN %(start)s AND %(end)s
                     AND a.date BETWEEN b.namedt AND b.nameenddt
                     ORDER BY a.permno, a.date'''
            params = dict(tickers=tuple(tickers),
                          start=(panel.date.min()-pd.Timedelta(days=60)).strftime('%Y-%m-%d'),
                          end=panel.date.max().strftime('%Y-%m-%d'))
            (out / 'query.sql').write_text(sql, encoding='utf-8')
            print('AUDIT ONLY | WRDS read | no GPU/training/cache changes', flush=True)
            db = wrds.Connection(wrds_username=args.username)
            try:
                raw = db.raw_sql(sql, params=params, date_cols=['date'])
            finally:
                db.close()
        if raw.empty:
            raise ValueError('WRDS returned no rows.')
        raw.to_csv(out / 'crsp_daily_raw.csv', index=False)
        prices = prepare_prices(raw)
        ambiguous = prices.duplicated(['date', 'ticker'], keep=False)
        prices.loc[ambiguous].to_csv(out / 'ambiguous_ticker_dates.csv', index=False)
        counts = prices.groupby('ticker')['permno'].nunique()
        report['tickers_with_multiple_permnos'] = {str(k): int(v) for k, v in counts[counts > 1].items()}
        report['ambiguous_source_rows_excluded'] = int(ambiguous.sum())
        merged = panel.merge(prices.loc[~ambiguous], on=['date', 'ticker'], how='left',
                             validate='one_to_one', indicator=True)
        matched = merged['_merge'].eq('both')
        report['matched_cache_rows'] = int(matched.sum())
        report['unmatched_cache_rows'] = int((~matched).sum())
        for name in ['raw_close', 'adjusted_close']:
            valid = matched & merged[name].notna()
            report[f'{name}_valid_rows'] = int(valid.sum())
            report[f'cache_matches_{name}_rows'] = int((valid & np.isclose(
                merged['close'], merged[name], rtol=1e-5, atol=1e-6)).sum())
        event = merged['factor_changed'].eq(True)
        material = event & merged['return_difference'].abs().gt(0.01)
        report['factor_change_days_in_cache'] = int(event.sum())
        report['factor_change_days_return_difference_gt_1pp'] = int(material.sum())
        report['raw_drop_gt_20pct_adjusted_drop_lt_5pct'] = int((matched
            & merged.raw_price_return.lt(-0.20) & merged.adjusted_price_return.gt(-0.05)).sum())
        merged.loc[event | merged.return_difference.abs().gt(0.01)].to_csv(
            out / 'price_discrepancies.csv', index=False)
        report['notes'] = [
            'CFACPR changes are adjustment events, not all necessarily stock splits.',
            'Returns are between available source rows within PERMNO; gaps need separate review.',
            'Adjusted price returns are not dividend-inclusive portfolio returns.',
            'RET/RETX exported unchanged; delisting returns are NOT included by this audit.',
            'Unmatched rows may include synthetic delisting rows or ambiguous identifiers.',
            'These counts do not measure how many labels change. Do not retrain from this export.',
            'Source may have been revised since cache creation; discrepancies require inspection.'
        ]
        if sha256(cache_path) != before:
            raise RuntimeError('Cache changed during audit, possibly by another process.')
        report['status'] = 'PASS_AUDIT_COMPLETED_NOT_DATA_VALIDATED'
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (out / 'price_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report: {out / "price_audit.json"}', flush=True)


if __name__ == '__main__':
    main()
