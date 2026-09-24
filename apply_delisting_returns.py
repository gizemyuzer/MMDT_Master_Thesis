"""Offline return ledger. Outcome columns MUST NOT enter model features.

Uses the legacy DSF RET (before delisting return), combines same-day DLRET.
Missing regular/delisting returns remain missing; no -30% or zero imputation.
Cash-equivalent settlement at the delisting date is an evaluation convention,
not a claim about actual payment timing. Labels/features are NOT built here.
"""
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def combine(prices, events):
    p, d = prices.copy(), events.copy()
    for frame in [p,d]:
        ids = pd.to_numeric(frame['permno'], errors='raise')
        if ids.isna().any() or (ids % 1 != 0).any():
            raise ValueError('Invalid PERMNO')
        frame['permno'] = ids.astype('int64')
    p['date'] = pd.to_datetime(p['date'], errors='raise')
    d['date'] = pd.to_datetime(d['dlstdt'], errors='raise')
    if p.date.isna().any() or d.date.isna().any():
        raise ValueError('Missing dates')
    if p.duplicated(['permno','date']).any():
        raise ValueError('Duplicate prices')
    code = pd.to_numeric(d['dlstcd'], errors='raise')
    if code.isna().any():
        raise ValueError('Missing exit code')
    active_count = int(code.eq(100).sum())
    d = d.loc[~code.eq(100), ['permno','date','dlstcd','dlret']].copy()
    if d.permno.duplicated().any():
        raise ValueError('Multiple exits per PERMNO require separate review')
    p = p.sort_values(['permno','date']).reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(p[['permno','date']])
    if not pd.MultiIndex.from_frame(d[['permno','date']]).isin(keys).all():
        raise ValueError('Exit without same-day price row: not supported by this source-specific step')
    last = p.groupby('permno')['date'].max()
    if not d['date'].eq(d.permno.map(last)).all():
        raise ValueError('Prices continue after an exit; inspect identity/timing')
    # Raw regular return is required, never reuse a previously combined return.
    r = pd.to_numeric(p['CRSP_RET'], errors='coerce')
    p['RegularReturn'] = r.where(np.isfinite(r) & r.ge(-1))
    d['DelistingReturn'] = pd.to_numeric(d['dlret'], errors='coerce')
    d['DelistingReturn'] = d['DelistingReturn'].where(
        np.isfinite(d.DelistingReturn) & d.DelistingReturn.ge(-1))
    p = p.merge(d[['permno','date','dlstcd','DelistingReturn']],
                on=['permno','date'], how='left', validate='one_to_one')
    p['ExitAfterReturn'] = p.dlstcd.notna()
    p['TotalReturn'] = p.RegularReturn
    exits = p.ExitAfterReturn
    p.loc[exits,'TotalReturn'] = ((1+p.loc[exits,'RegularReturn']) *
                                (1+p.loc[exits,'DelistingReturn']) - 1)
    p['UnresolvedExit'] = exits & p.TotalReturn.isna()
    report = dict(status='RETURN_LEDGER_BUILT_NOT_TRAINING_READY', rows=len(p),
        code100_records_ignored=active_count, exit_rows=int(exits.sum()),
        exits_combined=int((exits & p.TotalReturn.notna()).sum()),
        exits_missing_dlret=int((exits & p.DelistingReturn.isna()).sum()),
        exits_missing_regular_return=int((exits & p.RegularReturn.isna()).sum()),
        unresolved_exit_rows=int(p.UnresolvedExit.sum()),
        total_missing_return_rows=int(p.TotalReturn.isna().sum()),
        imputed_returns=0,
        notes=['TotalReturn=(1+RET)*(1+DLRET)-1 on exits; RET otherwise.',
               'No missing returns filled; held missing returns must block a backtest.',
               'Exit settlement at dlstdt is a cash-equivalent convention.',
               'Do not expose exit/outcome metadata to model input.',
               'Price-based labels still require separate terminal-outcome handling.',
               'Adjusted OHLC unchanged. No training or cache replacement.'])
    return p, report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',default='datasets/permno_sources_20260917T191937Z_f78c6a09/price_base_20260918T131725Z_a8743561')
    args=ap.parse_args()
    base=Path(args.base)
    files=[base/'adjusted_price_base.csv',base/'delisting_review.csv']
    hashes={f.name:sha(f) for f in files}
    p, report=combine(pd.read_csv(files[0]),pd.read_csv(files[1]))
    out=base/('return_ledger_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    cols=['permno','date','FormationTicker','Close','CRSP_RET','RegularReturn',
          'dlstcd','DelistingReturn','TotalReturn','ExitAfterReturn','UnresolvedExit']
    p[cols].to_csv(out/'return_ledger.csv',index=False)
    p.loc[p.ExitAfterReturn,cols].to_csv(out/'exit_outcomes.csv',index=False)
    p.loc[p.UnresolvedExit,cols].to_csv(out/'unresolved_exits.csv',index=False)
    if hashes != {f.name:sha(f) for f in files}:
        raise RuntimeError('Sources changed during run')
    report.update(source_hashes=hashes,output_folder=str(out))
    (out/'return_ledger_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('\nUnresolved exits:')
    print(p.loc[p.UnresolvedExit,cols].to_string(index=False))


if __name__=='__main__':
    main()
