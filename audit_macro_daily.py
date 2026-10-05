"""Audit the historical nine-column macro block without loading a large CSV at once.
Run from the project root: python audit_macro_daily.py
This script reads data only. It does not train or choose a macro specification.
"""
from pathlib import Path
import csv
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'datasets/final_dataset.csv'
MACRO=['VIX_Close','SPY_Trend_50','SPY_Trend_200','VIX_5d_Delta',
       'VIX_10d_Delta','SPY_Trend_Accel','Yield_Spread_10Y2Y',
       'Is_Yield_Inverted','Yield_Spread_Delta_20d']
if not SOURCE.is_file():raise SystemExit(f'Missing source: {SOURCE}')
with SOURCE.open(encoding='utf-8-sig',newline='') as f:
    header=next(csv.reader(f))
missing=[c for c in MACRO if c not in header]
if missing:raise SystemExit(f'Missing macro columns: {missing}; header: {header}')
date=next((c for c in ('date','Date','Unnamed: 0','') if c in header),None)
if date is None:raise SystemExit(f'Cannot identify source date column; header: {header}')
print('SOURCE',SOURCE,'DATE_COLUMN',repr(date),'MACRO_COLUMNS',MACRO,flush=True)
parts=[];invalid_dates=0;rows=0
for chunk in pd.read_csv(SOURCE,usecols=[date]+MACRO,chunksize=150000,low_memory=False):
    parsed=pd.to_datetime(chunk[date],errors='coerce')
    invalid_dates+=int(parsed.isna().sum());rows+=len(chunk)
    chunk[MACRO]=chunk[MACRO].apply(pd.to_numeric,errors='raise')
    chunk.insert(0,'_date',parsed)
    parts.append(chunk.loc[parsed.notna(),['_date']+MACRO].groupby('_date').agg(['min','max','count']))
if invalid_dates:raise SystemExit(f'Invalid dates: {invalid_dates}')
# Reduce chunk statistics per date, retaining extrema across chunk boundaries.
full=pd.concat(parts)
summary=full.groupby(level=0).agg(['min','max'])
report={'source':str(SOURCE),'rows':rows,'dates':len(summary),'first':str(summary.index.min().date()),
        'last':str(summary.index.max().date()),'features':{}}
for col in MACRO:
    lo=summary[(col,'min','min')];hi=summary[(col,'max','max')]
    spread=hi-lo
    inconsistent=spread.gt(1e-8).fillna(False)
    examples=[{'date':str(d.date()),'min':float(lo.loc[d]),'max':float(hi.loc[d])}
              for d in spread.index[inconsistent][:5]]
    report['features'][col]={'days_with_multiple_values':int(inconsistent.sum()),
                             'maximum_within_day_spread':float(spread.max()) if spread.notna().any() else None,
                             'examples':examples}
print(json.dumps(report,indent=2))
(ROOT/'macro_source_audit.json').write_text(json.dumps(report,indent=2))
print('REPORT',ROOT/'macro_source_audit.json')
