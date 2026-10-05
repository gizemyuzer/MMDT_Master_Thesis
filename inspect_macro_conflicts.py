"""Diagnose inconsistent date-level yield features; read only required CSV columns."""
from pathlib import Path
import csv
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'datasets/final_dataset.csv'
COLS=['Yield_Spread_10Y2Y','Is_Yield_Inverted','Yield_Spread_Delta_20d']
with SOURCE.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
date=next((c for c in ('date','Date','Unnamed: 0','') if c in header),None)
if date is None or any(c not in header for c in COLS):raise SystemExit(f'Invalid source schema: {header}')
chunks=[]
for chunk in pd.read_csv(SOURCE,usecols=[date]+COLS,chunksize=150000,low_memory=False):
    chunk['_date']=pd.to_datetime(chunk[date],errors='raise')
    for col in COLS:chunk[col]=pd.to_numeric(chunk[col],errors='raise')
    chunks.append(chunk[['_date']+COLS])
frame=pd.concat(chunks,ignore_index=True)
report={'source':str(SOURCE),'rows':len(frame),'features':{}}
for col in COLS:
    counts=frame.groupby(['_date',col],dropna=False).size().rename('n').reset_index()
    distinct=counts.groupby('_date').size()
    days=distinct.index[distinct.gt(1)]
    detail=[]
    for day in days:
        day_rows=counts[counts['_date'].eq(day)].sort_values('n',ascending=False)
        total=int(day_rows.n.sum())
        detail.append({'date':str(day.date()),'total':total,'modal_share':float(day_rows.n.iloc[0]/total),
                       'values':[{'value':None if pd.isna(v) else float(v),'rows':int(n)}
                                 for v,n in zip(day_rows[col],day_rows.n)]})
    report['features'][col]={'inconsistent_days':len(detail),'minimum_modal_share':min((x['modal_share'] for x in detail),default=1.),
                             'first_five':detail[:5],'last_five':detail[-5:]}
out=ROOT/'macro_conflicts_detail.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));print('REPORT',out)
