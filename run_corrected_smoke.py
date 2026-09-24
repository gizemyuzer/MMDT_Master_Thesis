"""Corrected I+II loader integration and 5+5 batch smoke test.
No WRDS, text join, full training, test evaluation, or historical output changes.
Loader functions are reusable by the subsequent full experiment runner.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

TECH = ['RSI','MACD_Hist_Rel','MACD_Hist_Slope_Rel','BB_Pct','BB_Width','Returns','Vol_20d','Vol_5d',
'Current_Drawdown_5d','Current_Drawdown_20d','Current_Return_5d','ADX_14','Price_vs_SMA50','Price_vs_SMA200',
'ROC_10','Dist_to_Max252d','Rel_Volume','Vol_Trend','Price_vs_Max20d','RSI_vs_Max20d','RSI_Divergence',
'Vol_Dist_Ratio','Rel_Returns','Rel_RSI','Rel_Vol_20d','Rel_MACD_Ratio','Rel_ROC_10','Rel_ATR_Ratio',
'CCI_14','MFI_14','WILLR_14','OBV_Z20']
FUND = ['Liabilities_to_Assets','Net_Profit_Margin_TTM','Current_Ratio','Altman_Z_TTM_Proxy',
'Retained_Earnings_TA','Market_Value_to_Liab_QuarterEnd']
FLAGS = ['TrainEndpoint','ValidationEndpoint','TestEndpoint']
META = ['date','permno','SessionID','SegmentID','Target','LabelEndDate']+FLAGS

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_bool(s):
    if s.dtype==bool:return s
    t=s.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if t.isna().any():raise ValueError('Invalid partition boolean')
    return t.astype(bool)

def check_allowlist(path, expected):
    d=pd.read_csv(path)
    if len(d.columns)!=1 or d.iloc[:,0].tolist()!=expected:
        raise ValueError(f'Unexpected feature allowlist: {path}')

def join_panel(technical,fundamental):
    if technical.duplicated(['date','permno']).any() or fundamental.duplicated(['date','permno']).any():
        raise ValueError('Duplicate date/PERMNO keys')
    if len(technical)!=len(fundamental):raise ValueError('Sidecar and panel row counts differ')
    merged=technical.merge(fundamental,on=['date','permno'],how='left',validate='one_to_one',indicator=True)
    if not merged._merge.eq('both').all():raise ValueError('Sidecar key mismatch')
    return merged.drop(columns='_merge').sort_values(['permno','date']).reset_index(drop=True)

def check_endpoints(frame):
    if frame[['date','permno','SessionID','SegmentID']].isna().any().any():raise ValueError('Missing sequence metadata')
    for c in FLAGS:frame[c]=read_bool(frame[c])
    if frame[FLAGS].sum(axis=1).gt(1).any():raise ValueError('Overlapping endpoint partitions')
    bounds=[('2010-01-01','2019-12-31'),('2020-01-01','2021-12-31'),('2022-01-01','2024-12-31')]
    for flag,(start,end) in zip(FLAGS,bounds):
        x=frame.loc[frame[flag]]
        if not x.date.between(start,end).all():raise ValueError('Endpoint outside split')
        if not x.Target.isin([0,1]).all():raise ValueError('Invalid endpoint target')
        if x.LabelEndDate.isna().any() or (x.LabelEndDate>pd.Timestamp(end)).any():
            raise ValueError('Label horizon crosses split boundary')
        if (x.LabelEndDate<x.date).any():raise ValueError('Label ends before decision')

def window_positions(frame, values, flag, length=20):
    # Whole panel remains present: no stitching across dropped rows or partitions.
    same=(frame.permno.eq(frame.permno.shift()) & frame.SegmentID.eq(frame.SegmentID.shift()) &
          frame.SessionID.sub(frame.SessionID.shift()).eq(1))
    run=(~same).cumsum()
    enough=frame.groupby(run).cumcount().ge(length-1).to_numpy()
    valid=np.isfinite(values).all(axis=1)
    finite_window=pd.Series(valid).rolling(length,min_periods=length).sum().eq(length).to_numpy()
    requested=frame[flag].to_numpy(dtype=bool)
    if (requested & ~(enough & finite_window)).any():
        raise ValueError(f'{flag}: declared endpoint has a gap or invalid input window')
    return np.flatnonzero(requested)

def preprocess(frame):
    from sklearn.preprocessing import RobustScaler
    mask=frame.TrainEndpoint.to_numpy(dtype=bool)
    if not mask.any():raise ValueError('No training endpoints')
    columns=TECH+FUND
    raw=frame[columns].replace([np.inf,-np.inf],np.nan)
    medians=raw.loc[mask,FUND].median()
    if medians.isna().any():raise ValueError('A fundamental feature is entirely missing in training')
    # Technical windows must be originally complete; only fundamentals are imputed.
    raw[FUND]=raw[FUND].fillna(medians)
    if not np.isfinite(raw.loc[mask,columns].to_numpy()).all():raise ValueError('Invalid scaler fitting rows')
    scaler=RobustScaler().fit(raw.loc[mask,columns])
    values=scaler.transform(raw[columns]).astype(np.float32)
    state={'scaler':scaler,'fundamental_medians':medians,'columns':columns,
           'fit_rule':'Only TrainEndpoint rows, selected smoke subset if applicable'}
    return values,state

class EndpointDataset:
    def __init__(self,frame,values,positions,length=20):
        self.values=values;self.positions=positions;self.length=length
        self.labels=frame.Target.to_numpy(dtype=np.float32)[positions]
        self.dates=pd.DatetimeIndex(frame.date.iloc[positions])
        self.permnos=frame.permno.to_numpy(dtype=np.int64)[positions]
    def __len__(self):return len(self.positions)
    def __getitem__(self,k):
        import torch
        end=self.positions[k];seq=self.values[end-self.length+1:end+1]
        return {'tech_seq':torch.from_numpy(seq[:,:len(TECH)]),
                'fund_seq':torch.from_numpy(seq[:,len(TECH):]),'label':torch.tensor(self.labels[k])}

def load_corrected(tech_path,fund_path,max_permnos=8):
    tech_path=Path(tech_path);fund_path=Path(fund_path)
    check_allowlist(tech_path.with_name('feature_columns.csv'),TECH)
    check_allowlist(fund_path.with_name('fund_feature_columns.csv'),FUND)
    tr=json.loads(tech_path.with_name('build_report.json').read_text(encoding='utf-8'))
    fr=json.loads(fund_path.with_name('fundamental_build_report.json').read_text(encoding='utf-8'))
    tech_hash=sha(tech_path)
    if fr['source_hashes']['technical']!=tech_hash:raise ValueError('Fundamentals were built against a different technical panel')
    t=pd.read_csv(tech_path,usecols=META+TECH,parse_dates=['date','LabelEndDate'])
    f=pd.read_csv(fund_path,usecols=['date','permno']+FUND,parse_dates=['date'])
    if len(t)!=tr['output_rows_including_2009_context'] or len(f)!=fr['daily_rows']:
        raise ValueError('Input truncated or row count differs from report')
    frame=join_panel(t,f);check_endpoints(frame)
    full_counts={c:int(frame[c].sum()) for c in FLAGS}
    for c,key in zip(FLAGS,['Train','Validation','Test']):
        if full_counts[c]!=tr['splits'][key]['retained']:raise ValueError('Technical report endpoint mismatch')
    if max_permnos is not None:
        counts=frame.groupby('permno')[FLAGS[:2]].sum()
        ids=counts.loc[(counts[FLAGS[:2]]>=320).all(axis=1)].index.sort_values()[:max_permnos]
        if len(ids)!=max_permnos:raise ValueError('Insufficient smoke subset')
        frame=frame.loc[frame.permno.isin(ids)].reset_index(drop=True)
    values,state=preprocess(frame)
    positions={flag:window_positions(frame,values,flag) for flag in FLAGS[:2]}
    audit={'source_sha256':{'technical':tech_hash,'fundamental':sha(fund_path)},'full_endpoint_counts':full_counts,
           'selected_permnos':frame.permno.drop_duplicates().astype(int).tolist(),'loaded_rows':len(frame),
           'sequence_counts':{k:len(v) for k,v in positions.items()},'technical_columns':TECH,'fundamental_columns':FUND}
    return frame,values,state,positions,audit

def self_test():
    # Context may cross split boundary but may not cross a session gap.
    dates=pd.bdate_range('2019-12-02',periods=50)
    f=pd.DataFrame({'date':dates,'permno':1,'SessionID':np.arange(50),'SegmentID':1,
                    'Target':0.,'LabelEndDate':dates})
    for flag in FLAGS:f[flag]=False
    for c in TECH+FUND:f[c]=np.arange(50,dtype=float)
    f.loc[19,'TrainEndpoint']=True;f.loc[30,'ValidationEndpoint']=True
    v,state=preprocess(f)
    assert window_positions(f,v,'ValidationEndpoint').tolist()==[30]
    f2=f.copy();f2.loc[30,FUND]=1e8
    _,state2=preprocess(f2)
    assert np.array_equal(state['scaler'].center_,state2['scaler'].center_)
    bad=f.copy();bad.loc[25:,'SessionID']+=1
    try:window_positions(bad,v,'ValidationEndpoint')
    except ValueError:pass
    else:raise AssertionError('Gap was accepted')
    bad=f.copy();bad.loc[19,'LabelEndDate']=pd.Timestamp('2020-01-03')
    try:check_endpoints(bad)
    except ValueError:pass
    else:raise AssertionError('Leaking endpoint was accepted')
    assert join_panel(f[META+TECH],f[['date','permno']+FUND].sample(frac=1,random_state=1))[FUND].equals(f[FUND])
    print('PASS | join alignment, context, session gaps, train-only scaling, purge boundary')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
    p.add_argument('--fundamental',default='datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args();self_test()
    if a.self_test_only:return
    import torch
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from models.transformer_model import DualEncoderTransformer
    from models.losses import FocalLoss
    import joblib
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; activate the H: CUDA environment or use --device cpu')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    out=Path('results')/('corrected_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();report={'status':'RUNNING','seed':42,'device':a.device,'test_evaluated':False}
    try:
        print('SMOKE ONLY | corrected I+II | 8 PERMNOs | 5 train + 5 validation batches',flush=True)
        frame,values,state,pos,audit=load_corrected(a.technical,a.fundamental)
        report.update(audit)
        print('PERMNOs:',audit['selected_permnos'],'| sequences:',audit['sequence_counts'],flush=True)
        td=EndpointDataset(frame,values,pos['TrainEndpoint']);vd=EndpointDataset(frame,values,pos['ValidationEndpoint'])
        counts=np.bincount(td.labels.astype(int),minlength=2)
        if (counts==0).any():raise ValueError('Both training classes are required')
        weights=1./counts[td.labels.astype(int)]
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(42))
        tl=DataLoader(td,batch_size=64,sampler=sampler,num_workers=0,drop_last=True)
        vl=DataLoader(vd,batch_size=64,shuffle=False,num_workers=0)
        config=dict(tech_dim=32,fund_dim=6,seq_len=20,d_model=64,n_heads=4,n_layers=2,
                    ffn_dim=256,dropout=.15,modality='multimodal',fusion_type='cross_attention')
        model=DualEncoderTransformer(**config).to(a.device)
        loss_fn=FocalLoss(alpha=1.,gamma=2.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        before=[q.detach().clone() for q in model.parameters() if q.requires_grad]
        losses={}
        for name,loader,training in [('train',tl,True),('validation',vl,False)]:
            model.train(training);ls=[]
            with torch.set_grad_enabled(training):
                for k,batch in enumerate(loader,1):
                    x=batch['tech_seq'].to(a.device);z=batch['fund_seq'].to(a.device);y=batch['label'].to(a.device)
                    assert x.shape[1:]==(20,32) and z.shape[1:]==(20,6)
                    assert torch.isfinite(x).all() and torch.isfinite(z).all()
                    if training:optimizer.zero_grad(set_to_none=True)
                    logits=model(x,z).reshape(-1)
                    if logits.shape!=y.shape:raise ValueError('Target/logit shape mismatch')
                    loss=loss_fn(logits,y)
                    if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                    if training:
                        loss.backward()
                        grads=[q.grad for q in model.parameters() if q.grad is not None]
                        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Invalid gradients')
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                    ls.append(float(loss.detach().cpu()))
                    print(f'{name} batch {k}/5 | loss={ls[-1]:.6f}',flush=True)
                    if k==5:break
            if len(ls)!=5:raise ValueError('Too few smoke batches')
            losses[name]=ls
        after=[q for q in model.parameters() if q.requires_grad]
        if not any(not torch.equal(x,y) for x,y in zip(before,after)):raise ValueError('No optimizer update')
        report.update(status='PASS',losses=losses,config=config,torch_version=torch.__version__,cuda_build=torch.version.cuda,
                      sampler='Balanced replacement; dedicated generator seed42',focal_alpha=1.,focal_gamma=2.,
                      limitations=['Smoke subset only; not a model-performance estimate.','Fundamentals use RDQ with next-session availability, not vintage point-in-time financial statements.'])
        joblib.dump(state,out/'smoke_preprocessing.joblib')
        report['code_sha256']={str(q):sha(q) for q in [Path(__file__),Path('models/transformer_model.py'),Path('models/losses.py')]}
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:
        report['elapsed_minutes']=(time.perf_counter()-started)/60
        (out/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS | corrected loaders, finite losses/gradients, optimizer update; test evaluation NOT RUN')
    print('Report:',out/'smoke_report.json')

if __name__=='__main__':main()
