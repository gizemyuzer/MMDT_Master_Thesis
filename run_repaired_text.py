"""Corrected I+II+IV runner with two documented CIK repairs; other links remain candidates. Default: smoke only.
Keep run_corrected_smoke.py beside this file. Full training remains provisional.
Uses existing models modules. New output directory; no old checkpoint resume.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from uuid import uuid4

@contextmanager
def isolated(path):
    old=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)

class Progress:
    def __init__(self,loader,name):self.loader,self.name=loader,name
    @property
    def dataset(self):return self.loader.dataset
    def __len__(self):return len(self.loader)
    def __iter__(self):
        start=time.perf_counter()
        for n,batch in enumerate(self.loader,1):
            if n==1 or n%500==0 or n==len(self):
                print(f'[{self.name}] batch {n}/{len(self)} | {(time.perf_counter()-start)/60:.1f} min',flush=True)
            yield batch

TEXT = ['EK_Bankruptcy','EK_DebtTrigger','EK_Impairment','EK_ListingWarn',
        'EK_AuditorChange','EK_Restatement','EK_MgmtChange','EK_Activity_60d','EK_DaysSince',
        'LM_Negative','LM_Positive','LM_Uncertainty','LM_Litigious','LM_Constraining',
        'LM_WordCount','TXT_SimPrev','TXT_LenChange','TXT_DaysSinceFiling','TXT_HasCoverage']
TEXT_META = ['TextLinkCandidate','TextCIK','TextSourceFilingDate',
             'TextSourceAvailableDate','TextEventHistorySessions']


def align_text(frame, text):
    import numpy as np
    import pandas as pd
    keys = ['date', 'permno']
    if text[keys].isna().any().any() or text.duplicated(keys).any():
        raise ValueError('Text has missing or duplicate date/PERMNO keys')
    if len(frame) != len(text):
        raise ValueError('Text must cover exactly the same full panel, including 2009 context')
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Base panel keys invalid')
    left = frame[keys].copy(); left['_order'] = np.arange(len(left))
    joined = left.merge(text[keys+TEXT+TEXT_META], on=keys, how='left',
                        validate='one_to_one', indicator=True).sort_values('_order')
    if not joined['_merge'].eq('both').all():
        raise ValueError('Text/base key mismatch; rows must not be silently dropped')
    return joined.drop(columns=['_order','_merge']).reset_index(drop=True)


def validate_text(frame, text):
    import numpy as np
    import pandas as pd
    import run_corrected_smoke as adapter
    if not text['TXT_HasCoverage'].isin([0.,1.]).all():
        raise ValueError('Text coverage must be observed binary 0/1')
    linked = adapter.read_bool(text.TextLinkCandidate)
    if (linked & text.TextCIK.isna()).any():
        raise ValueError('Linked row without CIK')
    raw = text[TEXT].apply(pd.to_numeric, errors='raise')
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError('Infinite text value')
    language = TEXT[9:18]
    has_language = raw[language].notna().any(axis=1)
    filing = pd.to_datetime(text.TextSourceFilingDate, errors='raise')
    available = pd.to_datetime(text.TextSourceAvailableDate, errors='raise')
    used = filing.notna() | available.notna() | has_language
    bad = used & (filing.isna() | available.isna() | (filing >= text.date) |
                  (available > text.date) | (available <= filing))
    if bad.any():
        raise ValueError('Language filing not strictly prior/available by decision date')
    age = (text.date-filing).dt.days
    if (used & ((age > 365) | (age < 1))).any():
        raise ValueError('Stale or future language source')
    if (used & raw.TXT_DaysSinceFiling.ne(age)).any():
        raise ValueError('Filing age does not match source date')
    if ((raw[TEXT[:-1]].notna().any(axis=1) | raw.TXT_HasCoverage.eq(1)) & ~linked).any():
        raise ValueError('Text feature without candidate link')
    for c in TEXT[9:14]+['TXT_SimPrev']:
        v=raw[c].dropna()
        if ((v < -1e-8) | (v > 1+1e-8)).any():raise ValueError('Invalid ratio: '+c)
    for c in TEXT[:8]:
        v=raw[c].dropna()
        if (v.lt(0) | (v-v.round()).abs().gt(1e-8)).any():raise ValueError('Invalid event count: '+c)
    for c in ['EK_DaysSince','TXT_DaysSinceFiling','LM_WordCount']:
        if raw[c].dropna().lt(0).any():raise ValueError('Negative text count/age: '+c)
    # Counts are meaningful only after the builder has observed a full window.
    history=pd.to_numeric(text.TextEventHistorySessions,errors='raise')
    for c in TEXT[:8]:
        width=60 if c=='EK_Activity_60d' else (126 if c=='EK_MgmtChange' else 252)
        if (raw[c].notna() & (history.isna() | history.lt(width))).any():
            raise ValueError('Incomplete event history treated as observed count: '+c)
    return raw


def fit_text(raw, train_mask):
    import numpy as np
    from sklearn.preprocessing import RobustScaler
    medians=raw.loc[train_mask,TEXT].median()
    if medians.isna().any():
        raise ValueError('Text columns entirely missing in training: '+str(medians.index[medians.isna()].tolist()))
    filled=raw[TEXT].fillna(medians)
    scaler=RobustScaler().fit(filled.loc[train_mask,TEXT])
    values=scaler.transform(filled[TEXT]).astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Non-finite scaled text')
    return values, {'columns':TEXT,'medians':medians,'scaler':scaler,
                    'fit_rule':'Full-panel TrainEndpoint rows only; no test/validation fit'}


def attach_text(frame, values, text_path, tech_hash):
    import numpy as np
    import pandas as pd
    import run_corrected_smoke as adapter
    path=Path(text_path)
    report_path=path.with_name('text_daily_report.json')
    tr=json.loads(report_path.read_text(encoding='utf-8-sig'))
    if tr['status']!='TARGETED_REPAIRS_BUILT_REVIEW_REQUIRED':
        raise ValueError('This runner requires the two-repair text panel')
    expected='618f5c70583bbf71a652cc57d663bb0d3f63774131308a4394b4500d11094286'
    if adapter.sha(path)!=expected:
        raise ValueError('Text panel differs from the frozen two-repair version')
    rr=json.loads(path.with_name('identity_repair_report.json').read_text(encoding='utf-8-sig'))
    if rr.get('status')!='TARGETED_REPAIRS_BUILT_AND_COMPARED' or rr.get('new_text_sha256')!=expected:
        raise ValueError('Missing or inconsistent identity repair report')
    expected_repairs={(11644,23002,'1987-08-28','2018-08-13',23194),
                      (88992,140033,'2010-12-06','2012-12-31',1126294)}
    actual={(int(r['permno']),int(r['gvkey']),r['start'],r['end'],int(r['new_cik'])) for r in rr['repairs']}
    if actual!=expected_repairs or len(rr['repairs'])!=2:
        raise ValueError('Unexpected identity repairs')
    if adapter.sha(path.with_name('dated_links_repaired.csv'))!=tr['source_hashes']['links']:
        raise ValueError('Repaired links do not match text build report')
    if tr['source_hashes']['technical']!=tech_hash:
        raise ValueError('Text built against another technical panel')
    adapter.check_allowlist(path.with_name('text_feature_columns.csv'),TEXT)
    text=pd.read_csv(path,usecols=['date','permno']+TEXT+TEXT_META,
                     parse_dates=['date','TextSourceFilingDate','TextSourceAvailableDate'])
    if len(text)!=tr['rows'] or text.permno.nunique()!=tr['permnos']:
        raise ValueError('Text row/security count differs from build report')
    text=align_text(frame,text)
    raw=validate_text(frame,text)
    coverage={}
    for flag in adapter.FLAGS:
        mask=frame[flag].to_numpy(dtype=bool)
        counts={'endpoints':int(mask.sum()),
                'recent_filing_coverage':int(raw.loc[mask,'TXT_HasCoverage'].eq(1).sum()),
                'language_available':int(raw.loc[mask,'LM_Negative'].notna().sum()),
                'all_event_counts_available':int(raw.loc[mask,TEXT[:7]].notna().all(axis=1).sum())}
        if counts!=tr['splits'][flag]:raise ValueError('Text coverage differs from build report: '+flag)
        coverage[flag]=dict(counts,missing_rates=raw.loc[mask].isna().mean().to_dict())
    tv,ts=fit_text(raw,frame.TrainEndpoint.to_numpy(dtype=bool))
    combined=np.concatenate([values,tv],axis=1)
    audit={'text_columns':TEXT,'auxiliary_columns':adapter.FUND+TEXT,
           'text_sha256':adapter.sha(path),'text_report_sha256':adapter.sha(report_path),
           'text_build_report':tr,'text_coverage':coverage,
           'text_linkage_status':'TWO_REPAIRS_OTHER_LINKS_NOT_HISTORICALLY_VERIFIED',
           'final_thesis_ready':False,
           'stream_layout':'32 technical | 6 fundamental + 19 text in second encoder; no third encoder',
           'text_controls':'key/date/history checks do not verify historical issuer identity or archive completeness'}
    return combined,ts,audit


def self_test():
    import numpy as np
    import pandas as pd
    dates=pd.bdate_range('2020-01-01',periods=6)
    base=pd.DataFrame({'date':dates,'permno':1})
    t=base.copy()
    for c in TEXT:t[c]=np.arange(6,dtype=float)
    t['TXT_HasCoverage']=1.
    for c in TEXT_META:t[c]=np.nan
    shuffled=align_text(base,t.sample(frac=1,random_state=42))
    assert shuffled[TEXT].equals(t[TEXT])
    for bad in [t.iloc[:-1],pd.concat([t.iloc[:-1],t.iloc[[0]]]),t.assign(permno=2)]:
        try:align_text(base,bad)
        except ValueError:pass
        else:raise AssertionError('Bad text keys accepted')
    mask=np.array([True,True,True,False,False,False])
    raw=t[TEXT].copy();raw.loc[1,'LM_Negative']=np.nan
    v,s=fit_text(raw,mask)
    changed=raw.copy();changed.loc[~mask,TEXT]=1e9
    _,s2=fit_text(changed,mask)
    assert s['medians'].equals(s2['medians'])
    assert np.array_equal(s['scaler'].center_,s2['scaler'].center_)
    assert np.array_equal(s['scaler'].scale_,s2['scaler'].scale_)
    valid=t.copy()
    for c in TEXT:valid[c]=0.
    valid['TXT_HasCoverage']=1.;valid['TextLinkCandidate']=True;valid['TextCIK']=123
    valid['TextSourceFilingDate']=dates-pd.Timedelta(days=1)
    valid['TextSourceAvailableDate']=dates
    valid['TXT_DaysSinceFiling']=1.;valid['TextEventHistorySessions']=252.
    validate_text(base,valid)
    for col,value in [('TextSourceFilingDate',dates[0]),('TextSourceAvailableDate',dates[1]),
                      ('TextEventHistorySessions',10.)]:
        bad=valid.copy();bad.loc[0,col]=value
        try:validate_text(base,bad)
        except ValueError:pass
        else:raise AssertionError('Future filing/incomplete event history accepted')
    print('PASS | text key alignment, duplicate/missing keys, train-only fit, availability and event history')


def smoke(model, loaders, device, loss_fn):
    import torch
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    model.to(device)
    before=[p.detach().clone() for p in model.parameters() if p.requires_grad]
    results={};text_gradient_seen=False
    for split,training in [('train',True),('val',False)]:
        model.train(training);losses=[]
        with torch.set_grad_enabled(training):
            for batch in loaders[split]:
                x=batch['tech_seq'].to(device);z=batch['fund_seq'].to(device);y=batch['label'].to(device)
                if x.shape[1:]!=(20,32) or z.shape[1:]!=(20,25):raise ValueError('Stream shape mismatch')
                if not torch.isfinite(x).all() or not torch.isfinite(z).all():raise ValueError('Invalid inputs')
                if training:optimizer.zero_grad(set_to_none=True)
                logits=model(x,z).reshape(-1)
                if logits.shape!=y.shape:raise ValueError('Output/target shape mismatch')
                loss=loss_fn(logits,y)
                if not torch.isfinite(loss):raise ValueError('Non-finite smoke loss')
                if training:
                    loss.backward()
                    grads=[p.grad for p in model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Non-finite gradients')
                    projection=model.fund_encoder.proj.weight
                    text_gradient_seen = text_gradient_seen or (
                        projection.grad is not None and bool(torch.count_nonzero(projection.grad[:,6:]).item()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
                print(f'{split} batch {len(losses)}/5 | loss={losses[-1]:.6f}',flush=True)
                if len(losses)==5:break
        if len(losses)!=5:raise ValueError('Too few smoke batches')
        results[split]=losses
    after=[p for p in model.parameters() if p.requires_grad]
    if not any(not torch.equal(a,b) for a,b in zip(before,after)):raise ValueError('No optimizer update')
    if not text_gradient_seen:raise ValueError('No gradient reached the text projection in smoke batches')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
    p.add_argument('--fundamental',default='datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv')
    p.add_argument('--text',default='datasets/text_identity_repair_20260923T172930Z_fe9420be/text_daily.csv')
    p.add_argument('--mode',choices=['smoke','train'],default='smoke')
    p.add_argument('--allow-candidate-linkage',action='store_true',
                   help='Run provisional full training despite unverified historical CIK linkage')
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    self_test()
    if a.self_test_only:return
    if a.mode=='train' and not a.allow_candidate_linkage:
        p.error('Historical CIK linkage is unverified. Use smoke first; provisional training requires --allow-candidate-linkage.')
    if not 0<=a.seed<2**32:p.error('Invalid seed')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,WeightedRandomSampler
    import matplotlib
    matplotlib.use('Agg')
    import joblib
    import run_corrected_smoke as adapter
    from models.transformer_model import DualEncoderTransformer
    from models.losses import FocalLoss
    from models.pytorch_trainer import train_pytorch_model,_evaluate_on_loader,_compute_classification_metrics
    adapter.self_test()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    root=Path.cwd().resolve()
    out=root/'results'/(('corrected_I_II_IV_'+a.mode+'_two_repairs_seed')+str(a.seed)+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rp=out/'experiment_report.json'
    report=dict(status='RUNNING',protocol='corrected_prices_permno_I_II_IV_two_repairs_v1',mode=a.mode,final_thesis_ready=False,seed=a.seed,output_folder=str(out),
                epochs_max=30,batch_size=64,monitor='pr_auc',lr=1e-4,weight_decay=.01,early_stop_patience=8,
                scheduler='ReduceLROnPlateau max/patience4/factor0.5',focal_alpha=1.,focal_gamma=2.,
                device=a.device,torch_version=torch.__version__,cuda_build=torch.version.cuda,test_evaluated=False,
                limitations=['Compustat RDQ alignment is not vintage point-in-time financial data.',
                             'Historical CIK links are candidate mappings, not verified issuer identity. Results are provisional.',
                             'Text coverage and content are combined; no content-only causal interpretation.',
                             'Compustat and text imputation/scaling use training data only. No macro inputs.'])
    def save():
        report['elapsed_minutes']=(time.perf_counter()-start)/60
        rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();print('CORRECTED I+II+IV |',a.mode,'| CANDIDATE LINKAGE | seed',a.seed,flush=True)
    print('Output folder:',out,flush=True)
    try:
        frame,values,state,pos,audit=adapter.load_corrected(a.technical,a.fundamental,max_permnos=None)
        expected={'technical':'b7a9de11255319edebcb0c1042ca4cd60da2c66ab3fe906beaca3549eea45287',
                  'fundamental':'f7763fde9f81758ccf3c3b0df3265d5b6a782a809defb1d135ffe2ce55ebe731'}
        if audit['source_sha256']!=expected:raise ValueError('I+II data source mismatch')
        values,text_state,text_audit=attach_text(frame,values,a.text,audit['source_sha256']['technical'])
        state['text_preprocessing']=text_state
        state['model_columns']=adapter.TECH+adapter.FUND+TEXT
        state['stream_split_index']=len(adapter.TECH)
        report.update(text_audit)
        audit['source_sha256']['text']=text_audit['text_sha256']
        if a.mode=='smoke':
            counts=frame.groupby('permno')[adapter.FLAGS[:2]].sum()
            ids=counts.loc[(counts>=320).all(axis=1)].index.sort_values()[:8]
            if len(ids)!=8:raise ValueError('Insufficient smoke securities')
            keep=frame.permno.isin(ids).to_numpy()
            frame=frame.loc[keep].reset_index(drop=True);values=values[keep]
            audit['selected_permnos']=[int(x) for x in ids];audit['loaded_rows']=len(frame)
        flags=adapter.FLAGS if a.mode=='train' else adapter.FLAGS[:2]
        pos={flag:adapter.window_positions(frame,values,flag) for flag in flags}
        audit['sequence_counts']={flag:len(v) for flag,v in pos.items()}
        report.update(audit)
        state['fit_rule']='Full-panel TrainEndpoint rows only'
        joblib.dump(state,out/'preprocessing.joblib')
        ds={s:adapter.EndpointDataset(frame,values,pos[f]) for s,f in
            ([('train','TrainEndpoint'),('val','ValidationEndpoint')]+([('test','TestEndpoint')] if a.mode=='train' else []))}
        report['splits']={s:dict(sequences=len(d),date_min=str(d.dates.min().date()),date_max=str(d.dates.max().date())) for s,d in ds.items()}
        print('Sequences:',{s:len(d) for s,d in ds.items()},flush=True)
        counts=np.bincount(ds['train'].labels.astype(int),minlength=2)
        if (counts==0).any():raise ValueError('Both training classes required')
        weights=1./counts[ds['train'].labels.astype(int)]
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(a.seed))
        loaders={s:DataLoader(d,batch_size=64,sampler=sampler if s=='train' else None,
                             shuffle=False,num_workers=0,drop_last=False,
                             generator=torch.Generator().manual_seed(a.seed)) for s,d in ds.items()}
        report['sampling']='Balanced replacement; independent generator seeded per run; drop_last=False'
        config=dict(tech_dim=32,fund_dim=25,seq_len=20,d_model=64,n_heads=4,n_layers=2,ffn_dim=256,
                    dropout=.15,modality='multimodal',fusion_type='cross_attention')
        report['config']=config
        sources=[Path(__file__).resolve(),Path(adapter.__file__).resolve(),root/'models/transformer_model.py',root/'models/losses.py',root/'models/pytorch_trainer.py']
        report['code_sha256']={str(x):adapter.sha(x) for x in sources};save()
        class CheckedLoss(FocalLoss):
            def forward(self,logits,targets):
                loss=super().forward(logits,targets)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                return loss
        model=DualEncoderTransformer(**config)
        report['parameter_count']=sum(x.numel() for x in model.parameters())
        if a.mode=='smoke':
            report['losses']=smoke(model,loaders,a.device,CheckedLoss(alpha=1.,gamma=2.))
            report['status']='SMOKE_PASS';save()
            print('PASS | I+II+IV smoke, finite gradients, text projection active; test NOT RUN',flush=True)
            print('Report:',rp,flush=True)
            return
        training_start=time.perf_counter()
        with isolated(out):
            result=train_pytorch_model(model,Progress(loaders['train'],'train'),Progress(loaders['val'],'validation'),
                test_loader=None,model_name=f'CORRECTED_I_II_IV_CANDIDATE_seed{a.seed}',epochs=30,device=a.device,
                criterion=CheckedLoss(alpha=1.,gamma=2.),monitor='pr_auc',lr=1e-4,weight_decay=.01,
                early_stop_patience=8,use_warmup=False)
        report['training_and_validation_minutes']=(time.perf_counter()-training_start)/60
        threshold=float(result['threshold'])
        if not np.isfinite(threshold):raise ValueError('Invalid validation threshold')
        report['threshold']=threshold
        torch.save(dict(state_dict=result['model'].state_dict(),config=config,seed=a.seed,threshold=threshold,
                        technical_columns=adapter.TECH,fundamental_columns=adapter.FUND,text_columns=TEXT,auxiliary_columns=adapter.FUND+TEXT,
                        text_linkage_status='TWO_REPAIRS_OTHER_LINKS_NOT_HISTORICALLY_VERIFIED',
                        protocol=report['protocol'],source_sha256=audit['source_sha256']),out/'selected_model.pt')
        report['status']='TRAINED_EXPORTING';save()
        # Test was never supplied to the trainer. Evaluate it once after selection.
        for split in ['val','test']:
            preds,targets,_=_evaluate_on_loader(result['model'],Progress(loaders[split],split+'_export'),a.device)
            d=ds[split]
            if len(preds)!=len(d) or not np.array_equal(targets,d.labels):raise ValueError('Export alignment failure')
            if not np.isfinite(preds).all() or ((preds<0)|(preds>1)).any():raise ValueError('Invalid probabilities')
            np.savez_compressed(out/f'{split}_predictions.npz',dates=d.dates.to_numpy(dtype='datetime64[ns]'),
                                permnos=d.permnos,y_true=targets,y_prob=preds,threshold=np.asarray(threshold))
            metrics={k:float(v) for k,v in _compute_classification_metrics(preds,targets,threshold).items()}
            if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Invalid metric')
            report[split+'_metrics']=metrics
            if split=='test':report['test_evaluated']=True
            save();print(split.upper(),report['splits'][split],json.dumps(metrics,indent=2),flush=True)
        report['status']='PROVISIONAL_PASS'
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:save()
    print('PROVISIONAL_PASS | I+II+IV training/export complete; historical CIK linkage still unverified',flush=True)
    print('Report:',rp,flush=True)

if __name__=='__main__':main()
