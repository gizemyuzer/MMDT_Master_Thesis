"""Full corrected I+II experiment. Keep run_corrected_smoke.py beside this file.
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
    p.add_argument('--fundamental',default='datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv')
    a=p.parse_args()
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
    out=root/'results'/('corrected_I_II_seed'+str(a.seed)+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rp=out/'experiment_report.json'
    report=dict(status='RUNNING',protocol='corrected_prices_permno_I_II_v1',seed=a.seed,output_folder=str(out),
                epochs_max=30,batch_size=64,monitor='pr_auc',lr=1e-4,weight_decay=.01,early_stop_patience=8,
                scheduler='ReduceLROnPlateau max/patience4/factor0.5',focal_alpha=1.,focal_gamma=2.,
                device=a.device,torch_version=torch.__version__,cuda_build=torch.version.cuda,test_evaluated=False,
                limitations=['Compustat RDQ alignment is not vintage point-in-time financial data.',
                             'I+II only; no text or macro. One seed does not establish a modality effect.'])
    def save():
        report['elapsed_minutes']=(time.perf_counter()-start)/60
        rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();print('FULL CORRECTED I+II | seed',a.seed,'| max 30 epochs',flush=True)
    print('Output folder:',out,flush=True)
    try:
        frame,values,state,pos,audit=adapter.load_corrected(a.technical,a.fundamental,max_permnos=None)
        pos['TestEndpoint']=adapter.window_positions(frame,values,'TestEndpoint')
        report.update(audit)
        state['fit_rule']='Full-panel TrainEndpoint rows only'
        joblib.dump(state,out/'preprocessing.joblib')
        ds={s:adapter.EndpointDataset(frame,values,pos[f]) for s,f in
            [('train','TrainEndpoint'),('val','ValidationEndpoint'),('test','TestEndpoint')]}
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
        config=dict(tech_dim=32,fund_dim=6,seq_len=20,d_model=64,n_heads=4,n_layers=2,ffn_dim=256,
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
        training_start=time.perf_counter()
        with isolated(out):
            result=train_pytorch_model(model,Progress(loaders['train'],'train'),Progress(loaders['val'],'validation'),
                test_loader=None,model_name=f'CORRECTED_I_II_seed{a.seed}',epochs=30,device=a.device,
                criterion=CheckedLoss(alpha=1.,gamma=2.),monitor='pr_auc',lr=1e-4,weight_decay=.01,
                early_stop_patience=8,use_warmup=False)
        report['training_and_validation_minutes']=(time.perf_counter()-training_start)/60
        threshold=float(result['threshold'])
        if not np.isfinite(threshold):raise ValueError('Invalid validation threshold')
        report['threshold']=threshold
        torch.save(dict(state_dict=result['model'].state_dict(),config=config,seed=a.seed,threshold=threshold,
                        technical_columns=adapter.TECH,fundamental_columns=adapter.FUND,
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
        report['status']='PASS'
    except Exception as e:
        report.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:save()
    print('PASS | corrected full I+II training, evaluation and indexed export',flush=True)
    print('Report:',rp,flush=True)

if __name__=='__main__':main()
