"""Corrected XGBoost comparison. Default: smoke only, no test scoring.
Run beside run_corrected_smoke.py and run_repaired_text.py in the project root.
Uses identical 20-session feature histories and endpoint keys as Transformers.
"""
import argparse, gc, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, matthews_corrcoef,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

PROTOCOL='corrected_xgb_flat20_validation_search_v1'
HASHES={'technical':'b7a9de11255319edebcb0c1042ca4cd60da2c66ab3fe906beaca3549eea45287',
        'fundamental':'f7763fde9f81758ccf3c3b0df3265d5b6a782a809defb1d135ffe2ce55ebe731',
        'text':'618f5c70583bbf71a652cc57d663bb0d3f63774131308a4394b4500d11094286'}
REF_PROTOCOLS={'I':'corrected_prices_permno_I_v1','I+II':'corrected_prices_permno_I_II_v1',
               'I+II+IV':'corrected_prices_permno_I_II_IV_two_repairs_v1'}
WIDTHS={'I':32,'I+II':38,'I+II+IV':57}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def negative_average_precision(y,p):
 return -float(average_precision_score(y,p))

def threshold_mcc(y,p):
 """Same 300-point percentile grid and tie rule as the Transformer trainer."""
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 if np.unique(y).size<2:return .5,0.
 lo,hi=np.percentile(p,[.5,99.5])
 if not np.isfinite([lo,hi]).all() or hi<=lo:lo,hi=float(p.min()),float(p.max())
 if hi<=lo:return .5,0.
 best=-2.;threshold=.5;valid=0
 for t in np.linspace(lo,hi,300):
  pred=p>=t
  if pred.sum() in (0,len(pred)):continue
  valid+=1;m=matthews_corrcoef(y,pred)
  if m>best:best=float(m);threshold=float(t)
 return (threshold,best) if valid else (float(np.median(p)),0.)

def metrics(y,p,t):
 if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
 pred=p>=t
 return dict(roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
  mcc=float(matthews_corrcoef(y,pred)),accuracy=float(accuracy_score(y,pred)),
  precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
  f1=float(f1_score(y,pred,zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def make_model(params,seed,device,jobs,rounds=1200,patience=50):
 import xgboost as xgb
 # No duplicate keyword arguments. Minimize negative AP = maximize sklearn AP.
 cfg=dict(objective='binary:logistic',tree_method='hist',device=device,n_jobs=jobs,
          random_state=seed,n_estimators=rounds,max_bin=256,
          eval_metric=negative_average_precision,early_stopping_rounds=patience)
 cfg.update(params)
 return xgb.XGBClassifier(**cfg)

def flatten(values,positions,width,path):
 """Lag-major order: t-19 features ... t features. Float32 disk-backed matrix."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 ans=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(positions),20*width))
 for k in range(0,len(positions),2048):
  ids=positions[k:k+2048,None]+np.arange(-19,1)[None,:]
  block=values[ids,:width].reshape(len(ids),20*width)
  if not np.isfinite(block).all():raise ValueError('Nonfinite history in matrix')
  ans[k:k+len(ids)]=block
 ans.flush();return ans

def close_matrix(x,path):
 if hasattr(x,'_mmap'):x._mmap.close()
 Path(path).unlink(missing_ok=True)

def find_references(root,configs):
 refs={}
 for path in Path(root).rglob('experiment_report.json'):
  try:r=json.loads(path.read_text(encoding='utf-8-sig'))
  except (ValueError,OSError):continue
  for config in configs:
   wanted={k:v for k,v in HASHES.items() if k!='text' or config=='I+II+IV'}
   if (r.get('protocol')==REF_PROTOCOLS[config] and r.get('seed')==42 and
       r.get('status') in ['PASS','PROVISIONAL_PASS'] and r.get('source_sha256')==wanted):
    if all(path.with_name(s+'_predictions.npz').exists() for s in ['val','test']):
     refs.setdefault(config,path)
 missing=set(configs)-set(refs)
 if missing:raise FileNotFoundError('Completed seed42 Transformer reports/NPZ missing for '+str(sorted(missing))+'. Set --reference-root to their parent folder.')
 return refs

def verify_keys(frame,pos,refs):
 audit={}
 for config,path in refs.items():
  audit[config]={'report':str(path.resolve()),'report_sha256':sha(path),'predictions':{}}
  for split,flag in [('val','ValidationEndpoint'),('test','TestEndpoint')]:
   ix=pos[flag];file=path.with_name(split+'_predictions.npz')
   with np.load(file,allow_pickle=False) as z:
    for key,expected in [('dates',frame.date.to_numpy(dtype='datetime64[ns]')[ix]),
                         ('permnos',frame.permno.to_numpy(dtype=np.int64)[ix]),
                         ('y_true',frame.Target.to_numpy()[ix])]:
     actual=z[key].astype('datetime64[ns]') if key=='dates' else z[key]
     if not np.array_equal(actual,expected):raise ValueError(f'Endpoint alignment failed: {config}/{split}/{key}')
   audit[config]['predictions'][split]=sha(file)
 return audit

def data(args):
 import run_corrected_smoke as adapter
 frame,values,state,_,audit=adapter.load_corrected(args.technical,args.fundamental,max_permnos=None)
 if audit['source_sha256']!={k:HASHES[k] for k in ['technical','fundamental']}:raise ValueError('Frozen technical/fundamental source mismatch')
 columns=adapter.TECH+adapter.FUND
 if 'I+II+IV' in args.configs:
  from run_repaired_text import attach_text,TEXT
  values,ts,ta=attach_text(frame,values,args.text,HASHES['technical'])
  state['text_preprocessing']=ts;columns+=TEXT;audit['source_sha256']['text']=ta['text_sha256']
 pos={f:adapter.window_positions(frame,values,f) for f in adapter.FLAGS}
 expected={'TrainEndpoint':756423,'ValidationEndpoint':109943,'TestEndpoint':150436}
 if {f:len(v) for f,v in pos.items()}!=expected:raise ValueError('Unexpected corrected endpoint counts')
 state['fit_rule']='Full-panel TrainEndpoint rows only'
 return frame,values,state,pos,audit,columns

def sample_parameters(trial):
 return dict(max_depth=trial.suggest_int('max_depth',2,6),learning_rate=trial.suggest_float('learning_rate',.01,.1,log=True),
   subsample=trial.suggest_float('subsample',.5,1.),colsample_bytree=trial.suggest_float('colsample_bytree',.5,1.),
   min_child_weight=trial.suggest_float('min_child_weight',1.,20.,log=True),gamma=trial.suggest_float('gamma',0.,2.),
   reg_alpha=trial.suggest_float('reg_alpha',1e-4,10.,log=True),reg_lambda=trial.suggest_float('reg_lambda',.1,20.,log=True))

def self_test():
 import tempfile
 # Independent reference loop for tied scores and one-class predictions.
 rng=np.random.default_rng(7);y=np.tile([0,1],40);p=np.round(rng.random(80),1)
 t,m=threshold_mcc(y,p);assert np.isclose(m,matthews_corrcoef(y,p>=t))
 assert threshold_mcc(y,np.ones(80))==(.5,0.)
 with tempfile.TemporaryDirectory() as d:
  v=np.arange(70*3,dtype=np.float32).reshape(70,3);ix=np.array([19,20,69]);path=Path(d)/'x.npy'
  x=flatten(v,ix,3,path)
  assert np.array_equal(x,np.stack([v[i-19:i+1].flatten() for i in ix]));close_matrix(x,path)
 # No data fetchers/torch/model checkpoints imported.
 print('PASS | feature history order and threshold diagnostics',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['smoke','run','resume'],default='smoke')
 p.add_argument('--configs',nargs='+',choices=list(WIDTHS),default=list(WIDTHS))
 p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,53)))
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--jobs',type=int,default=8)
 p.add_argument('--trials',type=int,default=30);p.add_argument('--search-seed',type=int,default=20260924)
 p.add_argument('--rounds',type=int,default=1200)
 p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
 p.add_argument('--fundamental',default='datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv')
 p.add_argument('--text',default='datasets/text_identity_repair_20260923T172930Z_fe9420be/text_daily.csv')
 p.add_argument('--reference-root',default='results');p.add_argument('--resume-dir')
 p.add_argument('--self-test-only',action='store_true')
 a=p.parse_args();self_test()
 if a.self_test_only:return
 if len(set(a.configs))!=len(a.configs) or len(set(a.seeds))!=len(a.seeds):p.error('Duplicate configs/seeds')
 if any(s<0 or s>=2**32 for s in a.seeds) or a.jobs<1 or a.trials<1 or a.rounds<2:p.error('Invalid positive argument')
 if (a.mode=='resume')!=bool(a.resume_dir):p.error('Use --mode resume together with --resume-dir')
 import xgboost as xgb,joblib
 if int(xgb.__version__.split('.')[0])<2:raise RuntimeError('Requires XGBoost >=2.0; tested on 3.2.0')
 if a.mode=='run':
  import optuna
 refs=find_references(a.reference_root,a.configs)
 print('Reading frozen panels; validating full sequence endpoints...',flush=True)
 frame,values,state,pos,audit,columns=data(a)
 refaudit=verify_keys(frame,pos,refs)
 print('PASS | exact Transformer validation/test date-PERMNO-label alignment (no test scoring)',flush=True)
 codefiles=[Path(__file__),Path('run_corrected_smoke.py')]
 if 'I+II+IV' in a.configs:codefiles.append(Path('run_repaired_text.py'))
 signature=dict(protocol=PROTOCOL,configs=a.configs,seeds=a.seeds,source_sha256=audit['source_sha256'],
  code_sha256={f.name:sha(f) for f in codefiles},xgboost_version=xgb.__version__,
  numpy_version=np.__version__,pandas_version=pd.__version__,
  sklearn_version=__import__('sklearn').__version__,
  trials=a.trials,search_seed=a.search_seed,rounds=a.rounds,device=a.device,jobs=a.jobs,
  representation='20 sessions flattened oldest-to-newest; each lag uses explicit feature allowlist',
  preprocessing='TrainEndpoint median/scaler identical to Transformer; no numeric-column auto-selection',
  monitor='sklearn average precision via negative_average_precision minimized',
  threshold='Transformer 300-point percentile-grid MCC selection on validation',
  balancing='scale_pos_weight=negative/positive TRAIN count; no resampling',
  tuning='TPE on fixed validation 2020-2021; same seed42 and budget per config; not historical CV',
  limitations=['Post-Transformer-test model-family extension; test not used in XGBoost tuning.',
   'Hyperparameter budgets/losses differ between model families; not equal-compute comparison.',
   'CIK bridge partly unverified; no content-only interpretation.',
   'Training seeds are not independent market histories.'])
 if a.mode=='resume':
  out=Path(a.resume_dir).resolve();previous=json.loads((out/'protocol.json').read_text())
  if previous!=signature:raise ValueError('Resume protocol/source/version/code mismatch; use original settings/files')
  if not (out/'ALL_PARAMETERS_FROZEN.json').exists():raise ValueError('Only resume completed tuning; incomplete tuning has not frozen all configs')
 else:
  out=(Path('results')/('corrected_xgb_'+a.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])).resolve()
  out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',signature)
 print('OUTPUT:',out,flush=True);write(out/'alignment_report.json',refaudit);joblib.dump(state,out/'preprocessing.joblib')
 target=frame.Target.to_numpy();dates=frame.date.to_numpy(dtype='datetime64[ns]');permnos=frame.permno.to_numpy(dtype=np.int64)
 train=pos['TrainEndpoint'];val=pos['ValidationEndpoint'];test=pos['TestEndpoint']
 ytrain=target[train].astype(int);yval=target[val].astype(int)
 spw=float((ytrain==0).sum()/(ytrain==1).sum())
 if a.mode=='smoke':
  rng=np.random.default_rng(42);train=np.sort(rng.choice(train,min(6000,len(train)),replace=False));val=np.sort(rng.choice(val,min(3000,len(val)),replace=False))
  ytrain=target[train].astype(int);yval=target[val].astype(int)
  smoke={}
  for config in a.configs:
   width=WIDTHS[config];trp=out/'train_matrix.npy';vap=out/'val_matrix.npy'
   xtr=flatten(values,train,width,trp);xv=flatten(values,val,width,vap)
   model=make_model(dict(max_depth=2,learning_rate=.1,subsample=.8,colsample_bytree=.8,scale_pos_weight=spw),42,a.device,a.jobs,rounds=12,patience=3)
   model.fit(xtr,ytrain,eval_set=[(xv,yval)],verbose=False);prob=model.predict_proba(xv)[:,1]
   threshold,_=threshold_mcc(yval,prob);metrics(yval,prob,threshold)
   file=out/(config.replace('+','_')+'_smoke.json');model.save_model(file)
   loaded=xgb.XGBClassifier();loaded.load_model(file)
   if not np.allclose(prob,loaded.predict_proba(xv)[:,1],atol=1e-7):raise ValueError('Reload mismatch')
   actual_device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
   if a.device=='cuda' and not actual_device.startswith('cuda'):raise RuntimeError('CUDA requested but XGBoost fell back to CPU')
   smoke[config]=dict(features=width*20,train=len(train),validation=len(val),best_iteration=int(model.best_iteration),actual_device=actual_device)
   print('PASS |',config,'| smoke fit, finite probabilities, save/reload',flush=True)
   del model,loaded;close_matrix(xtr,trp);close_matrix(xv,vap);gc.collect()
  write(out/'smoke_report.json',dict(status='PASS',test_evaluated=False,configs=smoke,source_sha256=audit['source_sha256'],alignment='exact full val/test keys verified; smoke trained on subset'))
  print('PASS | smoke only; test evaluation NOT RUN. Report:',out/'smoke_report.json',flush=True);return
 # Freeze all configurations before scoring any test predictions.
 if a.mode=='run':
  import optuna
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  for config in a.configs:
   cfgdir=out/config.replace('+','_');cfgdir.mkdir(exist_ok=True)
   trp=out/'train_matrix.npy';vap=out/'val_matrix.npy';width=WIDTHS[config]
   print('Building matrices:',config,'| input columns',width*20,flush=True)
   xtr=flatten(values,train,width,trp);xv=flatten(values,val,width,vap)
   sampler=optuna.samplers.TPESampler(seed=a.search_seed)
   study=optuna.create_study(direction='maximize',sampler=sampler)
   def objective(trial):
    params=sample_parameters(trial);params['scale_pos_weight']=spw
    model=make_model(params,42,a.device,a.jobs,rounds=a.rounds)
    model.fit(xtr,ytrain,eval_set=[(xv,yval)],verbose=False)
    actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if a.device=='cuda' and not actual.startswith('cuda'):raise RuntimeError('Requested CUDA unavailable in XGBoost')
    score=float(average_precision_score(yval,model.predict_proba(xv)[:,1]));trial.set_user_attr('best_iteration',int(model.best_iteration))
    print(f'{config} | tuning {trial.number+1}/{a.trials} | validation AP={score:.6f}',flush=True)
    del model;gc.collect();return score
   def progress(study,trial):study.trials_dataframe().to_csv(cfgdir/'search_trials.csv',index=False)
   study.optimize(objective,n_trials=a.trials,callbacks=[progress],gc_after_trial=True)
   params=dict(study.best_params,scale_pos_weight=spw)
   write(cfgdir/'frozen_parameters.json',dict(params=params,validation_ap=study.best_value,search_seed=a.search_seed,trials=a.trials,test_used=False))
   close_matrix(xtr,trp);close_matrix(xv,vap);gc.collect()
  write(out/'ALL_PARAMETERS_FROZEN.json',{c:sha(out/c.replace('+','_')/'frozen_parameters.json') for c in a.configs})
 frozen=json.loads((out/'ALL_PARAMETERS_FROZEN.json').read_text())
 for config in a.configs:
  cfgdir=out/config.replace('+','_');paramfile=cfgdir/'frozen_parameters.json'
  if sha(paramfile)!=frozen[config]:raise ValueError('Frozen parameters changed')
  params=json.loads(paramfile.read_text())['params'];width=WIDTHS[config]
  # Save the exact feature order for later TreeSHAP/aggregation across lags.
  names=[f'{col}__lag{lag:02d}' for lag in range(19,-1,-1) for col in columns[:width]]
  write(cfgdir/'feature_names.json',names)
  paths={s:out/(s+'_matrix.npy') for s in ['train','val','test']}
  matrices={s:flatten(values,ix,width,paths[s]) for s,ix in [('train',train),('val',val),('test',test)]}
  for seed in a.seeds:
   run=cfgdir/f'seed{seed}';rp=run/'experiment_report.json'
   if rp.exists():
    r=json.loads(rp.read_text())
    if r.get('status')=='PROVISIONAL_PASS':
     for name,digest in r['output_sha256'].items():
      if sha(run/name)!=digest:raise ValueError('Completed artifact modified: '+str(run/name))
     print('SKIP completed:',config,seed,flush=True);continue
   run.mkdir(exist_ok=True);started=time.perf_counter()
   r=dict(status='RUNNING',protocol=PROTOCOL,config=config,seed=seed,source_sha256=audit['source_sha256'],test_evaluated=False,final_thesis_ready=False)
   write(rp,r)
   try:
    print('TRAIN:',config,'| seed',seed,flush=True)
    model=make_model(params,seed,a.device,a.jobs,rounds=a.rounds)
    model.fit(matrices['train'],ytrain,eval_set=[(matrices['val'],yval)],verbose=100)
    vp=model.predict_proba(matrices['val'])[:,1];threshold,_=threshold_mcc(yval,vp)
    r.update(threshold=threshold,best_iteration=int(model.best_iteration),val_metrics=metrics(yval,vp,threshold),params=params)
    model.save_model(run/'xgboost_model.json');write(run/'learning_curve.json',model.evals_result())
    for split,ix,prob in [('val',val,vp),('test',test,model.predict_proba(matrices['test'])[:,1])]:
     np.savez_compressed(run/(split+'_predictions.npz'),dates=dates[ix],permnos=permnos[ix],y_true=target[ix].astype(np.float32),y_prob=prob,threshold=np.float64(threshold))
     if split=='test':r.update(test_metrics=metrics(target[ix],prob,threshold),test_evaluated=True)
    r.update(status='PROVISIONAL_PASS',elapsed_minutes=(time.perf_counter()-started)/60,
      output_sha256={name:sha(run/name) for name in ['xgboost_model.json','val_predictions.npz','test_predictions.npz','learning_curve.json']})
    write(rp,r);del model;gc.collect();print('PASS |',config,'seed',seed,flush=True)
   except Exception as e:
    r.update(status='FAILED',error=f'{type(e).__name__}: {e}');write(rp,r);raise
  for s in matrices:close_matrix(matrices[s],paths[s])
  gc.collect()
 write(out/'completed_runs.json',{'status':'PROVISIONAL_PASS','configs':a.configs,'seeds':a.seeds,'runs':len(a.configs)*len(a.seeds),'protocol':PROTOCOL})
 print('PASS | completed XGBoost comparisons:',out,flush=True)

if __name__=='__main__':main()
