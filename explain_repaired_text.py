"""Grouped Kernel SHAP for saved repaired I+II+IV seed42. No fitting/training.
Each player is one feature's entire 20-session trajectory (57 players).
Coalition output averages probabilities over fixed TRAIN background sequences.
Run from project root alongside run_corrected_smoke.py and run_repaired_text.py.
"""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run-dir',default='results/corrected_I_II_IV_train_two_repairs_seed42_20260923T175023Z_2de97c52')
 p.add_argument('--technical',default='datasets/corrected_technical_20260919T155939Z_31e99bbd/technical_panel.csv')
 p.add_argument('--fundamental',default='datasets/fundamental_panel_20260919T220205Z_2aed9ab0/fundamental_daily.csv')
 p.add_argument('--text',default='datasets/text_identity_repair_20260923T172930Z_fe9420be/text_daily.csv')
 p.add_argument('--samples',type=int,default=64);p.add_argument('--background',type=int,default=16)
 p.add_argument('--nsamples',type=int,default=256);p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
 a=p.parse_args()
 if a.samples<2 or a.background<2 or a.nsamples<128:p.error('Use samples/background >=2 and nsamples >=128')
 import torch,joblib,shap
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import run_corrected_smoke as adapter
 from run_repaired_text import TEXT,TEXT_META,align_text,validate_text
 from models.transformer_model import DualEncoderTransformer
 run=Path(a.run_dir);r=json.loads((run/'experiment_report.json').read_text(encoding='utf-8-sig'))
 if r['seed']!=42 or r['protocol']!='corrected_prices_permno_I_II_IV_two_repairs_v1' or r['status']!='PROVISIONAL_PASS':raise ValueError('Requires completed repaired seed42')
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
 for k in ['technical','fundamental','text']:
  print('Checking source:',k,flush=True)
  if sha(getattr(a,k))!=r['source_sha256'][k]:raise ValueError('Source hash mismatch: '+k)
 model_path=Path('models/transformer_model.py')
 # Use only user's own trusted training artifacts.
 ck=torch.load(run/'selected_model.pt',map_location='cpu',weights_only=True)
 if ck['source_sha256']!=r['source_sha256'] or ck['config']!=r['config']:raise ValueError('Checkpoint/report mismatch')
 state=joblib.load(run/'preprocessing.joblib')
 columns=adapter.TECH+adapter.FUND+TEXT
 if state['model_columns']!=columns or state['columns']!=adapter.TECH+adapter.FUND:raise ValueError('Preprocessing column order mismatch')
 print('Loading panels and saved preprocessing (no refitting)...',flush=True)
 t=pd.read_csv(a.technical,usecols=adapter.META+adapter.TECH,parse_dates=['date','LabelEndDate'])
 f=pd.read_csv(a.fundamental,usecols=['date','permno']+adapter.FUND,parse_dates=['date'])
 frame=adapter.join_panel(t,f);adapter.check_endpoints(frame)
 raw=frame[state['columns']].replace([np.inf,-np.inf],np.nan)
 raw[adapter.FUND]=raw[adapter.FUND].fillna(state['fundamental_medians'])
 v=state['scaler'].transform(raw).astype(np.float32)
 tx=pd.read_csv(a.text,usecols=['date','permno']+TEXT+TEXT_META,parse_dates=['date','TextSourceFilingDate','TextSourceAvailableDate'])
 tx=align_text(frame,tx);traw=validate_text(frame,tx);ts=state['text_preprocessing']
 if ts['columns']!=TEXT:raise ValueError('Text column order mismatch')
 tv=ts['scaler'].transform(traw.fillna(ts['medians'])).astype(np.float32)
 values=np.concatenate([v,tv],axis=1)
 train=adapter.window_positions(frame,values,'TrainEndpoint');test=adapter.window_positions(frame,values,'TestEndpoint')
 rng=np.random.default_rng(20260924)
 selected=np.sort(rng.choice(len(test),size=min(a.samples,len(test)),replace=False))
 bgpos=rng.choice(train,size=min(a.background,len(train)),replace=False)
 def windows(pos):return np.stack([values[j-19:j+1] for j in pos])
 background=windows(bgpos);examples=windows(test[selected])
 model=DualEncoderTransformer(**ck['config']).to(a.device);model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 @torch.inference_mode()
 def predict(x):
  ans=[]
  for j in range(0,len(x),128):
   q=torch.as_tensor(x[j:j+128],dtype=torch.float32,device=a.device)
   ans.append(torch.sigmoid(model(q[:,:,:32],q[:,:,32:]).reshape(-1)).cpu().numpy())
  return np.concatenate(ans)
 pred=predict(examples)
 with np.load(run/'test_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['permnos'],frame.permno.to_numpy()[test]) or not np.array_equal(z['dates'].astype('datetime64[ns]'),frame.date.to_numpy()[test]):raise ValueError('Archived prediction ordering differs')
  err=float(np.max(np.abs(pred-z['y_prob'][selected])))
  if err>2e-5:raise ValueError(f'Prediction reproduction failed: {err}')
 print('PASS | saved predictions reproduced; max error',err,flush=True)
 out=Path('results')/('shap_repaired_seed42_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
 vals=[];bases=[]
 np.random.seed(20260924)
 for n,x in enumerate(examples):
  def coalition(masks):
   masks=np.asarray(masks);res=[]
   for mask in masks:
    mixed=np.where(mask[None,None,:]>.5,x[None,:,:],background)
    res.append(float(predict(mixed).mean()))
   return np.asarray(res)
  ex=shap.KernelExplainer(coalition,np.zeros((1,len(columns))),link='identity')
  sv=np.asarray(ex.shap_values(np.ones((1,len(columns))),nsamples=a.nsamples,l1_reg=0.0,silent=True)).reshape(-1)
  if len(sv)!=len(columns) or not np.isfinite(sv).all():raise ValueError('Invalid SHAP output')
  vals.append(sv);bases.append(float(ex.expected_value))
  print(f'SHAP {n+1}/{len(examples)}',flush=True)
 vals=np.asarray(vals);bases=np.asarray(bases)
 residual=np.abs(bases+vals.sum(axis=1)-pred)
 if residual.max()>1e-4:raise ValueError('SHAP additivity failed')
 np.savez_compressed(out/'shap_values.npz',values=vals,base_values=bases,predictions=pred,features=np.array(columns),
  window_mean_scaled=examples.mean(axis=1),dates=frame.date.to_numpy()[test[selected]],permnos=frame.permno.to_numpy()[test[selected]],
  background_dates=frame.date.to_numpy()[bgpos],background_permnos=frame.permno.to_numpy()[bgpos])
 importance=np.abs(vals).mean(axis=0);order=np.argsort(importance)[-20:]
 plt.figure(figsize=(9,8));plt.barh(np.array(columns)[order],importance[order],color='#246A8D')
 plt.xlabel('Mean absolute grouped SHAP value (predicted probability)');plt.title('Repaired I+II+IV, seed 42 — sampled test observations');plt.tight_layout()
 for ext in ['png','pdf']:plt.savefig(out/('shap_importance.'+ext),dpi=300,bbox_inches='tight')
 plt.close()
 shap.summary_plot(vals,examples.mean(axis=1),feature_names=columns,max_display=20,show=False,color_bar_label='Mean scaled feature over 20 sessions')
 plt.title('Grouped SHAP — complete feature trajectories');plt.tight_layout()
 for ext in ['png','pdf']:plt.savefig(out/('shap_beeswarm.'+ext),dpi=300,bbox_inches='tight')
 plt.close()
 pd.DataFrame({'feature':columns,'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'feature_importance.csv',index=False)
 report=dict(status='PASS',seed=42,samples=len(examples),background=len(background),nsamples=a.nsamples,
  method='Kernel SHAP over 57 feature-trajectory groups; probability output; fixed training background; l1_reg=0',
  selection='Uniform test endpoint sample, fixed RNG 20260924; seed42 chosen without performance selection',
  prediction_max_error=err,additivity_max_error=float(residual.max()),
  source_hashes=r['source_sha256'],checkpoint_sha256=sha(run/'selected_model.pt'),preprocessing_sha256=sha(run/'preprocessing.joblib'),
  shap_version=shap.__version__,torch_version=torch.__version__,output_folder=str(out),
  limitations=['One seed and sampled endpoints, not an 11-seed global attribution.',
   'Approximate interventional SHAP; hybrid feature trajectories may be off-distribution; not causal effects.',
   'Temporal lags are grouped; beeswarm colour is mean scaled input over the window, not the masking variable.',
   'CIK mapping limitations remain; no word-level text explanation.'])
 (out/'shap_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2));print('PASS | SHAP plots saved; no model training')
if __name__=='__main__':main()
