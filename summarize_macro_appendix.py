"""Package I, I+II and exploratory macro predictions; compute paired appendix comparison."""
from pathlib import Path
import json
import zipfile
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
BASE={'I':('corrected_I_seed','corrected_prices_permno_I_v1','PASS'),
      'I+II':('corrected_I_II_seed','corrected_prices_permno_I_II_v1','PASS'),
      'I+II+III':('corrected_I_II_III_macro_seed','corrected_prices_permno_I_II_III_macro_consensus_v1','EXPLORATORY_PASS')}

def find(config,seed):
    prefix,protocol,status=BASE[config];matches=[]
    for d in (ROOT/'results').glob(f'{prefix}{seed}_*'):
        rp=d/'experiment_report.json'
        if rp.exists():
            r=json.loads(rp.read_text())
            if (r.get('protocol'),r.get('status'),r.get('seed'))==(protocol,status,seed):matches.append(d)
    if len(matches)!=1:raise RuntimeError(f'Expected exactly one {config}/{seed}; found {matches}')
    return matches[0]

def main():
    out=ROOT/'results/macro_appendix_evaluation';out.mkdir(parents=True,exist_ok=True)
    runs={(c,s):find(c,s) for c in BASE for s in range(42,53)}
    archive=out/'matched_predictions.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
        for (config,seed),d in runs.items():
            for split in ('val','test'):
                file=d/f'{split}_predictions.npz'
                if not file.exists():raise FileNotFoundError(file)
                z.write(file,f'npz/{config}/{seed}/{split}_predictions.npz')
    inputs=ROOT/'portfolio_inputs_20260923T110426840595Z.zip'
    cmd=[sys.executable,str(ROOT/'evaluate_macro_portfolios.py'),str(inputs),str(archive),str(out/'portfolio')]
    with (out/'portfolio_log.txt').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    result=json.loads((out/'portfolio/portfolio_results.json').read_text())
    vals={(r['strategy'],int(r['seed'])):r for r in result['portfolios'] if r['seed'] is not None}
    paired=[]
    for seed in range(42,53):
        old=vals['I+II',seed];new=vals['I+II+III',seed]
        oldmetrics=json.loads((runs['I+II',seed]/'experiment_report.json').read_text())['test_metrics']
        newmetrics=json.loads((runs['I+II+III',seed]/'experiment_report.json').read_text())['test_metrics']
        paired.append({'seed':seed,'delta_calmar':new['calmar']-old['calmar'],
                       'delta_cagr':new['cagr']-old['cagr'],
                       'delta_max_drawdown':new['max_drawdown']-old['max_drawdown'],
                       'delta_pr_auc':newmetrics['pr_auc']-oldmetrics['pr_auc'],
                       'delta_roc_auc':newmetrics['roc_auc']-oldmetrics['roc_auc'],
                       'delta_mcc':newmetrics['mcc']-oldmetrics['mcc']})
    metrics=['calmar','cagr','max_drawdown','pr_auc','roc_auc','mcc']
    summary={'status':'EXPLORATORY_POST_TEST','contrast':'I+II+III minus I+II, paired seed 42-52',
             'macro_consensus_min_support':.99,'macro_lag':'strictly prior source date',
             'source_vintage_verified':False,
             'means':{k:sum(r['delta_'+k] for r in paired)/11 for k in metrics},'paired':paired}
    (out/'macro_appendix_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['means'],indent=2));print('OUTPUT',out)
if __name__=='__main__':main()
