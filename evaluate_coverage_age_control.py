"""Evaluate archived coverage-age Transformer runs against the original full-text runs.
Run from repository root after all 11 control seeds finish. No training or model selection.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
CONFIGS={
 'I':'corrected_I_seed',
 'I+II':'corrected_I_II_seed',
 'I+II+IV':'corrected_I_II_IV_train_two_repairs_seed',
}

def single(pattern,seed,protocol):
    candidates=[]
    for directory in (ROOT/'results').glob(f'{pattern}{seed}_*'):
        report=directory/'experiment_report.json'
        if report.exists():
            r=json.loads(report.read_text())
            if r.get('status')=='PROVISIONAL_PASS' and r.get('protocol')==protocol and r.get('seed')==seed:
                candidates.append(directory)
    if len(candidates)!=1:raise RuntimeError(f'Expected exactly one completed {pattern}{seed}: {candidates}')
    return candidates[0]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',default='portfolio_inputs_20260923T110426840595Z.zip')
    parser.add_argument('--output',default='results/coverage_age_evaluation')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    baseline={}
    for config,prefix in CONFIGS.items():
        protocol=('corrected_prices_permno_I_II_IV_two_repairs_v1' if config=='I+II+IV'
                  else 'corrected_prices_permno_'+config.replace('+','_')+'_v1')
        for seed in range(42,53):
            dirs=list((ROOT/'results').glob(f'{prefix}{seed}_*'))
            completed=[d for d in dirs if (d/'experiment_report.json').exists() and
                       json.loads((d/'experiment_report.json').read_text()).get('status') in ('PASS','PROVISIONAL_PASS') and
                       json.loads((d/'experiment_report.json').read_text()).get('protocol')==protocol]
            if len(completed)!=1:raise RuntimeError(f'Expected one original {config}/{seed}, found {completed}')
            baseline[(config,seed)]=completed[0]
    control={seed:single('coverage_control_train_seed',seed,'coverage_age_only_same_25d_encoder_v1') for seed in range(42,53)}
    def archive(dest,use_control):
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
            for config in CONFIGS:
                for seed in range(42,53):
                    directory=control[seed] if (use_control and config=='I+II+IV') else baseline[(config,seed)]
                    for split in ('val','test'):
                        file=directory/f'{split}_predictions.npz'
                        if not file.exists():raise FileNotFoundError(file)
                        z.write(file,f'npz/{config}/{seed}/{split}_predictions.npz')
    original=out/'original_predictions.zip';counter=out/'coverage_control_predictions.zip'
    archive(original,False);archive(counter,True)
    inp=ROOT/args.inputs
    for tag,archive_path in [('original',original),('coverage_control',counter)]:
        cmd=[sys.executable,str(ROOT/'evaluate_portfolios.py'),str(inp),str(archive_path),str(out/tag)]
        with (out/f'{tag}_portfolio_log.txt').open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    old=json.loads((out/'original/portfolio_results.json').read_text())
    new=json.loads((out/'coverage_control/portfolio_results.json').read_text())
    def rows(result):return {int(r['seed']):r for r in result['portfolios'] if r['strategy']=='I+II+IV' and r['seed'] is not None}
    a=rows(old);b=rows(new)
    assert set(a)==set(b)==set(range(42,53))
    paired=[{'seed':s,**{m+'_original':a[s][m] for m in ('cagr','max_drawdown','calmar')},
             **{m+'_control':b[s][m] for m in ('cagr','max_drawdown','calmar')},
             **{'delta_'+m:a[s][m]-b[s][m] for m in ('cagr','max_drawdown','calmar')}} for s in sorted(a)]
    classification=[]
    for seed in range(42,53):
        full=json.loads((baseline[('I+II+IV',seed)]/'experiment_report.json').read_text())['test_metrics']
        ctrl=json.loads((control[seed]/'experiment_report.json').read_text())['test_metrics']
        classification.append({'seed':seed,**{f'delta_{metric}':full[metric]-ctrl[metric]
                           for metric in ('roc_auc','pr_auc','mcc')}})
    summary={'mean_delta_pr_auc':sum(x['delta_pr_auc'] for x in classification)/11,
             'classification_paired':classification,
             'contrast':'full-text minus availability-and-age-only Transformer, paired by seed',
             'control_columns':['TXT_HasCoverage','TXT_DaysSinceFiling'],
             'ex_post_exploratory':True,
             'mean_delta_calmar':sum(x['delta_calmar'] for x in paired)/11,
             'paired':paired}
    (out/'paired_portfolio_comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'mean_delta_calmar':summary['mean_delta_calmar'],'output':str(out)},indent=2))
if __name__=='__main__':main()
