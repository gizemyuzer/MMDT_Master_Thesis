"""Sequential purged experiments: I, I+II, I+II+IV; seeds 42-52. Run with --batch.
Run from project root with the CUDA-enabled virtual environment.
Every run uses a new results/purged_* directory; matching completed runs are reused.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from uuid import uuid4


@contextmanager
def isolated_outputs(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


class ProgressLoader:
    """Transparent loader wrapper; reports progress without changing batches."""
    def __init__(self, loader, name):
        self.loader, self.name = loader, name

    @property
    def dataset(self):
        return self.loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        start = time.perf_counter()
        for i, batch in enumerate(self.loader, 1):
            if i == 1 or i % 500 == 0 or i == len(self):
                print(f'[{self.name}] batch {i}/{len(self)} | elapsed {(time.perf_counter()-start)/60:.1f} min', flush=True)
            yield batch



CELLS = {
    'tech_only': ('I', ('fund',), 'tech_only'),
    'multi_pure': ('I+II', ('fund',), 'multimodal'),
    'multi_text': ('I+II+IV', ('fund', 'text'), 'multimodal'),
}
SOURCE_FILES = ['datasets/feature_engineering.py', 'models/transformer_model.py',
                'models/pytorch_trainer.py', 'models/losses.py']


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def complete_run(report_path, cell, seed, expected):
    """Accept only matching completed runs, never choose by a test score."""
    try:
        r = json.loads(report_path.read_text(encoding='utf-8'))
        old_i_ii = r.get('ablation') is None and str(r.get('purpose', '')).startswith('purged_I_II_seed')
        actual_cell = 'multi_pure' if old_i_ii else r.get('ablation')
        if actual_cell != cell or r.get('seed') != seed:
            return None
        if r.get('status') != 'PASS' or r.get('test_evaluated') is not True:
            return None
        for k, value in {'protocol':'purged_fixed_alpha1_v1', 'epochs':30, 'batch_size':64,
                         'focal_alpha':1.0, 'focal_gamma':2.0, 'purge_labels':True,
                         'label_horizon':20, 'torch_version':expected['torch_version'],
                         'cuda_build':expected['cuda_build'], 'device':expected['device'],
                         'source_sha256':expected['source_sha256']}.items():
            if r.get(k) != value:
                return None
        config = dict(tech_dim=32, fund_dim=25 if cell=='multi_text' else 6,
                      seq_len=20, d_model=64, n_heads=4, n_layers=2, ffn_dim=256,
                      dropout=0.15, modality=CELLS[cell][2], fusion_type='cross_attention')
        if r.get('config') != config:
            return None
        if r.get('technical_columns') != expected['technical_columns']:
            return None
        if r.get('fundamental_columns') != expected['fundamental_columns'][cell]:
            return None
        if r.get('dataset_sha256') not in (None, expected['dataset_sha256']):
            return None
        if r.get('dataset_sha256') is None and not old_i_ii:
            return None
        if cell == 'multi_text' and (r.get('text_sha256') != expected['text_sha256'] or
                r.get('text_coverage_rule') != 'current_row_wordcount_and_filing_age_or_positive_event_count_v1'):
            return None
        for name in ('selected_model.pt','preprocessing.joblib','val_predictions.npz','test_predictions.npz'):
            path=report_path.parent/name
            if not path.is_file() or path.stat().st_size == 0:
                return None
        return r
    except (ValueError, OSError):
        return None


def run_queue(args):
    import ast
    import csv
    import subprocess
    import sys
    import torch
    if any(s < 0 or s >= 2**32 for s in args.seeds) or len(args.seeds)!=len(set(args.seeds)):
        raise SystemExit('Seeds must be unique integers between 0 and 2**32-1.')
    root = Path.cwd().resolve()
    needed = SOURCE_FILES + ['datasets/final_dataset.csv','datasets/text_features.csv','datasets/universe.csv']
    for p in needed:
        if not (root/p).is_file():
            raise SystemExit('Missing local file: '+p+'. Queue has not started.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable in this Python environment.')
    # Read constant feature declarations without importing WRDS/universe modules.
    tree = ast.parse((root/'datasets/feature_engineering.py').read_text(encoding='utf-8-sig'))
    wanted = {'TECHNICAL_COLS','FIRM_FUNDAMENTAL_COLS','TEXT_EVENT_COLS','TEXT_LM_COLS','TEXT_COVERAGE_COL'}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = ast.literal_eval(node.value)
    text_cols = constants['TEXT_EVENT_COLS']+constants['TEXT_LM_COLS']+[constants['TEXT_COVERAGE_COL']]
    with open(root/'datasets/text_features.csv', newline='', encoding='utf-8-sig') as stream:
        header=next(csv.reader(stream))
    missing=[c for c in text_cols if c not in header and c!=constants['TEXT_COVERAGE_COL']]
    if missing or 'date' not in header or not ({'ticker','Ticker'} & set(header)):
        raise SystemExit(f'Text file columns invalid; missing features: {missing}. Queue has not started.')
    expected = dict(torch_version=torch.__version__, cuda_build=torch.version.cuda, device=args.device,
                    source_sha256={p:file_hash(root/p) for p in SOURCE_FILES},
                    dataset_sha256=file_hash(root/'datasets/final_dataset.csv'),
                    text_sha256=file_hash(root/'datasets/text_features.csv'),
                    technical_columns=constants['TECHNICAL_COLS'],
                    fundamental_columns={c:constants['FIRM_FUNDAMENTAL_COLS']+(text_cols if c=='multi_text' else []) for c in CELLS})
    frozen = {p: ((root/p).stat().st_size, (root/p).stat().st_mtime_ns) for p in needed}
    queue_dir = root/'results'/('purged_queue_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    queue_dir.mkdir(parents=True, exist_ok=False)
    (queue_dir/'protocol.json').write_text(json.dumps(dict(seeds=args.seeds, cells=list(CELLS), expected=expected,
        note='Same seed IDs do not guarantee identical minibatch streams across architectures. Legacy I+II reports lack a dataset hash; reuse assumes the cache was not changed.'),indent=2),encoding='utf-8')
    index=[]
    def save_index():
        (queue_dir/'completed_runs.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
        with open(queue_dir/'summary.csv','w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['ablation','seed','reused','report','dataset_hash_verified','test_mcc','test_pr_auc','test_roc_auc'])
            writer.writeheader(); writer.writerows(index)
    def matches(cell, seed):
        found=[]
        for p in sorted((root/'results').glob('purged_*/experiment_report.json')):
            r=complete_run(p,cell,seed,expected)
            if r is not None: found.append((p,r))
        if len(found)>1:
            raise RuntimeError(f'Multiple matching completed runs for {cell} seed {seed}; stopping to avoid silent selection.')
        return found
    print(f'QUEUE: {len(args.seeds)*len(CELLS)} cells; sequential execution; completed matching runs reused.',flush=True)
    print('Queue directory:',queue_dir,flush=True)
    for seed in args.seeds:
        for cell in CELLS:
            for p, signature in frozen.items():
                st=(root/p).stat()
                if (st.st_size,st.st_mtime_ns)!=signature:
                    raise RuntimeError('Source/data changed during queue: '+p)
            found=matches(cell,seed)
            reused=bool(found)
            if found:
                print(f'SKIP completed: {cell} seed {seed}',flush=True)
            else:
                print(f'START: {cell} seed {seed}',flush=True)
                log=queue_dir/f'{cell}_seed{seed}.log'
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--device',args.device,'--ablation',cell,'--seed',str(seed)]
                with open(log,'w',encoding='utf-8') as stream:
                    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1'
                    proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
                    try:
                        for line in proc.stdout:
                            stream.write(line);stream.flush()
                            try: print(line,end='',flush=True)
                            except UnicodeEncodeError: print(line.encode('ascii','replace').decode(),end='',flush=True)
                        code=proc.wait()
                    except BaseException:
                        proc.terminate();proc.wait()
                        raise
                if code:
                    raise SystemExit(f'QUEUE STOPPED: {cell} seed {seed}; exit {code}. Log: {log}')
                found=matches(cell,seed)
                if len(found)!=1:
                    raise RuntimeError(f'No valid completed report after {cell} seed {seed}.')
            p,r=found[0]
            if r.get('dataset_sha256') is None:
                print('Legacy I+II reused: settings/sources match; historical report has no cache hash.',flush=True)
            index.append(dict(ablation=cell,seed=seed,reused=reused,report=str(p),
                              dataset_hash_verified=r.get('dataset_sha256')==expected['dataset_sha256'],
                              test_mcc=r['test_metrics']['mcc'],test_pr_auc=r['test_metrics']['pr_auc'],test_roc_auc=r['test_metrics']['roc_auc']))
            save_index()
    print('QUEUE COMPLETE:',len(index),'runs. Summary:',queue_dir/'summary.csv',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablation', choices=list(CELLS), default='multi_pure')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 53)))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32-1')
    if args.batch:
        run_queue(args)
        return
    cell_label, fund_groups, modality = CELLS[args.ablation]
    root = Path.cwd().resolve()
    for path in ('datasets/final_dataset.csv', 'datasets/universe.csv',
                 'datasets/feature_engineering.py', 'models/pytorch_trainer.py'):
        if not (root / path).is_file():
            raise SystemExit(f'Missing {path}. Run from the project root.')

    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from datasets.feature_engineering import (
        get_dual_stream_dataloaders, TECHNICAL_COLS, FIRM_FUNDAMENTAL_COLS,
        cached_label_end_dates, cached_purge_mask, TEXT_COLS, TEXT_LM_COLS,
        TEXT_EVENT_COLS, TEXT_COVERAGE_COL, attach_text_features,
    )
    from models.transformer_model import DualEncoderTransformer
    from models.pytorch_trainer import train_pytorch_model, _evaluate_on_loader
    from models.losses import FocalLoss

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Use the CUDA-enabled H: virtual environment.')
    device = torch.device(args.device)
    def seed_all():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    seed_all()
    out = root / 'results' / (f'purged_{args.ablation}_seed{args.seed}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    print(f'FULL EXPERIMENT | {cell_label} | seed {args.seed} | up to 30 epochs | batch size 64', flush=True)
    print('Output directory:', out, flush=True)
    print('Test is evaluated only after validation-based checkpoint selection.', flush=True)
    if args.device == 'cuda':
        print('GPU:', torch.cuda.get_device_name(), flush=True)
    started = time.perf_counter()
    panel = pd.read_csv(root / 'datasets/final_dataset.csv', index_col=0, parse_dates=True)
    cols = list(TECHNICAL_COLS) + list(FIRM_FUNDAMENTAL_COLS)
    if args.ablation == 'multi_text':
        text_path = root / 'datasets/text_features.csv'
        if not text_path.is_file():
            raise FileNotFoundError('Missing local datasets/text_features.csv; no downloads will be started.')
        panel = attach_text_features(panel, path=str(text_path))
        absent = [c for c in TEXT_COLS if c not in panel.columns]
        if absent:
            raise ValueError(f'Missing text columns: {absent}')
        # The legacy helper marks coverage by membership in the entire filing archive.
        # Replace this with current-row availability, avoiding a future-filer indicator.
        panel[TEXT_COVERAGE_COL] = (
            (panel['LM_WordCount'].gt(0) & panel['TXT_DaysSinceFiling'].ge(0))
            | panel[[c for c in TEXT_EVENT_COLS if c != 'EK_DaysSince']].fillna(0).gt(0).any(axis=1)
        ).astype(float)
        if not panel[TEXT_COVERAGE_COL].any():
            raise ValueError('No available text signal joined to this panel.')
        cols += list(TEXT_COLS)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f'Missing required inputs: {missing}')
    # Keep all dates so cached label-end upper bounds remain available.
    tl, vl, testl, scalers, (tc, fc) = get_dual_stream_dataloaders(
        panel, seq_len=20, batch_size=64, tech_groups=('tech',),
        fund_groups=fund_groups, purge_labels=True, label_horizon=20,
    )
    print(f'Actual test dates: {testl.dataset.dates.min()} to {testl.dataset.dates.max()}', flush=True)
    print('Full-panel train sequences:', len(tl.dataset), flush=True)
    print('Full-panel validation sequences:', len(vl.dataset), flush=True)
    # Persist the exact imputation/scaling state, not a refit at export time.
    import joblib
    end_dates = cached_label_end_dates(panel, 20)
    train_mask, train_audit = cached_purge_mask(panel, end_dates, '2010-01-01', '2019-12-31')
    _, val_audit = cached_purge_mask(panel, end_dates, '2020-01-01', '2021-12-31')
    imputation_medians = panel.loc[train_mask, tc + fc].median()
    joblib.dump({'scalers': scalers, 'imputation_medians': imputation_medians,
                 'technical_columns': tc, 'fundamental_columns': fc}, out / 'preprocessing.joblib')
    del panel, scalers, end_dates, train_mask
    gc.collect()
    for loader in (tl, vl, testl):
        ds = loader.dataset
        for values in (ds.tech_sequences, ds.fund_sequences, ds.labels):
            # Chunked finite check avoids a large extra boolean allocation.
            for start in range(0, len(values), 10000):
                if not np.isfinite(values[start:start+10000]).all():
                    raise ValueError('Non-finite scaled features or labels.')
    config = dict(tech_dim=len(tc), fund_dim=len(fc), seq_len=20, d_model=64,
                  n_heads=4, n_layers=2, ffn_dim=256, dropout=0.15,
                  modality=modality, fusion_type='cross_attention')
    report = dict(status='running', protocol='purged_fixed_alpha1_v1', purpose=f'purged_{args.ablation}_seed{args.seed}_full_training',
                  ablation=args.ablation, text_coverage_rule='current_row_wordcount_and_filing_age_or_positive_event_count_v1' if args.ablation == 'multi_text' else None,
                  dataset_sha256=file_hash(root / 'datasets/final_dataset.csv'),
                  text_sha256=file_hash(root / 'datasets/text_features.csv') if args.ablation == 'multi_text' else None,
                  runner_sha256=file_hash(Path(__file__)),
                  seed=args.seed, epochs=30, batch_size=64, device=str(device),
                  torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  train_sequences=len(tl.dataset), validation_sequences=len(vl.dataset),
                  test_evaluated=False, train_purge_audit=train_audit, validation_purge_audit=val_audit, config=config, technical_columns=tc,
                  fundamental_columns=fc, purge_labels=True, label_horizon=20,
                  focal_alpha=1.0, focal_gamma=2.0,
                  loss_note='Fixed alpha=1 with balanced replacement sampling. Historical runner estimated alpha from balanced sampled batches; this removes random alpha variation and must be disclosed.',
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                 for p in ['datasets/feature_engineering.py', 'models/transformer_model.py',
                                           'models/pytorch_trainer.py', 'models/losses.py']})
    report_path = out / 'experiment_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    seed_all()
    model = DualEncoderTransformer(**config)
    training_started = time.perf_counter()
    try:
        # The original trainer writes relative paths; isolate its entire invocation.
        with isolated_outputs(out):
            result = train_pytorch_model(
                model=model, train_loader=ProgressLoader(tl, 'train'),
                val_loader=ProgressLoader(vl, 'validation'), test_loader=ProgressLoader(testl, 'test'),
                model_name=f'PURGED_{args.ablation}_seed{args.seed}', epochs=30, device=device,
                criterion=FocalLoss(alpha=1.0, gamma=2.0), monitor='pr_auc',
                lr=1e-4, weight_decay=1e-2, early_stop_patience=8, use_warmup=False,
            )
        for split in ('val', 'test'):
            if not all(np.isfinite(v) for v in result[f'{split}_metrics'].values()):
                raise ValueError(f'Non-finite {split} metric.')
        # Preserve ordering directly from each non-shuffled dataset.
        # This export repeats inference only; it does not change checkpoint or threshold.
        for split, loader in [('val', vl), ('test', testl)]:
            predictions, targets, _ = _evaluate_on_loader(result['model'], loader, device)
            ds = loader.dataset
            if len(predictions) != len(ds.dates) or not np.array_equal(targets, ds.labels):
                raise AssertionError('Prediction export alignment failed.')
            np.savez_compressed(out / f'{split}_predictions.npz',
                                dates=ds.dates.to_numpy(dtype='datetime64[ns]'),
                                tickers=np.asarray(ds.tickers, dtype=str),
                                y_true=targets, y_prob=predictions,
                                threshold=np.asarray(result['threshold']))
        torch.save({'state_dict': result['model'].state_dict(), 'config': config,
                    'seed': args.seed, 'threshold': result['threshold'],
                    'technical_columns': tc, 'fundamental_columns': fc,
                    'purge_labels': True, 'label_horizon': 20, 'focal_alpha': 1.0},
                   out / 'selected_model.pt')
        report.update(status='PASS', training_and_validation_minutes=(time.perf_counter()-training_started)/60,
                      total_minutes=(time.perf_counter()-started)/60,
                      validation_metrics=result['val_metrics'], test_metrics=result['test_metrics'],
                      test_evaluated=True, threshold=result['threshold'])
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"PASS | purged {cell_label} seed{args.seed} | training + evaluation + export: {report['training_and_validation_minutes']:.1f} min", flush=True)
    print('Historical outputs unchanged. Selected model, preprocessing and indexed predictions saved.', flush=True)
    print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
