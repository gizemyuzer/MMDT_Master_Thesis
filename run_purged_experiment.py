"""Purged full-panel I+II experiment: seed 42, at most 30 epochs.
Run from project root with the CUDA-enabled virtual environment.
All trainer outputs are isolated inside a new results/purged_I_II_* directory.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
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
        cached_label_end_dates, cached_purge_mask,
    )
    from models.transformer_model import DualEncoderTransformer
    from models.pytorch_trainer import train_pytorch_model, _evaluate_on_loader
    from models.losses import FocalLoss

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Use the CUDA-enabled H: virtual environment.')
    device = torch.device(args.device)
    def seed_all():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    seed_all()
    out = root / 'results' / ('purged_I_II_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    print('FULL EXPERIMENT | I+II | seed 42 | up to 30 epochs | batch size 64', flush=True)
    print('Output directory:', out, flush=True)
    print('Test is evaluated only after validation-based checkpoint selection.', flush=True)
    if args.device == 'cuda':
        print('GPU:', torch.cuda.get_device_name(), flush=True)
    started = time.perf_counter()
    panel = pd.read_csv(root / 'datasets/final_dataset.csv', index_col=0, parse_dates=True)
    cols = list(TECHNICAL_COLS) + list(FIRM_FUNDAMENTAL_COLS)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ValueError(f'Missing required inputs: {missing}')
    # Keep all dates so cached label-end upper bounds remain available.
    tl, vl, testl, scalers, (tc, fc) = get_dual_stream_dataloaders(
        panel, seq_len=20, batch_size=64, tech_groups=('tech',),
        fund_groups=('fund',), purge_labels=True, label_horizon=20,
    )
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
                  modality='multimodal', fusion_type='cross_attention')
    report = dict(status='running', protocol='purged_fixed_alpha1_v1', purpose='purged_I_II_seed42_full_training',
                  seed=42, epochs=30, batch_size=64, device=str(device),
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
                model_name='PURGED_I_II_seed42', epochs=30, device=device,
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
                    'seed': 42, 'threshold': result['threshold'],
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
    print(f"PASS | purged I+II seed42 | training + evaluation + export: {report['training_and_validation_minutes']:.1f} min", flush=True)
    print('Historical outputs unchanged. Selected model, preprocessing and indexed predictions saved.', flush=True)
    print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
