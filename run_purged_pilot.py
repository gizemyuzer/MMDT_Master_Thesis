"""Full-panel, one-epoch I+II pilot. Not a final thesis experiment.
Run from project root with the CUDA-enabled virtual environment.
All trainer outputs are isolated inside a new results/purged_pilot_* directory.
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
    )
    from models.transformer_model import DualEncoderTransformer
    from models.pytorch_trainer import train_pytorch_model
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
    out = root / 'results' / ('purged_pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    print('PILOT ONLY | full panel | I+II | seed 42 | one epoch | batch size 64', flush=True)
    print('Output directory:', out, flush=True)
    print('Test predictions will NOT be evaluated.', flush=True)
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
    # Test windows are created by the shared loader, but not evaluated in this pilot.
    del testl, panel, scalers
    gc.collect()
    for loader in (tl, vl):
        ds = loader.dataset
        for values in (ds.tech_sequences, ds.fund_sequences, ds.labels):
            # Chunked finite check avoids a large extra boolean allocation.
            for start in range(0, len(values), 10000):
                if not np.isfinite(values[start:start+10000]).all():
                    raise ValueError('Non-finite scaled features or labels.')
    config = dict(tech_dim=len(tc), fund_dim=len(fc), seq_len=20, d_model=64,
                  n_heads=4, n_layers=2, ffn_dim=256, dropout=0.15,
                  modality='multimodal', fusion_type='cross_attention')
    report = dict(status='running', purpose='one_epoch_resource_pilot_not_final_results',
                  seed=42, epochs=1, batch_size=64, device=str(device),
                  torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  train_sequences=len(tl.dataset), validation_sequences=len(vl.dataset),
                  test_evaluated=False, config=config, technical_columns=tc,
                  fundamental_columns=fc, purge_labels=True, label_horizon=20,
                  focal_alpha=1.0, focal_gamma=2.0,
                  loss_note='Pilot uses alpha=1, as in smoke check. Full experimental loss protocol is separate.',
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                                 for p in ['datasets/feature_engineering.py', 'models/transformer_model.py',
                                           'models/pytorch_trainer.py', 'models/losses.py']})
    report_path = out / 'pilot_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    seed_all()
    model = DualEncoderTransformer(**config)
    training_started = time.perf_counter()
    try:
        # The original trainer writes relative paths; isolate its entire invocation.
        with isolated_outputs(out):
            result = train_pytorch_model(
                model=model, train_loader=ProgressLoader(tl, 'train'),
                val_loader=ProgressLoader(vl, 'validation'), test_loader=None,
                model_name='PILOT_ONLY_PURGED_I_II_seed42', epochs=1, device=device,
                criterion=FocalLoss(alpha=1.0, gamma=2.0), monitor='pr_auc',
                lr=1e-4, weight_decay=1e-2, early_stop_patience=8, use_warmup=False,
            )
        if not all(np.isfinite(v) for v in result['val_metrics'].values()):
            raise ValueError('Non-finite validation metric.')
        report.update(status='PASS', training_and_validation_minutes=(time.perf_counter()-training_started)/60,
                      total_minutes=(time.perf_counter()-started)/60,
                      validation_metrics=result['val_metrics'], threshold=result['threshold'])
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"PASS | full-panel one-epoch pilot | training + validation: {report['training_and_validation_minutes']:.1f} min", flush=True)
    print('Historical outputs unchanged. Test evaluation NOT RUN.', flush=True)
    print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
