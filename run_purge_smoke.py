"""One small integration check; never use its output as thesis performance.
Run from the project root: python run_purge_smoke.py
Requires the updated datasets/feature_engineering.py and existing local caches.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def select_tickers(panel, limit=8):
    """Select by historical coverage only, without inspecting test outcomes."""
    train = panel.loc[(panel.index >= '2010-01-01') & (panel.index <= '2019-12-31')]
    val = panel.loc[(panel.index >= '2020-01-01') & (panel.index <= '2021-12-31')]
    tc = train.groupby('Ticker').size()
    vc = val.groupby('Ticker').size()
    candidates = sorted(set(tc[tc >= 80].index) & set(vc[vc >= 80].index))
    if not candidates:
        raise ValueError('No ticker has sufficient train/validation history.')
    return candidates[:limit]


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    required = ['datasets/final_dataset.csv', 'datasets/universe.csv',
                'datasets/feature_engineering.py', 'models/transformer_model.py',
                'models/losses.py']
    missing = [p for p in required if not Path(p).is_file()]
    if missing:
        raise SystemExit('Run from project root. Missing local files: ' + ', '.join(missing))

    import numpy as np
    import pandas as pd
    import torch
    from datasets.feature_engineering import (
        get_dual_stream_dataloaders, cached_label_end_dates,
        TECHNICAL_COLS, FIRM_FUNDAMENTAL_COLS,
    )
    from models.transformer_model import DualEncoderTransformer
    from models.losses import FocalLoss

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if mps_available:
        torch.mps.manual_seed(seed)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else ('mps' if mps_available else 'cpu')
    device = torch.device(device_name)
    print('SMOKE ONLY | seed=42 | max 8 tickers | 5 train + 5 validation batches', flush=True)
    print('Device:', device, flush=True)

    # Read the existing panel directly: no prepare_dataset/cache rebuild.
    panel = pd.read_csv(required[0], index_col=0, parse_dates=True)
    tickers = select_tickers(panel)
    panel = panel.loc[panel['Ticker'].isin(tickers)].copy()
    expected_cols = list(TECHNICAL_COLS) + list(FIRM_FUNDAMENTAL_COLS)
    missing_cols = [c for c in expected_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError('Missing model inputs: ' + ', '.join(missing_cols))
    print('Subset tickers:', tickers, '| rows:', len(panel), flush=True)
    print('Counts below refer to this subset, not the full-panel audit.', flush=True)
    tl, vl, _, _, (tc, fc) = get_dual_stream_dataloaders(
        panel, seq_len=20, batch_size=64, tech_groups=('tech',),
        fund_groups=('fund',), purge_labels=True, label_horizon=20,
    )
    if len(tl) < 5 or len(vl) < 5:
        raise ValueError('Insufficient batches for the five-step check.')
    # Check actual sequence endpoints, including positional date/ticker alignment.
    ends = cached_label_end_dates(panel, 20)
    lookup = pd.Series(ends.to_numpy(), index=pd.MultiIndex.from_arrays(
        [panel.index, panel['Ticker'].to_numpy()]))
    for loader, boundary in [(tl, '2019-12-31'), (vl, '2021-12-31')]:
        ds = loader.dataset
        keys = pd.MultiIndex.from_arrays([ds.dates, ds.tickers])
        selected_ends = lookup.reindex(keys)
        if selected_ends.isna().any() or (selected_ends > pd.Timestamp(boundary)).any():
            raise AssertionError('Unsafe sequence endpoint survived purge.')

    config = dict(tech_dim=len(tc), fund_dim=len(fc), seq_len=20, d_model=64,
                  n_heads=4, n_layers=2, ffn_dim=256, dropout=0.15,
                  modality='multimodal', fusion_type='cross_attention')
    model = DualEncoderTransformer(**config).to(device)
    # Smoke-only weight; the full experimental loss protocol is reviewed separately.
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    losses = {}
    for split, loader in [('train', tl), ('validation', vl)]:
        model.train(split == 'train')
        split_losses = []
        iterator = iter(loader)
        for step in range(5):
            batch = next(iterator)
            tech = batch['tech_seq'].to(device)
            fund = batch['fund_seq'].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)
            if not all(torch.isfinite(x).all().item() for x in (tech, fund, labels)):
                raise ValueError('Non-finite feature or label detected.')
            with torch.set_grad_enabled(split == 'train'):
                logits = model(tech, fund)
                if logits.shape != labels.shape:
                    raise ValueError(f'Output/label shape mismatch: {logits.shape}, {labels.shape}')
                loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise ValueError('Non-finite loss.')
                if split == 'train':
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
            split_losses.append(float(loss.detach().cpu()))
            print(f'{split} batch {step + 1}/5 | loss={split_losses[-1]:.6f}', flush=True)
        losses[split] = split_losses
    if torch.equal(before, first_parameter.detach()):
        raise AssertionError('Optimizer did not update the checked model parameter.')

    out = Path('results') / ('purge_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='PASS', purpose='integration_smoke_only_not_thesis_results',
                  seed=seed, device=str(device), tickers=[str(t) for t in tickers],
                  train_sequences=len(tl.dataset), validation_sequences=len(vl.dataset),
                  train_batches=5, validation_batches=5, test_evaluated=False,
                  config=config, technical_columns=tc, fundamental_columns=fc,
                  focal_alpha=1.0, focal_gamma=2.0, losses=losses,
                  torch_version=torch.__version__,
                  source_sha256={p: source_hash(p) for p in required[2:]})
    (out / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'config': config,
                'purpose': 'smoke_only_do_not_use_for_evaluation'}, out / 'smoke_only.pt')
    print('PASS | purge endpoints, tensor shapes, finite loss/gradients, optimizer update', flush=True)
    print('Test model evaluation: NOT RUN. Historical result/checkpoint files: unchanged.', flush=True)
    print('Report:', out / 'smoke_report.json', flush=True)


if __name__ == '__main__':
    main()
