"""
export_preds.py — tüm model tahminlerini TEK ve İNDEKSLİ şemaya getirir.

SORUN
    results/preds/ altında iki farklı şema var:
      eski (MS_*, TUNED_*) : yalnızca 'test' ve 'val' olasılık dizileri
      yeni (WL_*)          : 'dates', 'tickers', 'prob', 'target'

    Eski dosyalarda (tarih, hisse) anahtarı yok. Bu yüzden ne ortak
    örneklem kurulabiliyor ne de yeni portföy motoruna beslenebiliyor.
    Ayrıca hangi tablonun hangi tahminden geldiği izlenemiyor.

BU SCRIPT
    Checkpoint'lerden yeniden skorlar ve HERKESİ yeni şemaya yazar:
        results/preds_v2/<hücre>_seed<N>.npz
        anahtarlar: dates, tickers, prob, target, split

    Skorlama yolu run_portfolio_simulation.py ile birebir aynıdır
    (aynı dataloader, aynı _evaluate_on_loader), dolayısıyla mevcut
    sayılarla tutarlıdır — sadece indeks ekleniyor.

KULLANIM
    python export_preds.py --ablations multi_pure multi_text tech_only
    python export_preds.py --ablations multi_pure --seeds 42 43 44 45 46
    python export_preds.py --list          # mevcut checkpoint'leri göster
"""
import os, glob, argparse
import numpy as np, pandas as pd, torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import _evaluate_on_loader
from run_modality_v2 import ABLATIONS, BASE_CONFIG, fusion_for

OUTDIR = os.path.join('results', 'preds_v2')


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ckpt_path(ablation, seed):
    return os.path.join('checkpoints', f'best_v2_{ablation}_seed{seed}.pth'.lower())


def list_checkpoints():
    print("mevcut checkpoint'ler:")
    found = {}
    for p in sorted(glob.glob(os.path.join('checkpoints', 'best_v2_*.pth'))):
        b = os.path.basename(p)[len('best_v2_'):-len('.pth')]
        cell, _, seed = b.rpartition('_seed')
        found.setdefault(cell, []).append(seed)
    for c, s in sorted(found.items()):
        mark = '✓' if c in ABLATIONS else '?'
        print(f"  {mark} {c:<22} seed: {', '.join(sorted(s))}")
    if not found:
        print("  (hiç bulunamadı)")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablations', nargs='+', default=['multi_pure'])
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--splits', nargs='+', default=['test', 'val'],
                    choices=['test', 'val'])
    args = ap.parse_args()

    if args.list:
        list_checkpoints(); return

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    print(f"cihaz: {device}")

    ds_all = prepare_dataset(force_refresh=False)
    loader_cache = {}
    written = 0

    for ab in args.ablations:
        if ab not in ABLATIONS:
            print(f"⚠️ bilinmeyen hücre: {ab} — atlandı"); continue
        tech_groups, fund_groups, modality, _ = ABLATIONS[ab]
        key = (tuple(tech_groups), tuple(fund_groups))
        if key not in loader_cache:
            _, vl, tl, _, (tc, fc) = get_dual_stream_dataloaders(
                ds_all, seq_len=BASE_CONFIG['seq_len'],
                batch_size=args.batch_size,
                tech_groups=tech_groups, fund_groups=fund_groups)
            loader_cache[key] = (vl, tl, tc, fc)
        vl, tl, tech_cols, fund_cols = loader_cache[key]

        for seed in args.seeds:
            ck = ckpt_path(ab, seed)
            if not os.path.exists(ck):
                continue
            mdl = DualEncoderTransformer(
                tech_dim=len(tech_cols), fund_dim=len(fund_cols),
                seq_len=BASE_CONFIG['seq_len'], d_model=BASE_CONFIG['d_model'],
                n_heads=BASE_CONFIG['n_heads'], n_layers=BASE_CONFIG['n_layers'],
                dropout=BASE_CONFIG['dropout'], modality=modality,
                fusion_type=fusion_for(ab))
            mdl.load_state_dict(torch.load(ck, map_location=device,
                                           weights_only=True))
            mdl.to(device).eval()

            parts = []
            for split, loader in (('test', tl), ('val', vl)):
                if split not in args.splits:
                    continue
                dset = loader.dataset
                p, _, _ = _evaluate_on_loader(mdl, loader, device)
                p = np.asarray(p, dtype=np.float32).ravel()
                d = np.asarray(dset.dates.values)
                t = np.asarray(dset.tickers)
                y = np.asarray(dset.labels, dtype=np.float32)
                assert len(p) == len(d) == len(t) == len(y), (
                    f"{ab} seed{seed} {split}: uzunluk uyuşmuyor "
                    f"({len(p)}, {len(d)}, {len(t)}, {len(y)})")
                parts.append(pd.DataFrame({'dates': d, 'tickers': t,
                                           'prob': p, 'target': y,
                                           'split': split}))
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            out = os.path.join(args.outdir, f'{ab}_seed{seed}.npz')
            np.savez_compressed(
                out,
                dates=df['dates'].values.astype('datetime64[ns]'),
                tickers=df['tickers'].values.astype(str),
                prob=df['prob'].values.astype(np.float32),
                target=df['target'].values.astype(np.float32),
                split=df['split'].values.astype(str))
            written += 1
            n_te = int((df.split == 'test').sum())
            print(f"  ✓ {ab} seed={seed} → {os.path.basename(out)} "
                  f"({n_te:,} test satırı)")

            del mdl
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"\n{written} dosya yazıldı → {args.outdir}/")
    if written:
        print("\nSonraki adım:")
        print(f'  python run_portfolio_v2.py --preds-dir {args.outdir} '
              f'--pattern "multi_pure_seed*.npz" --pct-sweep --cash-variants')
        print(f'  python run_common_sample.py --preds-dir {args.outdir} '
              f'--pattern "multi_pure_seed*.npz"')


if __name__ == '__main__':
    main()