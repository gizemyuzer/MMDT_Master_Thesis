"""
analyze_cluster_robust_ci.py
──────────────────────────────
Cross-sectional bağımlılık kontrolü.

SORUN: Test setindeki gözlemler i.i.d. değil. 395 hissenin sequence'leri
aynı takvim günlerini paylaşıyor — sistemik bir şok (ör. 2022 bear market),
aynı tarihte birçok hissede eşzamanlı drawdown yaratır. Standart bootstrap
(gözlemleri TEK TEK resample eder) bu bağımlılığı görmezden gelir ve
MCC'nin güven aralığını olduğundan DAR gösterebilir — yani istatistiksel
kesinlik iddiası yanıltıcı olabilir.

YÖNTEM: MCC'nin %95 güven aralığını iki farklı bootstrap ile hesapla:
  1. i.i.d. bootstrap      — gözlemleri tek tek resample eder (YANLIŞ varsayım)
  2. Date-block bootstrap  — TARİHLERİ resample eder, o tarihteki TÜM
                              gözlemler birlikte alınır (DOĞRU: cross-sectional
                              bağımlılığı hesaba katar)

İki CI arasındaki fark, i.i.d. varsayımının ne kadar iyimser olduğunu gösterir.

KULLANIM:
    python analyze_cluster_robust_ci.py \\
        --checkpoint checkpoints/best_dualencoder_gated_cross_attention.pth \\
        --fusion gated_cross_attention

ÇIKTI:
    results/cluster_robust_ci.csv
    Konsola karşılaştırmalı rapor
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import _evaluate_on_loader, find_best_threshold_mcc


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def iid_bootstrap_mcc(preds, targets, threshold, n_boot=2000, seed=0):
    """Gözlem bazlı (i.i.d. varsayımlı) bootstrap — YANLIŞ varsayım, referans için."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    y_pred_all = (preds >= threshold).astype(int)
    mccs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = targets[idx], y_pred_all[idx]
        if len(set(yt)) < 2 or len(set(yp)) < 2:
            continue
        mccs.append(matthews_corrcoef(yt, yp))
    return np.array(mccs)


def date_block_bootstrap_mcc(preds, targets, dates, threshold, n_boot=2000, seed=0):
    """
    Tarih bazlı blok bootstrap — DOĞRU yöntem. Her iterasyonda benzersiz
    tarihler kendi aralarında (replacement ile) resample edilir; bir tarih
    seçildiğinde o tarihteki TÜM hisseler birlikte alınır. Böylece aynı
    günün sistemik şoku tek bir "birim" olarak resample edilir, gözlemler
    arası cross-sectional korelasyon korunur.
    """
    rng = np.random.default_rng(seed)
    y_pred_all = (preds >= threshold).astype(int)
    unique_dates = np.unique(dates)

    date_to_idx = {d: np.where(dates == d)[0] for d in unique_dates}

    mccs = []
    for _ in range(n_boot):
        sampled_dates = rng.choice(unique_dates, len(unique_dates), replace=True)
        idx = np.concatenate([date_to_idx[d] for d in sampled_dates])
        yt, yp = targets[idx], y_pred_all[idx]
        if len(set(yt)) < 2 or len(set(yp)) < 2:
            continue
        mccs.append(matthews_corrcoef(yt, yp))
    return np.array(mccs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='Değerlendirilecek eğitimli model checkpoint (.pth)')
    ap.add_argument('--fusion', type=str, default='gated_cross_attention',
                    choices=['concat', 'cross_attention', 'gated_cross_attention', 'film'])
    ap.add_argument('--modality', type=str, default='multimodal',
                    choices=['tech_only', 'fund_only', 'multimodal'])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("[1/4] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    print("[2/4] Dataloader'lar kuruluyor...")
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=args.batch_size)

    print(f"[3/4] Model yükleniyor: {args.checkpoint}")
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols), fund_dim=len(fund_cols),
        seq_len=20, d_model=64, n_heads=4, n_layers=2, dropout=0.15,
        modality=args.modality, fusion_type=args.fusion,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # Val'dan threshold seç — metodoloji tutarlı, test'e leak yok
    val_preds, val_targets, _ = _evaluate_on_loader(model, val_loader, device)
    threshold, _ = find_best_threshold_mcc(val_targets, val_preds)
    print(f"      MCC-optimal threshold (val'dan): {threshold:.4f}")

    print("[4/4] Test setinde tahminler + bootstrap...")
    test_preds, test_targets, _ = _evaluate_on_loader(model, test_loader, device)
    test_dates = test_loader.dataset.dates.values

    point_mcc = matthews_corrcoef(test_targets.astype(int), (test_preds >= threshold).astype(int))
    n_unique_dates = len(np.unique(test_dates))
    print(f"\n  Nokta tahmini test MCC: {point_mcc:.4f}")
    print(f"  Test seti: {len(test_targets):,} gözlem | {n_unique_dates:,} benzersiz tarih")
    print(f"  Ortalama gözlem/tarih: {len(test_targets) / n_unique_dates:.1f}  "
          f"(1'den büyükse i.i.d. varsayımı şüpheli)")

    print(f"\n  Bootstrap koşuluyor (n_boot={args.n_boot})...")
    iid_dist = iid_bootstrap_mcc(test_preds, test_targets, threshold, n_boot=args.n_boot)
    block_dist = date_block_bootstrap_mcc(test_preds, test_targets, test_dates, threshold, n_boot=args.n_boot)

    iid_ci = np.percentile(iid_dist, [2.5, 97.5])
    block_ci = np.percentile(block_dist, [2.5, 97.5])

    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    print(f"  Nokta tahmini MCC                    : {point_mcc:.4f}")
    print(f"\n  [YANLIŞ VARSAYIM] i.i.d. bootstrap (gözlem bazlı):")
    print(f"    %95 CI  : [{iid_ci[0]:.4f}, {iid_ci[1]:.4f}]  (genişlik: {iid_ci[1] - iid_ci[0]:.4f})")
    print(f"    std     : {iid_dist.std():.4f}")
    print(f"\n  [DOĞRU] Date-block bootstrap (tarih bazlı, cross-sectional bağımlılık dahil):")
    print(f"    %95 CI  : [{block_ci[0]:.4f}, {block_ci[1]:.4f}]  (genişlik: {block_ci[1] - block_ci[0]:.4f})")
    print(f"    std     : {block_dist.std():.4f}")

    widening = (block_ci[1] - block_ci[0]) / max(iid_ci[1] - iid_ci[0], 1e-9)
    print(f"\n  Block-bootstrap CI, i.i.d. CI'dan {widening:.2f}x daha geniş.")
    if widening > 1.3:
        print("  → Cross-sectional bağımlılık istatistiksel olarak önemli; i.i.d. varsayımı")
        print("    kesinliği olduğundan fazla gösteriyordu. Tez metninde block-bootstrap")
        print("    CI'ları raporla, i.i.d. CI'ları DEĞİL.")
    else:
        print("  → Fark küçük; bu test setinde cross-sectional bağımlılığın CI genişliğine")
        print("    etkisi sınırlı görünüyor — i.i.d. varsayımı burada makul bir yaklaşım.")
    print("=" * 70)

    out = pd.DataFrame([
        {'method': 'point_estimate', 'mcc': point_mcc, 'ci_low': np.nan, 'ci_high': np.nan},
        {'method': 'iid_bootstrap', 'mcc': iid_dist.mean(), 'ci_low': iid_ci[0], 'ci_high': iid_ci[1]},
        {'method': 'date_block_bootstrap', 'mcc': block_dist.mean(), 'ci_low': block_ci[0], 'ci_high': block_ci[1]},
    ])
    out_path = os.path.join(args.outdir, 'cluster_robust_ci.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSonuçlar kaydedildi → {out_path}")


if __name__ == '__main__':
    main()