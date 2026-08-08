"""
run_late_fusion.py
───────────────────
Üçüncü hipotez: GEÇ FÜZYON (late fusion / stacking).

Şu ana kadar denenen tüm füzyon mekanizmaları (concat, cross-attention, gated,
FiLM) TEMSİL seviyesinde birleşiyor — iki encoder'ın embedding'leri, classifier
görmeden önce karışıyor. Bu script farklı bir hipotezi test ediyor: iki
modalite TAMAMEN AYRI öğrensin, sadece KARAR seviyesinde (olasılık çıktısı)
birleşsin.

YÖNTEM:
    1. tech_only ve fund_only modelleri (DualEncoderTransformer, modality=
       'tech_only'/'fund_only') train set üzerinde bağımsız eğitilir.
       Mimari/hyperparam'lar merdivenle birebir aynı (d_model=64, n_layers=2,
       dropout=0.15, FocalLoss) — karşılaştırma adil kalsın diye.
    2. Her iki model, VALIDATION ve TEST setlerinde olasılık tahmini üretir
       (train set'teki WeightedRandomSampler'dan dolayı train tahminleri
       sırasız/tekrarlı olurdu — bu yüzden meta-learner val'da eğitilir).
    3. Basit bir lojistik regresyon (2 feature: p_tech, p_fund) VALIDATION
       tahminleri üzerinde eğitilir — bu "stacking" adımı.
    4. MCC-optimal threshold yine SADECE val'dan seçilir (metodoloji tutarlı).
    5. Test'te nihai değerlendirme.

NOT — bilinen limit: base modeller zaten val ile early-stop/model-seçimi
yapıyor, meta-learner da val'da fit ediliyor. Ayrı bir üçüncü "stacking fold"
olmadığı için hafif bir bilgi tekrar kullanımı var (test'e leak YOK, sadece
val iki kez kullanılıyor). Bu, projenin train/val/test yapısında pragmatik
bir seçim — dipnot olarak tez metnine yazılmalı.

KULLANIM:
    python run_late_fusion.py                  # 5 seed (önerilen)
    python run_late_fusion.py --seeds 42 43 44

ÇIKTI:
    results/late_fusion_raw.csv     — her seed'in taban model + stacked metrikleri
    results/late_fusion_summary.csv — ortalama ± std özet
    Konsola karşılaştırma tablosu
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import (
    train_pytorch_model, _evaluate_on_loader,
    find_best_threshold_mcc, _compute_classification_metrics,
)
from models.losses import FocalLoss
from run_multiseed import set_seed, get_device, compute_focal_alpha  # reuse seed/device utils


BASE_CONFIG = dict(
    seq_len=20,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.15,
)
EPOCHS = 30
MONITOR = 'pr_auc'


def train_base_model(modality, tech_dim, fund_dim, train_loader, val_loader,
                      test_loader, seed, alpha, device):
    """tech_only veya fund_only modelini eğitir, trained model + val/test olasılıklarını döner."""
    set_seed(seed)
    model = DualEncoderTransformer(
        tech_dim=tech_dim,
        fund_dim=fund_dim,
        modality=modality,
        fusion_type='cross_attention',  # tech/fund_only modda kullanılmıyor, sadece constructor gereksinimi
        **BASE_CONFIG,
    )
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    result = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=f"LateFusion_{modality}_seed{seed}",
        epochs=EPOCHS,
        device=device,
        criterion=criterion,
        monitor=MONITOR,
    )

    trained_model = result['model']
    val_probs, val_targets, _ = _evaluate_on_loader(trained_model, val_loader, device)
    test_probs, test_targets, _ = _evaluate_on_loader(trained_model, test_loader, device)

    return {
        'model': trained_model,
        'val_probs': val_probs, 'val_targets': val_targets,
        'test_probs': test_probs, 'test_targets': test_targets,
        'base_val_metrics': result['val_metrics'],
        'base_test_metrics': result['test_metrics'],
    }


def run_one_seed(train_loader, val_loader, test_loader, tech_dim, fund_dim,
                  seed, alpha, device):
    """Bir seed için: tech_only + fund_only eğit, stack et, metrikleri döndür."""
    print(f"\n  [tech_only] eğitiliyor (seed={seed})...")
    tech = train_base_model('tech_only', tech_dim, fund_dim,
                             train_loader, val_loader, test_loader, seed, alpha, device)

    print(f"\n  [fund_only] eğitiliyor (seed={seed})...")
    fund = train_base_model('fund_only', tech_dim, fund_dim,
                             train_loader, val_loader, test_loader, seed, alpha, device)

    # Val hedefleri iki model için de aynı sırada olmalı (deterministik loader — shuffle=False)
    assert np.array_equal(tech['val_targets'], fund['val_targets']), \
        "val_targets uyuşmuyor — loader sırası deterministik olmalıydı"
    assert np.array_equal(tech['test_targets'], fund['test_targets']), \
        "test_targets uyuşmuyor"

    y_val = tech['val_targets'].astype(int)
    y_test = tech['test_targets'].astype(int)

    # ── Stacking: lojistik regresyon, 2 feature (p_tech, p_fund), val'da fit ──
    X_val_meta = np.column_stack([tech['val_probs'], fund['val_probs']])
    X_test_meta = np.column_stack([tech['test_probs'], fund['test_probs']])

    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(X_val_meta, y_val)

    stacked_val_probs = stacker.predict_proba(X_val_meta)[:, 1]
    stacked_test_probs = stacker.predict_proba(X_test_meta)[:, 1]

    # MCC-optimal threshold SADECE val'dan (metodoloji tutarlı)
    best_thresh, _ = find_best_threshold_mcc(y_val, stacked_val_probs)

    stacked_val_metrics = _compute_classification_metrics(stacked_val_probs, y_val, best_thresh)
    stacked_test_metrics = _compute_classification_metrics(stacked_test_probs, y_test, best_thresh)

    row = {
        'seed': seed,
        'threshold': best_thresh,
        'stacker_coef_tech': float(stacker.coef_[0][0]),
        'stacker_coef_fund': float(stacker.coef_[0][1]),
        'stacker_intercept': float(stacker.intercept_[0]),
    }
    for split, m in [('val', stacked_val_metrics), ('test', stacked_test_metrics)]:
        for k, v in m.items():
            row[f'stacked_{split}_{k}'] = v
    for split, m in [('val', tech['base_val_metrics']), ('test', tech['base_test_metrics'])]:
        for k, v in m.items():
            row[f'tech_only_{split}_{k}'] = v
    for split, m in [('val', fund['base_val_metrics']), ('test', fund['base_test_metrics'])]:
        for k, v in m.items():
            row[f'fund_only_{split}_{k}'] = v

    return row


def print_report(df):
    print("\n" + "═" * 78)
    print("LATE FUSION (STACKING) SONUÇLARI")
    print("═" * 78)

    metrics_to_show = [
        ('stacked_test_mcc', 'STACKED — TEST MCC'),
        ('tech_only_test_mcc', 'tech_only — TEST MCC (referans)'),
        ('fund_only_test_mcc', 'fund_only — TEST MCC (referans)'),
        ('stacked_test_pr_auc', 'STACKED — TEST PR-AUC'),
    ]
    for col, title in metrics_to_show:
        if col not in df.columns:
            continue
        print(f"\n── {title} ──")
        print(f"  ortalama: {df[col].mean():.4f}  ±std: {df[col].std(ddof=1):.4f}  "
              f"min: {df[col].min():.4f}  max: {df[col].max():.4f}  n={len(df)}")

    print(f"\n── STACKER KATSAYILARI (öğrenilmiş ağırlıklar) ──")
    print(f"  tech katsayısı:  {df['stacker_coef_tech'].mean():+.4f} (±{df['stacker_coef_tech'].std(ddof=1):.4f})")
    print(f"  fund katsayısı:  {df['stacker_coef_fund'].mean():+.4f} (±{df['stacker_coef_fund'].std(ddof=1):.4f})")
    print(f"  (büyük pozitif katsayı → o modalite kararda daha ağır basıyor)")

    print("\n" + "═" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--force', action='store_true',
                    help='Var olan late_fusion_raw.csv\'yi yok say, sıfırdan başlat')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'late_fusion_raw.csv')

    # ── RESUME: bağlantı kopması / kesinti durumunda kaldığı yerden devam ──
    # late_fusion_raw.csv zaten var ve içinde tamamlanmış seed'ler varsa,
    # onları atla — sadece eksik seed'leri koştur. --force ile tamamen
    # sıfırdan başlatılabilir.
    rows = []
    if os.path.exists(raw_path) and not args.force:
        existing = pd.read_csv(raw_path)
        completed_seeds = set(existing['seed'].tolist())
        rows = existing.to_dict('records')
        skip = [s for s in args.seeds if s in completed_seeds]
        args_seeds_remaining = [s for s in args.seeds if s not in completed_seeds]
        if skip:
            print(f"  [RESUME] {raw_path} bulundu — zaten tamamlanmış seed'ler atlanıyor: {skip}")
        args.seeds = args_seeds_remaining
        if not args.seeds:
            print("  Tüm seed'ler zaten tamamlanmış. --force ile sıfırdan başlatabilirsin.")
            df = pd.DataFrame(rows)
            print_report(df)
            return

    print("═" * 78)
    print("LATE FUSION (STACKING) — çok-seed doğrulama")
    print("═" * 78)
    print(f"  Seed'ler (kalan): {args.seeds}")
    print(f"  Base config: {BASE_CONFIG}")
    print(f"  Çıktı: {raw_path}\n")

    print("[1/3] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    print("[2/3] Dataloader'lar kuruluyor (bir kez, tüm seed'lerde paylaşılacak)...")
    set_seed(0)
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=BASE_CONFIG['seq_len'],
                                    batch_size=args.batch_size)

    alpha = compute_focal_alpha(train_loader)
    device = get_device()
    print(f"       tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | "
          f"FocalLoss alpha={alpha:.2f} | device={device}")

    print(f"[3/3] {len(args.seeds)} seed başlıyor...\n")
    for i, seed in enumerate(args.seeds, 1):
        print("\n" + "▄" * 78)
        print(f"SEED {i}/{len(args.seeds)} — seed={seed}")
        print("▄" * 78)
        t0 = time.time()
        try:
            row = run_one_seed(train_loader, val_loader, test_loader,
                               len(tech_cols), len(fund_cols), seed, alpha, device)
            row['minutes'] = round((time.time() - t0) / 60, 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"\n  ✓ seed={seed} | stacked test MCC={row.get('stacked_test_mcc', float('nan')):.4f} "
                  f"| {row['minutes']} dk | kaydedildi → {raw_path}")
        except Exception as e:
            print(f"\n  ✗ SEED BAŞARISIZ (seed={seed}): {e}")
            import traceback
            traceback.print_exc()

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    if not rows:
        print("\nHiçbir seed tamamlanamadı.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    print_report(df)

    summ_path = os.path.join(args.outdir, 'late_fusion_summary.csv')
    summary_rows = []
    for col in ['stacked_test_mcc', 'stacked_test_pr_auc', 'tech_only_test_mcc', 'fund_only_test_mcc']:
        if col in df.columns:
            summary_rows.append({
                'metric': col, 'mean': df[col].mean(), 'std': df[col].std(ddof=1),
                'min': df[col].min(), 'max': df[col].max(), 'n': len(df),
            })
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)
    print(f"\nÖzet kaydedildi → {summ_path}")
    print(f"Ham sonuçlar     → {raw_path}")
    print(f"\nToplam süre: {df['minutes'].sum():.0f} dakika")


if __name__ == '__main__':
    main()