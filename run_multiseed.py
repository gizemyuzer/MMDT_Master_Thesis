"""
run_multiseed.py
────────────────
Ablation merdivenini (concat → cross-attention → gated) VE iki ek mimari
hipotezini (static_context, film) birden fazla seed ile koşturur, ortalama
± standart sapma raporlar.

NEDEN GEREKLİ:
    Mevcut kodda PyTorch tarafında hiç seed yok. Yani model ağırlık
    başlatması, dropout maskeleri ve WeightedRandomSampler çekimleri her
    koşuda farklı. Tek koşuluk sonuçlar TEKRARLANABİLİR DEĞİL.

    Gated'in cross-attention'a üstünlüğü tek koşuda 0.006 idi. Bu fark
    seed gürültüsünün içinde kalabilir. Multi-seed olmadan "gated daha iyi"
    denemez.

MİMARİLER (--fusions ile seçilir):
    concat / cross_attention / gated_cross_attention  — orijinal merdiven
    film             — FiLM koşullandırma (Perez 2018): fundamental, technical
                        embedding'i gamma/beta ile ölçekler. Cross-attention'dan
                        daha kısıtlı — token-seviyesi attention yok.
    static_context   — DualStreamRiskModel (models/transformer_model.py).
                        Fundamental'i her timestep'te değil, sequence başına TEK
                        statik context vektörü olarak kullanır (çeyreklik veri
                        günlük gibi davranmasın diye). Daha önce kodlanmış ama
                        hiç multi-seed ile doğrulanmamıştı.
    Tüm mimariler AYNI FocalLoss + epochs=30 ile eğitilir — fark sadece
    mimariden gelsin, kayıp fonksiyonundan değil.

KULLANIM:
    python run_multiseed.py                    # merdiven, 5 seed (önerilen)
    python run_multiseed.py --seeds 42 43 44   # 3 seed (daha hızlı)
    python run_multiseed.py --fusions concat cross_attention gated_cross_attention
    python run_multiseed.py --fusions film static_context   # yeni hipotezler

ÇIKTI:
    results/multiseed_raw.csv      — her koşunun tüm metrikleri
    results/multiseed_summary.csv  — ortalama ± std özet
    Konsola karşılaştırma tablosu
"""
import os
import sys
import time
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer, DualStreamRiskModel
from models.pytorch_trainer import train_pytorch_model
from models.losses import FocalLoss


# ══════════════════════════════════════════════════════════════════
# Seed yönetimi
# ══════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    """
    Tüm rastgelelik kaynaklarını sabitler.

    Kapsam:
      - torch: model ağırlık başlatması, dropout maskeleri
      - torch (CUDA/MPS): cihaz üzerindeki işlemler
      - WeightedRandomSampler: torch'un global RNG'sini kullanır,
        dolayısıyla torch.manual_seed ile kontrol edilir
      - numpy, python random: yardımcı hesaplar
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_focal_alpha(train_loader, max_batches=200):
    """
    Sınıf dengesizliğinden focal loss alpha'sı hesaplar.

    ÖNEMLİ: Bu fonksiyon seed döngüsünün DIŞINDA bir kez çağrılır.
    Alpha verinin bir özelliği, seed'in değil — her koşuda aynı olmalı.
    Ayrıca loader'ı tüketmek RNG durumunu ilerletir, bu da seed
    kontrolünü bozardı.
    """
    pos = neg = 0
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        lbl = batch['label']
        pos += int((lbl == 1).sum())
        neg += int((lbl == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


# ══════════════════════════════════════════════════════════════════
# Merdiven config — üç füzyon tipi de BİREBİR aynı
# ══════════════════════════════════════════════════════════════════
LADDER_CONFIG = dict(
    seq_len=20,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.15,
    modality='multimodal',
)
EPOCHS = 30
MONITOR = 'pr_auc'

FUSION_LABELS = {
    'concat':                'concat (statik)',
    'cross_attention':       'cross-attention (dinamik)',
    'gated_cross_attention': 'gated (kapılı)',
    'film':                  'FiLM (koşullandırma)',
    'static_context':        'static-context (DualStreamRiskModel)',
}

# DualEncoderTransformer'ın anladığı fusion_type'lar (LADDER_CONFIG ile uyumlu)
_DUAL_ENCODER_FUSIONS = {'concat', 'cross_attention', 'gated_cross_attention', 'film'}


def build_model(fusion_type, tech_dim, fund_dim):
    """Mimari adına göre doğru model sınıfını inşa eder."""
    if fusion_type in _DUAL_ENCODER_FUSIONS:
        return DualEncoderTransformer(
            tech_dim=tech_dim,
            fund_dim=fund_dim,
            fusion_type=fusion_type,
            **LADDER_CONFIG,
        )
    if fusion_type == 'static_context':
        # LADDER_CONFIG'teki d_model/n_heads ile hizalı ama kendi n_layers=2
        # (Transformer stream'i) sabit — DualStreamRiskModel'in kendi tasarımı.
        return DualStreamRiskModel(
            tech_input_dim=tech_dim,
            fund_input_dim=fund_dim,
            seq_len=LADDER_CONFIG['seq_len'],
            hidden_dim=LADDER_CONFIG['d_model'],
            num_heads=LADDER_CONFIG['n_heads'],
            dropout=LADDER_CONFIG['dropout'],
        )
    raise ValueError(f"Bilinmeyen fusion_type: {fusion_type}")


def run_one(train_loader, val_loader, test_loader,
            tech_dim, fund_dim, fusion_type, seed, alpha, device):
    """Tek bir (mimari, seed) kombinasyonunu eğitir ve metrikleri döndürür."""
    set_seed(seed)          # ← model init + dropout + sampler çekimleri

    model = build_model(fusion_type, tech_dim, fund_dim)
    # AYNI kayıp fonksiyonu her mimaride — fark mimariden gelsin, loss'tan değil
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    t0 = time.time()
    result = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=f"MS_{fusion_type}_seed{seed}",
        epochs=EPOCHS,
        device=device,
        criterion=criterion,
        monitor=MONITOR,
    )
    elapsed = time.time() - t0

    row = {
        'fusion': fusion_type,
        'seed': seed,
        'minutes': round(elapsed / 60, 1),
        'n_params': sum(p.numel() for p in model.parameters()),
    }
    for split, m in [('val', result['val_metrics']), ('test', result['test_metrics'])]:
        if m is None:
            continue
        for k, v in m.items():
            row[f'{split}_{k}'] = v
    row['threshold'] = result['threshold']

    # Gated ise kapı değerlerini, FiLM ise gamma/beta'yı kaydet (yorumlanabilirlik)
    if hasattr(model, 'last_gate_t'):
        row['gate_t'] = model.last_gate_t
        row['gate_f'] = model.last_gate_f
    if hasattr(model, 'last_film_gamma'):
        row['film_gamma'] = model.last_film_gamma
        row['film_beta'] = model.last_film_beta

    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    return row


# ══════════════════════════════════════════════════════════════════
# Raporlama
# ══════════════════════════════════════════════════════════════════
def summarize(df, metric='test_mcc'):
    """Füzyon tipi bazında ortalama ± std tablosu."""
    g = df.groupby('fusion')[metric]
    out = pd.DataFrame({
        'mean': g.mean(),
        'std': g.std(ddof=1),
        'min': g.min(),
        'max': g.max(),
        'n': g.count(),
    })
    return out


def print_report(df):
    print("\n" + "═" * 78)
    print("MULTI-SEED SONUÇLARI")
    print("═" * 78)

    order = [f for f in ['concat', 'cross_attention', 'gated_cross_attention',
                         'film', 'static_context']
             if f in df['fusion'].unique()]

    for metric, title in [('test_mcc', 'TEST MCC'),
                          ('test_pr_auc', 'TEST PR-AUC'),
                          ('val_mcc', 'VAL MCC')]:
        if metric not in df.columns:
            continue
        s = summarize(df, metric)
        print(f"\n── {title} ──")
        print(f"{'Füzyon':<28} {'ortalama':>9} {'±std':>8} {'min':>8} {'max':>8} {'n':>3}")
        print("-" * 70)
        for f in order:
            if f not in s.index:
                continue
            r = s.loc[f]
            print(f"{FUSION_LABELS.get(f, f):<28} {r['mean']:>9.4f} "
                  f"{r['std']:>8.4f} {r['min']:>8.4f} {r['max']:>8.4f} {int(r['n']):>3}")

    # ── Genelleme farkı ──
    if {'val_mcc', 'test_mcc'}.issubset(df.columns):
        df = df.copy()
        df['gap'] = df['test_mcc'] - df['val_mcc']
        s = summarize(df, 'gap')
        print(f"\n── GENELLEME FARKI (test MCC − val MCC) ──")
        print(f"{'Füzyon':<28} {'ortalama':>9} {'±std':>8}")
        print("-" * 47)
        for f in order:
            if f not in s.index:
                continue
            r = s.loc[f]
            print(f"{FUSION_LABELS.get(f, f):<28} {r['mean']:>+9.4f} {r['std']:>8.4f}")

    # ── Eşleştirilmiş karşılaştırma: gated vs cross-attention ──
    if {'gated_cross_attention', 'cross_attention'}.issubset(set(df['fusion'])):
        print(f"\n── EŞLEŞTİRİLMİŞ KARŞILAŞTIRMA (aynı seed'lerde) ──")
        piv = df.pivot_table(index='seed', columns='fusion', values='test_mcc')
        if {'gated_cross_attention', 'cross_attention'}.issubset(piv.columns):
            diff = piv['gated_cross_attention'] - piv['cross_attention']
            wins = int((diff > 0).sum())
            n = len(diff)
            print(f"  Seed bazında fark (gated − cross-attention):")
            for sd, d in diff.items():
                mark = "✓" if d > 0 else "✗"
                print(f"    seed {sd}: {d:+.4f}  {mark}")
            print(f"\n  Gated {wins}/{n} seed'de önde | "
                  f"ortalama fark: {diff.mean():+.4f} (±{diff.std(ddof=1):.4f})")
            if n >= 3:
                # Basit etki büyüklüğü — t-testi için n çok küçük olabilir
                if diff.std(ddof=1) > 1e-9:
                    cohen_d = diff.mean() / diff.std(ddof=1)
                    print(f"  Etki büyüklüğü (Cohen's d, eşleştirilmiş): {cohen_d:+.2f}")
                if wins == n:
                    print(f"  → Tüm seed'lerde tutarlı üstünlük")
                elif wins == 0:
                    print(f"  → Hiçbir seed'de üstün değil")
                else:
                    print(f"  → Tutarsız — fark muhtemelen gürültü içinde")

    # ── Kapı değerleri ──
    if 'gate_t' in df.columns and df['gate_t'].notna().any():
        gd = df[df['gate_t'].notna()]
        print(f"\n── KAPI DEĞERLERİ (gated) ──")
        print(f"  tech ← fund: {gd['gate_t'].mean():.4f} (±{gd['gate_t'].std(ddof=1):.4f})")
        print(f"  fund ← tech: {gd['gate_f'].mean():.4f} (±{gd['gate_f'].std(ddof=1):.4f})")
        print(f"  (başlangıç 0.5 — sapma öğrenilmiş davranışı gösterir)")

    # ── FiLM gamma/beta ──
    if 'film_gamma' in df.columns and df['film_gamma'].notna().any():
        fd = df[df['film_gamma'].notna()]
        print(f"\n── FiLM PARAMETRELERİ ──")
        print(f"  gamma (ölçek): {fd['film_gamma'].mean():.4f} (±{fd['film_gamma'].std(ddof=1):.4f})")
        print(f"  beta (kaydırma): {fd['film_beta'].mean():.4f} (±{fd['film_beta'].std(ddof=1):.4f})")
        print(f"  (başlangıç gamma=1, beta=0 — sapma öğrenilmiş davranışı gösterir)")

    print("\n" + "═" * 78)


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                    help='Kullanılacak seed listesi (varsayılan: 5 seed)')
    ap.add_argument('--fusions', type=str, nargs='+',
                    default=['concat', 'cross_attention', 'gated_cross_attention'],
                    help="Test edilecek mimariler: concat, cross_attention, "
                         "gated_cross_attention, film, static_context")
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--force', action='store_true',
                    help='Var olan multiseed_raw.csv\'yi yok say, sıfırdan başlat')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'multiseed_raw.csv')

    # ── RESUME: bağlantı kopması / kesinti durumunda kaldığı yerden devam ──
    rows = []
    completed_pairs = set()
    if os.path.exists(raw_path) and not args.force:
        existing = pd.read_csv(raw_path)
        completed_pairs = set(zip(existing['fusion'], existing['seed']))
        rows = existing.to_dict('records')
        if completed_pairs:
            print(f"  [RESUME] {raw_path} bulundu — {len(completed_pairs)} koşu zaten tamamlanmış, atlanacak.")

    n_runs = sum(1 for f in args.fusions for s in args.seeds if (f, s) not in completed_pairs)
    if n_runs == 0:
        print("  Tüm (fusion, seed) kombinasyonları zaten tamamlanmış. --force ile sıfırdan başlatabilirsin.")
        df = pd.DataFrame(rows)
        print_report(df)
        return
    print("═" * 78)
    print("MULTI-SEED ABLATION MERDİVENİ")
    print("═" * 78)
    print(f"  Füzyon tipleri : {args.fusions}")
    print(f"  Seed'ler       : {args.seeds}")
    print(f"  Toplam koşu    : {n_runs}")
    print(f"  Config         : d_model={LADDER_CONFIG['d_model']}, "
          f"n_layers={LADDER_CONFIG['n_layers']}, "
          f"dropout={LADDER_CONFIG['dropout']}, epochs={EPOCHS}")
    print(f"  Çıktı          : {raw_path}")
    print()

    # ── Veri: cache'ten oku, YENİDEN ÇEKME ──
    print("[1/3] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    # ── Loader'ları BİR KEZ kur ──
    # Sequence dataset deterministik (sadece veriye bağlı). Rastgelelik
    # sampler'ın ÇEKİMLERİNDE, o da her koşudan önce set_seed ile
    # kontrol ediliyor. Yani bir kez kurup tekrar kullanmak güvenli
    # ve 750K sequence'i her seferinde yeniden kurmaktan çok daha hızlı.
    print("[2/3] Dataloader'lar kuruluyor (bir kez, tüm koşularda paylaşılacak)...")
    set_seed(0)
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=LADDER_CONFIG['seq_len'],
                                    batch_size=args.batch_size)

    # Alpha'yı bir kez hesapla — verinin özelliği, seed'in değil
    alpha = compute_focal_alpha(train_loader)
    device = get_device()
    print(f"       tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | "
          f"FocalLoss alpha={alpha:.2f} | device={device}")

    # ── Koşular ──
    print(f"[3/3] {n_runs} koşu başlıyor...\n")
    run_i = 0
    for fusion in args.fusions:
        for seed in args.seeds:
            if (fusion, seed) in completed_pairs:
                continue
            run_i += 1
            print("\n" + "▄" * 78)
            print(f"KOŞU {run_i}/{n_runs} — fusion={fusion} | seed={seed}")
            print("▄" * 78)
            try:
                row = run_one(train_loader, val_loader, test_loader,
                              len(tech_cols), len(fund_cols),
                              fusion, seed, alpha, device)
                rows.append(row)
                # Her koşudan sonra kaydet — çökerse veri kaybolmasın
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print(f"\n  ✓ seed={seed} | test MCC={row.get('test_mcc', float('nan')):.4f} "
                      f"| {row['minutes']} dk | kaydedildi → {raw_path}")
            except Exception as e:
                print(f"\n  ✗ KOŞU BAŞARISIZ (fusion={fusion}, seed={seed}): {e}")
                import traceback
                traceback.print_exc()

    if not rows:
        print("\nHiçbir koşu tamamlanamadı.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)

    # Özet
    print_report(df)

    summ_path = os.path.join(args.outdir, 'multiseed_summary.csv')
    parts = []
    for metric in ['test_mcc', 'test_pr_auc', 'test_f1', 'val_mcc']:
        if metric in df.columns:
            s = summarize(df, metric)
            s['metric'] = metric
            parts.append(s.reset_index())
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(summ_path, index=False)
        print(f"\nÖzet kaydedildi → {summ_path}")
    print(f"Ham sonuçlar     → {raw_path}")
    print(f"\nToplam süre: {df['minutes'].sum():.0f} dakika")


if __name__ == '__main__':
    main()