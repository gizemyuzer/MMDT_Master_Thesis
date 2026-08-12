"""
run_modality_v2.py
───────────────────
FUNDAMENTAL / MAKRO AYRIMI SONRASI çok-seed ablation merdiveni.

═══════════════════════════════════════════════════════════════════════
NEDEN BU DENEY
═══════════════════════════════════════════════════════════════════════
Önceki tüm koşularda "fund_only" aslında FUNDAMENTAL DEĞİLDİ: 19 kolonun
sadece 6'sı firma-spesifik muhasebe oranıydı; 9'u makro seri, 4'ü mikro-makro
çarpımıydı. Yani "fundamental modalite işe yaramıyor" sonucumuz, aslında
büyük ölçüde makro hakkındaydı.

Kritik ayrım: MAKRO kolonlar belirli bir günde tüm hisseler için AYNI değeri
alır → kesitsel ayrım güçleri sıfırdır. "Piyasa ne zaman düşecek"i öğrenirler,
"hangi hisse düşecek"i değil. Val dönemi (2020-21 COVID) tek ortak şok olduğu
için bu değişkenler orada parlıyor, test döneminde (2022-24, firmaya özgü
düşüşler) çöküyor. fund_only'nin val 0.26 → test −0.018 çöküşünün açıklaması
büyük ihtimalle bu.

Bu script iki soruyu ayrı ayrı yanıtlar:
  (1) TEŞHİS  — eski fund_only sonucunu sürükleyen makro muydu, bilanço mu?
  (2) TEDAVİ  — firma değişkenlerini KESİTSEL normalize edersek (tarih-içi
                yüzdelik dilim) fundamental akış gerçek ayrım gücü kazanır mı?

═══════════════════════════════════════════════════════════════════════
ABLATION'LAR
═══════════════════════════════════════════════════════════════════════
  macro_only        fund akışı = 9 makro   → TEŞHİS: eski fund_only'yi bu mu
                                              sürüklüyordu? (beklenti: evet,
                                              val yüksek + test çöküşü tekrarlar)
  fund_pure_only    fund akışı = 6 firma   → TEŞHİS: bilanço tek başına ne yapar?
  fund_xs_only      fund akışı = 6 firma + 6 kesitsel → TEDAVİ testi
  multi_pure        tech + 6 firma         → makrosuz temiz füzyon
  multi_xs          tech + firma + kesitsel → ANA HİPOTEZ: XGBoost'u geçer mi?

Etkileşim terimleri (Debt_x_VIX_Delta vb.) BİLİNÇLİ olarak hiçbirinde yok:
onlar firma×makro çarpımı, yani füzyonun kendisinin keşfetmesi gereken şeyi
elle vermek olur. Çıkarmak, füzyon hipotezinin temiz testini sağlar.

KARŞILAŞTIRMA REFERANSLARI (mevcut sonuçlar, test MCC):
  XGBoost            0.1100  (tek seed, 100-trial Optuna)
  tech_only          0.1118 ± 0.0145  (5 seed)
  concat/gated/FiLM/static  0.083–0.092  (5'er seed)
  late-fusion stacked 0.0306 ± 0.0173  (5 seed)

KULLANIM:
    python run_modality_v2.py                      # 5 seed, tüm ablation'lar
    python run_modality_v2.py --seeds 42 43 44     # hızlı ilk okuma (~9 saat)
    python run_modality_v2.py --ablations multi_xs # tek ablation
    python run_modality_v2.py --force              # sıfırdan başla

Resume desteklidir: yarıda kesilirse tamamlanan (ablation, seed) çiftleri
atlanır. --force ile sıfırlanır.
"""
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import train_pytorch_model
from models.losses import FocalLoss


# ══════════════════════════════════════════════════════════════════
# Ablation tanımları: ad → (tech_groups, fund_groups, modality, açıklama)
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# TAM FAKTÖRİYEL TASARIM — I / II / III
# ══════════════════════════════════════════════════════════════════
#   I   = teknik (32 fiyat göstergesi)
#   II  = fundamental (6 firma muhasebe oranı)
#   III = makro (9 piyasa geneli seri)
#
# 2³−1 = 7 hücrenin tamamı koşulur. Bu, "şu kombinasyonu denemediniz"
# itirazını tamamen kapatır ve marjinal katkı analizini mümkün kılar:
#     Δ(II | I)      = (I+II) − I          bilanço, fiyata ne katıyor?
#     Δ(III | I)     = (I+III) − I         makro, fiyata ne katıyor?
#     Δ(III | I+II)  = (I+II+III) − (I+II) makro, ikisinin üstüne ne katıyor?
#
# AKIŞ ATAMA KURALI (a priori, sonuçlara bakılmadan sabitlendi):
#   Akış A (teknik encoder)  ← I
#   Akış B (bağlam encoder)  ← II ve/veya III
# Gerekçe: hızlı/fiyat kaynaklı sinyaller ayrı, yavaş/bağlamsal sinyaller
# ayrı encoder'da. Mimari iki akışlı olduğu için üç modalite bu kuralla
# yerleştirilir; kural tüm hücrelerde AYNI uygulanır.
#
# Tek modaliteli hücrelerde (I, II, III, II+III) kullanılmayan akış için
# bir "dummy" grup verilir — model onu modality parametresiyle yok sayar,
# sonucu etkilemez, sadece loader'ın iki akış beklemesini karşılar.
ABLATIONS = {
    # ── Tekli modaliteler ──
    'tech_only': (
        ('tech',), ('fund',), 'tech_only',
        '[I] Sadece teknik (fund akışı kurulur ama model yok sayar)',
    ),
    'fund_pure_only': (
        ('tech',), ('fund',), 'fund_only',
        '[II] Sadece 6 firma muhasebe oranı (ham seviye)',
    ),
    'macro_only': (
        ('tech',), ('macro',), 'fund_only',
        '[III] Sadece makro (kesitsel ayrım gücü SIFIR olan kontrol grubu)',
    ),
    # ── İkili kombinasyonlar ──
    'multi_pure': (
        ('tech',), ('fund',), 'multimodal',
        '[I+II] Teknik + firma oranları, cross-attention (makrosuz füzyon)',
    ),
    'tech_macro': (
        ('tech',), ('macro',), 'multimodal',
        '[I+III] Teknik + makro, cross-attention',
    ),
    'fund_macro': (
        ('tech',), ('fund', 'macro'), 'fund_only',
        '[II+III] Firma oranları + makro (fiyat sinyali yok)',
    ),
    # ── Üçlü ──
    'all_three': (
        ('tech',), ('fund', 'macro'), 'multimodal',
        '[I+II+III] Tam model — orijinal kurulumun makrolu hali',
    ),
    # ── Robustness varyantı (faktöriyelin parçası DEĞİL) ──
    # Kesitsel normalizasyon, II'nin bir varyantıdır; dördüncü faktör olarak
    # eklenirse tasarım 16 hücreye çıkar. Ayrı bir robustness satırı olarak
    # raporlanmalı: II-ham vs II-kesitsel.
    'fund_xs_only': (
        ('tech',), ('fund', 'fund_xs'), 'fund_only',
        '[II-kesitsel] Firma oranları + tarih-içi yüzdelik dilimleri',
    ),
    'multi_xs': (
        ('tech',), ('fund', 'fund_xs'), 'multimodal',
        '[I+II-kesitsel] Teknik + firma + kesitsel, cross-attention',
    ),
}

BASE_CONFIG = dict(
    seq_len=20,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.15,
    fusion_type='cross_attention',
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_focal_alpha(train_loader, max_batches=200):
    """
    Seed döngüsünün DIŞINDA bir kez çağrılır — alpha verinin özelliği, seed'in
    değil. Ayrıca loader'ı tüketmek RNG'yi ilerletir, bu da seed kontrolünü bozar.
    """
    pos = neg = 0
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        lbl = batch['label']
        pos += int((lbl == 1).sum())
        neg += int((lbl == 0).sum())
    return 1.0 if pos == 0 else float(neg / pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--ablations', type=str, nargs='+',
                    default=list(ABLATIONS), choices=list(ABLATIONS))
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--force', action='store_true',
                    help='Tamamlanmış koşuları da yeniden çalıştır')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'modality_v2_raw.csv')

    print("═" * 78)
    print("MODALITY V2 — fundamental / makro ayrımı sonrası ablation merdiveni")
    print("═" * 78)
    for name in args.ablations:
        tg, fg, mod, desc = ABLATIONS[name]
        print(f"  {name:<16} fund={str(tuple(fg)):<22} modality={mod:<11} {desc}")
    print(f"  Seed'ler    : {args.seeds}")
    print(f"  Toplam koşu : {len(args.ablations) * len(args.seeds)}")
    print()

    # ── Resume ──
    done = set()
    rows = []
    if os.path.exists(raw_path) and not args.force:
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict('records')
        done = {(r['ablation'], int(r['seed'])) for r in rows}
        if done:
            print(f"  [Resume] {len(done)} koşu zaten tamamlanmış, atlanacak.")
            print(f"           Sıfırdan başlatmak için --force kullan.\n")

    print("[1/3] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    # Kesitsel kolonlar gerçekten üretilmiş mi?
    xs_missing = [c for c in ['Altman_Z_XS', 'Debt_to_Equity_XS']
                  if c not in dataset_out.columns]
    if xs_missing:
        print(f"\n  ⚠️ Kesitsel kolonlar eksik: {xs_missing}")
        print("     feature_engineering.py güncel mi? _enrich_macro_features "
              "bunları üretmeli. Devam ediliyor ama *_xs ablation'ları hatalı olur.")
    else:
        print("  ✓ Kesitsel (_XS) kolonlar mevcut")

    device = get_device()
    print(f"  Device: {device}\n")

    # ── Dataloader'lar ablation'a göre değişir; grup-spec başına bir kez kur ──
    loader_cache = {}

    def get_loaders(tech_groups, fund_groups):
        key = (tuple(tech_groups), tuple(fund_groups))
        if key not in loader_cache:
            print(f"\n[Loader] Kuruluyor → tech={key[0]} | fund={key[1]}")
            tl, vl, testl, _, (tc, fc) = get_dual_stream_dataloaders(
                dataset_out,
                seq_len=BASE_CONFIG['seq_len'],
                batch_size=args.batch_size,
                tech_groups=tech_groups,
                fund_groups=fund_groups,
            )
            alpha = compute_focal_alpha(tl)
            loader_cache[key] = (tl, vl, testl, tc, fc, alpha)
            print(f"  tech_dim={len(tc)} | fund_dim={len(fc)} | FocalLoss alpha={alpha:.2f}")
        return loader_cache[key]

    total = len(args.ablations) * len(args.seeds)
    n = 0
    print("[2/3] Koşular başlıyor...\n")

    for name in args.ablations:
        tech_groups, fund_groups, modality, desc = ABLATIONS[name]
        tl, vl, testl, tech_cols, fund_cols, alpha = get_loaders(tech_groups, fund_groups)

        for seed in args.seeds:
            n += 1
            if (name, seed) in done:
                print(f"[{n}/{total}] {name} seed={seed} — atlandı (tamamlanmış)")
                continue

            print("\n" + "▄" * 78)
            print(f"KOŞU {n}/{total} — ablation={name} | seed={seed}")
            print(f"  {desc}")
            print(f"  tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | modality={modality}")
            print("▄" * 78)

            set_seed(seed)
            model = DualEncoderTransformer(
                tech_dim=len(tech_cols),
                fund_dim=len(fund_cols),
                seq_len=BASE_CONFIG['seq_len'],
                d_model=BASE_CONFIG['d_model'],
                n_heads=BASE_CONFIG['n_heads'],
                n_layers=BASE_CONFIG['n_layers'],
                dropout=BASE_CONFIG['dropout'],
                modality=modality,
                fusion_type=fusion_for(name),
            )
            )
            criterion = FocalLoss(alpha=alpha, gamma=2.0)

            t0 = time.time()
            try:
                res = train_pytorch_model(
                    model=model,
                    train_loader=tl,
                    val_loader=vl,
                    test_loader=testl,
                    model_name=f"V2_{name}_seed{seed}",
                    epochs=args.epochs,
                    device=device,
                    criterion=criterion,
                    monitor='pr_auc',
                )
            except Exception as e:
                print(f"  ✗ HATA: {e}")
                import traceback
                traceback.print_exc()
                continue

            elapsed = round((time.time() - t0) / 60, 1)
            row = {
                'ablation': name, 'seed': seed, 'modality': modality,
                'fund_groups': '+'.join(fund_groups),
                'tech_dim': len(tech_cols), 'fund_dim': len(fund_cols),
                'threshold': res.get('threshold'), 'minutes': elapsed,
            }
            for split in ('val', 'test'):
                m = res.get(f'{split}_metrics') or {}
                for k, v in m.items():
                    row[f'{split}_{k}'] = v
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"  ✓ {name} seed={seed} | test MCC={row.get('test_mcc', float('nan')):.4f} "
                  f"| {elapsed} dk | kaydedildi")

    if not rows:
        print("\nHiç sonuç yok.")
        return

    # ── Özet ──
    print("\n[3/3] Özet hesaplanıyor...\n")
    df = pd.DataFrame(rows)
    summaries = []
    print("═" * 78)
    print("MODALITY V2 SONUÇLARI")
    print("═" * 78)

    for metric in ('test_mcc', 'test_pr_auc', 'val_mcc'):
        if metric not in df.columns:
            continue
        print(f"\n── {metric.upper()} ──")
        print(f"{'Ablation':<18}{'ortalama':>10}{'±std':>9}{'min':>9}{'max':>9}{'n':>4}")
        print("-" * 60)
        g = df.groupby('ablation')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        for abl in args.ablations:
            if abl not in g.index:
                continue
            r = g.loc[abl]
            print(f"{abl:<18}{r['mean']:>10.4f}{(r['std'] if pd.notna(r['std']) else 0):>9.4f}"
                  f"{r['min']:>9.4f}{r['max']:>9.4f}{int(r['count']):>4}")
            summaries.append({'ablation': abl, 'metric': metric, 'mean': r['mean'],
                              'std': r['std'], 'min': r['min'], 'max': r['max'],
                              'n': int(r['count'])})

    # Genelleme farkı — tezin ana argümanı buradan çıkıyor
    if {'test_mcc', 'val_mcc'} <= set(df.columns):
        df['gen_gap'] = df['test_mcc'] - df['val_mcc']
        print(f"\n── GENELLEME FARKI (test MCC − val MCC) ──")
        print(f"{'Ablation':<18}{'ortalama':>10}{'±std':>9}{'negatif seed':>14}")
        print("-" * 55)
        for abl in args.ablations:
            sub = df[df['ablation'] == abl]
            if sub.empty:
                continue
            neg = int((sub['gen_gap'] < 0).sum())
            print(f"{abl:<18}{sub['gen_gap'].mean():>10.4f}"
                  f"{sub['gen_gap'].std() if len(sub) > 1 else 0:>9.4f}"
                  f"{neg:>10}/{len(sub)}")
            summaries.append({'ablation': abl, 'metric': 'gen_gap',
                              'mean': sub['gen_gap'].mean(),
                              'std': sub['gen_gap'].std(), 'min': sub['gen_gap'].min(),
                              'max': sub['gen_gap'].max(), 'n': len(sub)})

    summ_path = os.path.join(args.outdir, 'modality_v2_summary.csv')
    pd.DataFrame(summaries).to_csv(summ_path, index=False)

    # ── Yorum yardımcısı ──
    print("\n" + "═" * 78)
    print("YORUM REHBERİ")
    print("═" * 78)
    print("  Referanslar: XGBoost test MCC = 0.1100 | tech_only = 0.1118 ± 0.0145")
    print()
    print("  macro_only ≈ eski fund_only (val yüksek, test çöküş) ise")
    print("    → 'fundamental işe yaramadı' sonucu aslında MAKRO hakkındaymış; teşhis doğrulandı.")
    print("  fund_xs_only >> fund_pure_only ise")
    print("    → kesitsel normalizasyon firma bilançosuna gerçek ayrım gücü kazandırdı.")
    print("  multi_xs > 0.1118 + 2×0.0145 (≈0.141) ise")
    print("    → füzyon hem tech_only'yi hem XGBoost'u anlamlı biçimde geçti (ANA SONUÇ).")
    print("  multi_xs ≈ multi_pure ≈ 0.09 ise")
    print("    → kesitsel normalizasyon da kurtarmadı; negatif bulgu artık çok daha güçlü,")
    print("       çünkü 'fundamental'ı adil biçimde temsil ettiğinizi kanıtlamış olursunuz.")

    print(f"\nÖzet  → {summ_path}")
    print(f"Ham   → {raw_path}")


if __name__ == '__main__':
    main()