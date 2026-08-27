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
düşüşler) çöküyor.

═══════════════════════════════════════════════════════════════════════
MODALİTE IV — METİN (SEC EDGAR)
═══════════════════════════════════════════════════════════════════════
Dördüncü modalite eklendi: 10-K/10-Q metninden Loughran-McDonald duygu
oranları + ardışık dosyalar arası benzerlik (Cohen, Malloy & Nguyen 2020),
ve 8-K item kodlarından türetilen sıkıntı olayı bayrakları.

XGBoost tarafındaki ilk bulgu: metin, POZİTİF genelleme farkı olan tek
modalite (val 0.020 → test 0.094). Makronun tam aynası — çünkü metin
firmaya özgü, makro değil. Bu, tezin kesitsel argümanını doğruluyor.

KULLANIM:
    python run_modality_v2.py --ablations multi_text --seeds 42 43 44
    python run_modality_v2.py --ablations multi_pure_gated   # attribution
    python run_modality_v2.py --force                        # seçili hücreleri yenile

Resume desteklidir: yarıda kesilirse tamamlanan (ablation, seed) çiftleri
atlanır.
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
# TAM FAKTÖRİYEL TASARIM — I / II / III
# ══════════════════════════════════════════════════════════════════
#   I   = teknik (32 fiyat göstergesi)
#   II  = fundamental (6 firma muhasebe oranı)
#   III = makro (9 piyasa geneli seri)
#   IV  = metin (SEC EDGAR: 8-K olayları + LM duygu + benzerlik)
#
# 2³−1 = 7 hücrenin tamamı koşulur; marjinal katkı analizi:
#     Δ(II | I)      = (I+II) − I
#     Δ(III | I)     = (I+III) − I
#     Δ(III | I+II)  = (I+II+III) − (I+II)
#     Δ(IV | I+II)   = (I+II+IV) − (I+II)
#
# AKIŞ ATAMA KURALI (a priori, sonuçlara bakılmadan sabitlendi):
#   Akış A (teknik encoder)  ← I
#   Akış B (bağlam encoder)  ← II, III ve/veya IV
# Gerekçe: hızlı/fiyat kaynaklı sinyaller ayrı, yavaş/bağlamsal sinyaller
# ayrı encoder'da. Kural tüm hücrelerde AYNI uygulanır.
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
    'fund_xs_only': (
        ('tech',), ('fund', 'fund_xs'), 'fund_only',
        '[II-kesitsel] Firma oranları + tarih-içi yüzdelik dilimleri',
    ),
    'multi_xs': (
        ('tech',), ('fund', 'fund_xs'), 'multimodal',
        '[I+II-kesitsel] Teknik + firma + kesitsel, cross-attention',
    ),
    # ══════════════════════════════════════════════════════════════
    # MODALİTE IV — METİN
    # ══════════════════════════════════════════════════════════════
    # TAM FAKTÖRİYEL YAPILMIYOR: dört faktör 2⁴−1 = 15 hücre × 5 seed = 75
    # koşu, ~60 saat GPU. Ayrıca metin, makro gibi bir hipotez testi değil,
    # bir EKLEME sorusu: "fiyat ve bilançonun üstüne ne katıyor?" Cevap tek
    # kontrastta: Δ(IV | I+II).
    #
    # 'text_only', II ve III için yaptığımızın aynısı: modalitenin tek başına
    # bilgi taşıyıp taşımadığını gösteren kontrol hücresi. Onsuz "metin katkı
    # yapmadı" sonucu, metnin hiç bilgi içermediğini mi yoksa fiyatla
    # örtüştüğünü mü gösterdiği belirsiz kalırdı.
    'text_only': (
        ('tech',), ('text',), 'fund_only',
        '[IV] Sadece metin (8-K olayları + LM duygu + benzerlik)',
    ),
    'multi_text': (
        ('tech',), ('fund', 'text'), 'multimodal',
        '[I+II+IV] Teknik + firma + metin — ANA METİN TESTİ',
    ),
    # Katman ayrımı: metin çuvallarsa hangi katman sorumlu?
    # 8-K olayları günlük çözünürlükte ve NLP gerektirmiyor; LM çeyreklik.
    # Tek grup olsaydı olayların katkısı LM gürültüsünde kaybolabilirdi.
    'multi_text_event': (
        ('tech',), ('fund', 'text_event'), 'multimodal',
        '[I+II+IV-olay] Sadece 8-K olay bayrakları (LM yok)',
    ),
    'multi_text_lm': (
        ('tech',), ('fund', 'text_lm'), 'multimodal',
        '[I+II+IV-lm] Sadece Loughran-McDonald (8-K yok)',
    ),
    # ── Yorumlanabilirlik varyantları (faktöriyelin parçası DEĞİL) ──
    # Aynı I+II özellik kümesi, farklı füzyon mekanizması. Amaç performans
    # karşılaştırması değil ATTRIBUTION: kapı (gate) ve FiLM (gamma) değerleri
    # ancak bu mimarilerde gözlemlenebilir. Daha önce ölçülen kapı=0.4892 ve
    # gamma=0.7751 MAKRO İÇEREN eski özellik kümesinden geliyordu ve tezin
    # geri kalanıyla tutarsızdı.
    'multi_pure_gated': (
        ('tech',), ('fund',), 'multimodal',
        '[I+II gated] Teknik + firma, öğrenilebilir kapı (attribution için)',
    ),
    'multi_pure_film': (
        ('tech',), ('fund',), 'multimodal',
        '[I+II FiLM] Teknik + firma, feature-wise modülasyon (attribution için)',
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

# ── Hücre-bazlı füzyon tipi ────────────────────────────────────────
# Ayrı sözlük olarak tutuluyor çünkü ABLATIONS 4'lü tuple olarak dışarıdan
# (ör. run_regime_analysis.py, run_portfolio_simulation.py) unpack ediliyor;
# tuple'ı genişletmek onları kırardı.
ABLATION_FUSION = {
    'multi_pure_gated': 'gated_cross_attention',
    'multi_pure_film': 'film',
}


def fusion_for(ablation_name: str) -> str:
    """Hücrenin füzyon tipi — override yoksa taban konfigürasyon."""
    return ABLATION_FUSION.get(ablation_name, BASE_CONFIG['fusion_type'])


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
                    help='SEÇİLEN ablation-seed çiftlerini yeniden çalıştır')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'modality_v2_raw.csv')

    print("═" * 78)
    print("MODALITY V2 — modalite ablation merdiveni")
    print("═" * 78)
    for name in args.ablations:
        tg, fg, mod, desc = ABLATIONS[name]
        print(f"  {name:<17} fund={str(tuple(fg)):<22} modality={mod:<11} "
              f"fusion={fusion_for(name):<22} {desc}")
    print(f"  Seed'ler    : {args.seeds}")
    print(f"  Toplam koşu : {len(args.ablations) * len(args.seeds)}")
    print()

    # ══════════════════════════════════════════════════════════════
    # Resume
    # ══════════════════════════════════════════════════════════════
    # DİKKAT — bu blok bir VERİ KAYBI hatasını önlüyor:
    # Eskiden --force verildiğinde `rows` boş listeyle başlıyordu ve koşu
    # sonunda tüm CSV bu boş listeden yeniden yazılıyordu. Yani
    #     python run_modality_v2.py --ablations multi_pure --force
    # komutu, seçilmeyen ablation'ların TÜM satırlarını SESSİZCE SİLİYORDU.
    # Artık geçmiş satırlar her zaman okunur; --force yalnızca SEÇİLEN
    # ablation'ların satırlarını düşürür ve önce yedek alır.
    done = set()
    rows = []
    if os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        if args.force:
            backup = raw_path.replace('.csv', '_backup.csv')
            prev.to_csv(backup, index=False)
            dropped = prev[prev['ablation'].isin(args.ablations)]
            prev = prev[~prev['ablation'].isin(args.ablations)]
            print(f"  [--force] {len(dropped)} satır yeniden koşulacak; "
                  f"{len(prev)} satır korunuyor.")
            print(f"            Yedek → {backup}\n")
        rows = prev.to_dict('records')
        done = {(r['ablation'], int(r['seed'])) for r in rows}
        if done and not args.force:
            print(f"  [Resume] {len(done)} koşu zaten tamamlanmış, atlanacak.")
            print(f"           Seçili hücreleri yeniden koşmak için --force kullan.\n")

    print("[1/3] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    # Kesitsel kolonlar gerçekten üretilmiş mi?
    xs_missing = [c for c in ['Altman_Z_XS', 'Debt_to_Equity_XS']
                  if c not in dataset_out.columns]
    if xs_missing:
        print(f"\n  ⚠️ Kesitsel kolonlar eksik: {xs_missing}")
    else:
        print("  ✓ Kesitsel (_XS) kolonlar mevcut")

    # Metin kolonları — modalite IV hücreleri seçildiyse ŞART
    text_cells = {'text_only', 'multi_text', 'multi_text_event', 'multi_text_lm'}
    if text_cells & set(args.ablations):
        n_text = len([c for c in dataset_out.columns
                      if c.startswith(('EK_', 'LM_', 'TXT_'))])
        if n_text == 0:
            raise SystemExit(
                "\n  ✗ Metin hücreleri seçildi ama panelde metin kolonu yok.\n"
                "    Önce: python build_text_features.py --stage all\n")
        print(f"  ✓ Metin (modalite IV) kolonları mevcut: {n_text}")

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
            print(f"  tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | "
                  f"modality={modality} | fusion={fusion_for(name)}")
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
                'fusion': fusion_for(name),
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
        print(f"{'Ablation':<19}{'ortalama':>10}{'±std':>9}{'min':>9}{'max':>9}{'n':>4}")
        print("-" * 61)
        g = df.groupby('ablation')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        for abl in args.ablations:
            if abl not in g.index:
                continue
            r = g.loc[abl]
            print(f"{abl:<19}{r['mean']:>10.4f}{(r['std'] if pd.notna(r['std']) else 0):>9.4f}"
                  f"{r['min']:>9.4f}{r['max']:>9.4f}{int(r['count']):>4}")
            summaries.append({'ablation': abl, 'metric': metric, 'mean': r['mean'],
                              'std': r['std'], 'min': r['min'], 'max': r['max'],
                              'n': int(r['count'])})

    # Genelleme farkı — tezin ana argümanı buradan çıkıyor
    if {'test_mcc', 'val_mcc'} <= set(df.columns):
        df['gen_gap'] = df['test_mcc'] - df['val_mcc']
        print(f"\n── GENELLEME FARKI (test MCC − val MCC) ──")
        print(f"{'Ablation':<19}{'ortalama':>10}{'±std':>9}{'negatif seed':>14}")
        print("-" * 56)
        for abl in args.ablations:
            sub = df[df['ablation'] == abl]
            if sub.empty:
                continue
            neg = int((sub['gen_gap'] < 0).sum())
            print(f"{abl:<19}{sub['gen_gap'].mean():>10.4f}"
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
    print("  Referanslar (5 seed, test MCC):")
    print("    tech_only [I]      0.1110 ± 0.0142")
    print("    multi_pure [I+II]  0.1273 ± 0.0122   ← metin karşılaştırma tabanı")
    print()
    print("  multi_text > 0.1273 + 0.005 ise")
    print("    → metin, fiyat ve bilançonun üstüne gerçek katkı veriyor.")
    print("      (0.005 = XGBoost tarafında ölçülen arama gürültüsü tabanı)")
    print("  text_only >> fund_pure_only (0.0068) ise")
    print("    → metin, muhasebe oranlarından daha fazla firmaya özgü bilgi taşıyor.")
    print("  multi_text'in GENELLEME FARKI pozitifse")
    print("    → XGBoost'taki bulgu tekrarlandı: metin, makronun aynası.")
    print("      Makro val'da parlar test'te çöker; metin val'da zayıf test'te güçlü.")
    print("      İkisinin de açıklaması aynı: kesitsel varyans.")

    print(f"\nÖzet  → {summ_path}")
    print(f"Ham   → {raw_path}")


if __name__ == '__main__':
    main()