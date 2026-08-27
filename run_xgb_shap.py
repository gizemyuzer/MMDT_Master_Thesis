"""
run_xgb_shap.py
────────────────
XGBoost faktöriyel hücresi için SHAP — modalite düzeyinde toplulaştırmalı.

═══════════════════════════════════════════════════════════════════════
NEDEN AYRI BIR SCRIPT
═══════════════════════════════════════════════════════════════════════
models/xgboost_model.py içindeki mevcut SHAP kodu eski tek-hücre akışına
bağlı ve ürettiği grafik makro içeren eski özellik kümesinden geliyor.
Metin modalitesi orada hiç yok.

Bu script kayıtlı Optuna parametrelerini okuyup istenen faktöriyel hücreyi
eğitir ve SHAP'i GÜNCEL kurulumla üretir.

═══════════════════════════════════════════════════════════════════════
ASIL KATKI: MODALİTE DÜZEYİNDE TOPLULAŞTIRMA
═══════════════════════════════════════════════════════════════════════
Tek tek değişken önemleri ilginç ama tezin sorusu modalite düzeyinde:
"teknik / firma / makro / metin ne kadar iş görüyor?"

|SHAP| değerlerini gruplara göre toplamak bunu doğrudan cevaplar VE
run_attribution.py'nin transformer için ürettiği grup-düzeyi permütasyon
önemiyle karşılaştırılabilir hale getirir. İki model ailesi, aynı soru,
iki farklı yöntem — jüri karşısında güçlü bir tablo.

KULLANIM:
    python run_xgb_shap.py                      # I+II+IV (en iyi hücre)
    python run_xgb_shap.py --cell I+II          # metinsiz karşılaştırma
    python run_xgb_shap.py --sample 100000      # daha büyük örneklem

Çıktılar:
    visualization/shap/shap_<hücre>_summary.png     beeswarm
    visualization/shap/shap_<hücre>_modality.png    modalite toplamı
    visualization/shap/shap_<hücre>_regime.png      yıl bazında kırılım
    results/shap_<hücre>_features.csv               değişken bazında |SHAP|
"""
import os
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.feature_engineering import (
    prepare_dataset, resolve_groups, FEATURE_GROUPS,
    TECHNICAL_COLS, FIRM_FUNDAMENTAL_COLS, FIRM_FUNDAMENTAL_XS_COLS,
    MACRO_COLS, TEXT_EVENT_COLS, TEXT_LM_COLS, TEXT_COVERAGE_COL,
)
from run_xgb_factorial import CELLS, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END

OUT_FIG = os.path.join('visualization', 'shap')
OUT_CSV = 'results'

# Modalite atama — kolon adından gruba
MODALITY = [
    ('I — teknik',            set(TECHNICAL_COLS),                    '#2E86AB'),
    ('II — firma',            set(FIRM_FUNDAMENTAL_COLS) |
                              set(FIRM_FUNDAMENTAL_XS_COLS),          '#A23B72'),
    ('III — makro',           set(MACRO_COLS),                        '#F18F01'),
    ('IV — metin (8-K olay)', set(TEXT_EVENT_COLS),                   '#6A994E'),
    ('IV — metin (LM/dil)',   set(TEXT_LM_COLS) | {TEXT_COVERAGE_COL}, '#8E7DBE'),
]


def modality_of(col):
    for name, members, color in MODALITY:
        if col in members:
            return name, color
    return 'diğer', '#999999'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='I+II+IV', choices=list(CELLS))
    ap.add_argument('--sample', type=int, default=50_000,
                    help='SHAP için test örneklemi (0 = tamamı)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_CSV, exist_ok=True)

    par_path = os.path.join(args.outdir, 'xgb_factorial_params.csv')
    if not os.path.exists(par_path):
        raise SystemExit(f"\n{par_path} yok. Önce: python run_xgb_factorial.py\n")
    pdf = pd.read_csv(par_path)
    row = pdf[pdf['cell'] == args.cell]
    if row.empty:
        raise SystemExit(
            f"\n'{args.cell}' hücresi parametre dosyasında yok.\n"
            f"Mevcut: {pdf['cell'].tolist()}\n"
            f"Üretmek için: python run_xgb_factorial.py --cells {args.cell}\n")

    groups, desc = CELLS[args.cell]
    print("═" * 74)
    print(f"SHAP — hücre {args.cell}   ({desc})")
    print("═" * 74)

    ds = prepare_dataset(force_refresh=False)
    cols = resolve_groups(ds, groups)
    print(f"  {len(cols)} özellik")

    tr = ds[ds.index <= TRAIN_END]
    va = ds[(ds.index >= VAL_START) & (ds.index <= VAL_END)]
    te = ds[(ds.index >= TEST_START) & (ds.index <= TEST_END)]

    med = tr[cols].median()
    sc = RobustScaler()
    Xtr = sc.fit_transform(tr[cols].fillna(med).values)
    Xva = sc.transform(va[cols].fillna(med).values)
    Xte = sc.transform(te[cols].fillna(med).values)

    prm = {k: v for k, v in row.iloc[0].items()
           if k not in ('cell', 'n_features', 'cv_pr_auc') and pd.notna(v)}
    prm['tree_method'] = 'hist'
    prm['device'] = 'cpu'          # TreeSHAP CPU'da; GPU beklemeye gerek yok

    print("  Model eğitiliyor...")
    m = xgb.XGBClassifier(**prm, eval_metric='logloss', random_state=args.seed,
                          n_jobs=-1, early_stopping_rounds=50)
    m.fit(Xtr, tr['Target'].values,
          eval_set=[(Xva, va['Target'].values)], verbose=False)

    # ── SHAP ──
    import shap
    n = len(Xte) if args.sample == 0 else min(args.sample, len(Xte))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xte), n, replace=False) if n < len(Xte) else np.arange(len(Xte))
    idx = np.sort(idx)                       # tarih sırası korunur → rejim kırılımı için
    Xs = Xte[idx]
    print(f"  TreeSHAP hesaplanıyor ({n:,} gözlem)...")
    sv = shap.TreeExplainer(m).shap_values(Xs)

    mean_abs = np.abs(sv).mean(axis=0)
    F = pd.DataFrame({'feature': cols, 'mean_abs_shap': mean_abs})
    F['modality'] = [modality_of(c)[0] for c in F.feature]
    F['share_pct'] = 100 * F.mean_abs_shap / F.mean_abs_shap.sum()
    F = F.sort_values('mean_abs_shap', ascending=False)
    F.to_csv(os.path.join(OUT_CSV, f'shap_{args.cell.replace("+","_")}_features.csv'),
             index=False)

    print(f"\n── EN ÖNEMLİ 15 DEĞİŞKEN ──")
    print(f"  {'değişken':<26}{'modalite':<24}{'|SHAP|':>10}{'pay %':>8}")
    print("  " + "-" * 68)
    for _, r in F.head(15).iterrows():
        print(f"  {r.feature:<26}{r.modality:<24}{r.mean_abs_shap:>10.4f}{r.share_pct:>8.1f}")

    # ── Modalite düzeyinde toplam ──
    G = (F.groupby('modality')
           .agg(toplam=('mean_abs_shap', 'sum'), n=('feature', 'count'))
           .sort_values('toplam', ascending=False))
    G['pay_pct'] = 100 * G.toplam / G.toplam.sum()
    G['ozellik_basina'] = G.toplam / G.n

    print(f"\n── MODALİTE DÜZEYİNDE ──")
    print(f"  {'modalite':<24}{'kolon':>7}{'toplam':>10}{'pay %':>8}{'kolon başına':>14}")
    print("  " + "-" * 63)
    for k, r in G.iterrows():
        print(f"  {k:<24}{int(r.n):>7}{r.toplam:>10.4f}{r.pay_pct:>8.1f}{r.ozellik_basina:>14.4f}")
    print("\n  'kolon başına' sütunu kritik: teknik akış 32 kolonla doğal olarak")
    print("  büyük toplam üretir. Adil karşılaştırma birim kolon başınadır.")
    G.reset_index().to_csv(
        os.path.join(OUT_CSV, f'shap_{args.cell.replace("+","_")}_modality.csv'), index=False)

    # ── Şekil 1: beeswarm ──
    plt.figure(figsize=(9, max(6, len(cols) * 0.22)))
    shap.summary_plot(sv, Xs, feature_names=cols, show=False, max_display=25)
    plt.title(f'SHAP — {args.cell}  ({desc})', fontsize=11, loc='left')
    p = os.path.join(OUT_FIG, f'shap_{args.cell.replace("+","_")}_summary.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"\n  → {p}")

    # ── Şekil 2: modalite toplamı ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = [dict((m[0], m[2]) for m in MODALITY).get(k, '#999999') for k in G.index]
    axes[0].barh(G.index[::-1], G.toplam[::-1], color=colors[::-1])
    axes[0].set_xlabel('Toplam |SHAP|')
    axes[0].set_title('Modalite başına toplam katkı', fontsize=10, loc='left')
    axes[1].barh(G.sort_values('ozellik_basina').index,
                 G.sort_values('ozellik_basina').ozellik_basina,
                 color=[dict((m[0], m[2]) for m in MODALITY).get(k, '#999999')
                        for k in G.sort_values('ozellik_basina').index])
    axes[1].set_xlabel('Kolon başına |SHAP|')
    axes[1].set_title('Kolon başına katkı (adil karşılaştırma)', fontsize=10, loc='left')
    for a in axes:
        a.grid(alpha=0.25, axis='x')
    fig.suptitle(f'{args.cell} — modalite düzeyinde SHAP', fontsize=11, x=0.09, ha='left')
    p = os.path.join(OUT_FIG, f'shap_{args.cell.replace("+","_")}_modality.png')
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(f"  → {p}")

    # ── Şekil 3: rejim kırılımı ──
    years = te.index[idx].year.values
    rows = []
    for y in sorted(set(years)):
        m_ = years == y
        if m_.sum() < 100:
            continue
        ma = np.abs(sv[m_]).mean(axis=0)
        tmp = pd.DataFrame({'feature': cols, 'v': ma})
        tmp['modality'] = [modality_of(c)[0] for c in tmp.feature]
        s = tmp.groupby('modality')['v'].sum()
        rows.append(pd.Series(100 * s / s.sum(), name=y))
    if rows:
        R = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        R.plot(kind='bar', stacked=True, ax=ax,
               color=[dict((m[0], m[2]) for m in MODALITY).get(c, '#999999')
                      for c in R.columns])
        ax.set_ylabel('Toplam |SHAP| içindeki pay (%)')
        ax.set_xlabel('Yıl')
        ax.set_title('Modalite payları rejime göre değişiyor mu?\n'
                     '2022 ayı piyasası · 2023 toparlanma · 2024 boğa',
                     fontsize=11, loc='left')
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(alpha=0.25, axis='y')
        p = os.path.join(OUT_FIG, f'shap_{args.cell.replace("+","_")}_regime.png')
        fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
        print(f"  → {p}")
        print(f"\n── REJİME GÖRE MODALİTE PAYLARI (%) ──")
        print(R.round(1).to_string())

    print(f"\n  → {OUT_CSV}/shap_{args.cell.replace('+','_')}_features.csv")


if __name__ == '__main__':
    main()