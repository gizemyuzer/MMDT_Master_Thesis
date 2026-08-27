"""
run_xgb_factorial.py
─────────────────────
XGBoost için TAM FAKTÖRİYEL (I / II / III) — transformer'la birebir eşleşen tasarım.

═══════════════════════════════════════════════════════════════════════
NEDEN
═══════════════════════════════════════════════════════════════════════
Transformer tarafında 7 hücrenin tamamı × 5 seed koşuldu ve şu bulundu:

    Δ(III | I)     = -0.0410   p=0.034   0/5 seed pozitif
    Δ(III | I+II)  = -0.0407   p=0.010   0/5 seed pozitif
    Δ(II  | I)     = +0.0163   p=0.108   3/5 seed pozitif

Yani makro değişkenler test performansını anlamlı biçimde DÜŞÜRÜYOR.

Ama bu şu ana kadar tek bir model ailesinde gösterildi. Eğer aynı örüntü
gradyan artırmalı ağaçta da çıkarsa, bulgu "attention mimarisinin bir
kusuru" olmaktan çıkıp MODEL SINIFINDAN BAĞIMSIZ bir veri tasarımı olgusu
haline gelir — jüri karşısında çok daha güçlü bir iddia:

    "Makro değişkenlerin eklenmesi hem cross-attention transformer'da hem
     gradyan artırmalı ağaçta test performansını düşürmektedir."

XGBoost'ta şu an yalnızca 2 hücre var (I+II = 0.1328, I+II+III = 0.1174),
üstelik tek seed. Bu script eksik 5 hücreyi tamamlar ve seed varyasyonu ekler.

═══════════════════════════════════════════════════════════════════════
TASARIM — transformer harness'ıyla eşleştirilmiş
═══════════════════════════════════════════════════════════════════════
  I   = tech          (32 fiyat göstergesi)
  II  = fund          (6 firma muhasebe oranı)
  III = macro         (9 piyasa geneli seri)
  Etkileşim terimleri hiçbir hücrede yok (füzyon hipotezinin temiz testi için).

HİPERPARAMETRE ARAMASI HÜCRE BAŞINA BİR KEZ:
  Optuna her özellik kümesi için AYRI çalıştırılır (ör. 51 özellik için
  seçilmiş colsample_bytree, 38 özellikli sette anlamsızdır — aynı
  parametreleri kullanmak XGBoost'u haksız yere cezalandırır).
  Ama seed başına yeniden aranmaz: hiperparametre hücrenin özelliğidir,
  eğitim stokastisitesinin değil. Transformer tarafında da böyle yapıldı
  (tek Optuna araması → 5 seed final eğitim), yani iki taraf simetrik.

SEED VARYASYONU:
  Bulunan parametrelerle 5 farklı random_state ile yeniden eğitilir.
  Bu, ağaç örneklemesi/kolon örneklemesi kaynaklı varyansı yakalar ve
  transformer'la aynı eşleşmeli istatistiksel testleri mümkün kılar.

EŞİK:
  Her seed için YALNIZCA validation'dan MCC-optimal eşik seçilir.
  Test'e bakılarak hiçbir karar verilmez.

KULLANIM:
    python run_xgb_factorial.py                       # 7 hücre × 5 seed
    python run_xgb_factorial.py --trials 50           # arama bütçesi
    python run_xgb_factorial.py --cells I II III      # alt küme
    python run_xgb_factorial.py --force               # baştan
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    matthews_corrcoef, precision_score, recall_score, f1_score,
)

from datasets.feature_engineering import prepare_dataset, resolve_groups
from models.pytorch_trainer import find_best_threshold_mcc


# ══════════════════════════════════════════════════════════════════
# Faktöriyel hücreler — run_modality_v2.py'deki ABLATIONS ile eşleşir
# ══════════════════════════════════════════════════════════════════
CELLS = {
    'I':        (('tech',),                    'Sadece teknik'),
    'II':       (('fund',),                    'Sadece firma muhasebe oranları'),
    'III':      (('macro',),                   'Sadece makro'),
    'I+II':     (('tech', 'fund'),             'Teknik + firma (makrosuz)'),
    'I+III':    (('tech', 'macro'),            'Teknik + makro'),
    'II+III':   (('fund', 'macro'),            'Firma + makro (fiyat yok)'),
    'I+II+III': (('tech', 'fund', 'macro'),    'Tam model'),
    # Robustness (faktöriyelin parçası değil)
    'I+II-xs':  (('tech', 'fund', 'fund_xs'),  'Teknik + firma + kesitsel'),
}

# Transformer referansları (n=5) — çıktıda yan yana göstermek için
TRANSFORMER_REF = {
    'I': 0.1110, 'II': 0.0068, 'III': -0.0056, 'I+II': 0.1273,
    'I+III': 0.0700, 'II+III': -0.0061, 'I+II+III': 0.0866, 'I+II-xs': 0.1215,
}

TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def evaluate(y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    out = {}
    try:
        out['roc_auc'] = roc_auc_score(y_true, prob)
    except ValueError:
        out['roc_auc'] = 0.5
    try:
        out['pr_auc'] = average_precision_score(y_true, prob)
    except ValueError:
        out['pr_auc'] = 0.0
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['mcc'] = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    return out


def build_matrices(dataset_out, groups):
    cols = resolve_groups(dataset_out, groups)
    tr = dataset_out[dataset_out.index <= TRAIN_END]
    va = dataset_out[(dataset_out.index >= VAL_START) & (dataset_out.index <= VAL_END)]
    te = dataset_out[(dataset_out.index >= TEST_START) & (dataset_out.index <= TEST_END)]

    med = tr[cols].median()          # imputation SADECE train medyanıyla
    X = [d[cols].fillna(med).values for d in (tr, va, te)]
    y = [d['Target'].values for d in (tr, va, te)]

    sc = RobustScaler()
    X[0] = sc.fit_transform(X[0])
    X[1] = sc.transform(X[1])
    X[2] = sc.transform(X[2])
    return cols, X, y, tr.index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=str, nargs='+', default=list(CELLS),
                    choices=list(CELLS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--trials', type=int, default=50,
                    help='Hücre başına Optuna deneme sayısı (0 = cached params)')
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'xgb_factorial_raw.csv')
    par_path = os.path.join(args.outdir, 'xgb_factorial_params.csv')

    print("═" * 78)
    print("XGBOOST TAM FAKTÖRİYEL — I / II / III")
    print("═" * 78)
    print(f"  Hücreler : {args.cells}")
    print(f"  Seed'ler : {args.seeds}")
    print(f"  Optuna   : hücre başına {args.trials} deneme (seed başına DEĞİL)")
    print(f"  Eşik     : her seed için yalnızca val'dan seçilir")
    print()

    # ── Resume ──
    rows, done = [], set()
    if os.path.exists(raw_path) and not args.force:
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict('records')
        done = {(r['cell'], int(r['seed'])) for r in rows}
        if done:
            print(f"  [Resume] {len(done)} koşu tamamlanmış, atlanacak.\n")

    param_rows = []
    if os.path.exists(par_path) and not args.force:
        param_rows = pd.read_csv(par_path).to_dict('records')
    cached_params = {r['cell']: r for r in param_rows}

    dataset_out = prepare_dataset(force_refresh=False)

    for cell in args.cells:
        groups, desc = CELLS[cell]
        cols, X, y, train_dates = build_matrices(dataset_out, groups)
        Xtr, Xva, Xte = X
        ytr, yva, yte = y

        print("\n" + "▄" * 78)
        print(f"HÜCRE: {cell}   ({len(cols)} özellik)   {desc}")
        print("▄" * 78)

        # ── 1) Hiperparametre araması: hücre başına BİR KEZ ──
        if cell in cached_params and not args.force:
            best_params = {k: v for k, v in cached_params[cell].items()
                           if k not in ('cell', 'n_features', 'cv_pr_auc')}
            print(f"  [Params] Önceki aramadan alındı (CV PR-AUC="
                  f"{cached_params[cell].get('cv_pr_auc', float('nan')):.4f})")
        else:
            pos, neg = int((ytr == 1).sum()), int((ytr == 0).sum())
            base_spw = neg / max(pos, 1)
            if args.trials > 0:
                from models.xgboost_model import _run_optuna_search
                print(f"  [Params] Optuna araması ({args.trials} deneme)...")
                best_params, cv = _run_optuna_search(
                    Xtr, ytr, train_dates, base_spw, n_trials=args.trials)
            else:
                from models.xgboost_model import CACHED_BEST_PARAMS, CACHED_BEST_CV_PRAUC
                print("  [Params] ⚠️ cached (ön okuma; nihai raporlamada kullanma)")
                best_params, cv = dict(CACHED_BEST_PARAMS), CACHED_BEST_CV_PRAUC
            row = {'cell': cell, 'n_features': len(cols), 'cv_pr_auc': cv}
            row.update(best_params)
            param_rows = [p for p in param_rows if p.get('cell') != cell] + [row]
            pd.DataFrame(param_rows).to_csv(par_path, index=False)
            cached_params[cell] = row

        # tree_method/device best_params içinde yok, elle eklenir
        fit_params = dict(best_params)
        fit_params.pop('cell', None)
        fit_params.pop('n_features', None)
        fit_params.pop('cv_pr_auc', None)
        fit_params['tree_method'] = 'hist'
        fit_params['device'] = 'cuda'

        # ── 2) Seed başına final eğitim ──
        for seed in args.seeds:
            if (cell, seed) in done:
                print(f"  seed={seed} — atlandı (tamamlanmış)")
                continue
            t0 = time.time()
            model = xgb.XGBClassifier(
                **fit_params, eval_metric='logloss', random_state=seed,
                n_jobs=-1, early_stopping_rounds=50)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            p_va = model.predict_proba(Xva)[:, 1]
            p_te = model.predict_proba(Xte)[:, 1]
            thr, _ = find_best_threshold_mcc(yva, p_va)     # SADECE val'dan

            m_va, m_te = evaluate(yva, p_va, thr), evaluate(yte, p_te, thr)
            r = {'cell': cell, 'seed': seed, 'n_features': len(cols),
                 'threshold': thr, 'minutes': round((time.time() - t0) / 60, 2),
                 'best_iteration': getattr(model, 'best_iteration', None)}
            r.update({f'val_{k}': v for k, v in m_va.items()})
            r.update({f'test_{k}': v for k, v in m_te.items()})
            rows.append(r)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"  ✓ seed={seed} | val MCC={m_va['mcc']:.4f} → "
                  f"test MCC={m_te['mcc']:.4f} | {r['minutes']} dk")

    if not rows:
        print("\nSonuç yok.")
        return

    # ══ Özet ══
    df = pd.DataFrame(rows)
    df['gap'] = df['test_mcc'] - df['val_mcc']
    order = [c for c in ['I', 'II', 'III', 'I+II', 'I+III', 'II+III',
                         'I+II+III', 'I+II-xs'] if c in df['cell'].values]

    print("\n" + "═" * 78)
    print("XGBOOST FAKTÖRİYEL SONUÇLARI")
    print("═" * 78)
    g = df.groupby('cell').agg(
        nf=('n_features', 'first'), val=('val_mcc', 'mean'),
        test=('test_mcc', 'mean'), sd=('test_mcc', 'std'),
        pr=('test_pr_auc', 'mean'), roc=('test_roc_auc', 'mean'),
        gap=('gap', 'mean'), n=('seed', 'nunique'))
    print(f"{'hücre':<10}{'özk':>5}{'val MCC':>10}{'test MCC':>10}{'±sd':>9}"
          f"{'PR-AUC':>9}{'ROC':>8}{'gap':>9}{'Transf.':>10}{'fark':>9}")
    print("-" * 88)
    for c in order:
        r = g.loc[c]
        tref = TRANSFORMER_REF.get(c, float('nan'))
        print(f"{c:<10}{int(r.nf):>5}{r.val:>10.4f}{r.test:>10.4f}{r.sd:>9.4f}"
              f"{r.pr:>9.4f}{r.roc:>8.4f}{r.gap:>+9.4f}{tref:>10.4f}"
              f"{r.test - tref:>+9.4f}")

    # ── Marjinal katkılar (seed-eşleşmeli) ──
    try:
        from scipy import stats
        P = df.pivot_table(index='seed', columns='cell', values='test_mcc')
        print("\n── MARJİNAL KATKI (seed-eşleşmeli t-testi) ──")
        print(f"{'etki':<22}{'XGBoost':>22}{'Transformer (ref)':>22}")
        print("-" * 66)
        comps = [
            ('Δ(II | I)',      'I+II', 'I',        +0.0163, 0.108),
            ('Δ(II | I+III)',  'I+II+III', 'I+III', +0.0166, 0.317),
            ('Δ(III | I)',     'I+III', 'I',        -0.0410, 0.034),
            ('Δ(III | I+II)',  'I+II+III', 'I+II',  -0.0407, 0.010),
        ]
        for name, a, b, tref, tp in comps:
            if a in P.columns and b in P.columns:
                d_ = (P[a] - P[b]).dropna()
                if len(d_) >= 2:
                    t, p = stats.ttest_rel(P[a].dropna(), P[b].dropna())
                    print(f"{name:<22}{d_.mean():>+11.4f} (p={p:.3f}){tref:>+13.4f} (p={tp:.3f})")
    except ImportError:
        print("\n(scipy yok — marjinal katkı testleri atlandı)")

    print("\n── YORUM ──")
    print("  Δ(III | ·) XGBoost'ta da NEGATİF ise:")
    print("    → makro kirliliği model sınıfından bağımsız. Bulgu bir mimari")
    print("      kusur değil, veri tasarımı olgusu. En güçlü iddia bu.")
    print("  Δ(III | ·) XGBoost'ta pozitif/sıfır ise:")
    print("    → zarar attention mekanizmasına özgü. Yine ilginç ama daha dar")
    print("      bir iddia: 'attention işe yaramaz özelliklere dikkat harcıyor'.")

    summ = g.reset_index()
    summ.to_csv(os.path.join(args.outdir, 'xgb_factorial_summary.csv'), index=False)
    print(f"\nHam  → {raw_path}")
    print(f"Özet → {args.outdir}/xgb_factorial_summary.csv")
    print(f"Params → {par_path}")


if __name__ == '__main__':
    main()