"""
run_xgb_factorial.py
─────────────────────
XGBoost için TAM FAKTÖRİYEL — transformer'la birebir eşleşen tasarım.

═══════════════════════════════════════════════════════════════════════
NEDEN
═══════════════════════════════════════════════════════════════════════
Bir bulgu tek model ailesinde gösterilirse "mimarinin kusuru" diye
reddedilebilir. Aynı örüntü gradyan artırmalı ağaçta da çıkarsa MODEL
SINIFINDAN BAĞIMSIZ bir veri tasarımı olgusu haline gelir.

Şu ana kadar iki bulgu bu şekilde çift-doğrulandı:
    Makro zararı  : Δ(III|I+II) = -0.0407 (Tr) / -0.0119 (XGB)
    Metin katkısı : Δ(IV |I+II) = +0.0146 (Tr) / +0.0187 (XGB)

═══════════════════════════════════════════════════════════════════════
TASARIM
═══════════════════════════════════════════════════════════════════════
  I   = tech        (32 fiyat göstergesi)
  II  = fund        (6 firma muhasebe oranı)
  III = macro       (9 piyasa geneli seri)
  IV  = text        (8-K olay bayrakları + LM duygu + dosya benzerliği)

HİPERPARAMETRE ARAMASI HÜCRE BAŞINA BİR KEZ, seed başına değil:
  hiperparametre hücrenin özelliğidir, eğitim stokastisitesinin değil.
  Transformer tarafında da böyle yapıldı — iki taraf simetrik.

EŞİK: her seed için YALNIZCA validation'dan seçilir.

KULLANIM:
    python run_xgb_factorial.py --cells I+II --trials 30
    python run_xgb_factorial.py --cells IV I+II+IV --force
    python run_xgb_factorial.py --device cpu       # GPU dolu olduğunda
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
    # ── Modalite IV: metin (SEC EDGAR) ──
    'IV':        (('text',),                       'Sadece metin'),
    'I+II+IV':   (('tech', 'fund', 'text'),        'Teknik + firma + metin'),
    'I+II+IV-e': (('tech', 'fund', 'text_event'),  'Teknik + firma + 8-K olayları'),
    'I+II+IV-l': (('tech', 'fund', 'text_lm'),     'Teknik + firma + LM duygu'),
}

# Rapor sırası — CELLS ile aynı sırayı korur, eksik hücreleri atlar
CELL_ORDER = list(CELLS)

# Transformer referansları (n=5, test MCC) — çıktıda yan yana göstermek için
TRANSFORMER_REF = {
    'I': 0.1110, 'II': 0.0068, 'III': -0.0056, 'I+II': 0.1273,
    'I+III': 0.0700, 'II+III': -0.0061, 'I+II+III': 0.0866,
    'I+II-xs': 0.1215,
    'I+II+IV': 0.1419, 'I+II+IV-e': 0.1254, 'I+II+IV-l': 0.1324,
}

TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def pick_device(forced=None):
    """
    GPU'da yeterli boş bellek varsa 'cuda', yoksa 'cpu'.

    Neden gerekli: device='cuda' sabit yazıldığında, GPU başkası tarafından
    doluysa XGBoost cudaErrorMemoryAllocation ile patlıyor. Oysa bu model
    CPU'da da çalışır, sadece yavaştır. Otomatik düşüşte deney durmaz.
    """
    if forced in ('cpu', 'cuda'):
        print(f"  [Device] {forced} (elle belirtildi)")
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free > 2e9:
                print(f"  [Device] cuda ({free/1e9:.1f} GB boş)")
                return 'cuda'
            print(f"  [Device] GPU dolu ({free/1e9:.1f} GB boş) → cpu")
    except Exception as e:
        print(f"  [Device] GPU kontrolü başarısız ({type(e).__name__}) → cpu")
    return 'cpu'


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
    if not cols:
        raise SystemExit(f"\n  ✗ Grup {groups} için hiç kolon bulunamadı.\n"
                         f"    Metin grubu ise önce: python build_text_features.py --stage all\n")
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
    ap.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                    help='Belirtilmezse GPU boş alanına göre otomatik seçilir')
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--force', action='store_true',
                    help='SEÇİLEN hücreleri yeniden koş (diğerleri korunur)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, 'xgb_factorial_raw.csv')
    par_path = os.path.join(args.outdir, 'xgb_factorial_params.csv')

    print("═" * 78)
    print("XGBOOST FAKTÖRİYEL — I / II / III / IV")
    print("═" * 78)
    print(f"  Hücreler : {args.cells}")
    print(f"  Seed'ler : {args.seeds}")
    print(f"  Optuna   : hücre başına {args.trials} deneme (seed başına DEĞİL)")
    print(f"  Eşik     : her seed için yalnızca val'dan seçilir")
    device = pick_device(args.device)
    print()

    # ══════════════════════════════════════════════════════════════
    # Resume
    # ══════════════════════════════════════════════════════════════
    # DİKKAT — bu blok İKİ ayrı veri kaybı hatasını önlüyor:
    #
    # (1) Eskiden --force verildiğinde `rows` boş listeyle başlıyordu ve koşu
    #     sonunda raw CSV bu boş listeden yeniden yazılıyordu. Yani
    #         python run_xgb_factorial.py --cells IV --force
    #     komutu, seçilmeyen hücrelerin TÜM satırlarını siliyordu.
    #
    # (2) Aynı şey parametre dosyası için de geçerliydi ve daha sinsiydi:
    #     xgb_factorial_params.csv sıfırlanınca run_portfolio_simulation.py
    #     aradığı hücreyi bulamıyor ve XGBoost'u SESSİZCE atlıyordu —
    #     portföy tablosunda bir strateji eksik kalıyor, hata verilmiyordu.
    #
    # Artık geçmiş kayıtlar her zaman okunur; --force yalnızca SEÇİLEN
    # hücrelerin kayıtlarını düşürür ve önce yedek alır.
    rows, done = [], set()
    if os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        if args.force:
            backup = raw_path.replace('.csv', '_backup.csv')
            prev.to_csv(backup, index=False)
            n_drop = int(prev['cell'].isin(args.cells).sum())
            prev = prev[~prev['cell'].isin(args.cells)]
            print(f"  [--force] {n_drop} satır yeniden koşulacak; "
                  f"{len(prev)} satır korunuyor.  Yedek → {backup}")
        rows = prev.to_dict('records')
        done = {(r['cell'], int(r['seed'])) for r in rows}
        if done and not args.force:
            print(f"  [Resume] {len(done)} koşu tamamlanmış, atlanacak.\n")

    param_rows = []
    if os.path.exists(par_path):
        pprev = pd.read_csv(par_path)
        if args.force:
            pbackup = par_path.replace('.csv', '_backup.csv')
            pprev.to_csv(pbackup, index=False)
            kept = pprev[~pprev['cell'].isin(args.cells)]
            print(f"  [--force] Parametreler: {len(kept)} hücre korunuyor "
                  f"({sorted(kept['cell'])}).  Yedek → {pbackup}")
            pprev = kept
        param_rows = pprev.to_dict('records')
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
        if cell in cached_params:
            best_params = {k: v for k, v in cached_params[cell].items()
                           if k not in ('cell', 'n_features', 'cv_pr_auc')
                           and pd.notna(v)}
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
        for k in ('cell', 'n_features', 'cv_pr_auc', 'tree_method', 'device'):
            fit_params.pop(k, None)
        fit_params['tree_method'] = 'hist'
        fit_params['device'] = device

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
    # CELL_ORDER kullanılıyor — eskiden metin hücreleri sabit listede yoktu
    # ve özet tablosunda hiç GÖRÜNMÜYORDU.
    order = [c for c in CELL_ORDER if c in df['cell'].values]

    print("\n" + "═" * 78)
    print("XGBOOST FAKTÖRİYEL SONUÇLARI")
    print("═" * 78)
    g = df.groupby('cell').agg(
        nf=('n_features', 'first'), val=('val_mcc', 'mean'),
        test=('test_mcc', 'mean'), sd=('test_mcc', 'std'),
        pr=('test_pr_auc', 'mean'), roc=('test_roc_auc', 'mean'),
        gap=('gap', 'mean'), n=('seed', 'nunique'))
    print(f"{'hücre':<11}{'özk':>5}{'val MCC':>10}{'test MCC':>10}{'±sd':>9}"
          f"{'PR-AUC':>9}{'ROC':>8}{'gap':>9}{'Transf.':>10}{'fark':>9}")
    print("-" * 90)
    for c in order:
        r = g.loc[c]
        tref = TRANSFORMER_REF.get(c, float('nan'))
        sd = r.sd if pd.notna(r.sd) else 0.0
        print(f"{c:<11}{int(r.nf):>5}{r.val:>10.4f}{r.test:>10.4f}{sd:>9.4f}"
              f"{r.pr:>9.4f}{r.roc:>8.4f}{r.gap:>+9.4f}{tref:>10.4f}"
              f"{r.test - tref:>+9.4f}")

    # ── Marjinal katkılar (seed-eşleşmeli) ──
    try:
        from scipy import stats
        P = df.pivot_table(index='seed', columns='cell', values='test_mcc')
        print("\n── MARJİNAL KATKI (seed-eşleşmeli t-testi) ──")
        print(f"{'etki':<22}{'XGBoost':>24}{'Transformer (ref)':>24}")
        print("-" * 70)
        comps = [
            ('Δ(II | I)',        'I+II',      'I',      +0.0163, 0.108),
            ('Δ(III | I)',       'I+III',     'I',      -0.0410, 0.034),
            ('Δ(III | I+II)',    'I+II+III',  'I+II',   -0.0407, 0.010),
            ('Δ(IV | I+II)',     'I+II+IV',   'I+II',   +0.0146, 0.072),
            ('Δ(IV-olay | I+II)', 'I+II+IV-e', 'I+II',  -0.0019, 0.863),
            ('Δ(IV-lm | I+II)',  'I+II+IV-l', 'I+II',   +0.0052, 0.441),
        ]
        for name, a, b, tref, tp in comps:
            if a in P.columns and b in P.columns:
                pair = P[[a, b]].dropna()
                if len(pair) >= 2:
                    d_ = pair[a] - pair[b]
                    t, p = stats.ttest_rel(pair[a], pair[b])
                    print(f"{name:<22}{d_.mean():>+13.4f} (p={p:.3f})"
                          f"{tref:>+15.4f} (p={tp:.3f})")
    except ImportError:
        print("\n(scipy yok — marjinal katkı testleri atlandı)")

    print("\n── YORUM ──")
    print("  Δ(III | ·) her iki ailede de NEGATİF ise → makro kirliliği model")
    print("    sınıfından bağımsız; mimari kusur değil, veri tasarımı olgusu.")
    print("  Δ(IV | I+II) her iki ailede de POZİTİF ise → metin katkısı da")
    print("    çift-doğrulanmış olur. Alt katmanların (olay vs LM) hangisinin")
    print("    sorumlu olduğu iki ailede FARKLI çıkabilir — bu da raporlanmalı.")

    summ = g.reset_index()
    summ.to_csv(os.path.join(args.outdir, 'xgb_factorial_summary.csv'), index=False)
    print(f"\nHam    → {raw_path}")
    print(f"Özet   → {args.outdir}/xgb_factorial_summary.csv")
    print(f"Params → {par_path}")


if __name__ == '__main__':
    main()