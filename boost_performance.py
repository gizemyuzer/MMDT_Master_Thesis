"""
boost_performance.py
────────────────────
XGBoost'u geçmek için iki ucuz yol — YENİDEN EĞİTİM YOK.

  1. SEED ENSEMBLE : 5 seed'in tahminlerini ortala
  2. YÜZDELİK EŞİK : sabit skor eşiği yerine "en riskli %k'yı işaretle"

Kayıtlı checkpoint'lerden (checkpoints/best_ms_*.pth) tahminleri çıkarır,
XGBoost'u da 5 seed ile eğitip ADİL karşılaştırma yapar.

ADALET NOTLARI:
  - Derin model ensemble'ı varsa XGBoost da ensemble olmalı → ikisi de 5 seed
  - Eşik stratejisi her iki tarafa da aynı uygulanır
  - Test setleri hizalanır: sekans modelleri her ticker'ın ilk 19 gününü
    kullanamıyor (155,544 → 151,155). XGBoost'u aynı satırlara kısıtlıyoruz.
  - TÜM eşik/k seçimleri SADECE validation'da yapılır, test'e dokunulmaz

KULLANIM:
    python boost_performance.py
    python boost_performance.py --seeds 42 43 44 45 46
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    matthews_corrcoef, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import RobustScaler

from datasets.feature_engineering import (
    prepare_dataset, get_dual_stream_dataloaders, get_feature_groups,
)
from models.transformer_model import DualEncoderTransformer

# run_multiseed.py ile BİREBİR aynı olmalı — checkpoint'ler bu config'le eğitildi
LADDER_CONFIG = dict(seq_len=20, d_model=64, n_heads=4, n_layers=2,
                     dropout=0.15, modality='multimodal')

# XGBoost'un Optuna'nın bulduğu parametreleri (train.py çıktısından)
XGB_PARAMS = dict(
    max_depth=2, learning_rate=0.05533894705931091, n_estimators=1900,
    subsample=0.612290511361587, colsample_bytree=0.5038470300093549,
    gamma=3.69714205665728, min_child_weight=1,
    reg_alpha=1.4479885716146466, reg_lambda=0.003496116877652556,
    scale_pos_weight=1.6171278021543518,
)

TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END = (
    '2019-12-31', '2020-01-01', '2021-12-31', '2022-01-01', '2024-12-31')


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ══════════════════════════════════════════════════════════════════
# Metrikler
# ══════════════════════════════════════════════════════════════════
def metrics_at_threshold(y, scores, thresh):
    pred = (scores >= thresh).astype(int)
    return dict(
        mcc=matthews_corrcoef(y, pred),
        acc=accuracy_score(y, pred),
        f1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, scores),
        pr_auc=average_precision_score(y, scores),
        flag_rate=pred.mean(),
    )


def best_fixed_threshold(y, scores, n=300):
    """Sabit skor eşiği — MCC'yi maksimize eden (mevcut yöntem)."""
    lo, hi = np.percentile(scores, [0.5, 99.5])
    if hi <= lo:
        lo, hi = scores.min(), scores.max()
    best_m, best_t = -2.0, float(np.median(scores))
    for t in np.linspace(lo, hi, n):
        p = (scores >= t).astype(int)
        if p.sum() in (0, len(p)):
            continue
        m = matthews_corrcoef(y, p)
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t, best_m


def best_percentile(y, scores, grid=None):
    """
    Yüzdelik eşik — "en riskli %k'yı işaretle".

    Neden: val (2020-21) ve test (2022-24) farklı rejimler; skor dağılımı
    kayıyor. Sabit skor eşiği test'te bambaşka bir oranı işaretliyor.
    Yüzdelik, dağılım kaysa bile aynı oranı korur.
    """
    if grid is None:
        grid = np.arange(0.03, 0.61, 0.005)
    best_m, best_k = -2.0, 0.125
    for k in grid:
        t = np.quantile(scores, 1 - k)
        p = (scores >= t).astype(int)
        if p.sum() in (0, len(p)):
            continue
        m = matthews_corrcoef(y, p)
        if m > best_m:
            best_m, best_k = m, float(k)
    return best_k, best_m


def apply_percentile(scores, k):
    return float(np.quantile(scores, 1 - k))


# ══════════════════════════════════════════════════════════════════
# 1) Derin model tahminlerini checkpoint'lerden çıkar
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        logits = model(x_tech=batch['tech_seq'].to(device),
                       x_fund=batch['fund_seq'].to(device))
        out.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(out)


def extract_deep_predictions(fusions, seeds, val_loader, test_loader,
                             tech_dim, fund_dim, device, outdir):
    """checkpoints/best_ms_*.pth → tahminler. Eğitim yok, sadece çıkarım."""
    os.makedirs(outdir, exist_ok=True)
    preds = {}
    for fusion in fusions:
        for seed in seeds:
            name = f"MS_{fusion}_seed{seed}"
            ckpt = os.path.join('checkpoints', f'best_{name.lower()}.pth')
            if not os.path.exists(ckpt):
                print(f"  ⚠️  bulunamadı, atlanıyor: {ckpt}")
                continue

            cache = os.path.join(outdir, f'{name}.npz')
            if os.path.exists(cache):
                z = np.load(cache)
                preds[(fusion, seed)] = (z['val'], z['test'])
                print(f"  ✓ {name}  (cache'ten)")
                continue

            model = DualEncoderTransformer(tech_dim=tech_dim, fund_dim=fund_dim,
                                           fusion_type=fusion, **LADDER_CONFIG)
            model.load_state_dict(torch.load(ckpt, weights_only=True,
                                             map_location='cpu'))
            model.to(device)
            v = infer(model, val_loader, device)
            t = infer(model, test_loader, device)
            np.savez_compressed(cache, val=v, test=t)
            preds[(fusion, seed)] = (v, t)
            print(f"  ✓ {name}  (checkpoint'ten çıkarıldı)")

            del model
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
    return preds


# ══════════════════════════════════════════════════════════════════
# 2) XGBoost — 5 seed (adil ensemble karşılaştırması için)
# ══════════════════════════════════════════════════════════════════
def build_flat_arrays(df):
    tech_cols, fund_cols = get_feature_groups(df)
    cols = tech_cols + fund_cols

    tr = df[(df.index >= '2010-01-01') & (df.index <= TRAIN_END)]
    va = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    te = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    med = tr[cols].median()
    sc = RobustScaler()
    Xtr = sc.fit_transform(tr[cols].fillna(med).values)
    Xva = sc.transform(va[cols].fillna(med).values)
    Xte = sc.transform(te[cols].fillna(med).values)

    # Hizalama anahtarları — sekans modelleriyle aynı satırları eşleştirmek için
    key_va = list(zip(va['Ticker'].values, va.index))
    key_te = list(zip(te['Ticker'].values, te.index))

    return (Xtr, tr['Target'].values, Xva, va['Target'].values,
            Xte, te['Target'].values, key_va, key_te)


def train_xgb_seeds(Xtr, ytr, Xva, yva, Xte, seeds, outdir):
    import xgboost as xgb
    os.makedirs(outdir, exist_ok=True)
    preds = {}
    for seed in seeds:
        cache = os.path.join(outdir, f'XGB_seed{seed}.npz')
        if os.path.exists(cache):
            z = np.load(cache)
            preds[seed] = (z['val'], z['test'])
            print(f"  ✓ XGB seed={seed}  (cache'ten)")
            continue
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed,
                              eval_metric='logloss', early_stopping_rounds=50,
                              n_jobs=-1, tree_method='hist')
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        v = m.predict_proba(Xva)[:, 1]
        t = m.predict_proba(Xte)[:, 1]
        np.savez_compressed(cache, val=v, test=t)
        preds[seed] = (v, t)
        print(f"  ✓ XGB seed={seed}  ({m.best_iteration + 1} ağaç)")
    return preds


# ══════════════════════════════════════════════════════════════════
# 3) Sekans test setine hizalama
# ══════════════════════════════════════════════════════════════════
def sequence_keys(df, start, end, seq_len=20):
    """
    DualStreamSequenceDataset'in hangi satırları kullandığını yeniden üretir.
    Aynı groupby('Ticker') + sort_index() + range(seq_len-1, len) mantığı.
    """
    sub = df[(df.index >= start) & (df.index <= end)]
    keys = []
    for ticker, g in sub.groupby('Ticker'):
        g = g.sort_index()
        for i in range(seq_len - 1, len(g)):
            keys.append((ticker, g.index[i]))
    return keys


def align_mask(flat_keys, seq_keys):
    """flat dizideki hangi satırlar sekans setinde de var?"""
    s = set(seq_keys)
    return np.array([k in s for k in flat_keys], dtype=bool)


# ══════════════════════════════════════════════════════════════════
# 4) Değerlendirme
# ══════════════════════════════════════════════════════════════════
def evaluate(name, val_scores, y_val, test_scores, y_test, strategy):
    """
    strategy='fixed'      → val'da MCC-optimal skor eşiği, test'e aynen uygula
    strategy='percentile' → val'da MCC-optimal %k, test'te aynı %k'yı uygula
    Her iki durumda da seçim SADECE validation'da yapılır.
    """
    if strategy == 'fixed':
        thr, _ = best_fixed_threshold(y_val, val_scores)
        test_thr = thr
        detail = f"thr={thr:.4f}"
    else:
        k, _ = best_percentile(y_val, val_scores)
        test_thr = apply_percentile(test_scores, k)
        detail = f"top {k*100:.1f}%"

    m = metrics_at_threshold(y_test, test_scores, test_thr)
    m['name'] = name
    m['strategy'] = strategy
    m['detail'] = detail
    return m


def print_table(rows, title):
    print(f"\n{'═'*96}")
    print(title)
    print('═'*96)
    print(f"{'Model':<34} {'MCC':>8} {'PR-AUC':>8} {'F1':>7} {'ACC':>7} "
          f"{'P':>6} {'R':>6}  {'eşik':>12}")
    print('─'*96)
    for r in rows:
        print(f"{r['name']:<34} {r['mcc']:>8.4f} {r['pr_auc']:>8.4f} "
              f"{r['f1']:>7.4f} {r['acc']:>7.4f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f}  {r['detail']:>12}")


def compare_to_baseline(rows, baseline_name):
    """Her modeli baseline ile metrik metrik karşılaştır."""
    base = next((r for r in rows if r['name'] == baseline_name), None)
    if base is None:
        return
    print(f"\n{'─'*96}")
    print(f"BASELINE'A GÖRE FARK  ({baseline_name})")
    print('─'*96)
    keys = ['mcc', 'pr_auc', 'f1', 'acc']
    print(f"{'Model':<34} " + " ".join(f"{k.upper():>10}" for k in keys) + "   sonuç")
    for r in rows:
        if r['name'] == baseline_name:
            continue
        diffs = [r[k] - base[k] for k in keys]
        cells = " ".join(f"{d:>+10.4f}" for d in diffs)
        n_win = sum(d > 0 for d in diffs)
        verdict = ("TÜM metriklerde önde ✓" if n_win == len(keys)
                   else f"{n_win}/{len(keys)} metrikte önde")
        print(f"{r['name']:<34} {cells}   {verdict}")


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--fusions', type=str, nargs='+',
                    default=['cross_attention', 'gated_cross_attention'])
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results/preds')
    args = ap.parse_args()

    device = get_device()
    print("═"*96)
    print("ENSEMBLE + EŞİK OPTİMİZASYONU  (yeniden eğitim yok)")
    print("═"*96)
    print(f"  Seed'ler : {args.seeds}")
    print(f"  Füzyonlar: {args.fusions}")
    print(f"  Cihaz    : {device}\n")

    # ── Veri ──
    print("[1/5] Dataset (cache)...")
    df = prepare_dataset(force_refresh=False)

    print("\n[2/5] Dataloader'lar...")
    _, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(df, seq_len=20, batch_size=args.batch_size)

    y_val_seq = val_loader.dataset.labels.astype(int)
    y_test_seq = test_loader.dataset.labels.astype(int)

    # ── Derin model tahminleri ──
    print("\n[3/5] Checkpoint'lerden tahmin çıkarımı...")
    deep = extract_deep_predictions(args.fusions, args.seeds,
                                    val_loader, test_loader,
                                    len(tech_cols), len(fund_cols),
                                    device, args.outdir)
    if not deep:
        print("\n✗ Hiç checkpoint bulunamadı. checkpoints/ klasörünü kontrol et.")
        return

    # ── XGBoost ──
    print("\n[4/5] XGBoost (5 seed, adil ensemble için)...")
    Xtr, ytr, Xva, yva, Xte, yte, key_va, key_te = build_flat_arrays(df)
    xgb_preds = train_xgb_seeds(Xtr, ytr, Xva, yva, Xte, args.seeds, args.outdir)

    # Test/val setlerini hizala (sekans modelleri ilk 19 günü kullanamıyor)
    print("\n      Test setleri hizalanıyor...")
    mask_va = align_mask(key_va, sequence_keys(df, VAL_START, VAL_END))
    mask_te = align_mask(key_te, sequence_keys(df, TEST_START, TEST_END))
    print(f"      val : {len(key_va):,} → {mask_va.sum():,}")
    print(f"      test: {len(key_te):,} → {mask_te.sum():,} "
          f"(sekans seti: {len(y_test_seq):,})")

    if mask_te.sum() != len(y_test_seq):
        print("      ⚠️  Hizalama tam eşleşmedi — XGBoost kendi setinde raporlanacak")
        aligned = False
    else:
        aligned = True

    # ── Değerlendirme ──
    print("\n[5/5] Değerlendirme...\n")
    for strategy in ['fixed', 'percentile']:
        rows = []

        # XGBoost: tek seed (mevcut baseline) + ensemble
        xv0, xt0 = xgb_preds[args.seeds[0]]
        if aligned:
            xv0a, xt0a = xv0[mask_va], xt0[mask_te]
            yva_a, yte_a = yva[mask_va], yte[mask_te]
        else:
            xv0a, xt0a, yva_a, yte_a = xv0, xt0, yva, yte
        rows.append(evaluate("XGBoost (tek seed)", xv0a, yva_a, xt0a, yte_a, strategy))

        xv_ens = np.mean([xgb_preds[s][0] for s in xgb_preds], axis=0)
        xt_ens = np.mean([xgb_preds[s][1] for s in xgb_preds], axis=0)
        if aligned:
            xv_ens, xt_ens = xv_ens[mask_va], xt_ens[mask_te]
        rows.append(evaluate(f"XGBoost (ensemble×{len(xgb_preds)})",
                             xv_ens, yva_a, xt_ens, yte_a, strategy))

        # Derin modeller: seed ortalaması + ensemble
        for fusion in args.fusions:
            got = [(s, deep[(fusion, s)]) for s in args.seeds if (fusion, s) in deep]
            if not got:
                continue
            label = {'cross_attention': 'Cross-attention',
                     'gated_cross_attention': 'Gated'}.get(fusion, fusion)

            singles = [evaluate(f"{label} seed{s}", v, y_val_seq, t, y_test_seq, strategy)
                       for s, (v, t) in got]
            avg = {k: float(np.mean([s_[k] for s_ in singles]))
                   for k in ['mcc', 'pr_auc', 'f1', 'acc', 'precision', 'recall']}
            avg.update(name=f"{label} (tek model ort.)", strategy=strategy,
                       detail=f"n={len(singles)}")
            rows.append(avg)

            v_ens = np.mean([v for _, (v, _) in got], axis=0)
            t_ens = np.mean([t for _, (_, t) in got], axis=0)
            rows.append(evaluate(f"{label} (ensemble×{len(got)})",
                                 v_ens, y_val_seq, t_ens, y_test_seq, strategy))

        title = ("SABİT SKOR EŞİĞİ (mevcut yöntem)" if strategy == 'fixed'
                 else "YÜZDELİK EŞİK (rejim kaymasına dayanıklı)")
        print_table(rows, f"{title} — TEST SETİ")
        compare_to_baseline(rows, "XGBoost (tek seed)")

        pd.DataFrame(rows).to_csv(
            os.path.join('results', f'ensemble_{strategy}.csv'), index=False)

    print(f"\n{'═'*96}")
    print("Sonuçlar → results/ensemble_fixed.csv, results/ensemble_percentile.csv")
    print("Tahminler → results/preds/*.npz  (tekrar çalıştırınca cache'ten okunur)")
    print('═'*96)


if __name__ == '__main__':
    main()