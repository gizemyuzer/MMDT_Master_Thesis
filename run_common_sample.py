"""
run_common_sample.py — TÜM modellerin AYNI gözlemlerde değerlendirilmesi.

SORUN (danışman geri bildirimi C2)
    run_literature_baselines.py dört baseline için ortak örneklem kuruyor,
    ama Transformer/XGBoost sayıları bu kesişimde YENİDEN HESAPLANMIYOR;
    THESIS_REF içinde sabit olarak yazdırılıyor. Ayrıca ortak örneklem
    oranı tüm panel üzerinden hesaplanıp test kapsamı gibi raporlanmış
    (%80.3); test kümesine göre doğru oran %91.3'tür.

BU SCRIPT
    1) Dört baseline skorunu yeniden üretir (aynı fonksiyonları kullanır).
    2) Kayıtlı .npz model tahminlerini yükler.
    3) HEPSİNİN skor ürettiği (tarih, hisse) kesişimini kurar.
    4) Eşikleri validation üzerinde — aynı kesişim kuralıyla — seçer.
    5) Tüm metrikleri bu ortak örneklemde yeniden hesaplar.

    Böylece Bölüm 5.1'deki tablo gerçekten like-for-like olur ve
    "n kat daha iyi" türü ifadeler savunulabilir hâle gelir.

KULLANIM
    python run_common_sample.py
    python run_common_sample.py --pattern "MS_gated_cross_attention_seed*.npz"
"""
import os, glob, argparse
import numpy as np, pandas as pd
from sklearn.metrics import (matthews_corrcoef, average_precision_score,
                             roc_auc_score, precision_score, recall_score)

from datasets.feature_engineering import prepare_dataset
import run_literature_baselines as LB

VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
OUT = 'results'


# ══════════════════════════════════════════════════════════════════
def pick_threshold(y, s, n_grid=300):
    """Validation üzerinde MCC'yi maksimize eden eşik (Bölüm 4.12 ile aynı)."""
    ok = ~np.isnan(s)
    y, s = np.asarray(y)[ok], np.asarray(s)[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lo, hi = np.percentile(s, [0.5, 99.5])
    best_t, best_m = np.nan, -1.0
    for t in np.linspace(lo, hi, n_grid):
        pred = (s >= t).astype(int)
        if pred.min() == pred.max():
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return best_t, best_m


def metrics(y, s, thr):
    ok = ~np.isnan(s)
    y, s = np.asarray(y)[ok], np.asarray(s)[ok]
    pred = (s >= thr).astype(int)
    return {
        'n': int(len(y)),
        'mcc': matthews_corrcoef(y, pred) if pred.min() != pred.max() else 0.0,
        'pr_auc': average_precision_score(y, s),
        'roc_auc': roc_auc_score(y, s),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='multi_pure_seed*.npz')
    ap.add_argument('--preds-dir', default=os.path.join('results', 'preds_v2'))
    ap.add_argument('--outdir', default=OUT)
    args = ap.parse_args()

    tag = os.path.basename(args.pattern).replace('_seed*.npz','').replace('*','')
    ds = prepare_dataset(force_refresh=False)
    ds = ds.copy()
    ds['_date'] = ds.index
    key = pd.MultiIndex.from_arrays([ds['_date'], ds['Ticker']],
                                    names=['date', 'ticker'])
    print(f"panel: {len(ds):,} satır | {ds['Ticker'].nunique()} hisse")

    # ── 1) baseline skorları (yön: yüksek = riskli olacak şekilde çevrilir) ──
    S = {}
    S['Naive volatility (Vol_20d)'] = ds['Vol_20d'].values.astype(float)
    S['Inverse volatility'] = -ds['Vol_20d'].values.astype(float)   # hedefe uygun
    S['Altman Z (1968)'] = -ds['Altman_Z'].values.astype(float)     # düşük Z = riskli
    try:
        gv = LB.garch_scores(ds)
        if not gv.empty:
            S['GARCH(1,1)'] = (LB.align_to_panel(ds, gv, ['garch_vol'])
                               ['garch_vol'].values.astype(float))
    except ImportError:
        print("  ⚠️ 'arch' yok → GARCH atlandı")
    p = LB.chs_scores(ds)
    if p is not None:
        S['CHS (2008) logit'] = np.asarray(p, dtype=float)

    # ── 2) model tahminleri ──
    mfiles = sorted(glob.glob(os.path.join(args.preds_dir, args.pattern)))
    if not mfiles:
        raise SystemExit(f"{args.preds_dir}/{args.pattern} bulunamadı — "
                         f"önce export_preds.py koşun.")
    for f in mfiles:
        z = np.load(f, allow_pickle=True)
        if 'dates' not in z:
            raise SystemExit(f"{os.path.basename(f)} eski şemada; "
                             f"export_preds.py ile yeniden üretin.")
        name = os.path.basename(f).replace('.npz', '')
        ser = pd.Series(z['prob'].astype(float),
                        index=pd.MultiIndex.from_arrays(
                            [pd.to_datetime(z['dates']), z['tickers']],
                            names=['date', 'ticker']))
        ser = ser[~ser.index.duplicated()]
        S[name] = ser.reindex(key).values
    print(f"değerlendirilecek skor kaynağı: {len(S)}")

    # ── 3) ortak örneklem ──
    finite = np.ones(len(ds), dtype=bool)
    for v in S.values():
        finite &= np.isfinite(v)

    is_val = (ds['_date'] >= VAL_START) & (ds['_date'] <= VAL_END)
    is_test = (ds['_date'] >= TEST_START) & (ds['_date'] <= TEST_END)
    m_val = (finite & is_val).values
    m_test = (finite & is_test).values

    print()
    print("═" * 74)
    print("ORTAK ÖRNEKLEM")
    print("═" * 74)
    print(f"  validation: {m_val.sum():,} / {is_val.sum():,} "
          f"(%{100*m_val.sum()/max(is_val.sum(),1):.1f})")
    print(f"  test      : {m_test.sum():,} / {is_test.sum():,} "
          f"(%{100*m_test.sum()/max(is_test.sum(),1):.1f})")
    print(f"  panelin tamamına oranı: %{100*finite.mean():.1f}  "
          f"← tezde bu sayı test kapsamı sanılmıştı")

    y = ds['Target'].values.astype(int)

    # ── 4) eşik validation'da, metrik test'te ──
    rows = []
    for name, s in S.items():
        thr, vmcc = pick_threshold(y[m_val], s[m_val])
        if np.isnan(thr):
            print(f"  ⚠️ {name}: eşik seçilemedi, atlandı"); continue
        r = metrics(y[m_test], s[m_test], thr)
        rows.append({'model': name, 'threshold': thr, 'val_mcc': vmcc, **r})

    R = pd.DataFrame(rows).sort_values('mcc', ascending=False)
    R['grp'] = R['model'].str.replace(r'_seed\d+', '', regex=True)
    R.to_csv(os.path.join(args.outdir, f'common_sample_v2_{tag}.csv'), index=False)

    print()
    print("═" * 74)
    print(f"ORTAK ÖRNEKLEMDE TEST METRİKLERİ (n = {m_test.sum():,})")
    print("═" * 74)
    G = R.groupby('grp')[['mcc', 'pr_auc', 'roc_auc', 'precision', 'recall']].mean()
    G['n_seed'] = R.groupby('grp').size()
    print(f"{'model':<34}{'MCC':>9}{'PR-AUC':>9}{'ROC':>9}{'prec':>8}{'rec':>8}{'n':>4}")
    print("-" * 74)
    for k in G.sort_values('mcc', ascending=False).index:
        r = G.loc[k]
        print(f"{k[:33]:<34}{r.mcc:>9.4f}{r.pr_auc:>9.4f}{r.roc_auc:>9.4f}"
              f"{r.precision:>8.4f}{r.recall:>8.4f}{int(r.n_seed):>4}")

    print(f"\n→ {args.outdir}/common_sample_v2_{tag}.csv")
    print("\nNOT: Bu tablo Bölüm 5.1'in yerini alır. Eski tablodaki model")
    print("     satırları tam test kümesinden geliyordu; buradakiler")
    print("     baseline'larla AYNI gözlemlerden geliyor.")


if __name__ == '__main__':
    main()