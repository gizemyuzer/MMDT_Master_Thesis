"""
run_mechanism_2x2.py — hedef tanımı mı, ön işleme mi? (danışman maddesi E)

SORU
    Bölüm 6.1 şunu iddia ediyor: modeller ekonomik değer üretemiyor çünkü
    (i) hedef, volatiliteye göre ölçeklendiği için piyasa çapındaki düşüşleri
    değil, göreli düşüşleri işaretliyor; ve (ii) kesitsel normalizasyon
    piyasa SEVİYESİNİ siliyor, dolayısıyla modelin "bugün piyasa riskli mi"
    sorusuna erişimi yok.

    Bu iki açıklama tezde birlikte öne sürülüyor ama hiçbir yerde
    AYRIŞTIRILMIYOR. Bu haliyle bir hipotez, bir bulgu değil.

TASARIM — tam 2×2 faktöriyel
                     kesitsel norm (XS)      seviye koruyan (LEVEL)
    vol-ayarlı hedef        A                        B
    sabit %10 hedef         C                        D

    A = mevcut tez kurulumu.
    B = hedef aynı, model piyasa seviyesini görüyor  → ön işleme etkisi
    C = ön işleme aynı, hedef mutlak                 → hedef etkisi
    D = her ikisi de değişti                          → etkileşim

    Marjinal etkiler:
        ön işleme:  (B−A) ve (D−C)
        hedef:      (C−A) ve (D−B)
        etkileşim:  (D−C) − (B−A)

NEDEN XGBoost
    Dört hücre × birden çok seed'in Transformer ile koşulması GPU-günleri
    alır ve soru mimariyle ilgili değil. XGBoost hızlı, GPU gerektirmez ve
    Bölüm 5.3'te Transformer ile aynı yönde sonuç veriyor — mekanizma
    testi için doğru araç. Bulgu, model ailesinden bağımsız olmalıdır.

ORTAK DEĞERLENDİRME
    Dört hücre AYNI (tarih, hisse) anahtarlarında değerlendirilir; hem
    istatistiksel (MCC / PR-AUC / ROC) hem ekonomik (portfolio_engine ile
    Calmar / maksDD) sonuç aynı örneklemden gelir. Ekonomik karşılaştırma
    pasif referanslara karşı yapılır, birbirlerine karşı değil.

KULLANIM
    python run_mechanism_2x2.py                    # 3 seed, tam 2×2
    python run_mechanism_2x2.py --seeds 42 43 44 45 46
    python run_mechanism_2x2.py --cells A B        # yalnız bazı hücreler
"""
import os, argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import (matthews_corrcoef, average_precision_score,
                             roc_auc_score, precision_score, recall_score)

from datasets.feature_engineering import prepare_dataset
from portfolio_engine import simulate_positions, performance

warnings.filterwarnings('ignore')

TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
HORIZON = 20
OUT = 'results'

CELLS = {
    'A': ('vol_adj', 'xs',    'mevcut tez kurulumu'),
    'B': ('vol_adj', 'level', 'ön işleme değişti'),
    'C': ('fixed',   'xs',    'hedef değişti'),
    'D': ('fixed',   'level', 'her ikisi de'),
}


# ══════════════════════════════════════════════════════════════════
# 1. HEDEFLER — ikisi de AYNI ham düşüşten türetilir
# ══════════════════════════════════════════════════════════════════
def forward_drawdown(ds):
    """Her (hisse, gün) için önümüzdeki HORIZON günün maksimum düşüşü.

    dd[i] = (close[i] − min(close[i+1 .. i+HORIZON])) / close[i]

    Bölüm 4.3'teki döngüsel tanımla birebir aynı, vektörize edilmiş hâli;
    aşağıda mevcut 'Target' kolonuna karşı doğrulanır.

    ⚠️ KONUMSAL YAZIM: panel index'i tarih ve TEKRARLI (her tarihte ~400
    hisse). out.loc[g.index] = ... yazmak o tarihlerdeki bütün hisseleri
    seçer ve uzunluk hatası verir. Bu yüzden tamsayı konum kullanılıyor.
    """
    work = ds[['Ticker', 'Close']].copy()
    work['_pos'] = np.arange(len(work))
    out = np.full(len(work), np.nan)
    for _, g in work.groupby('Ticker', sort=False):
        s = g['Close'].astype(float).reset_index(drop=True)
        fmin = (s.shift(-1)
                 .rolling(HORIZON, min_periods=HORIZON).min()
                 .shift(-(HORIZON - 1)))
        out[g['_pos'].to_numpy()] = ((s - fmin) / s).to_numpy()
    return out                      # numpy dizisi — index hizalaması yok


def make_targets(ds, dd):
    """İki etiket: volatiliteye ölçekli eşik ve sabit %10 eşik.

    Karşılaştırmalar numpy üzerinde yapılır; tekrarlı index'te Series
    karşılaştırması hizalamaya girip kartezyen büyüme üretebilir.
    """
    vol = ds['Vol_20d'].to_numpy(dtype=float)
    ev = (vol / np.sqrt(252)) * np.sqrt(HORIZON)
    thr_vol = np.clip(1.5 * ev, 0.05, 0.25)
    ok = np.isfinite(dd)
    def lab(cond):
        y = np.full(len(dd), np.nan)
        y[ok] = cond[ok].astype(float)
        return y
    with np.errstate(invalid='ignore'):
        return {'vol_adj': lab(dd >= thr_vol),
                'fixed':   lab(dd >= 0.10)}


# ══════════════════════════════════════════════════════════════════
# 2. ÖN İŞLEME — tek fark, seviye bilgisinin korunup korunmadığı
# ══════════════════════════════════════════════════════════════════
def preprocess(X, dates, mode, train_mask):
    """
    xs    : tarih-içi yüzdelik sıra. Her gün [0,1]'e ölçeklenir, dolayısıyla
            o güne ait PİYASA SEVİYESİ tanım gereği silinir — bütün hisseler
            aynı anda ikiye katlansa da sıralama değişmez.
    level : eğitim döneminde fit edilen tek bir z-skor. Kesitsel bilgi
            korunur AMA seviye de korunur: bugün herkesin volatilitesi
            yüksekse, bütün satırlar yüksek değer alır.
    """
    if mode == 'xs':
        # dates bir numpy dizisi; Series geçilirse tekrarlı index hizalanır
        return X.groupby(dates, sort=False).rank(pct=True)
    mu = X[train_mask].mean()
    sd = X[train_mask].std().replace(0.0, 1.0)
    return ((X - mu) / sd).clip(-8, 8)


# ══════════════════════════════════════════════════════════════════
def pick_threshold(y, s, n_grid=300):
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


def stat_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    return {'mcc': matthews_corrcoef(y, pred) if pred.min() != pred.max() else 0.0,
            'pr_auc': average_precision_score(y, s),
            'roc_auc': roc_auc_score(y, s),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0)}


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--cells', nargs='+', default=list(CELLS))
    ap.add_argument('--exclude-pct', type=float, default=20.0)
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--n-estimators', type=int, default=400)
    ap.add_argument('--outdir', default=OUT)
    args = ap.parse_args()

    from xgboost import XGBClassifier

    ds = prepare_dataset(force_refresh=False).copy()
    ds = ds.sort_index()
    dates = ds.index.to_numpy()          # numpy: hizalama tuzağı yok

    # ── hedefleri kur ve mevcut etikete karşı doğrula ──
    print("\n[1/4] Hedefler kuruluyor...")
    dd = forward_drawdown(ds)                      # numpy
    TG = make_targets(ds, dd)                      # numpy sözlüğü
    tgt = ds['Target'].to_numpy(dtype=float)
    ok = np.isfinite(dd) & np.isfinite(tgt)
    agree = float((TG['vol_adj'][ok] == tgt[ok]).mean())
    print(f"  yeniden üretilen vol-ayarlı etiket ↔ mevcut 'Target': "
          f"%{100*agree:.2f} uyum  ({int(ok.sum()):,} satır)")
    if agree < 0.98:
        print("  ⚠️ Uyum düşük — etiket yeniden üretimi mevcut tanımdan sapıyor.")
    for k, v in TG.items():
        print(f"  {k:<8} olay oranı: %{100*np.nanmean(v):.2f}")

    # ── hedef–volatilite ilişkisi: iddianın doğrudan testi ──
    print("\n[2/4] Olay oranı × volatilite beşliği")
    q = (ds.groupby(dates, sort=False)['Vol_20d']
           .transform(lambda x: pd.qcut(x.rank(method='first'), 5,
                                        labels=False, duplicates='drop'))
           .to_numpy(dtype=float))
    print(f"  {'beşlik':<9}{'vol_adj':>10}{'fixed':>10}")
    print("  " + "-" * 29)
    rates = {}
    for k in TG:
        rates[k] = [100 * float(np.nanmean(TG[k][q == i])) for i in range(5)]
    for i in range(5):
        lab = f"Q{i+1}" + (" (düşük)" if i == 0 else " (yüksek)" if i == 4 else "")
        print(f"  {lab:<9}{rates['vol_adj'][i]:>9.2f}%{rates['fixed'][i]:>9.2f}%")
    print(f"  {'oran Q1/Q5':<9}{rates['vol_adj'][0]/max(rates['vol_adj'][4],1e-9):>9.2f}x"
          f"{rates['fixed'][0]/max(rates['fixed'][4],1e-9):>9.2f}x")
    print("  → vol_adj için oran > 1 ise hedef volatiliteyi TERSİNE çeviriyor;")
    print("    fixed için < 1 beklenir (volatil hisse daha çok düşer).")

    # ── özellikler ──
    drop = {'Target', 'Ticker', 'Sector', 'Close', 'Open', 'High', 'Low',
            'Volume', 'Adj Close'}
    feat = [c for c in ds.columns
            if c not in drop and pd.api.types.is_numeric_dtype(ds[c])
            and not c.endswith('_XS')]     # XS türevleri hariç: ön işlemeyi
                                           # burada biz kontrol ediyoruz
    X_raw = ds[feat].astype(float)
    print(f"\n[3/4] {len(feat)} özellik | panel {len(ds):,} satır")

    is_tr = (ds.index <= TRAIN_END)
    is_va = (ds.index >= VAL_START) & (ds.index <= VAL_END)
    is_te = (ds.index >= TEST_START) & (ds.index <= TEST_END)

    # ── ortak örneklem: dört hücrenin de skor ürettiği satırlar ──
    # Tümü numpy bool; tekrarlı index'te Series & Series hizalamaya girer.
    valid = (X_raw.notna().any(axis=1).to_numpy()
             & np.isfinite(dd)
             & np.isfinite(ds['Vol_20d'].to_numpy(dtype=float)))

    Xp = {m: preprocess(X_raw, dates, m, is_tr & valid)
          for m in ('xs', 'level')}

    # ── portföy paneli ──
    sub = ds[is_te]
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    rebal = set(ret.index[::args.rebalance])
    flat = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

    rows, preds = [], {}
    print(f"\n[4/4] {len(args.cells)} hücre × {len(args.seeds)} seed eğitiliyor...")
    for cell in args.cells:
        tgt_name, prep_name, desc = CELLS[cell]
        y = TG[tgt_name]                      # numpy, NaN'lı
        m_all = valid & np.isfinite(y)
        X = Xp[prep_name]
        mtr = is_tr & m_all
        mva = is_va & m_all
        mte = is_te & m_all
        yi = np.where(m_all, np.nan_to_num(y), 0).astype(int)

        for seed in args.seeds:
            clf = XGBClassifier(
                n_estimators=args.n_estimators, max_depth=5,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, eval_metric='logloss', tree_method='hist',
                random_state=seed, n_jobs=-1)
            clf.fit(X[mtr], yi[mtr])

            s_va = clf.predict_proba(X[mva])[:, 1]
            s_te = clf.predict_proba(X[mte])[:, 1]
            thr, vmcc = pick_threshold(yi[mva], s_va)
            r = stat_metrics(yi[mte], s_te, thr)

            # ekonomik: AYNI kural, AYNI panel
            sw = (pd.DataFrame({'date': ds.index.to_numpy()[mte],
                                'ticker': ds['Ticker'].to_numpy()[mte],
                                'p': s_te})
                  .pivot_table(index='date', columns='ticker', values='p')
                  .reindex(index=ret.index).reindex(columns=ret.columns))
            net, tov, inv, _, _ = simulate_positions(
                sw, ret, rebal, args.exclude_pct, args.cost_bps)
            e = performance(net, f'{cell}_seed{seed}')

            rows.append({'cell': cell, 'target': tgt_name, 'prep': prep_name,
                         'seed': seed, 'n_test': int(mte.sum()),
                         'threshold': thr, 'val_mcc': vmcc, **r,
                         'cagr': e['cagr'], 'max_drawdown': e['max_drawdown'],
                         'calmar': e['calmar'], 'turnover': tov})
            preds[f'{cell}_seed{seed}'] = sw
            print(f"  {cell} ({desc:<20}) seed={seed}  "
                  f"MCC={r['mcc']:.4f}  Calmar={e['calmar']:.3f}")

    # ── pasif referanslar, aynı panel ──
    bench = []
    for nm, rb in (('buy_hold_true', {ret.index[0]}), ('equal_weight_rebal', rebal)):
        net, tov, inv, _, _ = simulate_positions(flat, ret, rb, 0.0, args.cost_bps)
        bench.append({**performance(net, nm), 'turnover': tov})
    B = pd.DataFrame(bench)

    R = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    R.to_csv(os.path.join(args.outdir, 'mechanism_2x2.csv'), index=False)
    B.to_csv(os.path.join(args.outdir, 'mechanism_2x2_benchmarks.csv'), index=False)

    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("2×2 MEKANİZMA DENEYİ")
    print("═" * 78)
    G = R.groupby('cell')[['mcc', 'pr_auc', 'roc_auc', 'cagr',
                           'max_drawdown', 'calmar']].agg(['mean', 'std'])
    print(f"{'hücre':<6}{'hedef':<10}{'ön işleme':<11}{'MCC':>9}{'±':>7}"
          f"{'PR-AUC':>9}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}{'±':>7}")
    print("-" * 86)
    for c in [k for k in CELLS if k in G.index]:
        g = G.loc[c]; t, p, _ = CELLS[c]
        print(f"{c:<6}{t:<10}{p:<11}{g[('mcc','mean')]:>9.4f}"
              f"{np.nan_to_num(g[('mcc','std')]):>7.4f}{g[('pr_auc','mean')]:>9.4f}"
              f"{100*g[('cagr','mean')]:>8.2f}%{100*g[('max_drawdown','mean')]:>9.2f}%"
              f"{g[('calmar','mean')]:>9.3f}{np.nan_to_num(g[('calmar','std')]):>7.3f}")
    print("-" * 86)
    for _, b in B.iterrows():
        print(f"{b['strategy']:<27}{'':>25}{100*b['cagr']:>8.2f}%"
              f"{100*b['max_drawdown']:>9.2f}%{b['calmar']:>9.3f}")

    # ── marjinal etkiler ──
    def mean_of(c, col):
        v = R[R.cell == c][col]
        return float(v.mean()) if len(v) else np.nan

    print("\n" + "═" * 78)
    print("MARJİNAL ETKİLER")
    print("═" * 78)
    print(f"{'kontrast':<38}{'ΔMCC':>10}{'ΔCalmar':>11}{'ΔmaksDD':>11}")
    print("-" * 70)
    contrasts = [
        ('ön işleme | vol-ayarlı hedef  (B−A)', 'B', 'A'),
        ('ön işleme | sabit hedef       (D−C)', 'D', 'C'),
        ('hedef     | kesitsel norm     (C−A)', 'C', 'A'),
        ('hedef     | seviye koruyan    (D−B)', 'D', 'B'),
    ]
    got = {}
    for lab, hi, lo in contrasts:
        if hi not in R.cell.values or lo not in R.cell.values:
            continue
        dm = mean_of(hi, 'mcc') - mean_of(lo, 'mcc')
        dc = mean_of(hi, 'calmar') - mean_of(lo, 'calmar')
        dd_ = 100 * (mean_of(hi, 'max_drawdown') - mean_of(lo, 'max_drawdown'))
        got[lab] = (dm, dc)
        print(f"{lab:<38}{dm:>+10.4f}{dc:>+11.3f}{dd_:>+10.2f}pp")
    if len(got) == 4:
        k = list(got)
        inter = got[k[1]][1] - got[k[0]][1]
        print("-" * 70)
        print(f"{'etkileşim (Calmar)':<38}{'':>10}{inter:>+11.3f}")

    # ── yorum ──
    print("\n" + "═" * 78)
    print("YORUM")
    print("═" * 78)
    if all(c in R.cell.values for c in 'ABCD'):
        d_prep = 0.5 * ((mean_of('B','calmar') - mean_of('A','calmar')) +
                        (mean_of('D','calmar') - mean_of('C','calmar')))
        d_tgt  = 0.5 * ((mean_of('C','calmar') - mean_of('A','calmar')) +
                        (mean_of('D','calmar') - mean_of('B','calmar')))
        bm = float(B[B.strategy == 'equal_weight_rebal']['calmar'].iloc[0])
        print(f"  ortalama ön işleme etkisi (Calmar): {d_prep:+.3f}")
        print(f"  ortalama hedef etkisi     (Calmar): {d_tgt:+.3f}")
        dom = 'ön işleme' if abs(d_prep) > abs(d_tgt) else 'hedef tanımı'
        print(f"  → baskın faktör: {dom}")
        best = R.groupby('cell')['calmar'].mean().max()
        if best > bm:
            print(f"  → EN İYİ HÜCRE PASİF REFERANSI GEÇİYOR ({best:.3f} > {bm:.3f}).")
            print("    Bölüm 6.1'deki hipotez destekleniyor: ekonomik başarısızlık")
            print("    kurulum tercihlerinden kaynaklanıyor, öğrenilebilirlikten değil.")
        else:
            print(f"  → hiçbir hücre pasif referansı geçmiyor "
                  f"(en iyi {best:.3f} < {bm:.3f}).")
            print("    Bölüm 6.1'deki hipotez DESTEKLENMİYOR: hedefi ve ön işlemeyi")
            print("    değiştirmek ekonomik açığı kapatmıyor. Açıklama başka yerde.")
    else:
        print("  Tam 2×2 koşulmadı; marjinal etkiler eksik.")

    print(f"\n→ {args.outdir}/mechanism_2x2.csv")
    print(f"→ {args.outdir}/mechanism_2x2_benchmarks.csv")


if __name__ == '__main__':
    main()