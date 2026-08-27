"""
run_literature_baselines.py
────────────────────────────
LİTERATÜR BASELINE'LARI — danışman aksiyon maddesi #1.

═══════════════════════════════════════════════════════════════════════
NEDEN BU BASELINE'LAR
═══════════════════════════════════════════════════════════════════════
Tezin sorusuna (kısa ufuklu, kesitsel, olay bazlı drawdown sınıflandırması)
birebir uyan bir çalışma yok. Komşu literatürler farklı ufuk ya da farklı
hedef kullanıyor. Bu yüzden replike edilen şey makalelerin GÖREVİ değil,
paylaştıkları BASELINE'lardır:

  GARCH(1,1)  Sardelich & Manandhar (2018) ve Gupta (2025) — ikisinin de
              karşılaştırma ölçütü. Yalnızca fiyat kullanır, dolayısıyla
              "only the data types mentioned in the papers" şartını karşılar.
              naive_vol'ün düzgün parametrik versiyonu.

  CHS (2008)  Campbell, Hilscher & Szilagyi, Journal of Finance 63(6).
              Kesitsel, firma düzeyi, logit. Hem muhasebe hem piyasa
              değişkeni kullanır — tezin I+II hücresiyle aynı bilgi kümesi.
              Sıkıntı tahmininde Altman Z'nin yerini almış kanonik referans.

═══════════════════════════════════════════════════════════════════════
DÜRÜSTLÜK NOTLARI — tez metnine aynen geçmeli
═══════════════════════════════════════════════════════════════════════
· CHS orijinalinde İFLAS olasılığını tahmin eder. Burada aynı değişken
  kümesi ve aynı model sınıfı tezin etiketine uygulanır. Bu bir replikasyon
  değil, baseline TRANSFERİDİR.
· CHS'nin NIMTAAVG / EXRETAVG geometrik ağırlıklı ortalamaları
  uygulanmadı; anlık değerler kullanıldı.
· RSIZE, CRSP toplam piyasa değeri yerine EVREN İÇİ toplamla hesaplandı.
  Seviye kayması sabit olduğu için kesitsel sıralamayı değiştirmez.
· GARCH parametreleri YALNIZCA eğitim döneminde tahmin edilir, sonra sabit
  parametreyle ileri filtrelenir. Yuvarlanan yeniden tahmin YOK.
· Baseline'lar farklı kapsama oranlarına sahip. Farklı örneklemlerde ölçülen
  metrikler karşılaştırılamayacağı için ORTAK ÖRNEKLEM tablosu asıl sonuç
  olarak raporlanır.

KULLANIM:
    pip install arch
    python run_literature_baselines.py --stage fetch    # WRDS, bir kez
    python run_literature_baselines.py --stage run
"""
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score)

from datasets.feature_engineering import prepare_dataset
from models.pytorch_trainer import find_best_threshold_mcc

CHS_PATH = os.path.join('datasets', 'chs_variables.csv')
TRAIN_END = '2019-12-31'
VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'

CHS_COLS = ['NIMTA', 'TLMTA', 'EXRET', 'SIGMA', 'RSIZE', 'CASHMTA', 'MB', 'PRICE']

# Tezin modelleri — çıktıda yan yana göstermek için (n=5)
THESIS_REF = [
    ('Transformer I+II+IV', 0.1419, 0.2048, 0.6521),
    ('XGBoost I+II+IV',     0.1492, 0.2164, 0.6454),
    ('Transformer I+II',    0.1273, 0.1975, 0.6394),
    ('Transformer I (tech)', 0.1110, 0.1873, 0.6284),
]


# ══════════════════════════════════════════════════════════════════
# AŞAMA 1 — CHS değişkenlerini WRDS'ten çek
# ══════════════════════════════════════════════════════════════════
def stage_fetch():
    """ANA CACHE'E DOKUNULMAZ. Ayrı dosyaya yazılır, yalnızca burada birleşir."""
    from datasets.feature_engineering import TARGET_TICKERS
    import wrds

    print("═" * 74)
    print("AŞAMA 1 — CHS (2008) değişkenleri")
    print("═" * 74)
    tstr = "','".join(list(TARGET_TICKERS))
    db = wrds.Connection(wrds_username='gizemyuzer')
    try:
        print("  [1/3] CRSP fiyat + hisse sayısı...")
        px = db.raw_sql(f"""
            SELECT a.date, b.ticker, a.prc, a.shrout, a.ret
            FROM crsp.dsf AS a
            JOIN crsp.stocknames AS b ON a.permno = b.permno
            WHERE b.ticker IN ('{tstr}')
              AND a.date BETWEEN '2009-06-01' AND '2024-12-31'
              AND a.date >= b.namedt AND a.date <= b.nameenddt
        """)
        print("  [2/3] CRSP piyasa endeksi...")
        mkt = db.raw_sql("""
            SELECT date, vwretd FROM crsp.dsi
            WHERE date BETWEEN '2009-06-01' AND '2024-12-31'
        """)
        # NOT: 'tic' comp.fundq'nun kendi kolonu; comp.company'de yok.
        print("  [3/3] Compustat çeyreklik...")
        fund = db.raw_sql(f"""
            SELECT tic, datadate, rdq, niq, ltq, cheq, ceqq
            FROM comp.fundq
            WHERE tic IN ('{tstr}')
              AND datadate BETWEEN '2009-01-01' AND '2024-12-31'
              AND indfmt = 'INDL' AND datafmt = 'STD'
              AND popsrc = 'D' AND consol = 'C'
        """)
    finally:
        db.close()

    px['date'] = pd.to_datetime(px['date'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    # Bir ticker birden fazla permno'ya eşleşebilir (isim değişikliği, örtüşen
    # namedt aralıkları). Aynı gün için mükerrer satır bırakılırsa sonraki
    # birleştirmeler paneli şişirir.
    px = px.sort_values(['ticker', 'date']).drop_duplicates(['date', 'ticker'], keep='last')
    px = px.merge(mkt, on='date', how='left')
    px['prc'] = px['prc'].abs()
    px['ME'] = px['prc'] * px['shrout'] * 1000.0

    tot = px.groupby('date')['ME'].transform('sum')
    px['RSIZE'] = np.log(px['ME'] / tot)
    px['PRICE'] = np.log(px['prc'].clip(upper=15.0))

    px = px.sort_values(['ticker', 'date'])
    px['SIGMA'] = px.groupby('ticker')['ret'].transform(
        lambda s: s.rolling(63, min_periods=20).std()) * np.sqrt(252)
    px['exc'] = np.log1p(px['ret'].fillna(0)) - np.log1p(px['vwretd'].fillna(0))
    px['EXRET'] = px.groupby('ticker')['exc'].transform(
        lambda s: s.rolling(63, min_periods=20).sum())

    # POINT-IN-TIME: rdq ile hizala, yoksa datadate + 60 gün
    fund['rdq'] = pd.to_datetime(fund['rdq'])
    fund['datadate'] = pd.to_datetime(fund['datadate'])
    fund['avail'] = fund['rdq'].fillna(fund['datadate'] + pd.Timedelta(days=60))
    fund = fund.rename(columns={'tic': 'ticker'}).dropna(subset=['ticker'])
    fund = fund.drop_duplicates(['ticker', 'avail'], keep='last').sort_values('avail')

    out = []
    for tic, pg in px.groupby('ticker'):
        fg = fund[fund['ticker'] == tic]
        if fg.empty:
            continue
        m = pd.merge_asof(pg.sort_values('date'),
                          fg[['avail', 'niq', 'ltq', 'cheq', 'ceqq']],
                          left_on='date', right_on='avail', direction='backward')
        out.append(m)
    if not out:
        raise SystemExit("Hiç eşleşme yok — ticker eşlemesini kontrol edin.")
    d = pd.concat(out, ignore_index=True)

    MTA = d['ME'] + d['ltq'] * 1e6
    d['NIMTA'] = (d['niq'] * 1e6) / MTA
    d['TLMTA'] = (d['ltq'] * 1e6) / MTA
    d['CASHMTA'] = (d['cheq'] * 1e6) / MTA
    d['MB'] = d['ME'] / (d['ceqq'] * 1e6).replace(0, np.nan)

    d = d[['date', 'ticker'] + CHS_COLS].replace([np.inf, -np.inf], np.nan)
    for c in CHS_COLS:                       # CHS de %1/%99 winsorize eder
        lo, hi = d[c].quantile([0.01, 0.99])
        d[c] = d[c].clip(lo, hi)

    n0 = len(d)
    d = d.drop_duplicates(['date', 'ticker'], keep='last')
    if len(d) < n0:
        print(f"  {n0 - len(d):,} mükerrer (tarih, hisse) satırı düşürüldü")

    os.makedirs('datasets', exist_ok=True)
    d.to_csv(CHS_PATH, index=False)
    print(f"\n  {len(d):,} satır | {d.ticker.nunique()} hisse → {CHS_PATH}")
    for c in CHS_COLS:
        print(f"    {c:<9} %{100 * d[c].notna().mean():.1f}")


# ══════════════════════════════════════════════════════════════════
# Skor üreticiler — hepsi ds ile AYNI UZUNLUKTA numpy dizisi döner
# ══════════════════════════════════════════════════════════════════
def align_to_panel(ds, frame, value_cols, key_col='ticker'):
    """
    ÇOĞALMAYI YAPISAL OLARAK ÖNLER.

    merge() yerine reindex() kullanılır: hedef indeks benzersiz olduğu için
    sonuç HER ZAMAN len(ds) satırdır. Daha önce merge kullanılıyordu ve
    kaynak tarafta tek bir mükerrer (tarih, hisse) çifti bile paneli
    şişirip 'operands could not be broadcast' hatasına yol açıyordu.
    """
    f = frame.rename(columns={key_col: 'Ticker'}).copy()
    f['date'] = pd.to_datetime(f['date'])
    f = f.drop_duplicates(['date', 'Ticker'], keep='last').set_index(['date', 'Ticker'])
    assert f.index.is_unique, "Kaynak indeks benzersiz değil"
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(ds.index.values), ds['Ticker'].values])
    out = f.reindex(idx)[value_cols]
    assert len(out) == len(ds), f"Hizalama bozuk: {len(out)} vs {len(ds)}"
    return out.reset_index(drop=True)


def garch_scores(ds):
    """
    Parametreler YALNIZCA eğitim döneminde tahmin edilir, sonra sabit
    parametreyle filtrelenir:  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Yuvarlanan yeniden tahmin YOK — test dönemine bakılmaz.
    """
    from arch import arch_model
    print("\n  GARCH(1,1) uyduruluyor (hisse başına, sadece eğitim dönemi)...")
    rows, ok, fail = [], 0, 0
    for tic, g in ds.groupby('Ticker'):
        r = g.sort_index()['Returns'].dropna() * 100.0
        tr = r[r.index <= TRAIN_END]
        if len(tr) < 250:
            fail += 1
            continue
        try:
            res = arch_model(tr, vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
            w, a, b = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
        except Exception:
            fail += 1
            continue
        rv = r.values
        var = np.empty(len(rv))
        var[0] = float(tr.var())
        for i in range(1, len(rv)):
            var[i] = w + a * rv[i - 1] ** 2 + b * var[i - 1]
        rows.append(pd.DataFrame({'date': r.index, 'ticker': tic,
                                  'garch_vol': np.sqrt(var)}))
        ok += 1
    print(f"    {ok} hisse uyduruldu, {fail} atlandı")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def chs_scores(ds):
    """CHS logit — eğitim döneminde uydurulur, tüm panele uygulanır."""
    if not os.path.exists(CHS_PATH):
        print(f"  ⚠️ {CHS_PATH} yok → CHS atlandı. --stage fetch ile üretin")
        return None
    chs = pd.read_csv(CHS_PATH)
    X = align_to_panel(ds, chs, CHS_COLS)
    full = X.notna().all(axis=1).values
    y = ds['Target'].values.astype(int)
    dates = pd.to_datetime(ds.index.values)
    tr = full & (dates <= pd.Timestamp(TRAIN_END))
    if tr.sum() < 1000:
        print("  ⚠️ CHS: yeterli tam gözlem yok, atlandı")
        return None

    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.transform(X[tr]), y[tr])
    p = np.full(len(X), np.nan)
    p[full] = lr.predict_proba(sc.transform(X[full]))[:, 1]

    print("\n  CHS katsayıları (standartlaştırılmış, |büyüklüğe| göre):")
    for c, co in sorted(zip(CHS_COLS, lr.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"    {c:<9}{co:>+8.3f}")

    # Eksiklik sistematik mi? Tez metnine girecek bir sayı.
    print(f"\n  CHS kapsama: %{100 * full.mean():.1f}")
    print(f"    kapsanan   kriz oranı: %{100 * y[full].mean():.2f}")
    if (~full).sum():
        print(f"    kapsanmayan kriz oranı: %{100 * y[~full].mean():.2f}"
              f"   ← fark varsa metinde belirtilmeli")
    return p


# ══════════════════════════════════════════════════════════════════
def evaluate(name, score, ds, higher_risky=True, mask=None):
    """Tezin protokolü: eşik SADECE validation'dan seçilir."""
    s = np.asarray(score, dtype=float)
    s = s if higher_risky else -s
    valid = ~np.isnan(s)
    if mask is not None:
        valid &= mask
    dates = pd.to_datetime(ds.index.values)
    y = ds['Target'].values.astype(int)

    va = valid & (dates >= pd.Timestamp(VAL_START)) & (dates <= pd.Timestamp(VAL_END))
    te = valid & (dates >= pd.Timestamp(TEST_START)) & (dates <= pd.Timestamp(TEST_END))
    if va.sum() < 100 or te.sum() < 100 or len(set(y[te])) < 2:
        print(f"  {name}: yetersiz gözlem, atlandı")
        return None

    thr, _ = find_best_threshold_mcc(y[va], s[va])
    yp = (s[te] >= thr).astype(int)
    return {
        'baseline': name, 'n_test': int(te.sum()),
        'val_mcc': matthews_corrcoef(y[va], (s[va] >= thr).astype(int)),
        'test_mcc': matthews_corrcoef(y[te], yp),
        'test_pr_auc': average_precision_score(y[te], s[te]),
        'test_roc_auc': roc_auc_score(y[te], s[te]),
        'test_precision': precision_score(y[te], yp, zero_division=0),
        'test_recall': recall_score(y[te], yp, zero_division=0),
        'threshold': float(thr), 'coverage': float(valid.mean()),
    }


def stage_run(outdir):
    print("═" * 74)
    print("AŞAMA 2 — baseline değerlendirmesi")
    print("═" * 74)
    ds = prepare_dataset(force_refresh=False)
    print(f"  Panel: {len(ds):,} satır | {ds['Ticker'].nunique()} hisse")

    # ad → (skor dizisi, yüksek_skor_riskli_mi)
    S = {}
    S['Naive volatility (Vol_20d)'] = (ds['Vol_20d'].values, True)
    S['Altman Z (1968)'] = (ds['Altman_Z'].values, False)   # düşük Z = riskli

    try:
        gv = garch_scores(ds)
        if not gv.empty:
            S['GARCH(1,1)'] = (align_to_panel(ds, gv, ['garch_vol'])
                               ['garch_vol'].values, True)
    except ImportError:
        print("  ⚠️ 'arch' kurulu değil → GARCH atlandı.  pip install arch")

    p = chs_scores(ds)
    if p is not None:
        S['CHS (2008) logit'] = (p, True)

    if not S:
        print("\nHiç baseline üretilemedi.")
        return

    # ── Her yöntem kendi örnekleminde ──
    own = [r for n, (s, h) in S.items() if (r := evaluate(n, s, ds, h))]

    # ── ORTAK ÖRNEKLEM: tüm yöntemlerin skor ürettiği kesişim ──
    common = np.ones(len(ds), dtype=bool)
    for s, _ in S.values():
        common &= ~np.isnan(np.asarray(s, dtype=float))
    print(f"\n  Ortak örneklem: {common.sum():,} / {len(ds):,} "
          f"(%{100 * common.mean():.1f})")
    comm = [r for n, (s, h) in S.items()
            if (r := evaluate(n, s, ds, h, mask=common))]

    os.makedirs(outdir, exist_ok=True)
    if own:
        pd.DataFrame(own).to_csv(
            os.path.join(outdir, 'literature_baselines.csv'), index=False)
    if comm:
        pd.DataFrame(comm).to_csv(
            os.path.join(outdir, 'literature_baselines_common.csv'), index=False)

    def table(rows, title):
        print("\n" + "═" * 74)
        print(title)
        print("═" * 74)
        print(f"{'yöntem':<30}{'test MCC':>10}{'PR-AUC':>9}{'ROC-AUC':>9}{'kapsama':>9}")
        print("-" * 67)
        for r in sorted(rows, key=lambda x: -x['test_mcc']):
            print(f"{r['baseline']:<30}{r['test_mcc']:>10.4f}{r['test_pr_auc']:>9.4f}"
                  f"{r['test_roc_auc']:>9.4f}{100 * r['coverage']:>8.0f}%")
        print("-" * 67)
        for nm, mcc, pr, roc in THESIS_REF:
            print(f"{'→ ' + nm:<30}{mcc:>10.4f}{pr:>9.4f}{roc:>9.4f}{'—':>9}")

    if comm:
        table(comm, "ORTAK ÖRNEKLEM — ASIL SONUÇ")
    if own:
        table(own, "Her yöntem kendi örnekleminde (ikincil)")

    print("\n── YORUM ──")
    print("  Baseline'lar tezin modellerine yakınsa → füzyonun katkısı sınırlı,")
    print("    dürüstçe raporlanmalı.")
    print("  Belirgin gerideyse → tezin katkısı artık ölçülmüş, tahmin değil.")
    print("  GARCH ya da naive önde ise → walkforward bulgusuyla tutarlı")
    print("    (basit kural stres dönemlerinde modeli geçiyordu).")
    print(f"\n→ {outdir}/literature_baselines_common.csv  (asıl)")
    print(f"→ {outdir}/literature_baselines.csv         (ikincil)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['fetch', 'run', 'all'], default='all')
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()
    if args.stage in ('fetch', 'all'):
        if os.path.exists(CHS_PATH) and args.stage == 'all':
            print(f"[Atlandı] {CHS_PATH} mevcut\n")
        else:
            stage_fetch()
    if args.stage in ('run', 'all'):
        stage_run(args.outdir)


if __name__ == '__main__':
    main()