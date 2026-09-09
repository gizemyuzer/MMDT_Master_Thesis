"""
run_cohort_drift.py — evren yaşlanması kontrolü.

ELEŞTİRİ
    Evren 2009-12-31'de donduruldu ve hiç yenilenmedi. Bu hayatta kalma
    yanlılığını ortadan kaldırıyor (doğru), ama 2022-24'e gelindiğinde elde
    kalan şey güncel bir mid-cap kesiti değil, 2009 kohortunun on üç yıl
    boyunca iflas etmemiş, satın alınmamış ALT KÜMESİ. Bu alt küme
    sistematik olarak daha büyük, daha az kaldıraçlı ve daha istikrarlı
    olabilir — o zaman ölçtüğümüz kriz oranı ve model performansı seçilmiş
    bir gruba özgü olur.

İKİ TEST
    A) SÜRÜKLENME  — panelin özellikleri yıllar içinde nasıl değişti?
    B) SEÇİLİM     — 2010'da, sonradan hayatta kalanlar ile sonradan
                     ayrılanlar ZATEN farklı mıydı? (asıl test bu)

    (A) sadece piyasa koşullarının değişmesinden de kaynaklanabilir.
    (B) doğrudan kohort seçilimini ölçer ve eleştiriye asıl cevabı verir.

KULLANIM
    python run_cohort_drift.py
"""
import numpy as np, pandas as pd
from scipy import stats
from datasets.feature_engineering import prepare_dataset

# panelde mevcut, iktisadi olarak yorumlanabilir özellikler
COLS = ['Close', 'Vol_20d', 'Debt_to_Equity', 'Current_Ratio',
        'Net_Profit_Margin', 'Altman_Z', 'Retained_Earnings_TA',
        'Market_Value_to_Liab']

SURVIVOR_FROM = '2022-01-01'   # bu tarihten sonra panelde görünen = hayatta kalan
EARLY_YEAR    = 2010           # karşılaştırma yılı


def main():
    ds = prepare_dataset(force_refresh=False)
    ds = ds.copy()
    ds['year'] = ds.index.year
    cols = [c for c in COLS if c in ds.columns]
    print(f"panel: {len(ds):,} satır | {ds['Ticker'].nunique()} hisse | "
          f"{ds.index.min().date()} → {ds.index.max().date()}")
    print(f"kullanılan özellikler: {cols}\n")

    # ───────────────────────── A) SÜRÜKLENME ─────────────────────────
    print("═" * 72)
    print("A) PANEL ÖZELLİKLERİNİN YILLARA GÖRE SEYRİ (medyan)")
    print("═" * 72)
    drift = ds.groupby('year')[cols].median()
    drift.insert(0, 'n_stocks', ds.groupby('year')['Ticker'].nunique())
    print(drift.round(3).to_string())

    first, last = drift.index.min(), drift.index.max()
    chg = pd.DataFrame({
        f'{first}': drift.loc[first, cols],
        f'{last}':  drift.loc[last,  cols],
    })
    chg['degisim_%'] = 100 * (chg[f'{last}'] / chg[f'{first}'].replace(0, np.nan) - 1)
    print(f"\n{first} → {last} değişim:")
    print(chg.round(3).to_string())

    # ───────────────────────── B) SEÇİLİM ─────────────────────────
    print("\n" + "═" * 72)
    print(f"B) SEÇİLİM TESTİ — {EARLY_YEAR}'da hayatta kalanlar vs ayrılanlar")
    print("═" * 72)

    survivors = set(ds.loc[ds.index >= SURVIVOR_FROM, 'Ticker'].unique())
    early = ds[ds['year'] == EARLY_YEAR].copy()
    early_tickers = set(early['Ticker'].unique())
    leavers = early_tickers - survivors

    print(f"{EARLY_YEAR}'da panelde: {len(early_tickers)} hisse")
    print(f"  {SURVIVOR_FROM} sonrası hâlâ var : {len(early_tickers & survivors)}")
    print(f"  yolda ayrılan                    : {len(leavers)}\n")

    # her hisse için o yılın medyanı → hisse başına tek gözlem
    firm = early.groupby('Ticker')[cols].median()
    firm['survivor'] = firm.index.isin(survivors)

    rows = []
    for c in cols:
        a = firm.loc[firm.survivor,  c].dropna()
        b = firm.loc[~firm.survivor, c].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        t = stats.ttest_ind(a, b, equal_var=False)
        u = stats.mannwhitneyu(a, b, alternative='two-sided')
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({
            'ozellik': c,
            'hayatta_kalan': a.median(),
            'ayrilan': b.median(),
            'fark_%': 100 * (a.median() / b.median() - 1) if b.median() else np.nan,
            'welch_p': t.pvalue,
            'mannwhitney_p': u.pvalue,
            'cohen_d': (a.mean() - b.mean()) / sd if sd > 0 else np.nan,
        })
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))

    sig = R[R.mannwhitney_p < 0.05]
    print(f"\n{len(sig)}/{len(R)} özellikte anlamlı fark (Mann–Whitney, α=0.05)")
    if len(sig):
        print("anlamlı olanlar:", ', '.join(sig.ozellik))
    big = R[R.cohen_d.abs() > 0.5]
    print(f"{len(big)}/{len(R)} özellikte |d| > 0.5 (orta/büyük etki)")

    # ─────────────── C) kriz oranı: kalanlar vs ayrılanlar ───────────────
    if 'Target' in ds.columns:
        print("\n" + "═" * 72)
        print(f"C) {EARLY_YEAR} KRİZ ORANI")
        print("═" * 72)
        e = ds[ds['year'] == EARLY_YEAR].copy()
        e['survivor'] = e['Ticker'].isin(survivors)
        cr = e.groupby('survivor')['Target'].agg(['mean', 'size'])
        cr.index = ['ayrılan', 'hayatta kalan']
        print((cr.assign(kriz_orani_pct=lambda d: 100 * d['mean'])
                 .drop(columns='mean')).round(3).to_string())

    R.to_csv('results/cohort_drift.csv', index=False)
    drift.to_csv('results/cohort_drift_by_year.csv')
    print("\n→ results/cohort_drift.csv, results/cohort_drift_by_year.csv")

    # ─────────────── yorum rehberi ───────────────
    print("\n" + "─" * 72)
    print("NASIL OKUNUR")
    print("  B'de çok az özellik anlamlıysa ve |d| küçükse → kohort seçilimi zayıf,")
    print("  eleştiri büyük ölçüde kapanır.")
    print("  Çok sayıda özellik anlamlı ve |d| > 0.5 ise → seçilim gerçek;")
    print("  dürüstçe raporlanır ve sonuçların bu alt kümeye özgü olabileceği")
    print("  limitasyon olarak yazılır.")


if __name__ == '__main__':
    main()