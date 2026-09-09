"""
run_eda.py
───────────
KEŞİFSEL VERİ ANALİZİ — yeni veri setiyle baştan.

═══════════════════════════════════════════════════════════════════════
NEDEN BAŞTAN
═══════════════════════════════════════════════════════════════════════
İlk sunumdaki EDA grafikleri artık geçersiz. Veri seti maddi olarak değişti:
  · Delisting satırları eklendi (Shumway 1997 düzeltmesi)
  · 6 firma-fundamental kolon geldi (Altman bileşenleri)
  · Getiri eğrisi serisi düzeldi (önce hep sıfırdı)
  · Kesitsel (_XS) kolonlar eklendi
  · Modalite grupları yeniden tanımlandı: 19 kolonluk "fundamental"
    → 6 firma + 9 makro + 4 etkileşim

Bu script tüm EDA çıktılarını yeniden üretir ve TEZ İÇİN İKİ KRİTİK FİGÜRÜ
oluşturur:

  ŞEKİL 2 — Evrendeki aktif hisse sayısının zamanla azalması (395 → 231).
            Hayatta kalma yanlılığından kaçınıldığının GÖRSEL KANITI.
            "2024'te var olan şirketlerle 2010'u tahmin etmedim" iddiasının
            tek bakışta anlaşılan dayanağı.

  ŞEKİL 7 — Makro kolonların gün-içi kesitsel standart sapmasının SIFIR
            olduğunun gösterimi. Tezin ana bulgusunun (makro kirliliği)
            yapısal dayanağı. Bu grafik olmadan "makro kesitsel bilgi
            taşımaz" iddiası sözde kalır; bu grafikle kanıtlanmış olur.

KULLANIM:
    python run_eda.py
Çıktılar: visualization/eda/*.png  +  results/eda_*.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.feature_engineering import (
    prepare_dataset, TECHNICAL_COLS, FIRM_FUNDAMENTAL_COLS, MACRO_COLS,
    INTERACTION_COLS, FIRM_FUNDAMENTAL_XS_COLS,
)

warnings.filterwarnings('ignore')

OUT_FIG = os.path.join('visualization', 'eda')
OUT_CSV = 'results'
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

SPLITS = {
    'Train (2010–2019)': ('2010-01-01', '2019-12-31'),
    'Validation (2020–2021)': ('2020-01-01', '2021-12-31'),
    'Test (2022–2024)': ('2022-01-01', '2024-12-31'),
}
C = {'tech': '#2E86AB', 'fund': '#A23B72', 'macro': '#F18F01',
     'inter': '#6A994E', 'xs': '#8E7DBE', 'gray': '#666666'}


def save(fig, name):
    p = os.path.join(OUT_FIG, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    → {p}")


# ══════════════════════════════════════════════════════════════════
def sec1_overview(df):
    print("\n[1] Veri seti genel görünümü")
    rows = []
    for name, (a, b) in SPLITS.items():
        s = df[(df.index >= a) & (df.index <= b)]
        rows.append({'split': name, 'rows': len(s), 'stocks': s['Ticker'].nunique(),
                     'first': s.index.min().date(), 'last': s.index.max().date(),
                     'crisis_rate': s['Target'].mean()})
    t = pd.DataFrame(rows)
    print(f"{'Bölüm':<24}{'satır':>10}{'hisse':>8}{'kriz oranı':>12}")
    for _, r in t.iterrows():
        print(f"{r['split']:<24}{r['rows']:>10,}{r['stocks']:>8}{r['crisis_rate']*100:>11.1f}%")
    print(f"\n  TOPLAM: {len(df):,} satır · {df['Ticker'].nunique()} hisse · "
          f"kriz oranı %{df['Target'].mean()*100:.1f}")
    t.to_csv(os.path.join(OUT_CSV, 'eda_overview.csv'), index=False)
    return t


# ══════════════════════════════════════════════════════════════════
def sec2_attrition(df):
    """ŞEKİL 2 — hayatta kalma yanlılığından kaçınmanın görsel kanıtı."""
    print("\n[2] ŞEKİL 2 — Evrendeki aktif hisse sayısı (hayatta kalma yanlılığı kanıtı)")
    n = df.groupby(df.index)['Ticker'].nunique().resample('ME').last()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(n.index, n.values, lw=2, color=C['tech'])
    ax.fill_between(n.index, 0, n.values, alpha=0.12, color=C['tech'])
    for a, b in SPLITS.values():
        ax.axvline(pd.Timestamp(a), color=C['gray'], ls='--', lw=1, alpha=0.6)
    ymax = n.max()
    for name, (a, b) in SPLITS.items():
        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
        ax.text(mid, ymax * 1.04, name.split(' (')[0], ha='center',
                fontsize=9, color=C['gray'])
    first, last = int(n.iloc[0]), int(n.iloc[-1])
    ax.annotate(f'{first} hisse', xy=(n.index[0], first),
                xytext=(15, 12), textcoords='offset points', fontsize=10, fontweight='bold')
    ax.annotate(f'{last} hisse\n(%{100*(1-last/first):.0f} azalma)',
                xy=(n.index[-1], last), xytext=(-90, 22), textcoords='offset points',
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['gray']))
    ax.set_ylabel('Aktif hisse sayısı')
    ax.set_title('Evrendeki aktif hisse sayısı zamanla azalıyor\n'
                 'Evren 2009-12-31 anlık görüntüsüyle seçildi — iflas, delisting ve '
                 'birleşmeyle çıkanlar veride kalıyor (hayatta kalma yanlılığı yok)',
                 fontsize=11, loc='left')
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(alpha=0.25)
    save(fig, 'fig2_universe_attrition.png')
    print(f"    {first} → {last} hisse (%{100*(1-last/first):.0f} azalma)")
    n.to_frame('active_stocks').to_csv(os.path.join(OUT_CSV, 'eda_attrition.csv'))


# ══════════════════════════════════════════════════════════════════
def sec3_target(df):
    print("\n[3] Hedef değişken dağılımı")
    yearly = df.groupby(df.index.year)['Target'].agg(['mean', 'count'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    cols = [C['fund'] if y >= 2022 else (C['macro'] if y >= 2020 else C['tech'])
            for y in yearly.index]
    axes[0].bar(yearly.index, yearly['mean'] * 100, color=cols)
    axes[0].axhline(df['Target'].mean() * 100, color=C['gray'], ls='--', lw=1,
                    label=f"genel ort. %{df['Target'].mean()*100:.1f}")
    axes[0].set_ylabel('Kriz oranı (%)'); axes[0].set_xlabel('Yıl')
    axes[0].set_title('Yıllara göre kriz oranı', loc='left')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25, axis='y')

    if 'Sector' in df.columns:
        sec = df.groupby('Sector')['Target'].mean().sort_values() * 100
        axes[1].barh(sec.index, sec.values, color=C['fund'])
        axes[1].axvline(df['Target'].mean() * 100, color=C['gray'], ls='--', lw=1)
        axes[1].set_xlabel('Kriz oranı (%)')
        axes[1].set_title('Sektöre göre kriz oranı', loc='left')
        axes[1].grid(alpha=0.25, axis='x')
    save(fig, 'fig3_target_distribution.png')

    yearly.to_csv(os.path.join(OUT_CSV, 'eda_target_by_year.csv'))
    print(f"    Sınıf dengesizliği: 1'e {(1-df['Target'].mean())/df['Target'].mean():.1f}")


# ══════════════════════════════════════════════════════════════════
def sec4_missing(df):
    print("\n[4] Eksik veri haritası (feature grubuna göre)")
    groups = [('Teknik (I)', TECHNICAL_COLS, C['tech']),
              ('Firma fundamental (II)', FIRM_FUNDAMENTAL_COLS, C['fund']),
              ('Makro (III)', MACRO_COLS, C['macro']),
              ('Kesitsel (_XS)', FIRM_FUNDAMENTAL_XS_COLS, C['xs']),
              ('Etkileşim', INTERACTION_COLS, C['inter'])]
    names, vals, colors, rows = [], [], [], []
    for gname, cols, col in groups:
        present = [c for c in cols if c in df.columns]
        for c in present:
            names.append(c); vals.append(100 * df[c].isna().mean()); colors.append(col)
            rows.append({'group': gname, 'feature': c,
                         'missing_pct': 100 * df[c].isna().mean()})
    fig, ax = plt.subplots(figsize=(9, max(6, len(names) * 0.22)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel('Eksik veri (%)')
    ax.set_title('Feature bazında eksik veri oranı\n'
                 'Fundamental kolonlardaki ~%19, Compustat kapsamı olmayan '
                 'firmalardan gelir (train medyanıyla dolduruluyor)',
                 fontsize=10, loc='left')
    ax.grid(alpha=0.25, axis='x')
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups]
    ax.legend(hs, [g for g, _, _ in groups], fontsize=8, loc='lower right')
    save(fig, 'fig4_missing_data.png')
    pd.DataFrame(rows).to_csv(os.path.join(OUT_CSV, 'eda_missing.csv'), index=False)


# ══════════════════════════════════════════════════════════════════
def sec5_fundamentals(df):
    print("\n[5] Firma fundamental değişkenlerinin dağılımı")
    cols = [c for c in FIRM_FUNDAMENTAL_COLS if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, c in zip(axes.ravel(), cols):
        v = df[c].dropna()
        lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v[(v >= lo) & (v <= hi)], bins=60, color=C['fund'], alpha=0.8)
        ax.axvline(v.median(), color='black', ls='--', lw=1)
        ax.set_title(f'{c}\nmedyan={v.median():.2f} · eksik %{100*df[c].isna().mean():.0f}',
                     fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle('Firma fundamental değişkenleri (%1–%99 aralığı)',
                 fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    save(fig, 'fig5_fundamentals.png')
    df[cols].describe().T.to_csv(os.path.join(OUT_CSV, 'eda_fundamental_stats.csv'))


# ══════════════════════════════════════════════════════════════════
def sec6_correlation(df):
    print("\n[6] Gruplar arası korelasyon")
    sel = ([c for c in TECHNICAL_COLS if c in df.columns][:12]
           + [c for c in FIRM_FUNDAMENTAL_COLS if c in df.columns]
           + [c for c in MACRO_COLS if c in df.columns])
    corr = df[sel].corr()
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7)
    nt = len([c for c in TECHNICAL_COLS if c in df.columns][:12])
    nf = len([c for c in FIRM_FUNDAMENTAL_COLS if c in df.columns])
    for b in (nt - 0.5, nt + nf - 0.5):
        ax.axhline(b, color='black', lw=1.5); ax.axvline(b, color='black', lw=1.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Korelasyon matrisi — teknik | firma fundamental | makro\n'
                 'Siyah çizgiler modalite sınırları', fontsize=10, loc='left')
    save(fig, 'fig6_correlation.png')
    corr.to_csv(os.path.join(OUT_CSV, 'eda_correlation.csv'))


# ══════════════════════════════════════════════════════════════════
def sec7_macro_zero_variance(df):
    """ŞEKİL 7 — tezin ana bulgusunun yapısal kanıtı."""
    print("\n[7] ŞEKİL 7 — Makro kolonların kesitsel varyansı SIFIR")
    ex_macro = [c for c in ['VIX_Close', 'SPY_Trend_50', 'Yield_Spread_10Y2Y']
                if c in df.columns]
    ex_firm = [c for c in ['Altman_Z', 'Debt_to_Equity'] if c in df.columns]
    ex_tech = [c for c in ['RSI', 'Vol_20d'] if c in df.columns]

    rows = []
    for cols, gname, col in [(ex_tech, 'Teknik (I)', C['tech']),
                             (ex_firm, 'Firma fundamental (II)', C['fund']),
                             (ex_macro, 'Makro (III)', C['macro'])]:
        for c in cols:
            xs_std = df.groupby(df.index)[c].std()          # GÜN İÇİ kesitsel std
            rows.append({'feature': c, 'group': gname, 'color': col,
                         'mean_xs_std': float(xs_std.mean()),
                         'max_xs_std': float(xs_std.max())})
    t = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(t['feature'], t['mean_xs_std'], color=t['color'])
    axes[0].set_xlabel('Ortalama gün-içi kesitsel standart sapma')
    axes[0].set_title('Makro değişkenlerin kesitsel varyansı SIFIR\n'
                      'Belirli bir günde tüm hisseler için aynı değeri alırlar',
                      fontsize=10, loc='left')
    axes[0].grid(alpha=0.25, axis='x')
    for i, v in enumerate(t['mean_xs_std']):
        axes[0].text(v, i, f'  {v:.4f}', va='center', fontsize=8)

    # Tek bir günün kesiti — somut örnek
    d0 = df.index[len(df) // 2]
    day = df.loc[d0]
    if isinstance(day, pd.DataFrame) and len(day) > 5:
        pairs = [(ex_tech[0] if ex_tech else None, C['tech']),
                 (ex_firm[0] if ex_firm else None, C['fund']),
                 (ex_macro[0] if ex_macro else None, C['macro'])]
        for i, (c, col) in enumerate([p for p in pairs if p[0]]):
            v = day[c].dropna()
            if len(v) == 0:
                continue
            vn = (v - v.mean()) / (v.std() + 1e-12) if v.std() > 1e-12 else v * 0
            axes[1].scatter(vn, np.full(len(vn), i) + np.random.normal(0, .06, len(vn)),
                            s=6, alpha=.35, color=col, label=f'{c} (n={len(v)})')
        axes[1].set_yticks(range(len([p for p in pairs if p[0]])))
        axes[1].set_yticklabels([p[0] for p in pairs if p[0]], fontsize=9)
        axes[1].set_xlabel('Standartlaştırılmış değer (o güne ait kesit)')
        axes[1].set_title(f'{d0.date()} tarihli tek günün kesiti\n'
                          'Makro değişken tek bir noktaya çöküyor — ayrım gücü yok',
                          fontsize=10, loc='left')
        axes[1].grid(alpha=0.25, axis='x')
    save(fig, 'fig7_macro_zero_cross_sectional_variance.png')
    t.drop(columns='color').to_csv(os.path.join(OUT_CSV, 'eda_macro_variance.csv'),
                                   index=False)
    for _, r in t.iterrows():
        print(f"    {r['feature']:<22} ort. kesitsel std = {r['mean_xs_std']:.6f}")


# ══════════════════════════════════════════════════════════════════
def sec8_cross_sectional_effect(df):
    print("\n[8] Kesitsel normalizasyonun etkisi")
    pairs = [(c, c + '_XS') for c in FIRM_FUNDAMENTAL_COLS
             if c in df.columns and c + '_XS' in df.columns][:3]
    if not pairs:
        print("    (_XS kolonları yok, atlanıyor)")
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.2 * len(pairs), 6.5))
    axes = np.atleast_2d(axes)
    for j, (raw, xs) in enumerate(pairs):
        v = df[raw].dropna(); lo, hi = v.quantile([.01, .99])
        axes[0, j].hist(v[(v >= lo) & (v <= hi)], bins=50, color=C['fund'], alpha=.85)
        axes[0, j].set_title(f'{raw}\n(ham seviye)', fontsize=9)
        axes[1, j].hist(df[xs].dropna(), bins=50, color=C['xs'], alpha=.85)
        axes[1, j].set_title(f'{xs}\n(tarih-içi yüzdelik dilim)', fontsize=9)
        for a in (axes[0, j], axes[1, j]):
            a.grid(alpha=.2)
    fig.suptitle('Kesitsel normalizasyon: mutlak seviye → akranlar arası sıralama',
                 fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    save(fig, 'fig8_cross_sectional_normalization.png')


# ══════════════════════════════════════════════════════════════════
def sec9_regime(df):
    print("\n[9] Test dönemi rejim karakterizasyonu")
    te = df[(df.index >= '2022-01-01') & (df.index <= '2024-12-31')]
    rows = []
    for y in [2022, 2023, 2024]:
        s = te[te.index.year == y]
        if s.empty:
            continue
        pw = s.pivot_table(index=s.index, columns='Ticker', values='Close')
        eq = pw.pct_change().mean(axis=1).fillna(0)
        rows.append({'year': y, 'universe_return': float((1 + eq).prod() - 1),
                     'mean_vix': float(s['VIX_Close'].mean()) if 'VIX_Close' in s else np.nan,
                     'crisis_rate': float(s['Target'].mean()),
                     'n_stocks': s['Ticker'].nunique()})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, lab, fmt) in zip(axes, [
            ('universe_return', 'Evren getirisi (%)', 100),
            ('mean_vix', 'Ortalama VIX', 1),
            ('crisis_rate', 'Kriz oranı (%)', 100)]):
        vals = t[col] * fmt
        ax.bar(t['year'].astype(str), vals,
               color=[C['fund'] if v < 0 else C['tech'] for v in vals])
        ax.set_title(lab, fontsize=10); ax.grid(alpha=.25, axis='y')
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.1f}', ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=9)
    fig.suptitle('Test dönemi üç ayrı rejim içeriyor', fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    save(fig, 'fig9_test_regimes.png')
    t.to_csv(os.path.join(OUT_CSV, 'eda_regimes.csv'), index=False)
    print(t.round(3).to_string(index=False))


# ══════════════════════════════════════════════════════════════════
def main():
    print("═" * 70)
    print("KEŞİFSEL VERİ ANALİZİ — yeni veri seti")
    print("═" * 70)
    df = prepare_dataset(force_refresh=False)

    sec1_overview(df)
    sec2_attrition(df)
    sec3_target(df)
    sec4_missing(df)
    sec5_fundamentals(df)
    sec6_correlation(df)
    sec7_macro_zero_variance(df)
    sec8_cross_sectional_effect(df)
    sec9_regime(df)

    print("\n" + "═" * 70)
    print("TAMAMLANDI")
    print("═" * 70)
    print(f"  Şekiller → {OUT_FIG}/")
    print(f"  Tablolar → {OUT_CSV}/eda_*.csv")
    print("\n  SUNUMA MUTLAKA KOYULMASI GEREKEN İKİ ŞEKİL:")
    print("    fig2 — evren azalması → hayatta kalma yanlılığından kaçınma kanıtı")
    print("    fig7 — makro kesitsel varyans = 0 → ana bulgunun yapısal dayanağı")


if __name__ == '__main__':
    main()