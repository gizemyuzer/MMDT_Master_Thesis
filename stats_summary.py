"""
stat_summary.py — faktöriyel sonuçların hücre özeti + eşleşmeli testler.
Karışık seed sayılarını doğru işler: her karşılaştırma yalnızca İKİ hücrede
de bulunan seed'ler üzerinden eşleşmeli test edilir.

KULLANIM:  python stat_summary.py
"""
import pandas as pd, numpy as np
from scipy import stats

RAW = 'results/modality_v2_raw.csv'
d = pd.read_csv(RAW)

print('═' * 74)
print('HÜCRE ÖZETİ (test MCC)')
print('═' * 74)
g = (d.groupby('ablation')['test_mcc']
       .agg(n='count', ortalama='mean', sd='std', min='min', max='max')
       .sort_values('ortalama', ascending=False))
print(g.round(4).to_string())

print()
print('seed kapsamı:')
cov = d.pivot_table(index='ablation', columns='seed', values='test_mcc', aggfunc='size')
print(cov.fillna(0).astype(int).to_string())

P = d.pivot_table(index='seed', columns='ablation', values='test_mcc')

CONTRASTS = [
    ('D(II|I)',        'multi_pure',       'tech_only'),
    ('D(III|I)',       'tech_macro',       'tech_only'),
    ('D(III|I+II)',    'all_three',        'multi_pure'),
    ('D(IV|I+II)',     'multi_text',       'multi_pure'),
    ('D(IV-olay|I+II)','multi_text_event', 'multi_pure'),
    ('D(IV-LM|I+II)',  'multi_text_lm',    'multi_pure'),
    ('gated - ungated','multi_pure_gated', 'multi_pure'),
]

print()
print('═' * 74)
print('EŞLEŞMELİ KARŞILAŞTIRMALAR')
print('═' * 74)
print(f"{'karşılaştırma':18s}{'Δ':>9s}{'sd':>9s}{'n':>4s}{'poz':>7s}{'p':>9s}{'d':>7s}")
print('-' * 74)
rows = []
for lab, a, b in CONTRASTS:
    if a not in P.columns or b not in P.columns:
        print(f'{lab:18s}  — hücre yok ({a} veya {b})')
        continue
    sub = P[[a, b]].dropna()          # yalnızca ortak seed'ler
    if len(sub) < 2:
        print(f'{lab:18s}  — ortak seed yetersiz ({len(sub)})')
        continue
    x = sub[a] - sub[b]
    t = stats.ttest_rel(sub[a], sub[b])
    dz = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan
    print(f'{lab:18s}{x.mean():+9.4f}{x.std(ddof=1):9.4f}{len(sub):4d}'
          f'{int((x>0).sum()):4d}/{len(sub):<2d}{t.pvalue:9.4f}{dz:+7.2f}')
    rows.append({'karsilastirma': lab, 'delta': x.mean(), 'sd': x.std(ddof=1),
                 'n': len(sub), 'pozitif': int((x > 0).sum()),
                 'p': t.pvalue, 'cohen_d': dz})

pd.DataFrame(rows).to_csv('results/paired_contrasts.csv', index=False)
print()
print('→ results/paired_contrasts.csv')

print()
print('═' * 74)
print('SEED BAZINDA — ana hücreler')
print('═' * 74)
core = [c for c in ['tech_only', 'multi_pure', 'multi_text',
                    'multi_pure_gated', 'tech_macro', 'all_three'] if c in P.columns]
print(P[core].round(4).to_string())