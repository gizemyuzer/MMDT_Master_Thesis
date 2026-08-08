"""
check_fund_data.py
────────────────────
fund_only'deki NaN loss'un kaynağını bulmak için: final_dataset.csv'deki
fundamental kolonlarda Inf / aşırı büyük değer / all-NaN kolon var mı kontrol eder.
Sadece okuma yapar, hiçbir şeyi değiştirmez.

KULLANIM:
    python check_fund_data.py
"""
import numpy as np
import pandas as pd

from datasets.feature_engineering import FUNDAMENTAL_COLS

df = pd.read_csv('datasets/final_dataset.csv', index_col=0)
df.index = pd.to_datetime(df.index)

print(f"Toplam satır: {len(df):,}\n")

fund_cols = [c for c in FUNDAMENTAL_COLS if c in df.columns]
missing = [c for c in FUNDAMENTAL_COLS if c not in df.columns]
if missing:
    print(f"⚠️ Dataset'te olmayan kolonlar: {missing}\n")

print("=" * 90)
print(f"{'Kolon':<28}{'NaN%':>8}{'Inf sayısı':>12}{'Min':>14}{'Max':>14}{'Std':>14}")
print("=" * 90)

for c in fund_cols:
    col = df[c]
    n_nan = col.isna().sum()
    nan_pct = 100 * n_nan / len(col)
    finite = col.replace([np.inf, -np.inf], np.nan)
    n_inf = int(np.isinf(col.astype(float)).sum())
    cmin = finite.min()
    cmax = finite.max()
    cstd = finite.std()
    flag = ""
    if n_nan == len(col):
        flag = "  ⚠️ TAMAMEN NaN"
    elif n_inf > 0:
        flag = "  ⚠️ INF VAR"
    elif pd.notna(cmax) and abs(cmax) > 1e6:
        flag = "  ⚠️ AŞIRI BÜYÜK"
    elif pd.notna(cmin) and abs(cmin) > 1e6:
        flag = "  ⚠️ AŞIRI BÜYÜK (min)"
    print(f"{c:<28}{nan_pct:>7.1f}%{n_inf:>12}{cmin:>14.3f}{cmax:>14.3f}{cstd:>14.3f}{flag}")

print("\n" + "=" * 90)
print("Train split'te (2010-2019) medyan hesaplandığında NaN kalıyor mu kontrolü:")
print("=" * 90)
train_df = df[(df.index >= '2010-01-01') & (df.index <= '2019-12-31')]
train_medians = train_df[fund_cols].median()
for c in fund_cols:
    m = train_medians[c]
    flag = "  ⚠️ MEDYAN NaN (kolon train'de tamamen boş!)" if pd.isna(m) else ""
    print(f"  {c:<28} medyan={m}{flag}")

print("\nBitti.")