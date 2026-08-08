import pandas as pd
from datasets.feature_engineering import prepare_dataset

df = prepare_dataset(force_refresh=False)

train_mask = (df.index >= '2010-01-01') & (df.index <= '2019-12-31')
val_mask = (df.index >= '2020-01-01') & (df.index <= '2021-12-31')
test_mask = (df.index >= '2022-01-01') & (df.index <= '2024-12-31')
test_2022_mask = (df.index >= '2022-01-01') & (df.index <= '2022-12-31')

def print_stats(name, mask):
    subset = df[mask]
    n = len(subset)
    target_mean = subset['Target'].mean() * 100
    print(f"{name}: {n:,} rows | Crash Rate: {target_mean:.1f}%")

print_stats("Train (2010-2019)", train_mask)
print_stats("Validation (2020-2021)", val_mask)
print_stats("Full Test (2022-2024)", test_mask)
print_stats("Bear Market Test (2022)", test_2022_mask)
