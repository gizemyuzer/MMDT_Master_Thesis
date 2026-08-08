from datasets.feature_engineering import prepare_dataset
df = prepare_dataset(force_refresh=False)
print([c for c in df.columns])
