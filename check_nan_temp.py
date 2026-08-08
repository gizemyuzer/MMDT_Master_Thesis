from datasets.feature_engineering import prepare_dataset, get_feature_groups
df = prepare_dataset(force_refresh=False)
tech_cols, fund_cols = get_feature_groups(df)
cols = tech_cols + fund_cols

# dropna kaç satır siler + silinen satırların kriz oranı
mask_has_nan = df[cols].isna().any(axis=1)
print(f"dropna kaybı: {mask_has_nan.sum():,} satır ({mask_has_nan.mean()*100:.1f}%)")
print(f"Silinecek satırların kriz oranı: {df.loc[mask_has_nan, 'Target'].mean()*100:.1f}%")
print(f"Kalan satırların kriz oranı:     {df.loc[~mask_has_nan, 'Target'].mean()*100:.1f}%")
print(f"Genel kriz oranı:                {df['Target'].mean()*100:.1f}%")

# fundamental özelinde NaN oranı (bunlar warmup DEĞİL, gerçek eksik olabilir)
print("\nFundamental NaN oranları:")
print((df[fund_cols].isna().mean()*100).round(1).sort_values(ascending=False).to_string())
