
from pathlib import Path
import pandas as pd

path = Path("datasets/final_dataset.csv")
if not path.exists():
    raise SystemExit(f"Dosya bulunamadı: {path.resolve()}")

df = pd.read_csv(path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)

print("Satır sayısı:", len(df))
print("Hisse sayısı:", df["Ticker"].nunique())
print("Tarih aralığı:", df.index.min(), "→", df.index.max())

x = pd.DataFrame({
    "date": df.index,
    "ticker": df["Ticker"].to_numpy()
}).sort_values(["ticker", "date"])

print("Tekrarlı tarih–hisse:", x.duplicated(["ticker", "date"]).sum())

# Mevcut panelde sonraki 20. gözlemin tarihini kontrol eder.
# Bu bir ön kontroldür; gerçek etiket son tarihinin yerine geçmez.
x["next20"] = x.groupby("ticker")["date"].shift(-20)

for name, start, end, boundary in [
    ("Train", "2010-01-01", "2019-12-31", "2020-01-01"),
    ("Validation", "2020-01-01", "2021-12-31", "2022-01-01"),
]:
    part = x[x["date"].between(start, end)]
    crossing = (part["next20"] >= pd.Timestamp(boundary)).sum()
    unknown = part["next20"].isna().sum()
    print(f"{name}: {len(part)} satır; sınırı aşan={crossing}; "
          f"20. sonraki tarihi panelde bulunmayan={unknown}")
