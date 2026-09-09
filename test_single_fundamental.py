"""
test_pipeline_2tickers.py
──────────────────────────
prepare_dataset()'in GERÇEK akışını (fetch_price_data → fetch_fundamental_data
→ align_and_merge → create_labels) sadece 2 hisse için çalıştırır.

AMAÇ: 400 hisselik 1 saatlik force_refresh'i körlemesine tekrar başlatmadan
önce, fundamental verinin pipeline'ın SONUNDA hâlâ dolu olup olmadığını
1-2 dakikada doğrulamak. test_single_fundamental.py fonksiyonun tek başına
çalıştığını gösterdi; buradaki soru fonksiyonun ÇIKTISININ birleştirme
adımlarında kaybolup kaybolmadığı.

KULLANIM:
    python test_pipeline_2tickers.py
"""
import numpy as np
import pandas as pd

from datasets.data_fetcher import WRDSDataIngestion, fetch_macro_data
from datasets.feature_engineering import PreprocessingAndFeatures, TARGET_TICKERS

# Evrenin ilk 2 hissesini kullan (gerçek pipeline'la aynı hisseler)
TEST_TICKERS = list(TARGET_TICKERS)[:2]
print(f"Test edilecek hisseler: {TEST_TICKERS}\n")

FUND_BASE = ['Debt_to_Equity', 'Net_Profit_Margin', 'Current_Ratio',
             'Altman_Z', 'Retained_Earnings_TA', 'Market_Value_to_Liab']

ingestion = WRDSDataIngestion(TEST_TICKERS, start_date='2010-01-01', end_date='2024-12-31')
price_data = ingestion.fetch_price_data()

if not price_data:
    print("⚠️ Hiç fiyat verisi gelmedi — ticker isimleri CRSP'te bulunamadı olabilir.")
    raise SystemExit

macro_df = fetch_macro_data()
preprocessor = PreprocessingAndFeatures(lag_n=1)

for ticker, ticker_df in price_data.items():
    print("\n" + "=" * 72)
    print(f"TICKER: {ticker}  (fiyat satırı: {len(ticker_df):,})")
    print("=" * 72)

    # ── ADIM A: fetch_fundamental_data çıktısı ──
    fund_data = ingestion.fetch_fundamental_data(ticker, ticker_df.index)
    print(f"  [A] fetch_fundamental_data → shape={fund_data.shape}")
    for c in FUND_BASE:
        if c in fund_data.columns:
            pct = 100 * fund_data[c].isna().mean()
            flag = "  ⚠️ TAMAMEN NaN" if pct == 100 else ""
            print(f"      {c:<24} NaN%={pct:6.1f}{flag}")
        else:
            print(f"      {c:<24} ⚠️ KOLON YOK")

    # ── ADIM B: align_and_merge sonrası (concat + macro join) ──
    combined = preprocessor.align_and_merge(ticker_df, fund_data, macro_df)
    print(f"\n  [B] align_and_merge → shape={combined.shape}")
    for c in FUND_BASE:
        if c in combined.columns:
            pct = 100 * combined[c].isna().mean()
            flag = "  ⚠️ TAMAMEN NaN — KAYIP BURADA!" if pct == 100 else ""
            print(f"      {c:<24} NaN%={pct:6.1f}{flag}")
        else:
            print(f"      {c:<24} ⚠️ KOLON YOK — KAYIP BURADA!")

    # ── ADIM C: create_labels sonrası (son hali) ──
    labeled = preprocessor.create_labels(combined, combined['Close'], horizon=20)
    print(f"\n  [C] create_labels → shape={labeled.shape}")
    for c in FUND_BASE:
        if c in labeled.columns:
            pct = 100 * labeled[c].isna().mean()
            flag = "  ⚠️ TAMAMEN NaN — KAYIP BURADA!" if pct == 100 else ""
            print(f"      {c:<24} NaN%={pct:6.1f}{flag}")
        else:
            print(f"      {c:<24} ⚠️ KOLON YOK — KAYIP BURADA!")

    # ── İndeks uyumu teşhisi ──
    print(f"\n  [Teşhis] fiyat index dtype={ticker_df.index.dtype} | "
          f"fund index dtype={fund_data.index.dtype}")
    overlap = ticker_df.index.intersection(fund_data.index)
    print(f"  [Teşhis] index kesişimi: {len(overlap):,} / {len(ticker_df):,} fiyat günü")
    if len(overlap) == 0:
        print("  ⚠️ KESİŞİM SIFIR — concat hizalanamıyor, fundamental bu yüzden NaN oluyor!")

ingestion.db.close()
print("\nBitti.")