"""
verify_rdq.py
─────────────
Run this ONCE after fixing fetch_fundamental_data, BEFORE launching the full
pipeline. Confirms that rdq (report date) actually comes back from Compustat
and that the point-in-time alignment behaves as intended — so you don't spend
an hour re-fetching only to discover rdq is empty.

USAGE:
    python verify_rdq.py
"""
import pandas as pd
import wrds

# A few liquid tickers that definitely have quarterly fundamentals
TEST_TICKERS = ['AKAM', 'NFLX', 'CMG', 'ALB']

def main():
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username='gizemyuzer')

    for tic in TEST_TICKERS:
        sql = f"""
            SELECT datadate, rdq, atq, ltq
            FROM comp.fundq
            WHERE tic = '{tic}'
              AND datadate >= '2014-01-01'
              AND datadate <= '2016-12-31'
              AND indfmt = 'INDL' AND datafmt = 'STD'
              AND popsrc = 'D' AND consol = 'C'
            ORDER BY datadate
        """
        df = db.raw_sql(sql)
        if df is None or df.empty:
            print(f"\n{tic}: NO DATA")
            continue

        df['datadate'] = pd.to_datetime(df['datadate'])
        df['rdq'] = pd.to_datetime(df['rdq'], errors='coerce')
        df['lag_days'] = (df['rdq'] - df['datadate']).dt.days

        n_rdq = df['rdq'].notna().sum()
        print(f"\n{tic}: {len(df)} quarters | rdq present: {n_rdq}/{len(df)}")
        print(f"  reporting lag (rdq - datadate): "
              f"median={df['lag_days'].median():.0f} days, "
              f"range=[{df['lag_days'].min():.0f}, {df['lag_days'].max():.0f}]")
        print(df[['datadate', 'rdq', 'lag_days']].head(3).to_string(index=False))

    db.close()

    print("\n" + "=" * 60)
    print("WHAT TO CHECK:")
    print("  • rdq present should be close to full (most quarters have it)")
    print("  • lag should be ~30-60 days (typical earnings reporting delay)")
    print("  • if lag is negative or rdq mostly empty → investigate before")
    print("    the full rerun")
    print("=" * 60)


if __name__ == '__main__':
    main()