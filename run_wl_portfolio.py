"""
run_wl_portfolio.py — ağırlıklı kayıp modelinin portföy karşılaştırması.

Kaydedilmiş .npz tahminlerinden portföy koşar. Ağırlıklı model
sınıflandırmada geriledi (MCC 0.138 → 0.113); soru şu: ekonomik değerde
de geriledi mi, yoksa MCC-Calmar ayrışması burada da geçerli mi?

KULLANIM:
    python run_wl_portfolio.py --exclude-pct 8
"""
import os, glob, argparse
import numpy as np, pandas as pd
from datasets.feature_engineering import prepare_dataset
from run_portfolio_simulation import simulate, performance

TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-pct', type=float, default=8.0)
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    ds = prepare_dataset(force_refresh=False)
    sub = ds[(ds.index >= TEST_START) & (ds.index <= TEST_END)]
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    days = ret.index
    rebal = set(days[::args.rebalance])

    scores = {}
    for f in sorted(glob.glob(os.path.join(args.outdir, 'preds', 'WL_*.npz'))):
        z = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        w = pd.DataFrame({'date': pd.to_datetime(z['dates']),
                          'ticker': z['tickers'], 'p': z['prob']})
        scores[name] = (w.pivot_table(index='date', columns='ticker', values='p')
                        .reindex(index=days).reindex(columns=ret.columns))
    if not scores:
        raise SystemExit("WL_*.npz bulunamadı.")

    if 'Vol_20d' in sub.columns:
        scores['naive_vol'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                               values='Vol_20d')
                               .reindex(index=days).reindex(columns=ret.columns))
    scores['oracle'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                        values='Target')
                        .reindex(index=days).reindex(columns=ret.columns))

    rows = []
    flat = pd.DataFrame(0.0, index=days, columns=ret.columns)
    net, tov, exp_ = simulate(flat, ret, rebal, 0.0, args.cost_bps)
    rows.append({**performance(net, 'buy_hold'), 'exposure': exp_})
    for lab, sw in scores.items():
        net, tov, exp_ = simulate(sw, ret, rebal, args.exclude_pct, args.cost_bps)
        rows.append({**performance(net, lab), 'exposure': exp_})

    P = pd.DataFrame(rows)
    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)
    g = P.groupby('grp')[['cagr', 'max_drawdown', 'calmar', 'sortino']].mean()

    print("═" * 66)
    print(f"AĞIRLIKLI KAYIP — PORTFÖY  (dışlama %{args.exclude_pct:.0f})")
    print("═" * 66)
    print(f"{'strateji':<18}{'CAGR':>10}{'maksDD':>10}{'Calmar':>9}{'Sortino':>9}")
    print("-" * 56)
    for k in g.sort_values('calmar', ascending=False).index:
        r = g.loc[k]
        print(f"{k:<18}{100*r.cagr:>9.2f}%{100*r.max_drawdown:>9.2f}%"
              f"{r.calmar:>9.3f}{r.sortino:>9.3f}")
    print("-" * 56)
    print(f"{'referans: %20 ile':<18}{'':>10}{'':>10}{0.4318:>9.3f}")
    print(f"{'referans: %8 ile':<18}{'':>10}{'':>10}{0.4554:>9.3f}")

    P.to_csv(os.path.join(args.outdir, 'wl_portfolio.csv'), index=False)
    print(f"\n→ {args.outdir}/wl_portfolio.csv")


if __name__ == '__main__':
    main()