"""
run_economic_all.py — BÜTÜN ekonomik sonuçları TEK motorla yeniden üretir.

SORUN (danışman geri bildirimi #4)
    run_walkforward_bear.py ve run_wl_portfolio.py hâlâ eski
    run_portfolio_simulation motorunu çağırıyor. Bölüm 5.4 yeni motorla,
    ayı piyasası / walk-forward / sampler sonuçları eski motorla üretilmiş
    durumda. Seviyeleri kıyaslanamaz, ama tez bir yerde "her ekonomik sayı
    yeniden hesaplandı" diyordu. İkisi birden doğru olamaz.

BU SCRIPT
    Kayıtlı, indeksli tahminleri (results/preds_v2/*.npz) alır ve
    portfolio_engine ile — yani Bölüm 5.4 ile AYNI muhasebeyle —
    istenen tarih penceresinde koşar. Hangi deney olduğu yalnızca
    pencere ve hangi tahmin dosyaları kullanıldığıyla belirlenir.

    Uygulanan düzeltmeler:
      • ortak başlangıç tarihi (tüm stratejiler aynı gün başlar)
      • işlem görebilirlik maskesi (o gün fiyatı olmayan hisse alınamaz)
      • başlangıç NAV'ı drawdown zirvesi olarak sayılır
      • işlem maliyeti yatırımdan ÖNCE ayrılır (kaldıraç yok)
      • kapsam farkları ayrıca raporlanır

KULLANIM
    # ayı piyasası penceresi
    python run_economic_all.py --tag bear --start 2022-01-01 --end 2022-12-31 \
        --patterns "multi_text_seed*.npz" "multi_pure_seed*.npz" "tech_only_seed*.npz"

    # sampler deneyi (aynı pencere, farklı tahmin dizini)
    python run_economic_all.py --tag sampler --preds-dir results/preds_sampler \
        --patterns "sampler_seed*.npz" "uniform_seed*.npz" "weighted_seed*.npz"

    # walk-forward: her pencere için ayrı çağrı
    python run_economic_all.py --tag f2015 --start 2015-07-01 --end 2016-02-29 ...
"""
import os, glob, argparse
import numpy as np, pandas as pd

from datasets.feature_engineering import prepare_dataset
from portfolio_engine import simulate_positions, performance

__build__ = "2026-09-12a"   # tek motorla bear/walkforward/sampler


OUT = 'results'


def load_panel(start, end):
    ds = prepare_dataset(force_refresh=False)
    sub = ds[(ds.index >= start) & (ds.index <= end)]
    if sub.empty:
        raise SystemExit(f"{start} – {end} aralığında veri yok.")
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    return sub, ret, px.notna()


def load_scores(sub, ret, preds_dir, patterns):
    days, cols = ret.index, ret.columns
    out = {}
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(preds_dir, pat)))
        if not files:
            print(f"  ⚠️ eşleşme yok: {pat}")
        for f in files:
            z = np.load(f, allow_pickle=True)
            if 'dates' not in z:
                raise SystemExit(f"{os.path.basename(f)} indekssiz — "
                                 f"export_preds.py ile yeniden üretin.")
            d = pd.DataFrame({'date': pd.to_datetime(z['dates']),
                              'ticker': z['tickers'], 'p': z['prob']})
            if 'split' in z:
                d = d[np.asarray(z['split']) == 'test']
            name = os.path.basename(f).replace('.npz', '')
            out[name] = (d.pivot_table(index='date', columns='ticker', values='p')
                         .reindex(index=days).reindex(columns=cols))
    if 'Vol_20d' in sub.columns:
        nv = (sub.pivot_table(index=sub.index, columns='Ticker', values='Vol_20d')
              .reindex(index=days).reindex(columns=cols))
        out['naive_vol'] = nv
        out['inv_vol'] = -nv
    out['oracle'] = (sub.pivot_table(index=sub.index, columns='Ticker', values='Target')
                     .reindex(index=days).reindex(columns=cols))
    return out


def first_scored(sw, rebal, min_names=10):
    ok = [d for d in sorted(rebal)
          if d in sw.index and sw.loc[d].notna().sum() >= min_names]
    return ok[0] if ok else None


def _print_build():
    import hashlib, os
    h = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]
    print(f"[{os.path.basename(__file__)}  build {__build__}  sha256 {h}]")


def main():
    _print_build()
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--preds-dir', default=os.path.join('results', 'preds_v2'))
    ap.add_argument('--patterns', nargs='+', default=['multi_text_seed*.npz'])
    ap.add_argument('--exclude-pct', type=float, default=20.0)
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    args = ap.parse_args()

    sub, ret, tradable = load_panel(args.start, args.end)
    days = ret.index
    rebal = set(days[::args.rebalance])
    scores = load_scores(sub, ret, args.preds_dir, args.patterns)
    if not scores:
        raise SystemExit("Hiç skor yüklenemedi.")

    # ── ortak başlangıç ──
    firsts = {k: first_scored(v, rebal) for k, v in scores.items()}
    usable = [v for v in firsts.values() if v is not None]
    if not usable:
        raise SystemExit("Hiçbir strateji için yeterli skor yok.")
    start = max(usable)

    print("═" * 78)
    print(f"EKONOMİK YENİDEN ÜRETİM — {args.tag}   ({args.start} → {args.end})")
    print("═" * 78)
    print(f"  panel: {len(days)} gün × {ret.shape[1]} hisse | "
          f"{len(rebal)} rebalance | maliyet {args.cost_bps:.0f} bps")
    print("\n  KAPSAM")
    for k in sorted(firsts, key=lambda x: (firsts[x] is None, firsts[x])):
        f = firsts[k]
        lag = (f - days[0]).days if f is not None else None
        print(f"    {k[:38]:<40}{f.date() if f is not None else 'skor yok':>12}"
              f"{('  +%d gün' % lag) if lag else ''}")
    print(f"  → ORTAK BAŞLANGIÇ: {start.date()}  "
          f"(panel başından {(start - days[0]).days} gün sonra)")
    print("    Bütün stratejiler bu tarihten başlar; öncesi hiçbirine sayılmaz.")

    flat = pd.DataFrame(0.0, index=days, columns=ret.columns)
    rows = []

    net, tov, inv, _, _ = simulate_positions(
        flat, ret, {start}, 0.0, args.cost_bps,
        tradable=tradable, start_date=start)
    rows.append({**performance(net, 'buy_hold_true'), 'turnover': tov,
                 'avg_invested': inv})

    net, tov, inv, _, _ = simulate_positions(
        flat, ret, rebal, 0.0, args.cost_bps,
        tradable=tradable, start_date=start)
    rows.append({**performance(net, 'equal_weight_rebal'), 'turnover': tov,
                 'avg_invested': inv})

    for name, sw in scores.items():
        net, tov, inv, _, _ = simulate_positions(
            sw, ret, rebal, args.exclude_pct, args.cost_bps,
            tradable=tradable, start_date=start)
        skipped = len(getattr(simulate_positions, 'last_skipped', []))
        rows.append({**performance(net, name), 'turnover': tov,
                     'avg_invested': inv, 'skipped_rebalances': skipped})

    P = pd.DataFrame(rows)
    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)
    P['tag'] = args.tag
    os.makedirs(OUT, exist_ok=True)
    P.to_csv(os.path.join(OUT, f'economic_{args.tag}.csv'), index=False)

    g = (P.groupby('grp')[['cagr', 'max_drawdown', 'calmar', 'turnover',
                           'avg_invested']].mean()
         .sort_values('calmar', ascending=False))
    n_seed = P.groupby('grp').size()
    print("\n" + "═" * 78)
    print(f"{'strateji':<26}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}"
          f"{'devir':>8}{'yatırım':>9}{'n':>4}")
    print("-" * 75)
    for k in g.index:
        r = g.loc[k]
        print(f"{k[:25]:<26}{100*r.cagr:>8.2f}%{100*r.max_drawdown:>9.2f}%"
              f"{r.calmar:>9.3f}{100*r.turnover:>7.1f}%{100*r.avg_invested:>8.1f}%"
              f"{int(n_seed[k]):>4}")
    print("-" * 75)
    print("  yatırım oranı belirgin şekilde %100'ün altındaysa, o strateji bazı")
    print("  rebalance günlerini skor yetersizliğinden atlamış demektir.")
    print(f"\n→ {OUT}/economic_{args.tag}.csv")


if __name__ == '__main__':
    main()