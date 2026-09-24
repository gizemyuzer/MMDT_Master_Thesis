"""
run_portfolio_v2.py — ekonomik değerlendirmenin DÜZELTİLMİŞ yeniden üretimi.

DEĞİŞENLER (danışman geri bildirimi C1 ve C4)
  1) Muhasebe: portfolio_engine.simulate_positions kullanılıyor.
     Pozisyonlar fiyatla sürüklenir; işlem maliyeti yalnızca gerçekten
     işlem gören tutar üzerinden alınır. Eski motor sabit-ağırlık
     varsayıyordu ve örtük günlük rebalance'ın maliyetini almıyordu.

  2) İki pasif referans ayrıldı:
       buy_hold_true  — başta alınır, bir daha işlem yapılmaz
       equal_weight   — modelle AYNI sıklıkta eşit ağırlığa dönülür,
                        ama hiçbir hisse dışlanmaz
     İkincisi doğru kontroldür: modelden tek farkı dışlama kararıdır.
     Böylece "dışlama kararı" ile "rebalance temposu" ayrışır.

  3) timing ve timing_oracle AYNI burn-in ile koşulur; artık gerçekten
     yalnızca sinyal değişir.

KULLANIM
    python run_portfolio_v2.py                      # ana tablo
    python run_portfolio_v2.py --pct-sweep          # dışlama taraması
    python run_portfolio_v2.py --cash-variants      # nakit/timing
"""
import os, glob, argparse
import numpy as np, pandas as pd

from datasets.feature_engineering import prepare_dataset
from portfolio_engine import simulate_positions, performance

__build__ = "2026-09-12b"   # KAPSAM bloğu + ortak başlangıç + tradable maskesi


TEST_START, TEST_END = '2022-01-01', '2024-12-31'
OUT = 'results'


# ══════════════════════════════════════════════════════════════════
def load_panel(start=TEST_START, end=TEST_END):
    ds = prepare_dataset(force_refresh=False)
    sub = ds[(ds.index >= start) & (ds.index <= end)]
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    # İşlem görebilirlik: o gün fiyatı OLAN hisse. Pasif referansın skorları
    # sabit sıfır olduğu için bu maske olmadan fiyatsız hisseler de
    # alınabilir görünüyordu (danışman geri bildirimi #1).
    tradable = px.notna()
    return sub, ret, tradable


def common_start(scores: dict, rebal, min_names=10):
    """Tüm stratejilerin en az min_names skora sahip olduğu ilk rebalance günü.

    Model tahminleri test döneminin ilk 19 gününde yok (sekans penceresi
    dolmuyor), pasif referans ise 1. günden yatırımda. Aynı tarihten
    başlatmadan karşılaştırma adil değil.
    """
    firsts = {}
    for name, sw in scores.items():
        ok = [d for d in sorted(rebal)
              if d in sw.index and sw.loc[d].notna().sum() >= min_names]
        firsts[name] = ok[0] if ok else None
    usable = [v for v in firsts.values() if v is not None]
    return (max(usable) if usable else None), firsts


def build_scores(sub, ret, pattern, preds_dir):
    """İndeksli .npz tahminleri + naive_vol + inv_vol + oracle."""
    days, cols = ret.index, ret.columns
    out = {}
    files = sorted(glob.glob(os.path.join(preds_dir, pattern)))
    if not files:
        raise SystemExit(
            f"{preds_dir}/{pattern} bulunamadı.\n"
            f"Önce indeksli tahminleri üretin:\n"
            f"  python export_preds.py --list\n"
            f"  python export_preds.py --ablations <hücre>")
    for f in files:
        z = np.load(f, allow_pickle=True)
        if 'dates' not in z:
            raise SystemExit(
                f"{os.path.basename(f)} eski şemada (indeks yok).\n"
                f"export_preds.py ile yeniden üretin.")
        d = pd.DataFrame({'date': pd.to_datetime(z['dates']),
                          'ticker': z['tickers'], 'p': z['prob']})
        if 'split' in z:
            d = d[np.asarray(z['split']) == 'test']
        name = os.path.basename(f).replace('.npz', '')
        out[name] = (d.pivot_table(index='date', columns='ticker', values='p')
                     .reindex(index=days).reindex(columns=cols))
    if 'Vol_20d' in sub.columns:
        out['naive_vol'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                            values='Vol_20d')
                            .reindex(index=days).reindex(columns=cols))
        # hedefe uygun ters volatilite baseline'ı (Bölüm 7.2 E3)
        out['inv_vol'] = -out['naive_vol']
    out['oracle'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                     values='Target')
                     .reindex(index=days).reindex(columns=cols))
    return out


def flat_scores(ret):
    return pd.DataFrame(0.0, index=ret.index, columns=ret.columns)


# ══════════════════════════════════════════════════════════════════
def _print_build():
    import hashlib, os
    h = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]
    print(f"[{os.path.basename(__file__)}  build {__build__}  sha256 {h}]")


def main():
    _print_build()
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-pct', type=float, default=20.0)
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, nargs='+', default=[10, 25, 50])
    ap.add_argument('--pattern', default='multi_pure_seed*.npz')
    ap.add_argument('--preds-dir', default=os.path.join('results', 'preds_v2'))
    ap.add_argument('--pct-sweep', action='store_true')
    ap.add_argument('--cash-variants', action='store_true')
    ap.add_argument('--max-cash', type=float, default=40.0)
    ap.add_argument('--burnin', type=int, default=6)
    ap.add_argument('--cash-yield', type=float, default=0.0)
    args = ap.parse_args()

    tag = os.path.basename(args.pattern).replace('_seed*.npz', '').replace('*', '')
    sub, ret, tradable = load_panel()
    days = ret.index
    rebal = set(days[::args.rebalance])
    scores = build_scores(sub, ret, args.pattern, args.preds_dir)
    if not scores:
        raise SystemExit(f"{args.pattern} bulunamadı.")
    start, firsts = common_start(scores, rebal)
    if start is None:
        raise SystemExit("Hiçbir strateji için yeterli skor bulunamadı.")
    print(f"panel: {len(days)} gün × {ret.shape[1]} hisse | "
          f"{len(rebal)} rebalance | stratejiler: {len(scores)}")
    print("\n" + "═" * 74)
    print("KAPSAM VE ORTAK BAŞLANGIÇ")
    print("═" * 74)
    print(f"  panel ilk günü            : {days[0].date()}")
    print(f"  ilk rebalance             : {min(rebal).date()}")
    for name in sorted(firsts, key=lambda k: (firsts[k] is None, firsts[k])):
        f = firsts[name]
        print(f"  {name[:34]:<36}{f.date() if f is not None else 'skor yok'}")
    print(f"  → ORTAK BAŞLANGIÇ         : {start.date()}  "
          f"({(start - days[0]).days} gün sonra)")
    print("  Bütün stratejiler bu tarihten başlatılır; panel başından bu tarihe")
    print("  kadarki dönem hiçbir strateji için hesaba katılmaz.")

    # ── ana tablo: maliyet düzeyleri × stratejiler ──
    rows = []
    for cb in args.cost_bps:
        # 1) gerçek al-tut: yalnızca ilk gün kurulur
        net, tov, inv, csh, _ = simulate_positions(
            flat_scores(ret), ret, {start}, 0.0, cb,
            tradable=tradable, start_date=start)
        rows.append({**performance(net, 'buy_hold_true'), 'cost_bps': cb,
                     'turnover': tov, 'avg_invested': inv})

        # 2) eşit ağırlık, model temposuyla, dışlama yok  ← doğru kontrol
        net, tov, inv, csh, _ = simulate_positions(
            flat_scores(ret), ret, rebal, 0.0, cb,
            tradable=tradable, start_date=start)
        rows.append({**performance(net, 'equal_weight_rebal'), 'cost_bps': cb,
                     'turnover': tov, 'avg_invested': inv})

        # 3) skor tabanlı stratejiler
        for name, sw in scores.items():
            net, tov, inv, csh, _ = simulate_positions(
                sw, ret, rebal, args.exclude_pct, cb,
                tradable=tradable, start_date=start)
            rows.append({**performance(net, name), 'cost_bps': cb,
                         'turnover': tov, 'avg_invested': inv})

    P = pd.DataFrame(rows)
    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)
    P.to_csv(os.path.join(OUT, f'portfolio_v2_{tag}.csv'), index=False)

    print("\n" + "═" * 74)
    print(f"ANA TABLO — dışlama %{args.exclude_pct:.0f}, "
          f"{args.rebalance} günde bir rebalance")
    print("═" * 74)
    for cb in args.cost_bps:
        g = (P[P.cost_bps == cb].groupby('grp')[['cagr', 'max_drawdown',
                                                 'calmar', 'turnover']].mean()
             .sort_values('calmar', ascending=False))
        print(f"\n── {cb:.0f} bps ──")
        print(f"{'strateji':<24}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}{'devir':>9}")
        print("-" * 61)
        for k in g.index:
            r = g.loc[k]
            print(f"{k:<24}{100*r.cagr:>8.2f}%{100*r.max_drawdown:>9.2f}%"
                  f"{r.calmar:>9.3f}{100*r.turnover:>8.1f}%")

    # ── dışlama oranı taraması ──
    if args.pct_sweep:
        srows = []
        for pct in [5, 8, 10, 15, 20, 30]:
            for name, sw in scores.items():
                if name in ('oracle', 'inv_vol'):
                    continue
                net, tov, *_ = simulate_positions(sw, ret, rebal, pct,
                                                  args.cost_bps[0],
                                                  tradable=tradable,
                                                  start_date=start)
                srows.append({**performance(net, name), 'exclude_pct': pct,
                              'turnover': tov})
        S = pd.DataFrame(srows)
        S['grp'] = S['strategy'].str.replace(r'_seed\d+', '', regex=True)
        S.to_csv(os.path.join(OUT, f'pct_sweep_v2_{tag}.csv'), index=False)
        print("\n" + "═" * 74)
        print("DIŞLAMA ORANI TARAMASI")
        print("═" * 74)
        print(S.pivot_table(index='exclude_pct', columns='grp',
                            values=['calmar', 'max_drawdown']).round(4).to_string())

    # ── nakit / timing varyantları (burn-in EŞİT) ──
    if args.cash_variants:
        truth = (sub.pivot_table(index=sub.index, columns='Ticker', values='Target')
                 .reindex(index=days).reindex(columns=ret.columns))

        def make_timing(signal_wide, burnin):
            hist = []
            def fn(d):
                m = float(signal_wide.loc[d].dropna().mean()) if d in signal_wide.index else np.nan
                if np.isnan(m) or len(hist) < max(burnin, 1):
                    if not np.isnan(m):
                        hist.append(m)
                    return 0.0
                pct = float(np.mean(np.array(hist) < m))
                hist.append(m)
                return args.max_cash / 100.0 * float(np.clip((pct - 0.5) / 0.5, 0, 1))
            return fn

        crows = []
        cb = args.cost_bps[0]
        for name, sw in scores.items():
            if name in ('oracle', 'naive_vol', 'inv_vol'):
                continue
            for mode in ('renorm', 'cash', 'timing', 'timing_oracle'):
                if mode == 'renorm':
                    fn = None
                elif mode == 'cash':
                    fn = lambda d: args.exclude_pct / 100.0
                elif mode == 'timing':
                    fn = make_timing(sw, args.burnin)
                else:
                    fn = make_timing(truth, args.burnin)   # ← AYNI burn-in
                net, tov, inv, csh, _ = simulate_positions(
                    sw, ret, rebal, args.exclude_pct, cb,
                    cash_weight_fn=fn, cash_yield=args.cash_yield,
                    tradable=tradable, start_date=start)
                crows.append({**performance(net, f'{name}__{mode}'),
                              'mode': mode, 'turnover': tov,
                              'avg_invested': inv, 'avg_cash': csh})
        net, tov, inv, csh, _ = simulate_positions(
            flat_scores(ret), ret, rebal, 0.0, cb, cash_yield=args.cash_yield,
            tradable=tradable, start_date=start)
        crows.append({**performance(net, 'equal_weight_rebal'),
                      'mode': 'benchmark', 'turnover': tov,
                      'avg_invested': inv, 'avg_cash': 0.0})
        C = pd.DataFrame(crows)
        C.to_csv(os.path.join(OUT, f'cash_portfolio_v2_{tag}.csv'), index=False)
        print("\n" + "═" * 74)
        print(f"NAKİT VARYANTLARI (burn-in = {args.burnin} her ikisinde de, "
              f"nakit getirisi %{args.cash_yield:.1f})")
        print("═" * 74)
        g = C.groupby('mode')[['cagr', 'max_drawdown', 'calmar',
                               'avg_invested', 'avg_cash']].mean()
        print(g.round(4).sort_values('calmar', ascending=False).to_string())

    print(f"\n→ {OUT}/portfolio_v2_{tag}.csv")
    if args.pct_sweep:      print(f"→ {OUT}/pct_sweep_v2_{tag}.csv")
    if args.cash_variants:  print(f"→ {OUT}/cash_portfolio_v2_{tag}.csv")


if __name__ == '__main__':
    main()