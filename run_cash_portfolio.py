"""
run_cash_portfolio.py — dışlanan bütçe nakde giderse ne olur?

MEVCUT DURUM
    run_portfolio_simulation.simulate() dışlanan hisselerin ağırlığını
    kalanlara yeniden dağıtıyor:  new[keep] = 1/len(keep)
    Yani portföy her zaman %100 yatırımda. Sonuç: maxDD hiçbir ayarda
    değişmiyor (−0,21), ORACLE'da bile (−0,18). Strateji tasarımı gereği
    düşüş azaltamaz — yalnızca hangi isimleri tuttuğunu değiştirir.

BU SCRIPT DÖRT AĞIRLIKLANDIRMA DENER
    renorm         mevcut davranış (KONTROL — wl_portfolio.csv'yi yeniden
                   üretmeli; üretmiyorsa bir şey bozuk demektir)
    cash           dışlanan bütçe nakitte kalır, sabit oranda
    timing         nakit oranı modelin kesitsel ortalama riskiyle değişir
    timing_oracle  aynı kural ama GERÇEK kriz oranıyla — üst sınır teşhisi

NEDENSELLİK
    'timing' modunda t günündeki yüzdelik yalnızca t'den ÖNCEKİ rebalance
    günlerine göre hesaplanır (genişleyen pencere + burn-in). İleriye bakma
    yok. 'timing_oracle' kasten ileriye bakar; karşılaştırma noktası olarak
    vardır, strateji önerisi değildir.

KULLANIM
    python run_cash_portfolio.py --exclude-pct 8 --max-cash 40
    python run_cash_portfolio.py --exclude-pct 8 --max-cash 40 --cash-yield 4.5
"""
import os, glob, argparse
import numpy as np, pandas as pd

from datasets.feature_engineering import prepare_dataset
from run_portfolio_simulation import performance

TEST_START, TEST_END = '2022-01-01', '2024-12-31'
TRADING_DAYS = 252


############################################################
def simulate_cash(scores_wide, ret_wide, rebal_dates, exclude_pct, cost_bps,
                  mode='renorm', max_cash=0.0, burnin=6, cash_yield=0.0,
                  timing_signal=None):
    """
    Nakit pozisyonuna izin veren de-risking simülasyonu.

    mode:
      'renorm'  → new[keep] = 1/len(keep)              (toplam 1,0)
      'cash'    → new[keep] = 1/len(avail)             (toplam 1−k/N)
      'timing'  → new[keep] = (1−cash_t)/len(keep)     (cash_t zamanla değişir)

    timing_signal : (tarih × hisse) matris. Nakit oranını sürmek için
                    kullanılır. None ise scores_wide kullanılır.
                    timing_oracle için buraya gerçek Target matrisi verilir.
    """
    weights = pd.DataFrame(0.0, index=ret_wide.index, columns=ret_wide.columns)
    cur = pd.Series(0.0, index=ret_wide.columns)
    cur_cash = 0.0
    cash_series = pd.Series(0.0, index=ret_wide.index)
    turnover_log, cash_log = {}, {}

    sig = timing_signal if timing_signal is not None else scores_wide
    hist = []          # geçmiş kesitsel ortalamalar — yalnızca geçmiş

    for d in ret_wide.index:
        if d in rebal_dates:
            s = scores_wide.loc[d] if d in scores_wide.index else pd.Series(dtype=float)
            avail = s.dropna()
            avail = avail[avail.index.isin(ret_wide.columns)]

            if len(avail) >= 10:
                N = len(avail)
                k = int(np.ceil(N * exclude_pct / 100.0))

                # ── sıralama: artan = [en güvenli ... en riskli] ──
                ranked = avail.sort_values(ascending=True, kind='mergesort')
                keep = ranked.index[:N - k] if k > 0 else ranked.index

                if k > 0 and len(keep) > 0:
                    dropped = ranked.index[N - k:]
                    if avail[keep].mean() > avail[dropped].mean():
                        raise RuntimeError(
                            f"Sıralama yönü ters! tutulan={avail[keep].mean():.4f} "
                            f"> çıkarılan={avail[dropped].mean():.4f}")

                # ── nakit oranı ──
                if mode == 'renorm':
                    cash_t = 0.0
                elif mode == 'cash':
                    cash_t = k / N
                elif mode == 'timing':
                    m_t = float(sig.loc[d].dropna().mean()) if d in sig.index else np.nan
                    if np.isnan(m_t) or len(hist) < max(burnin, 1):
                        cash_t = 0.0
                    else:
                        pct = float(np.mean(np.array(hist) < m_t))   # [0,1]
                        cash_t = max_cash / 100.0 * np.clip((pct - 0.5) / 0.5, 0.0, 1.0)
                    if not np.isnan(m_t):
                        hist.append(m_t)          # SONRA ekle → t kendini görmez
                else:
                    raise ValueError(f"bilinmeyen mode: {mode}")

                new = pd.Series(0.0, index=ret_wide.columns)
                if len(keep) > 0:
                    if mode == 'renorm':
                        new[keep] = 1.0 / len(keep)
                    elif mode == 'cash':
                        new[keep] = 1.0 / N
                    else:
                        new[keep] = (1.0 - cash_t) / len(keep)

                # nakit değişimi de işlem maliyeti doğurur
                turnover_log[d] = float(((new - cur).abs().sum()
                                         + abs(cash_t - cur_cash)) / 2.0)
                cur, cur_cash = new, cash_t
                cash_log[d] = cash_t

        weights.loc[d] = cur
        cash_series.loc[d] = cur_cash

    # sinyal t'de, getiri t+1'de
    gross = (weights.shift(1).fillna(0.0) * ret_wide).sum(axis=1)

    # nakit getirisi (yıllık %cash_yield → günlük)
    if cash_yield != 0.0:
        daily_rf = (1.0 + cash_yield / 100.0) ** (1.0 / TRADING_DAYS) - 1.0
        gross = gross + cash_series.shift(1).fillna(0.0) * daily_rf

    cost = pd.Series(0.0, index=ret_wide.index)
    for d, t in turnover_log.items():
        cost.loc[d] = t * (cost_bps / 10_000.0)

    net = gross - cost
    avg_turnover = float(np.mean(list(turnover_log.values()))) if turnover_log else 0.0
    avg_invested = float(weights.sum(axis=1).mean())
    avg_cash = float(np.mean(list(cash_log.values()))) if cash_log else 0.0
    return net, avg_turnover, avg_invested, avg_cash


############################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude-pct', type=float, default=8.0)
    ap.add_argument('--max-cash', type=float, default=40.0,
                    help='timing modunda üst sınır, yüzde')
    ap.add_argument('--burnin', type=int, default=6,
                    help='timing devreye girmeden önceki rebalance sayısı')
    ap.add_argument('--cash-yield', type=float, default=0.0,
                    help='nakde uygulanacak yıllık getiri, yüzde (2022-24 ~4,5)')
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--pattern', default='WL_uniform_*.npz')
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    ds = prepare_dataset(force_refresh=False)
    sub = ds[(ds.index >= TEST_START) & (ds.index <= TEST_END)]
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    days = ret.index
    rebal = set(days[::args.rebalance])

    # gerçek kriz oranı — timing_oracle için
    truth = (sub.pivot_table(index=sub.index, columns='Ticker', values='Target')
             .reindex(index=days).reindex(columns=ret.columns))

    files = sorted(glob.glob(os.path.join(args.outdir, 'preds', args.pattern)))
    if not files:
        raise SystemExit(f"{args.pattern} bulunamadı.")

    rows = []

    # buy & hold
    flat = pd.DataFrame(1.0, index=days, columns=ret.columns)
    net, tov, inv, csh = simulate_cash(flat, ret, rebal, 0.0, args.cost_bps,
                                       mode='renorm')
    rows.append({**performance(net, 'buy_hold'), 'mode': 'buy_hold',
                 'avg_invested': inv, 'avg_cash': csh, 'turnover': tov})

    for f in files:
        z = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        sw = (pd.DataFrame({'date': pd.to_datetime(z['dates']),
                            'ticker': z['tickers'], 'p': z['prob']})
              .pivot_table(index='date', columns='ticker', values='p')
              .reindex(index=days).reindex(columns=ret.columns))

        for mode in ['renorm', 'cash', 'timing']:
            net, tov, inv, csh = simulate_cash(
                sw, ret, rebal, args.exclude_pct, args.cost_bps,
                mode=mode, max_cash=args.max_cash, burnin=args.burnin,
                cash_yield=args.cash_yield)
            rows.append({**performance(net, f'{name}__{mode}'), 'mode': mode,
                         'avg_invested': inv, 'avg_cash': csh, 'turnover': tov})

        # üst sınır: gerçek kriz oranıyla zamanlama
        net, tov, inv, csh = simulate_cash(
            sw, ret, rebal, args.exclude_pct, args.cost_bps,
            mode='timing', max_cash=args.max_cash, burnin=0,
            cash_yield=args.cash_yield, timing_signal=truth)
        rows.append({**performance(net, f'{name}__timing_oracle'),
                     'mode': 'timing_oracle', 'avg_invested': inv,
                     'avg_cash': csh, 'turnover': tov})

    P = pd.DataFrame(rows)
    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)

    g = (P.groupby('mode')[['cagr', 'max_drawdown', 'calmar', 'sortino',
                            'avg_invested', 'avg_cash']].mean())

    print("═" * 78)
    print(f"NAKİT VARYANTI  (dışlama %{args.exclude_pct:.0f}, "
          f"maks nakit %{args.max_cash:.0f}, nakit getirisi %{args.cash_yield:.1f})")
    print("═" * 78)
    print(f"{'mod':<16}{'CAGR':>9}{'maksDD':>10}{'Calmar':>9}"
          f"{'Sortino':>9}{'yatırım':>10}{'ort.nakit':>11}")
    print("-" * 78)
    for k in g.sort_values('calmar', ascending=False).index:
        r = g.loc[k]
        print(f"{k:<16}{100*r.cagr:>8.2f}%{100*r.max_drawdown:>9.2f}%"
              f"{r.calmar:>9.3f}{r.sortino:>9.3f}"
              f"{100*r.avg_invested:>9.1f}%{100*r.avg_cash:>10.1f}%")
    print("-" * 78)

    out = os.path.join(args.outdir, 'cash_portfolio.csv')
    P.to_csv(out, index=False)
    print(f"\n→ {out}")

    # ── kontrol ──
    rn = P[P['mode'] == 'renorm']['calmar'].mean()
    print(f"\nKONTROL: renorm Calmar = {rn:.4f} "
          f"(wl_portfolio.csv'deki WL_uniform 0,8083 ile eşleşmeli)")


if __name__ == '__main__':
    main()