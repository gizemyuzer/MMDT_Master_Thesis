"""
portfolio_engine.py — hisse adedi bazlı portföy simülasyonu (DÜZELTİLMİŞ).

NEDEN YENİ BİR MOTOR
    run_portfolio_simulation.simulate() ağırlıkları rebalance gününde
    belirleyip sonraki 20 gün SABİT tutuyor ve her gün getiriyi
    sum(w × r) ile hesaplıyordu. Bu, sabit-ağırlık (constant-mix)
    portföyüdür: örtük olarak HER GÜN hedef ağırlıklara geri döner ve
    bu günlük işlemin maliyeti hiç alınmaz.

    Al-tut istiyorsak pozisyonlar fiyatla sürüklenmeli; işlem maliyeti
    yalnızca gerçekten alım-satım yapılan günlerde ve gerçekten
    işlem gören tutar üzerinden alınmalıdır.

DOĞRULAMA
    self_test() danışmanın örneğini içerir: eşit ağırlıklı 10 hisseden
    biri önce %100 yükselip sonra %50 düşüyor, diğerleri sabit, arada
    işlem yok. Doğru cevap %0.00; eski motor %4.50 veriyordu.

KULLANIM
    from portfolio_engine import simulate_positions, performance
    python portfolio_engine.py --self-test
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════
def simulate_positions(scores_wide: pd.DataFrame,
                       ret_wide: pd.DataFrame,
                       rebal_dates,
                       exclude_pct: float,
                       cost_bps: float,
                       cash_weight_fn=None,
                       cash_yield: float = 0.0,
                       higher_is_riskier: bool = True,
                       initial_capital: float = 1.0):
    """
    Hisse-adedi (pozisyon değeri) bazlı simülasyon.

    scores_wide  : (tarih × hisse) risk skoru; NaN = o gün işlem görmüyor
    ret_wide     : (tarih × hisse) günlük getiri
    rebal_dates  : yeniden dengeleme günleri (küme)
    exclude_pct  : her rebalance'ta çıkarılacak en riskli yüzde
    cash_weight_fn : d -> [0,1] nakit oranı döndüren fonksiyon (None = 0)
    cash_yield   : nakde uygulanan YILLIK getiri, yüzde
    initial_capital : başlangıç sermayesi

    DÖNÜŞ: net günlük getiri serisi, ortalama tek yönlü devir hızı,
           ortalama yatırım oranı, ortalama nakit oranı, günlük ağırlık matrisi

    MUHASEBE
      • pozisyonlar her gün getiriyle çarpılır (ağırlıklar sürüklenir)
      • nakit günlük risksiz oranla büyür
      • rebalance gününde hedef pozisyonlar hesaplanır; GERÇEKTEN işlem
        gören tutar |hedef − mevcut| üzerinden maliyet düşülür
      • sinyal t'de belirlenir, portföy t kapanışında kurulur, getiri
        t+1'den itibaren yeni pozisyonlara işler
    """
    cols = ret_wide.columns
    dates = ret_wide.index
    daily_rf = ((1.0 + cash_yield / 100.0) ** (1.0 / TRADING_DAYS) - 1.0
                if cash_yield else 0.0)

    pos = pd.Series(0.0, index=cols)     # her hissedeki tutar
    cash = float(initial_capital)        # nakit
    equity = np.empty(len(dates))
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    turnover_log, cash_log = {}, {}

    for i, d in enumerate(dates):
        # ── 1) piyasa hareketi: pozisyonlar sürüklenir ──
        r = ret_wide.loc[d].fillna(0.0)
        pos = pos * (1.0 + r)
        cash = cash * (1.0 + daily_rf)

        # ── 2) rebalance: hedefe git, gerçek işlem maliyetini öde ──
        if d in rebal_dates:
            s = scores_wide.loc[d] if d in scores_wide.index else pd.Series(dtype=float)
            avail = s.dropna()
            avail = avail[avail.index.isin(cols)]

            if len(avail) >= 10:
                N = len(avail)
                k = int(np.ceil(N * exclude_pct / 100.0))
                ranked = avail.sort_values(ascending=higher_is_riskier,
                                           kind='mergesort')
                keep = ranked.index[:N - k] if k > 0 else ranked.index

                if k > 0 and len(keep) > 0:
                    dropped = ranked.index[N - k:]
                    if avail[keep].mean() > avail[dropped].mean():
                        raise RuntimeError(
                            f"Sıralama yönü ters! tutulan={avail[keep].mean():.4f} "
                            f"> çıkarılan={avail[dropped].mean():.4f}")

                cw = float(cash_weight_fn(d)) if cash_weight_fn else 0.0
                cw = min(max(cw, 0.0), 1.0)

                total = float(pos.sum() + cash)
                target = pd.Series(0.0, index=cols)
                if len(keep) > 0:
                    target[keep] = total * (1.0 - cw) / len(keep)

                traded = float((target - pos).abs().sum())      # tek yön toplam
                fee = traded * (cost_bps / 10_000.0)

                pos = target
                cash = total * cw - fee                         # maliyet nakitten
                turnover_log[d] = traded / total if total > 0 else 0.0
                cash_log[d] = cw

        equity[i] = pos.sum() + cash
        tot = equity[i]
        if tot > 0:
            weights.loc[d] = pos / tot

    eq = pd.Series(equity, index=dates)
    net = eq.pct_change().fillna(eq.iloc[0] / initial_capital - 1.0)
    avg_turnover = float(np.mean(list(turnover_log.values()))) if turnover_log else 0.0
    avg_invested = float(weights.sum(axis=1).mean())
    avg_cash = float(np.mean(list(cash_log.values()))) if cash_log else 0.0
    return net, avg_turnover, avg_invested, avg_cash, weights


# ══════════════════════════════════════════════════════════════════
def performance(ret: pd.Series, label: str = '') -> dict:
    """Günlük net getiri serisinden performans metrikleri. Şema sabittir."""
    keys = ['strategy', 'total_return', 'cagr', 'ann_vol', 'sharpe',
            'sortino', 'max_drawdown', 'calmar', 'n_days']
    r = ret.dropna()
    if len(r) < 20:
        out = {k: np.nan for k in keys}
        out['strategy'] = label
        out['n_days'] = len(r)
        return out
    cum = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(cum.iloc[-1] - 1)
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    sd = r.std(ddof=1)
    vol = float(sd * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    dn = r[r < 0]
    sortino = (float(r.mean() / dn.std(ddof=1) * np.sqrt(TRADING_DAYS))
               if len(dn) > 1 and dn.std(ddof=1) > 0 else np.nan)
    run_max = np.maximum.accumulate(cum.values)
    maxdd = float(((cum.values - run_max) / run_max).min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else np.nan
    return {'strategy': label, 'total_return': total, 'cagr': cagr,
            'ann_vol': vol, 'sharpe': sharpe, 'sortino': sortino,
            'max_drawdown': maxdd, 'calmar': calmar, 'n_days': len(r)}


# ══════════════════════════════════════════════════════════════════
def self_test() -> bool:
    """Motorun muhasebesini bilinen cevaplı örneklerle sınar."""
    ok = True

    def check(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) < tol
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}: {100*got:+.4f}%  (beklenen {100*want:+.4f}%)")

    print("═" * 62)
    print("TEST 1 — danışmanın örneği: al-tut, arada işlem yok")
    print("═" * 62)
    # 10 hisse eşit ağırlık; A +100% sonra −50%; diğerleri sabit
    dates = pd.date_range('2020-01-01', periods=3, freq='B')
    cols = [f'S{i}' for i in range(10)]
    ret = pd.DataFrame(0.0, index=dates, columns=cols)
    ret.iloc[1, 0] = 1.0      # A: +100%
    ret.iloc[2, 0] = -0.5     # A: −50%
    sc = pd.DataFrame(0.0, index=dates, columns=cols)
    net, tov, inv, csh, _ = simulate_positions(
        sc, ret, {dates[0]}, exclude_pct=0.0, cost_bps=0.0)
    check("toplam getiri", float((1 + net).prod() - 1), 0.0)

    print()
    print("═" * 62)
    print("TEST 2 — sabit-ağırlık (eski motorun örtük varsayımı)")
    print("═" * 62)
    # Her gün rebalance edilirse gerçekten +4.5% olmalı
    net2, *_ = simulate_positions(sc, ret, set(dates),
                                  exclude_pct=0.0, cost_bps=0.0)
    check("her gün rebalance", float((1 + net2).prod() - 1), 0.045, tol=1e-9)

    print()
    print("═" * 62)
    print("TEST 3 — işlem maliyeti yalnızca işlem gününde alınıyor")
    print("═" * 62)
    ret3 = pd.DataFrame(0.0, index=dates, columns=cols)
    net3, tov3, *_ = simulate_positions(sc, ret3, {dates[0]},
                                        exclude_pct=0.0, cost_bps=100.0)
    # tek kurulum: sermayenin tamamı işlem görür → 100 bps = %1
    check("tek kurulum maliyeti", float((1 + net3).prod() - 1), -0.01)
    print(f"  ✓ devir hızı: {tov3:.4f} (beklenen 1.0000)")
    ok &= abs(tov3 - 1.0) < 1e-9

    print()
    print("═" * 62)
    print("TEST 4 — nakit getirisi")
    print("═" * 62)
    sc4 = pd.DataFrame(np.nan, index=dates, columns=cols)   # hiç pozisyon yok
    net4, *_ = simulate_positions(sc4, ret3, {dates[0]}, exclude_pct=0.0,
                                  cost_bps=0.0, cash_yield=TRADING_DAYS * 0.0)
    check("sıfır getiri, sıfır faiz", float((1 + net4).prod() - 1), 0.0)

    print()
    print("═" * 62)
    print("SONUÇ:", "TÜM TESTLER GEÇTİ" if ok else "BAŞARISIZ")
    print("═" * 62)
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    print(__doc__)