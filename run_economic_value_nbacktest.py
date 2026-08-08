"""
run_economic_value_backtest.py
────────────────────────────────
Sınıflandırma metriklerini (MCC, PR-AUC) gerçekleşmiş ekonomik değere çevirir.

SORUN: Tez şu ana kadar "model kriz tahmininde ne kadar isabetli" sorusuna
cevap veriyor (MCC/PR-AUC), ama "bu tahmin bir portföy yöneticisi için
işlem maliyeti dahil gerçek bir değer yaratıyor mu" sorusuna cevap vermiyor.
Bu script o köprüyü kurar.

YÖNTEM — basit bir risk-overlay (de-risking) stratejisi:
    Her gün t'de, model bir hisse için P(kriz) >= threshold derse, o hisse
    portföyden çıkarılır (ağırlık=0); aksi halde eşit-ağırlıklı portföyde
    tutulur. Günlük yeniden dengelenir (daily rebalance).

    Benchmark: aynı evrende, de-risking KULLANMAYAN eşit-ağırlıklı buy&hold.

    İşlem maliyeti: her gün turnover'ın (ağırlık değişiminin) belirli bir
    baz puanı (varsayılan 10bps) kadar getiriden düşülür — gerçekçi bir
    basitleştirme, tam bir execution-cost modeli değil.

ÇIKTI: toplam getiri, yıllıklaştırılmış Sharpe, maksimum drawdown —
    strateji vs benchmark karşılaştırması. results/economic_value_backtest.csv

KULLANIM:
    python run_economic_value_backtest.py \\
        --checkpoint checkpoints/best_dualencoder_gated_cross_attention.pth \\
        --fusion gated_cross_attention \\
        --cost-bps 10
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import _evaluate_on_loader, find_best_threshold_mcc


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def annualized_sharpe(daily_returns, periods_per_year=252):
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if r.std(ddof=1) < 1e-12:
        return 0.0
    return (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(cum_returns):
    cum = np.asarray(cum_returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return dd.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--fusion', type=str, default='gated_cross_attention',
                    choices=['concat', 'cross_attention', 'gated_cross_attention', 'film'])
    ap.add_argument('--modality', type=str, default='multimodal',
                    choices=['tech_only', 'fund_only', 'multimodal'])
    ap.add_argument('--cost-bps', type=float, default=10.0,
                    help='Turnover başına işlem maliyeti (baz puan, tek yön)')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("[1/5] Dataset yükleniyor (cache)...")
    dataset_out = prepare_dataset(force_refresh=False)

    print("[2/5] Dataloader'lar kuruluyor...")
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(dataset_out, seq_len=20, batch_size=args.batch_size)

    print(f"[3/5] Model yükleniyor: {args.checkpoint}")
    model = DualEncoderTransformer(
        tech_dim=len(tech_cols), fund_dim=len(fund_cols),
        seq_len=20, d_model=64, n_heads=4, n_layers=2, dropout=0.15,
        modality=args.modality, fusion_type=args.fusion,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    val_preds, val_targets, _ = _evaluate_on_loader(model, val_loader, device)
    threshold, _ = find_best_threshold_mcc(val_targets, val_preds)
    print(f"      MCC-optimal threshold (val'dan): {threshold:.4f}")

    print("[4/5] Test tahminleri + fiyat serisi hizalanıyor...")
    test_preds, test_targets, _ = _evaluate_on_loader(model, test_loader, device)
    ds = test_loader.dataset
    pred_df = pd.DataFrame({
        'date': ds.dates.values,
        'ticker': ds.tickers,
        'prob': test_preds,
    })
    # Wide format: date x ticker olasılık matrisi
    prob_wide = pred_df.pivot_table(index='date', columns='ticker', values='prob')

    # Test dönemindeki günlük fiyatlardan getiri matrisini kur
    test_mask = (dataset_out.index >= '2022-01-01') & (dataset_out.index <= '2024-12-31')
    price_wide = dataset_out[test_mask].pivot_table(index=dataset_out[test_mask].index,
                                                     columns='Ticker', values='Close')
    ret_wide = price_wide.pct_change()

    # Ortak tarih ve ticker kesişimi
    common_dates = prob_wide.index.intersection(ret_wide.index).sort_values()
    common_tickers = prob_wide.columns.intersection(ret_wide.columns)
    prob_wide = prob_wide.loc[common_dates, common_tickers]
    ret_wide = ret_wide.loc[common_dates, common_tickers]
    # NaN olasılıkları "riskli değil" varsay (0), NaN getirileri 0 varsay (o gün pozisyon yok)
    prob_wide = prob_wide.ffill().fillna(0.0)
    ret_wide = ret_wide.fillna(0.0)

    print(f"      {len(common_dates):,} gün | {len(common_tickers)} hisse hizalandı")

    print("[5/5] Backtest koşuluyor...")
    flagged = (prob_wide >= threshold)

    # ── Strateji: flagged olmayan hisselere eşit ağırlık ──
    n_safe = (~flagged).sum(axis=1).replace(0, np.nan)  # gün gün "güvenli" hisse sayısı
    strat_weights = (~flagged).astype(float).div(n_safe, axis=0).fillna(0.0)

    # ── Benchmark: her gün TÜM hisselere eşit ağırlık (de-risking yok) ──
    n_all = len(common_tickers)
    bench_weights = pd.DataFrame(1.0 / n_all, index=common_dates, columns=common_tickers)

    # Getiriler bir gün ileri kaydırılmış ağırlıkla çarpılır (t günü sinyali,
    # t+1 getirisine uygulanır — look-ahead yok)
    strat_gross_ret = (strat_weights.shift(1).fillna(0.0) * ret_wide).sum(axis=1)
    bench_gross_ret = (bench_weights.shift(1).fillna(0.0) * ret_wide).sum(axis=1)

    # ── İşlem maliyeti: turnover * cost_bps ──
    cost_rate = args.cost_bps / 10_000.0
    strat_turnover = (strat_weights - strat_weights.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    bench_turnover = (bench_weights - bench_weights.shift(1).fillna(bench_weights.iloc[0])).abs().sum(axis=1) / 2.0

    strat_net_ret = strat_gross_ret - strat_turnover * cost_rate
    bench_net_ret = bench_gross_ret - bench_turnover * cost_rate

    strat_cum = (1 + strat_net_ret).cumprod()
    bench_cum = (1 + bench_net_ret).cumprod()

    results = {
        'strategy': {
            'total_return': strat_cum.iloc[-1] - 1,
            'annualized_sharpe': annualized_sharpe(strat_net_ret),
            'max_drawdown': max_drawdown(strat_cum.values),
            'avg_daily_turnover': strat_turnover.mean(),
        },
        'benchmark_buyhold': {
            'total_return': bench_cum.iloc[-1] - 1,
            'annualized_sharpe': annualized_sharpe(bench_net_ret),
            'max_drawdown': max_drawdown(bench_cum.values),
            'avg_daily_turnover': bench_turnover.mean(),
        },
    }

    print("\n" + "=" * 70)
    print(f"EKONOMİK DEĞER BACKTEST — {args.fusion} | test 2022-2024 | "
          f"cost={args.cost_bps}bps")
    print("=" * 70)
    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    Toplam getiri        : {r['total_return']*100:+.2f}%")
        print(f"    Yıllık Sharpe         : {r['annualized_sharpe']:.3f}")
        print(f"    Maksimum drawdown     : {r['max_drawdown']*100:.2f}%")
        print(f"    Ort. günlük turnover  : {r['avg_daily_turnover']*100:.2f}%")

    sharpe_diff = results['strategy']['annualized_sharpe'] - results['benchmark_buyhold']['annualized_sharpe']
    print(f"\n  Sharpe farkı (strateji - benchmark): {sharpe_diff:+.3f}")
    if sharpe_diff > 0.05:
        print("  → De-risking overlay, işlem maliyeti dahil risk-ayarlı getiriyi iyileştiriyor.")
    elif sharpe_diff < -0.05:
        print("  → De-risking overlay, işlem maliyeti dahil buy&hold'dan daha kötü —")
        print("    modelin sinyali, çıkış/giriş maliyetini karşılayacak kadar güçlü değil.")
    else:
        print("  → İki strateji arasında anlamlı bir fark yok (gürültü düzeyinde).")
    print("=" * 70)

    out_path = os.path.join(args.outdir, 'economic_value_backtest.csv')
    pd.DataFrame(results).T.to_csv(out_path)
    curves_path = os.path.join(args.outdir, 'economic_value_curves.csv')
    pd.DataFrame({'strategy_cum': strat_cum, 'benchmark_cum': bench_cum}).to_csv(curves_path)
    print(f"\nÖzet kaydedildi → {out_path}")
    print(f"Kümülatif getiri eğrileri → {curves_path}")


if __name__ == '__main__':
    main()