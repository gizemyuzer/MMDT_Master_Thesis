"""
run_regime_analysis.py
───────────────────────
TEST SETİNİ REJİMLERE BÖLEREK analiz eder: 2022 / 2023 / 2024.

═══════════════════════════════════════════════════════════════════════
NEDEN BU ANALİZ
═══════════════════════════════════════════════════════════════════════
Test setiniz (2022-2024) tek bir sayıya eziliyor, ama içinde ÜÇ farklı
piyasa rejimi var:
    2022 — faiz kaynaklı ayı piyasası (çarpan sıkışması)
    2023 — toparlanma
    2024 — boğa

Tezin merkezi iddiası "modalitelerin bilgi içeriği rejime bağlıdır". Bu
iddia şu ana kadar val→test farkıyla dolaylı olarak gösterildi. Bu script
DOĞRUDAN gösterir: hangi modalite hangi rejimde öne geçiyor?

Beklenti (deney öncesi kayda geçsin):
    - fundamental sinyal 2022'de (borçlanma maliyeti şoku) görece güçlü,
      2023-24'te zayıf olmalı
    - teknik sinyal tüm rejimlerde daha istikrarlı olmalı
    - taban oran (kriz oranı) 2022'de yüksek, 2024'te düşük olmalı

═══════════════════════════════════════════════════════════════════════
METODOLOJİ
═══════════════════════════════════════════════════════════════════════
KRİTİK: Karar eşiği YALNIZCA validation'dan seçilir ve ÜÇ YILA DA AYNI
eşik uygulanır. Yıl bazında eşik yeniden optimize edilmez — bu, test
setine bakarak karar vermek olurdu (leakage) ve rejim farkı bulgusunu
yapay olarak üretirdi.

Her (ablation, seed) çifti ayrı değerlendirilir, sonra yıl bazında
ortalama ± std raporlanır. Tek seed'e güvenilmez.

Ayrıca her yıl için ekonomik değer hesaplanır (de-risking overlay vs
eşit-ağırlıklı buy&hold, işlem maliyeti dahil) — MCC bir finans jürisi
için anlamlı değildir, Sharpe ve drawdown anlamlıdır.

KULLANIM:
    python run_regime_analysis.py
    python run_regime_analysis.py --ablations multi_pure fund_pure_only
    python run_regime_analysis.py --cost-bps 5 --no-backtest
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, accuracy_score,
)

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import _evaluate_on_loader, find_best_threshold_mcc
from run_modality_v2 import ABLATIONS, BASE_CONFIG

YEARS = [2022, 2023, 2024]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def annualized_sharpe(r, periods_per_year=252):
    r = np.asarray(r)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year))


def max_drawdown(cum):
    cum = np.asarray(cum)
    if len(cum) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cum)
    return float(((cum - running_max) / running_max).min())


def classification_metrics(y_true, prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    out = {'n': len(y_true), 'base_rate': float(y_true.mean())}
    try:
        out['roc_auc'] = roc_auc_score(y_true, prob)
    except ValueError:
        out['roc_auc'] = np.nan
    try:
        out['pr_auc'] = average_precision_score(y_true, prob)
    except ValueError:
        out['pr_auc'] = np.nan
    out['mcc'] = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan
    out['precision'] = precision_score(y_true, y_pred, zero_division=0)
    out['recall'] = recall_score(y_true, y_pred, zero_division=0)
    out['f1'] = f1_score(y_true, y_pred, zero_division=0)
    out['accuracy'] = accuracy_score(y_true, y_pred)
    out['flag_rate'] = float(y_pred.mean())   # modelin kaç gözlemi riskli dediği
    return out


def yearly_backtest(pred_df, dataset_out, threshold, cost_bps):
    """
    Yıl bazında de-risking overlay backtest'i.
    pred_df: date, ticker, prob sütunlu DataFrame (tek seed'in test tahminleri)

    Sinyal t gününde üretilir, t+1 getirisine uygulanır (shift(1)) — ileriye
    bakış yok. ile aynı mantık.
    """
    prob_wide = pred_df.pivot_table(index='date', columns='ticker', values='prob')

    mask = (dataset_out.index >= '2022-01-01') & (dataset_out.index <= '2024-12-31')
    sub = dataset_out[mask]
    price_wide = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret_wide = price_wide.pct_change()

    dates = prob_wide.index.intersection(ret_wide.index).sort_values()
    tickers = prob_wide.columns.intersection(ret_wide.columns)
    prob_wide = prob_wide.loc[dates, tickers].ffill().fillna(0.0)
    ret_wide = ret_wide.loc[dates, tickers].fillna(0.0)

    flagged = prob_wide >= threshold
    n_safe = (~flagged).sum(axis=1).replace(0, np.nan)
    strat_w = (~flagged).astype(float).div(n_safe, axis=0).fillna(0.0)
    bench_w = pd.DataFrame(1.0 / len(tickers), index=dates, columns=tickers)

    cost = cost_bps / 10_000.0
    strat_ret = ((strat_w.shift(1).fillna(0.0) * ret_wide).sum(axis=1)
                 - (strat_w - strat_w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0 * cost)
    bench_ret = ((bench_w.shift(1).fillna(0.0) * ret_wide).sum(axis=1)
                 - (bench_w - bench_w.shift(1).fillna(bench_w.iloc[0])).abs().sum(axis=1) / 2.0 * cost)

    rows = []
    for y in YEARS:
        m = strat_ret.index.year == y
        if m.sum() < 20:
            continue
        s, b = strat_ret[m], bench_ret[m]
        s_cum, b_cum = (1 + s).cumprod(), (1 + b).cumprod()
        rows.append({
            'year': y,
            'strat_return': float(s_cum.iloc[-1] - 1),
            'strat_sharpe': annualized_sharpe(s),
            'strat_maxdd': max_drawdown(s_cum.values),
            'bench_return': float(b_cum.iloc[-1] - 1),
            'bench_sharpe': annualized_sharpe(b),
            'bench_maxdd': max_drawdown(b_cum.values),
            'avg_flag_rate': float(flagged[m].mean().mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablations', type=str, nargs='+',
                    default=['multi_pure', 'multi_xs', 'fund_pure_only',
                             'fund_xs_only', 'macro_only'],
                    choices=list(ABLATIONS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--no-backtest', action='store_true',
                    help='Ekonomik değer hesabını atla (sadece sınıflandırma metrikleri)')
    ap.add_argument('--ckpt-dir', type=str, default='checkpoints')
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()

    print("═" * 78)
    print("REJİM ANALİZİ — test seti yıl bazında (2022 / 2023 / 2024)")
    print("═" * 78)
    print(f"  Ablation'lar: {args.ablations}")
    print(f"  Seed'ler    : {args.seeds}")
    print(f"  Device      : {device}")
    print(f"  Eşik        : val'dan seçilir, ÜÇ YILA DA aynı uygulanır (leakage yok)")
    print()

    dataset_out = prepare_dataset(force_refresh=False)

    # ── Rejim karakterizasyonu (modelden bağımsız, verinin kendisinden) ──
    print("\n" + "─" * 78)
    print("REJİM KARAKTERİZASYONU (evren geneli, modelden bağımsız)")
    print("─" * 78)
    test_mask = (dataset_out.index >= '2022-01-01') & (dataset_out.index <= '2024-12-31')
    tsub = dataset_out[test_mask]
    regime_rows = []
    print(f"{'Yıl':<6}{'evren getirisi':>16}{'ort. VIX':>11}{'ort. Vol_20d':>14}{'kriz oranı':>12}{'n':>10}")
    for y in YEARS:
        ys = tsub[tsub.index.year == y]
        if ys.empty:
            continue
        # Eşit ağırlıklı evren getirisi
        pw = ys.pivot_table(index=ys.index, columns='Ticker', values='Close')
        eq = pw.pct_change().mean(axis=1).fillna(0.0)
        tot = float((1 + eq).prod() - 1)
        row = {
            'year': y,
            'universe_return': tot,
            'mean_vix': float(ys['VIX_Close'].mean()) if 'VIX_Close' in ys else np.nan,
            'mean_vol20d': float(ys['Vol_20d'].mean()) if 'Vol_20d' in ys else np.nan,
            'crisis_rate': float(ys['Target'].mean()),
            'n_rows': len(ys),
        }
        regime_rows.append(row)
        print(f"{y:<6}{tot*100:>15.1f}%{row['mean_vix']:>11.1f}"
              f"{row['mean_vol20d']:>14.3f}{row['crisis_rate']*100:>11.1f}%{len(ys):>10,}")
    pd.DataFrame(regime_rows).to_csv(
        os.path.join(args.outdir, 'regime_characterization.csv'), index=False)

    # ── Loader cache (ablation'a göre fund_dim değişir) ──
    loader_cache = {}

    def get_loaders(tg, fg):
        key = (tuple(tg), tuple(fg))
        if key not in loader_cache:
            loader_cache[key] = get_dual_stream_dataloaders(
                dataset_out, seq_len=BASE_CONFIG['seq_len'],
                batch_size=args.batch_size, tech_groups=tg, fund_groups=fg)
        return loader_cache[key]

    cls_rows, bt_rows = [], []

    for abl in args.ablations:
        tech_groups, fund_groups, modality, desc = ABLATIONS[abl]
        _, val_loader, test_loader, _, (tech_cols, fund_cols) = get_loaders(
            tech_groups, fund_groups)
        test_ds = test_loader.dataset

        print("\n" + "▄" * 78)
        print(f"ABLATION: {abl}  ({desc})")
        print(f"  tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | modality={modality}")
        print("▄" * 78)

        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f'best_v2_{abl}_seed{seed}.pth')
            if not os.path.exists(ckpt):
                print(f"  ⚠️ checkpoint yok, atlanıyor: {ckpt}")
                continue

            model = DualEncoderTransformer(
                tech_dim=len(tech_cols), fund_dim=len(fund_cols),
                seq_len=BASE_CONFIG['seq_len'], d_model=BASE_CONFIG['d_model'],
                n_heads=BASE_CONFIG['n_heads'], n_layers=BASE_CONFIG['n_layers'],
                dropout=BASE_CONFIG['dropout'], modality=modality,
                fusion_type=BASE_CONFIG['fusion_type'])
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device,
                                                 weights_only=True))
            except Exception as e:
                print(f"  ⚠️ {ckpt} yüklenemedi: {e}")
                continue
            model.to(device)

            # Eşik SADECE validation'dan — üç yıla da aynısı uygulanacak
            v_pred, v_true, _ = _evaluate_on_loader(model, val_loader, device)
            thr, _ = find_best_threshold_mcc(v_true, v_pred)

            t_pred, t_true, _ = _evaluate_on_loader(model, test_loader, device)
            pred_df = pd.DataFrame({
                'date': test_ds.dates.values,
                'ticker': test_ds.tickers,
                'prob': t_pred,
                'target': t_true,
            })

            for y in YEARS:
                ys = pred_df[pd.DatetimeIndex(pred_df['date']).year == y]
                if len(ys) < 100:
                    continue
                m = classification_metrics(ys['target'].values, ys['prob'].values, thr)
                m.update({'ablation': abl, 'seed': seed, 'year': y, 'threshold': thr})
                cls_rows.append(m)

            if not args.no_backtest:
                for r in yearly_backtest(pred_df[['date', 'ticker', 'prob']],
                                         dataset_out, thr, args.cost_bps):
                    r.update({'ablation': abl, 'seed': seed})
                    bt_rows.append(r)

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f"  ✓ seed={seed} işlendi (eşik={thr:.4f})")

    if not cls_rows:
        print("\n⚠️ Hiç checkpoint bulunamadı. run_modality_v2.py çalıştırıldı mı?")
        return

    cls = pd.DataFrame(cls_rows)
    cls.to_csv(os.path.join(args.outdir, 'regime_classification_raw.csv'), index=False)

    # ══ Özet: yıl × ablation ══
    print("\n" + "═" * 78)
    print("SINIFLANDIRMA — yıl bazında (ortalama ± std, seed'ler üzerinden)")
    print("═" * 78)

    for metric in ['mcc', 'pr_auc', 'roc_auc']:
        print(f"\n── TEST {metric.upper()} ──")
        piv_m = cls.pivot_table(index='ablation', columns='year', values=metric, aggfunc='mean')
        piv_s = cls.pivot_table(index='ablation', columns='year', values=metric, aggfunc='std')
        header = f"{'Ablation':<18}" + "".join(f"{y:>18}" for y in piv_m.columns)
        print(header)
        print("-" * len(header))
        for abl in piv_m.index:
            line = f"{abl:<18}"
            for y in piv_m.columns:
                mu = piv_m.loc[abl, y]
                sd = piv_s.loc[abl, y] if not pd.isna(piv_s.loc[abl, y]) else 0.0
                line += f"{mu:>11.4f}±{sd:<6.4f}"
            print(line)

    # Taban oran ve flag rate — rejim farkını gösteren tanısal bilgi
    print(f"\n── TABAN ORAN vs MODELİN 'RİSKLİ' DEDİĞİ ORAN ──")
    br = cls.groupby('year')['base_rate'].mean()
    fr = cls.pivot_table(index='ablation', columns='year', values='flag_rate', aggfunc='mean')
    print(f"{'':<18}" + "".join(f"{y:>12}" for y in br.index))
    print(f"{'gerçek kriz oranı':<18}" + "".join(f"{br[y]*100:>11.1f}%" for y in br.index))
    for abl in fr.index:
        print(f"{abl:<18}" + "".join(f"{fr.loc[abl, y]*100:>11.1f}%" for y in fr.columns))

    summ = cls.groupby(['ablation', 'year']).agg(
        mcc_mean=('mcc', 'mean'), mcc_std=('mcc', 'std'),
        pr_auc_mean=('pr_auc', 'mean'), roc_auc_mean=('roc_auc', 'mean'),
        base_rate=('base_rate', 'mean'), flag_rate=('flag_rate', 'mean'),
        n_seeds=('seed', 'nunique')).reset_index()
    summ.to_csv(os.path.join(args.outdir, 'regime_classification_summary.csv'), index=False)

    # ══ Ekonomik değer ══
    if bt_rows:
        bt = pd.DataFrame(bt_rows)
        bt.to_csv(os.path.join(args.outdir, 'regime_backtest_raw.csv'), index=False)
        print("\n" + "═" * 78)
        print(f"EKONOMİK DEĞER — yıl bazında (cost={args.cost_bps}bps)")
        print("═" * 78)
        g = bt.groupby(['ablation', 'year']).agg(
            strat_sharpe=('strat_sharpe', 'mean'), bench_sharpe=('bench_sharpe', 'mean'),
            strat_return=('strat_return', 'mean'), bench_return=('bench_return', 'mean'),
            strat_maxdd=('strat_maxdd', 'mean'), bench_maxdd=('bench_maxdd', 'mean'),
        ).reset_index()
        g['sharpe_diff'] = g['strat_sharpe'] - g['bench_sharpe']
        g['maxdd_improvement'] = g['strat_maxdd'] - g['bench_maxdd']  # pozitif = daha az düşüş

        print(f"{'Ablation':<18}{'Yıl':>6}{'Sharpe(str)':>13}{'Sharpe(bmk)':>13}"
              f"{'fark':>9}{'MaxDD(str)':>12}{'MaxDD(bmk)':>12}")
        print("-" * 83)
        for _, r in g.iterrows():
            print(f"{r['ablation']:<18}{int(r['year']):>6}{r['strat_sharpe']:>13.3f}"
                  f"{r['bench_sharpe']:>13.3f}{r['sharpe_diff']:>+9.3f}"
                  f"{r['strat_maxdd']*100:>11.1f}%{r['bench_maxdd']*100:>11.1f}%")
        g.to_csv(os.path.join(args.outdir, 'regime_backtest_summary.csv'), index=False)

    # ══ Yorum rehberi ══
    print("\n" + "═" * 78)
    print("YORUM REHBERİ")
    print("═" * 78)
    print("  Tezin merkezi iddiası: modalitelerin bilgi içeriği rejime bağlıdır.")
    print()
    print("  Şunlara bakın:")
    print("   • fund_pure_only'nin MCC'si 2022'de diğer yıllardan belirgin yüksekse")
    print("     → bilanço sinyali yalnızca borçlanma-maliyeti şokunda çalışıyor. İDDİA DOĞRULANDI.")
    print("   • multi_pure tüm yıllarda tech'e yakınsa")
    print("     → füzyon istikrar katıyor ama seviye katmıyor.")
    print("   • macro_only'nin flag_rate'i yıllar arasında çok oynuyorsa")
    print("     → makro akış 'piyasa zamanlaması' yapıyor, kesitsel ayrım değil.")
    print("   • Taban oran yıllar arasında çok farklıysa (ör. 2022 >> 2024)")
    print("     → sabit eşik karşılaştırmasında bu farkın büyümesini bekleyin.")
    print()
    print(f"  Çıktılar → {args.outdir}/regime_*.csv")


if __name__ == '__main__':
    main()