"""
run_pct_sweep.py
─────────────────
DIŞLAMA YÜZDESİ TARAMASI — yeniden eğitim gerektirmez.

Dışlama bütçesi %20 sabit seçilmişti; taban kriz oranı ise %13. Aradaki
fark, her dönem riskli olmayan ~%7'nin gereksiz dışlanması demek — boğa
piyasasında doğrudan getiri kaybı.

Bu script yüzdeyi VALIDATION döneminde seçer ve seçilen değeri test'e
uygular. Test'e bakarak seçmek aşırı uyum olurdu; validation'da seçmek
mevcut eşik protokolünüzle aynı mantık.

KULLANIM:
    python run_pct_sweep.py --ablation multi_text
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from run_modality_v2 import ABLATIONS, BASE_CONFIG, fusion_for
from run_portfolio_simulation import simulate, performance

VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        out.append(torch.sigmoid(
            model(b['tech_seq'].to(device), b['fund_seq'].to(device))
        ).float().cpu().numpy().ravel())
    return np.concatenate(out)


def run_window(ds, scores_by_seed, a, b, pcts, rebal, cost):
    sub = ds[(ds.index >= a) & (ds.index <= b)]
    px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret = px.pct_change().fillna(0.0)
    days = ret.index
    rb = set(days[::rebal])

    rows = []
    flat = pd.DataFrame(0.0, index=days, columns=ret.columns)
    net, _, _ = simulate(flat, ret, rb, 0.0, cost)
    bh = performance(net, 'buy_hold')

    for pct in pcts:
        cal, cagr, dd = [], [], []
        for seed, w in scores_by_seed.items():
            sw = w.reindex(index=days).reindex(columns=ret.columns)
            net, _, _ = simulate(sw, ret, rb, pct, cost)
            p = performance(net, f'p{pct}')
            cal.append(p['calmar']); cagr.append(p['cagr']); dd.append(p['max_drawdown'])
        rows.append({'exclude_pct': pct, 'calmar': np.mean(cal),
                     'calmar_sd': np.std(cal, ddof=1) if len(cal) > 1 else 0.0,
                     'cagr': np.mean(cagr), 'max_drawdown': np.mean(dd),
                     'bh_calmar': bh['calmar'], 'bh_cagr': bh['cagr']})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', default='multi_text', choices=list(ABLATIONS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--pcts', type=float, nargs='+',
                    default=[5, 8, 10, 13, 15, 20, 25, 30])
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    device = get_device()
    ds = prepare_dataset(force_refresh=False)
    tg, fg, modality, _ = ABLATIONS[args.ablation]

    print("═" * 74)
    print(f"DIŞLAMA YÜZDESİ TARAMASI — {args.ablation}")
    print("═" * 74)
    print(f"  Taban kriz oranı: %{100*ds['Target'].mean():.1f}")
    print(f"  Denenen yüzdeler: {args.pcts}\n")

    # Val ve test için ayrı loader — dizi ısınması her pencerede kendi içinde
    out = {}
    for split, (a, b) in [('val', (VAL_START, VAL_END)),
                          ('test', (TEST_START, TEST_END))]:
        _, vl, tl, _, (tc, fc) = get_dual_stream_dataloaders(
            ds, seq_len=BASE_CONFIG['seq_len'], batch_size=256,
            tech_groups=tg, fund_groups=fg)
        loader = vl if split == 'val' else tl
        dset = loader.dataset
        sc = {}
        for seed in args.seeds:
            ck = os.path.join('checkpoints',
                              f'best_v2_{args.ablation}_seed{seed}.pth'.lower())
            if not os.path.exists(ck):
                print(f"  ⚠️ checkpoint yok: {ck}")
                continue
            m = DualEncoderTransformer(tech_dim=len(tc), fund_dim=len(fc),
                                       modality=modality,
                                       fusion_type=fusion_for(args.ablation),
                                       **{k: v for k, v in BASE_CONFIG.items()
                                          if k != 'fusion_type'})
            m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            m.to(device)
            p = predict(m, loader, device)
            sc[seed] = pd.DataFrame({'date': dset.dates.values,
                                     'ticker': dset.tickers, 'p': p}) \
                         .pivot_table(index='date', columns='ticker', values='p')
        if not sc:
            raise SystemExit("Hiç checkpoint bulunamadı.")
        out[split] = run_window(ds, sc, a, b, args.pcts,
                                args.rebalance, args.cost_bps)

    v, t = out['val'], out['test']
    best = v.loc[v['calmar'].idxmax(), 'exclude_pct']

    print(f"{'%':>5}{'VAL Calmar':>13}{'TEST Calmar':>14}{'TEST CAGR':>12}{'TEST maxDD':>13}")
    print("-" * 57)
    for _, r in t.iterrows():
        vr = v[v.exclude_pct == r.exclude_pct].iloc[0]
        mark = '  ← val seçimi' if r.exclude_pct == best else ''
        print(f"{r.exclude_pct:>5.0f}{vr.calmar:>13.3f}{r.calmar:>14.3f}"
              f"{100*r.cagr:>11.1f}%{100*r.max_drawdown:>12.1f}%{mark}")

    tb = t[t.exclude_pct == best].iloc[0]
    print("\n" + "─" * 57)
    print(f"  Validation'da seçilen: %{best:.0f}")
    print(f"  Test Calmar          : {tb.calmar:.3f}")
    print(f"  Mevcut (%20)         : {t[t.exclude_pct==20].iloc[0].calmar:.3f}")
    print(f"  Buy & hold           : {tb.bh_calmar:.3f}")
    print("\n  NOT: yüzde VALIDATION'da seçildi. Test satırları yalnızca")
    print("  raporlama içindir; en iyi test satırını seçmek aşırı uyum olurdu.")

    v['split'] = 'val'; t['split'] = 'test'
    pd.concat([v, t]).to_csv(os.path.join(args.outdir, 'pct_sweep.csv'), index=False)
    print(f"\n→ {args.outdir}/pct_sweep.csv")


if __name__ == '__main__':
    main()