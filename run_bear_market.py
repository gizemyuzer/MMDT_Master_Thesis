"""
run_bear_market.py
───────────────────
AYI PİYASASI DENEYİ — danışman toplantısı aksiyon maddesi.

═══════════════════════════════════════════════════════════════════════
İSTENEN
═══════════════════════════════════════════════════════════════════════
"Train the model on data leading up to 2022 and test its performance during
 that bear market period. Evaluate if the fusion model yields better results
 compared to the buy and hold strategy under these specific conditions."

═══════════════════════════════════════════════════════════════════════
NEDEN BU DENEY ANA KOŞUDAN FARKLI
═══════════════════════════════════════════════════════════════════════
Mevcut kurulum:  train 2010-2019 | val 2020-2021 | test 2022-2024
Bu deney:        train 2010-2020 | val 2021      | test 2022

İki maddi fark var:

(1) COVID EĞİTİME GİRİYOR. Şu anki modelin eğitim setinde tek bir büyük
    kriz yok — 2010-2019 kesintisiz bir boğa dönemi. 2020'yi eğitime almak,
    modele ilk kez "kriz nasıl görünür" örneği veriyor. Bu tek başına bir
    hipotez: kriz görmemiş bir model krizi tahmin edebilir mi?

(2) TEST TEK REJİM. 2022-2024 ortalaması, modelin 2022'deki davranışını
    iki boğa yılıyla seyreltiyordu. Mevcut sonuçlarda model YALNIZCA 2022'de
    buy & hold'u geçiyor (-%8,3 vs -%12,6). Bu deney o koşulu izole ediyor.

═══════════════════════════════════════════════════════════════════════
DÜRÜST SINIRLAR — tez metnine bunlar da yazılmalı
═══════════════════════════════════════════════════════════════════════
· 2022 TEK BİR AYI PİYASASI. n=1. Kesitte 50 binden fazla gözlem var ama
  rejim düzeyinde tek gözlem. "Model ayı piyasalarında çalışır" DENEMEZ;
  "bu ayı piyasasında çalıştı" denir.
· EŞİK KALİBRASYONU RİSKLİ. Validation 2021, sakin bir boğa yılı. Oradan
  seçilen eşik 2022'nin oynaklığına iyi oturmayabilir. Bu bilinçli bir
  tercih: kronolojik dürüstlüğü korumak için test'e bakılmıyor.
· SEÇİM YANLILIĞI RİSKİ. 2022'yi test olarak seçmemizin nedeni, mevcut
  sonuçlarda modelin orada iyi görünmesi. Bu bir keşif sonrası testtir,
  ön-kayıtlı bir hipotez değil. Metinde böyle raporlanmalı.

KULLANIM:
    python run_bear_market.py                       # 3 hücre × 5 seed
    python run_bear_market.py --cells I I+II        # alt küme
    python run_bear_market.py --no-train            # eğitilmiş checkpoint'lerle sadece portföy
    python run_bear_market.py --exclude-pct 20      # ana koşuyla eşleştirin!

Çıktılar: results/bear_*.csv
"""
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import train_pytorch_model, find_best_threshold_mcc
from models.losses import FocalLoss
from run_portfolio_simulation import simulate, performance

# ── Bölünme: ana koşudan FARKLI, bilinçli olarak ──
TRAIN_START, TRAIN_END = '2010-01-01', '2020-12-31'
VAL_START,   VAL_END   = '2021-01-01', '2021-12-31'
TEST_START,  TEST_END  = '2022-01-01', '2022-12-31'

# ABLATIONS ile aynı akış atama kuralı (a priori sabit)
CELLS = {
    'I':        (('tech',), ('fund',),          'tech_only',  'Sadece teknik'),
    'I+II':     (('tech',), ('fund',),          'multimodal', 'Teknik + firma (füzyon)'),
    'I+II+III': (('tech',), ('fund', 'macro'),  'multimodal', 'Teknik + firma + makro'),
    'I+II+IV': (('tech',), ('fund', 'text'), 'multimodal', 'Teknik + firma + metin'),
}
CONFIG = dict(seq_len=20, d_model=64, n_heads=4, n_layers=2, dropout=0.15,
              fusion_type='cross_attention')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def safe(name):
    return name.replace('+', '_').lower()


def focal_alpha(loader, max_batches=200):
    pos = neg = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        pos += int((b['label'] == 1).sum()); neg += int((b['label'] == 0).sum())
    return 1.0 if pos == 0 else float(neg / pos)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        logits = model(b['tech_seq'].to(device), b['fund_seq'].to(device))
        out.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=list(CELLS), choices=list(CELLS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--exclude-pct', type=float, default=20.0,
                    help='ANA KOŞUYLA AYNI OLMALI — yoksa karşılaştırma geçersiz')
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, nargs='+', default=[10, 25, 50])
    ap.add_argument('--no-train', action='store_true')
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()

    print("═" * 78)
    print("AYI PİYASASI DENEYİ — train ≤2020 | val 2021 | test 2022")
    print("═" * 78)
    print(f"  Hücreler   : {args.cells}")
    print(f"  Seed'ler   : {args.seeds}")
    print(f"  Dışlama    : en riskli %{args.exclude_pct:.0f}")
    print(f"  Device     : {device}")
    print()
    print("  ⚠️ --exclude-pct ana portföy koşusuyla AYNI olmalı, yoksa")
    print("     'ayı piyasasında daha iyi' iddiası karşılaştırılamaz olur.")
    print()

    dataset_out = prepare_dataset(force_refresh=False)

    # ── Eğitim ──
    rows = []
    raw_path = os.path.join(args.outdir, 'bear_classification.csv')
    if os.path.exists(raw_path):
        rows = pd.read_csv(raw_path).to_dict('records')
    done = {(r['cell'], int(r['seed'])) for r in rows}

    loaders = {}
    for cell in args.cells:
        tg, fg, modality, desc = CELLS[cell]
        key = (tg, fg)
        if key not in loaders:
            tl, vl, testl, _, (tc, fc) = get_dual_stream_dataloaders(
                dataset_out, seq_len=CONFIG['seq_len'], batch_size=args.batch_size,
                train_start=TRAIN_START, train_end=TRAIN_END,
                val_start=VAL_START, val_end=VAL_END,
                test_start=TEST_START, test_end=TEST_END,
                tech_groups=tg, fund_groups=fg)
            loaders[key] = (tl, vl, testl, tc, fc, focal_alpha(tl))
        tl, vl, testl, tc, fc, alpha = loaders[key]

        for seed in args.seeds:
            name = f"BEAR_{safe(cell)}_seed{seed}"
            ckpt = os.path.join('checkpoints', f'best_{name.lower()}.pth')
            if args.no_train or (cell, seed) in done:
                print(f"  [{cell} seed={seed}] atlandı")
                continue

            print("\n" + "▄" * 78)
            print(f"{cell} | seed={seed} | {desc}")
            print(f"  tech_dim={len(tc)} fund_dim={len(fc)} modality={modality}")
            print("▄" * 78)

            set_seed(seed)
            model = DualEncoderTransformer(
                tech_dim=len(tc), fund_dim=len(fc), modality=modality, **CONFIG)
            t0 = time.time()
            res = train_pytorch_model(
                model=model, train_loader=tl, val_loader=vl, test_loader=testl,
                model_name=name, epochs=args.epochs, device=device,
                criterion=FocalLoss(alpha=alpha, gamma=2.0), monitor='pr_auc')
            row = {'cell': cell, 'seed': seed, 'threshold': res.get('threshold'),
                   'minutes': round((time.time() - t0) / 60, 1)}
            for split in ('val', 'test'):
                for k, v in (res.get(f'{split}_metrics') or {}).items():
                    row[f'{split}_{k}'] = v
            rows.append(row)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"  ✓ 2022 test MCC = {row.get('test_mcc', float('nan')):.4f}")

    if rows:
        C = pd.DataFrame(rows)
        print("\n" + "═" * 78)
        print("SINIFLANDIRMA — 2022 test seti")
        print("═" * 78)
        g = C.groupby('cell').agg(n=('seed', 'count'), val_mcc=('val_mcc', 'mean'),
                                  test_mcc=('test_mcc', 'mean'),
                                  sd=('test_mcc', 'std'),
                                  pr_auc=('test_pr_auc', 'mean'))
        print(g.round(4).to_string())
        print("\n  Referans — ana koşuda (train ≤2019) 2022 alt dönemi: MCC ≈ 0.1160")
        print("  Bu koşu daha yüksekse, COVID'i eğitime almak işe yaramış demektir.")

    # ══════════════════════════════════════════════════════════════
    # PORTFÖY — 2022, buy & hold karşılaştırması
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("PORTFÖY — 2022")
    print("═" * 78)

    sub = dataset_out[(dataset_out.index >= TEST_START) & (dataset_out.index <= TEST_END)]
    price = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
    ret_wide = price.pct_change().fillna(0.0)
    days = ret_wide.index
    rebal = set(days[::args.rebalance])

    scores = {}
    for cell in args.cells:
        tg, fg, modality, _ = CELLS[cell]
        tl, vl, testl, tc, fc, _ = loaders[(tg, fg)]
        ds = testl.dataset
        for seed in args.seeds:
            name = f"bear_{safe(cell)}_seed{seed}"
            ckpt = os.path.join('checkpoints', f'best_{name}.pth')
            if not os.path.exists(ckpt):
                print(f"  ⚠️ checkpoint yok: {ckpt}")
                continue
            m = DualEncoderTransformer(tech_dim=len(tc), fund_dim=len(fc),
                                       modality=modality, **CONFIG)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.to(device)
            prob = predict(m, testl, device)
            w = pd.DataFrame({'date': ds.dates.values, 'ticker': ds.tickers, 'p': prob})
            scores[f'{cell}_seed{seed}'] = (w.pivot_table(index='date', columns='ticker', values='p')
                                            .reindex(index=days).reindex(columns=ret_wide.columns))
            print(f"  ✓ {cell} seed={seed} tahminleri hazır")

    if 'Vol_20d' in sub.columns:
        scores['naive_vol'] = (sub.pivot_table(index=sub.index, columns='Ticker', values='Vol_20d')
                               .reindex(index=days).reindex(columns=ret_wide.columns))
    scores['oracle'] = (sub.pivot_table(index=sub.index, columns='Ticker', values='Target')
                        .reindex(index=days).reindex(columns=ret_wide.columns))

    out = []
    for cost in args.cost_bps:
        # buy & hold: hiçbir şey dışlanmaz
        flat = pd.DataFrame(0.0, index=days, columns=ret_wide.columns)
        net, tov, exp_ = simulate(flat, ret_wide, rebal, 0.0, cost)
        out.append({**performance(net, 'buy_hold'), 'cost_bps': cost,
                    'avg_turnover': tov, 'exposure': exp_})
        for label, sw in scores.items():
            net, tov, exp_ = simulate(sw, ret_wide, rebal, args.exclude_pct, cost)
            out.append({**performance(net, label), 'cost_bps': cost,
                        'avg_turnover': tov, 'exposure': exp_})

    P = pd.DataFrame(out)
    P['exclude_pct'] = args.exclude_pct
    P['rebalance_days'] = args.rebalance
    P['period'] = '2022'
    P.to_csv(os.path.join(args.outdir, 'bear_portfolio.csv'), index=False)

    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)
    print(f"\n{'strateji':<14}{'getiri':>10}{'maxDD':>10}{'Calmar':>9}{'Sortino':>9}")
    print("-" * 52)
    base = P[P.cost_bps == args.cost_bps[0]].groupby('grp').agg(
        r=('total_return', 'mean'), d=('max_drawdown', 'mean'),
        c=('calmar', 'mean'), s=('sortino', 'mean'))
    for k in base.sort_values('c', ascending=False).index:
        b = base.loc[k]
        print(f"{k:<14}{b['r']*100:>9.1f}%{b['d']*100:>9.1f}%{b['c']:>9.3f}{b['s']:>9.3f}")

    if 'buy_hold' in base.index:
        bh = base.loc['buy_hold']
        print("\n" + "─" * 52)
        print("DANIŞMANIN SORUSU: füzyon modeli ayı piyasasında buy & hold'u geçti mi?")
        for k in base.index:
            if k in ('buy_hold', 'oracle', 'naive_vol'):
                continue
            b = base.loc[k]
            verdict = "GEÇTİ" if b['c'] > bh['c'] else "geçemedi"
            print(f"  {k:<12} Calmar {b['c']:.3f} vs buy&hold {bh['c']:.3f} → {verdict}")
        print("\n  ⚠️ 2022 TEK bir ayı piyasasıdır (rejim düzeyinde n=1). Sonuç ne olursa")
        print("     olsun 'ayı piyasalarında çalışır' değil, 'bu ayı piyasasında çalıştı'")
        print("     biçiminde raporlanmalıdır.")

    print(f"\nÇıktılar → {args.outdir}/bear_classification.csv, bear_portfolio.csv")


if __name__ == '__main__':
    main()