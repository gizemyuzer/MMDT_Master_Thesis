"""
run_walkforward_bear.py
────────────────────────
ÇOK KATLI AYI PİYASASI TESTİ — rejim düzeyinde n=1'den n=4'e.

═══════════════════════════════════════════════════════════════════════
NEDEN
═══════════════════════════════════════════════════════════════════════
run_bear_market.py 2022'de modelin buy & hold'u geçtiğini gösterdi
(-%6.80 vs -%12.78). Ama bu TEK bir ayı piyasası; "tesadüf olabilir"
itirazı haklı olur. Bu script aynı testi dört stres döneminde tekrarlar.

═══════════════════════════════════════════════════════════════════════
TAKVİM YILI DEĞİL, GERÇEK STRES PENCERESİ
═══════════════════════════════════════════════════════════════════════
2020 tüm yıl +%16 kapattı; ayı piyasası yalnızca Şubat-Nisan arasıydı.
2018 de -%6 kapattı ama düşüş Q4'te yoğunlaştı. O yüzden fiili düşüş
pencereleri kullanılıyor, takvim yılları değil.

═══════════════════════════════════════════════════════════════════════
KRİTİK: 20 GÜNLÜK ISINMA PAYI
═══════════════════════════════════════════════════════════════════════
Dizi veri seti test diliminden 20 günlük geçmiş kurar, dolayısıyla test
penceresinin İLK 20 GÜNÜNDE model tahmin üretemez. simulate() skor
yokken ağırlıkları sıfırda tutar → portföy o günlerde BOŞ kalır. Buy &
hold ise ilk günden yatırımdadır.

İlk sürümde bu asimetri F2020'de şu absürt sonucu verdi:
    I+II+IV  +11.85%   buy_hold  -25.14%   oracle  -22.98%
Model, COVID çöküşünün en sert kısmında piyasada değildi; yalnızca
toparlanmaya katıldı. Oracle'ın modelden 35 puan geride kalması bunun
işaretiydi — mükemmel öngörü tanımı gereği üst sınırdır.

İki önlem alındı:
  (1) Test pencereleri bir ay öne çekildi → model strese hazır girer.
  (2) Portföy penceresi, modelin skor ürettiği ilk güne kırpılır ve TÜM
      stratejiler aynı aralıkta değerlendirilir.
Ayrıca oracle bir stratejinin gerisinde kalırsa özet blokta UYARI basılır.

KULLANIM:
    python run_walkforward_bear.py
    python run_walkforward_bear.py --cells I+II+IV        # yarı yük
    python run_walkforward_bear.py --folds F2018 F2020
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
from models.pytorch_trainer import train_pytorch_model
from models.losses import FocalLoss
from run_portfolio_simulation import simulate, performance

# ── Katlar: (train_end, val_start, val_end, test_start, test_end, açıklama) ──
# Test pencereleri, asıl stres döneminden ~1 ay ÖNCE başlar. Bu, dizi
# ısınmasının stres kısmını yemesini önler (bkz. modül başlığı).
FOLDS = {
    'F2015': ('2013-12-31', '2014-01-01', '2015-06-30',
              '2015-07-01', '2016-02-15', 'Çin devalüasyonu + petrol çöküşü'),
    'F2018': ('2016-12-31', '2017-01-01', '2018-08-31',
              '2018-09-01', '2018-12-31', 'Fed sıkılaştırması, Q4 satışı'),
    'F2020': ('2018-12-31', '2019-01-01', '2020-01-15',
              '2020-01-16', '2020-04-15', 'COVID çöküşü'),
    'F2022': ('2020-12-31', '2021-01-01', '2021-12-31',
              '2022-01-01', '2022-12-31', 'Faiz şoku'),
}

CELLS = {
    'I':       (('tech',), ('fund',),         'tech_only',  'Sadece teknik'),
    'I+II':    (('tech',), ('fund',),         'multimodal', 'Teknik + firma'),
    'I+II+IV': (('tech',), ('fund', 'text'),  'multimodal', 'Teknik + firma + metin'),
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
    return torch.device('cpu')


def safe(x):
    return x.replace('+', '_').lower()


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
        out.append(torch.sigmoid(
            model(b['tech_seq'].to(device), b['fund_seq'].to(device))
        ).float().cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', default=['F2015', 'F2018', 'F2020'],
                    choices=list(FOLDS))
    ap.add_argument('--cells', nargs='+', default=['I+II', 'I+II+IV'],
                    choices=list(CELLS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--exclude-pct', type=float, default=20.0)
    ap.add_argument('--rebalance', type=int, default=20)
    ap.add_argument('--cost-bps', type=float, default=10.0)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    cls_path = os.path.join(args.outdir, 'walkforward_classification.csv')
    pf_path = os.path.join(args.outdir, 'walkforward_portfolio.csv')

    print("═" * 78)
    print("ÇOK KATLI AYI PİYASASI TESTİ")
    print("═" * 78)
    for f in args.folds:
        tr_e, v_s, v_e, t_s, t_e, d = FOLDS[f]
        print(f"  {f}  eğitim ≤{tr_e} | val {v_s}→{v_e} | test {t_s}→{t_e}")
        print(f"        {d}")
    print(f"  Hücreler: {args.cells} | Seed: {args.seeds} | Device: {device}")
    print(f"  Dışlama: %{args.exclude_pct:.0f} | rebalance: {args.rebalance} gün\n")

    ds = prepare_dataset(force_refresh=False)

    cls_rows = []
    if os.path.exists(cls_path):
        cls_rows = pd.read_csv(cls_path).to_dict('records')
    done = {(r['fold'], r['cell'], int(r['seed'])) for r in cls_rows}

    pf_rows = []
    if os.path.exists(pf_path):
        pf_rows = pd.read_csv(pf_path).to_dict('records')
    # Yeniden koşulan katların eski portföy satırlarını düşür
    pf_rows = [r for r in pf_rows if r.get('fold') not in args.folds]

    for fold in args.folds:
        tr_end, v_s, v_e, t_s, t_e, fdesc = FOLDS[fold]

        sub = ds[(ds.index >= t_s) & (ds.index <= t_e)]
        px = sub.pivot_table(index=sub.index, columns='Ticker', values='Close')
        ret = px.pct_change().fillna(0.0)
        days = ret.index
        bh_full = float((1 + ret.mean(axis=1)).prod() - 1)

        print("\n" + "▄" * 78)
        print(f"KAT {fold} — {fdesc}")
        print(f"  Test penceresi: {t_s} → {t_e}  ({len(days)} işlem günü)")
        print(f"  Evren getirisi (eşit ağırlık): {bh_full*100:+.1f}%")
        if bh_full >= 0:
            print("  ⚠️ Bu pencerede evren POZİTİF getirdi — ayı piyasası sayılmaz.")
            print("     Sonuç raporlanır ama 'stres testi' diye sunulmamalı.")
        print("▄" * 78)

        loaders = {}
        for cell in args.cells:
            tg, fg, modality, cdesc = CELLS[cell]
            key = (tg, fg)
            if key not in loaders:
                tl, vl, testl, _, (tc, fc) = get_dual_stream_dataloaders(
                    ds, seq_len=CONFIG['seq_len'], batch_size=args.batch_size,
                    train_start='2010-01-01', train_end=tr_end,
                    val_start=v_s, val_end=v_e,
                    test_start=t_s, test_end=t_e,
                    tech_groups=tg, fund_groups=fg)
                loaders[key] = (tl, vl, testl, tc, fc, focal_alpha(tl))
            tl, vl, testl, tc, fc, alpha = loaders[key]

            for seed in args.seeds:
                if (fold, cell, seed) in done:
                    print(f"  [{fold}/{cell}/seed{seed}] atlandı (tamamlanmış)")
                    continue
                name = f"WF_{fold}_{safe(cell)}_seed{seed}"
                print(f"\n  ── {fold} | {cell} | seed {seed} | "
                      f"tech={len(tc)} fund={len(fc)} ──")
                set_seed(seed)
                model = DualEncoderTransformer(
                    tech_dim=len(tc), fund_dim=len(fc), modality=modality, **CONFIG)
                t0 = time.time()
                res = train_pytorch_model(
                    model=model, train_loader=tl, val_loader=vl, test_loader=testl,
                    model_name=name, epochs=args.epochs, device=device,
                    criterion=FocalLoss(alpha=alpha, gamma=2.0), monitor='pr_auc')
                r = {'fold': fold, 'cell': cell, 'seed': seed,
                     'train_end': tr_end, 'test_start': t_s, 'test_end': t_e,
                     'bh_return': bh_full, 'threshold': res.get('threshold'),
                     'minutes': round((time.time() - t0) / 60, 1)}
                for sp in ('val', 'test'):
                    for k, v in (res.get(f'{sp}_metrics') or {}).items():
                        r[f'{sp}_{k}'] = v
                cls_rows.append(r)
                pd.DataFrame(cls_rows).to_csv(cls_path, index=False)
                print(f"    ✓ test MCC = {r.get('test_mcc', float('nan')):.4f}")

        # ══════════════════════════════════════════════════════════
        # PORTFÖY
        # ══════════════════════════════════════════════════════════
        scores = {}
        for cell in args.cells:
            tg, fg, modality, _ = CELLS[cell]
            tl, vl, testl, tc, fc, _ = loaders[(tg, fg)]
            dset = testl.dataset
            for seed in args.seeds:
                ck = os.path.join('checkpoints',
                                  f'best_wf_{fold.lower()}_{safe(cell)}_seed{seed}.pth')
                if not os.path.exists(ck):
                    print(f"  ⚠️ checkpoint yok, portföyde atlanıyor: {ck}")
                    continue
                m = DualEncoderTransformer(tech_dim=len(tc), fund_dim=len(fc),
                                           modality=modality, **CONFIG)
                m.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
                m.to(device)
                pr = predict(m, testl, device)
                w = pd.DataFrame({'date': dset.dates.values,
                                  'ticker': dset.tickers, 'p': pr})
                scores[f'{cell}_seed{seed}'] = (
                    w.pivot_table(index='date', columns='ticker', values='p')
                     .reindex(index=days).reindex(columns=ret.columns))

        if not scores:
            print("  Portföy atlandı: hiç model skoru yok.")
            continue

        if 'Vol_20d' in sub.columns:
            scores['naive_vol'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                                   values='Vol_20d')
                                   .reindex(index=days).reindex(columns=ret.columns))
        scores['oracle'] = (sub.pivot_table(index=sub.index, columns='Ticker',
                                            values='Target')
                            .reindex(index=days).reindex(columns=ret.columns))

        # ── ADALET KIRPMASI ──
        # Model ilk 20 günde dizi kuramaz; buy & hold ilk günden yatırımdadır.
        # Bu asimetri kısa pencerelerde modeli haksız avantajlı gösterir:
        # çöküş sırasında portföy boş kalır, yalnızca toparlanmaya katılır.
        # Tüm stratejiler modelin skor ürettiği ilk günden itibaren değerlendirilir.
        model_keys = [k for k in scores if k not in ('naive_vol', 'oracle')]
        valid = scores[model_keys[0]].notna().any(axis=1)
        if valid.any():
            first = valid.idxmax()
            n_before = len(days)
            days = days[days >= first]
            ret = ret.loc[days]
            scores = {k: v.reindex(index=days) for k, v in scores.items()}
            bh_win = float((1 + ret.mean(axis=1)).prod() - 1)
            print(f"\n  [Kırpma] Portföy penceresi {first.date()} → {t_e} "
                  f"({n_before} → {len(days)} gün, ısınma çıkarıldı)")
            print(f"           Kırpılmış pencerede evren getirisi: {bh_win*100:+.1f}%")
        else:
            bh_win = bh_full
        rebal = set(days[::args.rebalance])

        flat = pd.DataFrame(0.0, index=days, columns=ret.columns)
        net, tov, exp_ = simulate(flat, ret, rebal, 0.0, args.cost_bps)
        pf_rows.append({**performance(net, 'buy_hold'), 'fold': fold,
                        'window': f'{days[0].date()}→{days[-1].date()}',
                        'exclude_pct': args.exclude_pct, 'bh_window_return': bh_win,
                        'exposure': exp_, 'turnover': tov})
        for lab, sw in scores.items():
            net, tov, exp_ = simulate(sw, ret, rebal, args.exclude_pct, args.cost_bps)
            pf_rows.append({**performance(net, lab), 'fold': fold,
                            'window': f'{days[0].date()}→{days[-1].date()}',
                            'exclude_pct': args.exclude_pct, 'bh_window_return': bh_win,
                            'exposure': exp_, 'turnover': tov})
        pd.DataFrame(pf_rows).to_csv(pf_path, index=False)

    # ══════════════════════════════════════════════════════════════
    # ÖZET
    # ══════════════════════════════════════════════════════════════
    if not pf_rows:
        print("\nPortföy sonucu yok.")
        return
    P = pd.DataFrame(pf_rows)
    P['grp'] = P['strategy'].str.replace(r'_seed\d+', '', regex=True)

    print("\n" + "═" * 78)
    print("KAT BAZINDA PORTFÖY")
    print("═" * 78)
    warn = []
    for fold in [f for f in args.folds if f in set(P['fold'])]:
        f = P[P.fold == fold]
        g = f.groupby('grp')[['total_return', 'max_drawdown', 'exposure']].mean()
        bh = g.loc['buy_hold'] if 'buy_hold' in g.index else None
        print(f"\n── {fold}  ({f['window'].iloc[0]}) ──")
        print(f"  {'strateji':<16}{'getiri':>10}{'maksDD':>10}{'yatırımda':>11}"
              f"{'B&H farkı':>12}")
        for k in g.sort_values('total_return', ascending=False).index:
            diff = ((g.loc[k, 'total_return'] - bh['total_return']) * 100
                    if bh is not None else np.nan)
            print(f"  {k:<16}{g.loc[k,'total_return']*100:>9.1f}%"
                  f"{g.loc[k,'max_drawdown']*100:>9.1f}%"
                  f"{g.loc[k,'exposure']*100:>10.1f}%{diff:>+11.1f} puan")

        # ── SAĞLIK KONTROLÜ: oracle üst sınırdır ──
        models = [k for k in g.index if k not in ('buy_hold', 'naive_vol', 'oracle')]
        if 'oracle' in g.index and models:
            best = max(g.loc[k, 'total_return'] for k in models)
            if g.loc['oracle', 'total_return'] < best:
                warn.append(fold)
                print(f"\n  ⚠️ ORACLE MODELDEN GERİDE "
                      f"({g.loc['oracle','total_return']*100:+.1f}% vs {best*100:+.1f}%)")
                print(f"     Mükemmel öngörü tanımı gereği ÜST SINIRDIR — bu bir hata")
                print(f"     işaretidir, bu katın portföy sonucunu KULLANMAYIN.")
                print(f"     Olası sebep: pencere hâlâ çok kısa ya da kriz oranı ~%100")
                print(f"     olduğu için oracle sıralaması dejenere.")

    print("\n" + "═" * 78)
    print("DANIŞMANIN SORUSU: kaç stres döneminde buy & hold geçildi?")
    print("═" * 78)
    valid_folds = [f for f in P['fold'].unique() if f not in warn]
    for cell in args.cells:
        wins = tot = 0
        for fold in valid_folds:
            g = P[P.fold == fold].groupby('grp')['total_return'].mean()
            if cell in g.index and 'buy_hold' in g.index:
                tot += 1
                wins += int(g[cell] > g['buy_hold'])
        if tot:
            print(f"  {cell:<12} {wins}/{tot} dönemde buy & hold'u geçti")
    if warn:
        print(f"\n  ⚠️ Şu katlar sayıma DAHİL EDİLMEDİ (oracle kontrolü başarısız): {warn}")
    print("\n  NOT: katlar farklı uzunlukta ve farklı eğitim penceresi kullanıyor;")
    print("  erken katlarda eğitim verisi daha az. Bu metinde belirtilmelidir.")
    print(f"\nÇıktılar → {cls_path}")
    print(f"          {pf_path}")


if __name__ == '__main__':
    main()