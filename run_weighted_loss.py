"""
run_weighted_loss.py
─────────────────────
EĞİTİM TASARIMI DENEYİ — sınıf dengeleme ve amaç hizalaması.

═══════════════════════════════════════════════════════════════════════
ÜÇ MOD
═══════════════════════════════════════════════════════════════════════
  sampler   Mevcut boru hattının davranışı: WeightedRandomSampler ile sınıf
            dengeleme + FocalLoss alpha. get_dual_stream_dataloaders bunu
            uyguluyor; burada AYNI eğitim döngüsü içinde yeniden üretilir.

  uniform   Sampler YOK, sadece FocalLoss alpha. Aşırı örnekleme kaldırılmış.

  weighted  Sampler yok + maliyet duyarlı örnek ağırlıkları:
              Target=1 → gerçekleşen düşüşün büyüklüğü (kaçırmanın maliyeti)
              Target=0 → kaçırılan yükseliş (yanlış alarmın maliyeti)

═══════════════════════════════════════════════════════════════════════
NEDEN ÜÇÜ AYNI DÖNGÜDE
═══════════════════════════════════════════════════════════════════════
İlk koşuda 'uniform' modu, standart boru hattına göre MCC'yi neredeyse
değiştirmeden (0.1435 vs 0.1419) portföy Calmar'ını 0.455'ten 0.979'a
çıkardı — buy & hold'u (0.945) ilk kez geçerek.

Tek fark sampler'dı. Ama bu iki AYRI eğitim döngüsünün karşılaştırmasıydı;
fark döngü farklılığından da gelebilirdi. 'sampler' modu bu belirsizliği
kaldırır: aynı kod, aynı seed, tek değişken.

Hipotez: FocalLoss zaten alpha = neg/pos ≈ 6.5 uyguluyor. Sampler da sınıf
dengeliyor. İkisi birlikte pozitifleri doğal frekansının ~40 katına çıkarıyor.
MCC bunu telafi edebiliyor (eşik val'da optimize ediliyor) ama portföy
SIRALAMA kullanıyor ve sıralamanın tepesi bozuluyor.

═══════════════════════════════════════════════════════════════════════
GEÇERLİLİK
═══════════════════════════════════════════════════════════════════════
· Ağırlıklar YALNIZCA eğitimde. Val/test değerlendirmesi ağırlıksız ve
  mevcut protokolle birebir aynı: eşik sadece validation'dan.
· İleri getiri bir ÖZELLİK değil, eğitim zamanı ağırlığı. Etiketin kendisi
  de ileri veriden türetiliyor; aynı meşruiyet.
· Ağırlıkların ortalaması 1'e normalize — aksi halde kayıp ölçeği değişir
  ve aynı öğrenme oranı modlar arasında farklı davranır.

KULLANIM:
    python run_weighted_loss.py --mode sampler --seeds 42 43 44 45 46
    python run_weighted_loss.py --mode uniform
    python run_weighted_loss.py --mode all        # üçü sırayla
"""
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, precision_score, recall_score)

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.losses import FocalLoss
from models.pytorch_trainer import find_best_threshold_mcc

CONFIG = dict(seq_len=20, d_model=64, n_heads=4, n_layers=2, dropout=0.15,
              fusion_type='cross_attention')
TECH_G, FUND_G = ('tech',), ('fund', 'text')     # I+II+IV — en iyi hücre
HORIZON = 20


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WeightedDataset(Dataset):
    """Mevcut dataset'i sarar, örnek başına ağırlık ekler. Orijinali bozmaz."""

    def __init__(self, base, weights):
        assert len(base) == len(weights), f"{len(base)} vs {len(weights)}"
        self.base, self.w = base, weights.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item['weight'] = torch.tensor(self.w[i])
        return item


def build_weights(ds, dataset, cap_q, mode):
    """
    Maliyet duyarlı, SINIFA GÖRE ASİMETRİK ağırlık:

      Target=1 → gerçekleşen ileri maks düşüşün büyüklüğü
      Target=0 → kaçırılan yükseliş  max(0, ileri getiri)

    Simetrik |getiri| kullanmak yanlış olurdu: modelin asıl hatası yükselen
    isimleri yanlışlıkla işaretlemek, ve simetrik ağırlık bunu ayırt etmez.
    """
    if mode in ('uniform', 'sampler'):
        return np.ones(len(dataset), dtype=np.float32)

    px = ds.pivot_table(index=ds.index, columns='Ticker', values='Close')
    fwd = px.shift(-HORIZON) / px - 1.0
    roll_min = px[::-1].rolling(HORIZON, min_periods=1).min()[::-1]
    dd = (1.0 - roll_min / px).clip(lower=0.0)
    up = fwd.clip(lower=0.0)

    ddl = dd.stack().rename('v').reset_index()
    upl = up.stack().rename('v').reset_index()
    for f in (ddl, upl):
        f.columns = ['date', 'Ticker', 'v']
    dd_lut = ddl.set_index(['date', 'Ticker'])['v']
    up_lut = upl.set_index(['date', 'Ticker'])['v']

    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dataset.dates.values), np.asarray(dataset.tickers)])
    y = dataset.labels.astype(int)
    w = np.where(y == 1,
                 dd_lut.reindex(idx).values,
                 up_lut.reindex(idx).values).astype(float)

    med = np.nanmedian(w)
    w = np.where(np.isnan(w), med, w)
    w = np.clip(w, 0.0, np.quantile(w, cap_q))
    return (w / w.mean()).astype(np.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    p, y = [], []
    for b in loader:
        lg = model(b['tech_seq'].to(device), b['fund_seq'].to(device))
        p.append(torch.sigmoid(lg).float().cpu().numpy().ravel())
        y.append(b['label'].numpy().ravel())
    return np.concatenate(p), np.concatenate(y)


def train_one(model, tl, vl, device, alpha, epochs, lr, patience, tag):
    """reduction='none' → örnek başına kayıp → ağırlıkla çarp → ortalama."""
    crit = FocalLoss(alpha=alpha, gamma=2.0, reduction='none')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                       factor=0.5, patience=3)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in tl:
            xt = b['tech_seq'].to(device)
            xf = b['fund_seq'].to(device)
            y = b['label'].to(device).float().unsqueeze(1)
            w = b['weight'].to(device).float().unsqueeze(1)
            opt.zero_grad()
            loss = (crit(model(xt, xf), y) * w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)

        pv, yv = predict(model, vl, device)
        pr = average_precision_score(yv.astype(int), pv)
        sched.step(pr)
        flag = ''
        if pr > best:
            best, bad = pr, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            flag = '  ← en iyi'
        else:
            bad += 1
        print(f"    [{tag}] epoch {ep+1:>2}  loss={tot/max(n,1):.5f}  "
              f"val PR-AUC={pr:.4f}{flag}")
        if bad >= patience:
            print(f"    [{tag}] erken durdurma (epoch {ep+1})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def metrics(y, p, thr):
    y = y.astype(int); yp = (p >= thr).astype(int)
    return {'mcc': matthews_corrcoef(y, yp),
            'pr_auc': average_precision_score(y, p),
            'roc_auc': roc_auc_score(y, p),
            'precision': precision_score(y, yp, zero_division=0),
            'recall': recall_score(y, yp, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sampler', 'uniform', 'weighted', 'all'],
                    default='all')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--cap', type=float, default=0.99)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    device = get_device()
    print("═" * 78)
    print("EĞİTİM TASARIMI DENEYİ — I+II+IV hücresi")
    print("═" * 78)
    print(f"  Mod: {args.mode} | Seed: {args.seeds} | Device: {device}\n")

    ds = prepare_dataset(force_refresh=False)
    tl0, vl, testl, _, (tc, fc) = get_dual_stream_dataloaders(
        ds, seq_len=CONFIG['seq_len'], batch_size=args.batch_size,
        tech_groups=TECH_G, fund_groups=FUND_G)
    print(f"  tech_dim={len(tc)} | fund_dim={len(fc)}")

    pos = int((tl0.dataset.labels == 1).sum())
    neg = int((tl0.dataset.labels == 0).sum())
    alpha = neg / max(pos, 1)
    print(f"  FocalLoss alpha={alpha:.2f}")

    # ── Resume ── önceki koşular korunur, tamamlananlar atlanır
    raw_path = os.path.join(args.outdir, 'weighted_loss_raw.csv')
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    if os.path.exists(raw_path):
        rows = pd.read_csv(raw_path).to_dict('records')
        print(f"  [Resume] {len(rows)} önceki koşu okundu")
    done = {(r['mode'], int(r['seed'])) for r in rows}

    modes = ['sampler', 'uniform', 'weighted'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        w = build_weights(ds, tl0.dataset, args.cap, mode)

        print(f"\n{'▄'*78}\nMOD: {mode}")
        if mode == 'weighted':
            print(f"  ağırlık ort={w.mean():.3f} medyan={np.median(w):.3f} "
                  f"maks={w.max():.3f}")
        print("▄" * 78)

        if mode == 'sampler':
            # Mevcut boru hattının davranışı — get_dual_stream_dataloaders
            # ile birebir aynı sampler kurulumu.
            lab = tl0.dataset.labels.astype(int)
            cc = np.bincount(lab)
            sw_ = (1.0 / (cc + 1e-9))[lab]
            smp = WeightedRandomSampler(sw_, len(sw_), replacement=True)
            wtl = DataLoader(WeightedDataset(tl0.dataset, w),
                             batch_size=args.batch_size, sampler=smp,
                             drop_last=True, num_workers=0)
            print("  [Sampler] WeightedRandomSampler devrede")
        else:
            wtl = DataLoader(WeightedDataset(tl0.dataset, w),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=0)

        for seed in args.seeds:
            if (mode, seed) in done:
                print(f"  [{mode}/s{seed}] atlandı (tamamlanmış)")
                continue
            tag = f"{mode}/s{seed}"
            print(f"\n  ── {tag} ──")
            set_seed(seed)
            model = DualEncoderTransformer(tech_dim=len(tc), fund_dim=len(fc),
                                           modality='multimodal', **CONFIG).to(device)
            t0 = time.time()
            model = train_one(model, wtl, vl, device, alpha,
                              args.epochs, args.lr, args.patience, tag)

            pv, yv = predict(model, vl, device)
            pt, yt = predict(model, testl, device)
            thr, _ = find_best_threshold_mcc(yv, pv)      # SADECE val'dan
            mv, mt = metrics(yv, pv, thr), metrics(yt, pt, thr)

            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join('checkpoints', f'best_wl_{mode}_seed{seed}.pth'))
            os.makedirs(os.path.join(args.outdir, 'preds'), exist_ok=True)
            np.savez(os.path.join(args.outdir, 'preds', f'WL_{mode}_seed{seed}.npz'),
                     dates=testl.dataset.dates.values,
                     tickers=np.asarray(testl.dataset.tickers),
                     prob=pt, target=yt)

            r = {'mode': mode, 'seed': seed, 'threshold': thr,
                 'minutes': round((time.time() - t0) / 60, 1)}
            r.update({f'val_{k}': v for k, v in mv.items()})
            r.update({f'test_{k}': v for k, v in mt.items()})
            rows.append(r)
            done.add((mode, seed))
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print(f"    ✓ test MCC={mt['mcc']:.4f}  PR-AUC={mt['pr_auc']:.4f}  "
                  f"ROC={mt['roc_auc']:.4f}  ({r['minutes']} dk)")

    if not rows:
        return
    R = pd.DataFrame(rows)
    print("\n" + "═" * 78)
    print("SONUÇ")
    print("═" * 78)
    g = R.groupby('mode').agg(n=('seed', 'count'),
                              val_mcc=('val_mcc', 'mean'),
                              test_mcc=('test_mcc', 'mean'),
                              sd=('test_mcc', 'std'),
                              pr=('test_pr_auc', 'mean'),
                              roc=('test_roc_auc', 'mean'),
                              prec=('test_precision', 'mean'),
                              rec=('test_recall', 'mean'))
    print(g.round(4).to_string())
    print(f"\n  Referans (standart boru hattı, sampler'lı, n=5): "
          f"multi_text test MCC = 0.1419, Calmar(%8) = 0.455")

    if len(set(R['mode'])) >= 2:
        from scipy import stats
        import itertools
        P = R.pivot_table(index='seed', columns='mode', values='test_mcc')
        print("\n  Eşleşmeli farklar (test MCC):")
        for a, b in itertools.combinations(P.columns, 2):
            q = P[[a, b]].dropna()
            if len(q) >= 2:
                d = q[a] - q[b]
                p = stats.ttest_rel(q[a], q[b]).pvalue
                print(f"    {a:<9} − {b:<9} = {d.mean():+.4f} ± {d.std(ddof=1):.4f}"
                      f" | poz {int((d>0).sum())}/{len(d)} | p={p:.3f}")

    print("\n── ASIL TEST PORTFÖYDE ──")
    print("  python run_wl_portfolio.py --exclude-pct 8")
    print("  MCC farkı küçük kalıp Calmar farkı büyükse, sınıf dengeleme")
    print("  sınıflandırma metriğine görünmeden ekonomik değeri düşürüyor demektir.")
    print(f"\n→ {raw_path}")


if __name__ == '__main__':
    main()