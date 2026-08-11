"""
tune_transformer.py
───────────────────
Gated cross-attention için Optuna Bayesian hiperparametre araması.

NEDEN GEREKLİ:
  1. XGBoost 100 denemelik arama gördü, Transformer SIFIR. Asimetrik.
     "Her ikisi de tune edildi" diyebilmek için bu şart.
  2. Mevcut ayarlar modelin ezberlemesine izin veriyor:
        Epoch 1: train 0.713 | val 0.620   ← zirve
        Epoch 9: train 0.850 | val 0.583   ← ezber
     Model 1. epoch'ta zirve yapıp bozuluyor = kapasite fazla / lr yüksek.
  3. XGBoost'un Optuna'sı max_depth=2 seçti — bu problemde "çok basit
     model + erken dur" doğru strateji. Transformer için de küçültme
     yönü doğru olabilir (regularized varyant val PR-AUC 0.2388 ile en iyiydi).

METODOLOJİ:
  - Arama SADECE validation PR-AUC'ye göre. Test setine DOKUNULMAZ.
  - Hız için: eğitim verisinin bir kısmı + kısa epoch + Optuna pruning
  - En iyi config bulunduktan sonra TAM veriyle, çoklu seed ile yeniden eğitim
  - Final değerlendirme mevcut pipeline'la (train_pytorch_model) yapılır

KULLANIM:
    python tune_transformer.py                      # 30 deneme, %40 veri
    python tune_transformer.py --trials 20          # daha hızlı
    python tune_transformer.py --skip-search        # arama atla, kayıtlı en iyiyi eğit

SÜRE TAHMİNİ:
    ~10 dk/deneme (pruning ile ortalama) → 30 deneme ≈ 5 saat
    + final 5-seed eğitim ≈ 1.5 saat
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import average_precision_score

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer
from models.pytorch_trainer import train_pytorch_model
from models.losses import FocalLoss

STUDY_PATH = 'results/optuna_transformer.json'

# Füzyon tipi — main() içinde --fusion-type ile ayarlanır.
# Varsayılan 'cross_attention': run_modality_v2.py'deki multi_pure ile AYNI
# olmalı, aksi halde tune edilen model karşılaştırılan modelden farklı olur.
FUSION_TYPE = 'cross_attention'


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════
# Alt-örneklenmiş eğitim loader'ı (arama hızı için)
# ══════════════════════════════════════════════════════════════════
def make_search_loader(train_ds, frac, batch_size, seed=0):
    """
    Eğitim setinin frac kadarını alır, WeightedRandomSampler ile
    (orijinal pipeline'la aynı mantık) loader kurar.
    """
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(frac * n), replace=False)
    idx.sort()

    labels = train_ds.labels[idx].astype(int)
    class_counts = np.bincount(labels, minlength=2).astype(float)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    return DataLoader(Subset(train_ds, idx), batch_size=batch_size,
                      sampler=sampler, drop_last=True)


def compute_alpha(labels):
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return float(neg / pos) if pos else 1.0


# ══════════════════════════════════════════════════════════════════
# Hafif eğitim döngüsü (arama için — checkpoint yok, çıktı yok)
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def val_pr_auc(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        logit = model(x_tech=b['tech_seq'].to(device),
                      x_fund=b['fund_seq'].to(device))
        ps.append(torch.sigmoid(logit).cpu().numpy().ravel())
        ys.append(b['label'].numpy().ravel())
    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0
    return float(average_precision_score(y, p))


def train_short(model, train_loader, val_loader, device, criterion,
                lr, weight_decay, epochs, patience, trial=None):
    """Kısa eğitim + Optuna pruning. En iyi val PR-AUC'yi döndürür."""
    import optuna
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, bad = 0.0, 0

    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            logit = model(x_tech=b['tech_seq'].to(device),
                          x_fund=b['fund_seq'].to(device))
            loss = criterion(logit, b['label'].to(device).float().unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        score = val_pr_auc(model, val_loader, device)
        if score > best:
            best, bad = score, 0
        else:
            bad += 1

        if trial is not None:
            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if bad >= patience:
            break

    return best


# ══════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════
def build_objective(train_ds, val_loader, tech_dim, fund_dim, alpha,
                    device, args):
    def objective(trial):
        # ── Arama alanı ───────────────────────────────────────────
        # d_model = n_heads × head_dim  → bölünebilirlik GARANTİ, ve
        # arama alanı sabit (dinamik categorical TPE'yi bozar).
        # Üretilen d_model: 16, 24, 32, 48, 64, 96, 128
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        head_dim = trial.suggest_categorical('head_dim', [8, 12, 16])
        d_model = n_heads * head_dim

        n_layers = trial.suggest_int('n_layers', 1, 3)
        ffn_mult = trial.suggest_categorical('ffn_mult', [2, 4])
        dropout = trial.suggest_float('dropout', 0.10, 0.45)
        lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
        wd = trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True)
        gamma = trial.suggest_categorical('focal_gamma', [1.0, 2.0, 3.0])

        trial.set_user_attr('d_model', d_model)

        set_seed(args.search_seed)
        model = DualEncoderTransformer(
            tech_dim=tech_dim, fund_dim=fund_dim, seq_len=20,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ffn_dim=d_model * ffn_mult, dropout=dropout,
            modality='multimodal', fusion_type=FUSION_TYPE,
        ).to(device)

        train_loader = make_search_loader(train_ds, args.subsample,
                                          args.batch_size, seed=args.search_seed)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        try:
            score = train_short(model, train_loader, val_loader, device,
                                criterion, lr, wd,
                                epochs=args.search_epochs,
                                patience=args.search_patience, trial=trial)
        finally:
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

        return score

    return objective


# ══════════════════════════════════════════════════════════════════
# Final: en iyi config, tam veri, çoklu seed
# ══════════════════════════════════════════════════════════════════
def final_training(params, train_loader, val_loader, test_loader,
                   tech_dim, fund_dim, alpha, device, seeds):
    d_model = params['n_heads'] * params['head_dim']
    results = []
    for seed in seeds:
        print("\n" + "▄" * 70)
        print(f"FINAL EĞİTİM — seed={seed} (tam veri, tuned config)")
        print("▄" * 70)
        set_seed(seed)

        model = DualEncoderTransformer(
            tech_dim=tech_dim, fund_dim=fund_dim, seq_len=20,
            d_model=d_model, n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            ffn_dim=d_model * params['ffn_mult'],
            dropout=params['dropout'],
            modality='multimodal', fusion_type=FUSION_TYPE,
        )
        criterion = FocalLoss(alpha=alpha, gamma=params['focal_gamma'])

        r = train_pytorch_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader,
            model_name=f"TUNED_gated_seed{seed}",
            epochs=30, device=device, criterion=criterion, monitor='pr_auc',
            lr=params['lr'], weight_decay=params['weight_decay'],
            early_stop_patience=8,
        )
        row = {'seed': seed}
        for split, m in [('val', r['val_metrics']), ('test', r['test_metrics'])]:
            if m:
                row.update({f'{split}_{k}': v for k, v in m.items()})
        results.append(row)

        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════
# Tuned modellerin ensemble'ı (checkpoint'lerden, yeniden eğitim yok)
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def _infer(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        logit = model(x_tech=b['tech_seq'].to(device),
                      x_fund=b['fund_seq'].to(device))
        out.append(torch.sigmoid(logit).cpu().numpy().ravel())
    return np.concatenate(out)


def _best_threshold_mcc(y, scores, n=300):
    from sklearn.metrics import matthews_corrcoef
    lo, hi = np.percentile(scores, [0.5, 99.5])
    if hi <= lo:
        lo, hi = scores.min(), scores.max()
    best_m, best_t = -2.0, float(np.median(scores))
    for t in np.linspace(lo, hi, n):
        p = (scores >= t).astype(int)
        if p.sum() in (0, len(p)):
            continue
        m = matthews_corrcoef(y, p)
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def _metrics(y, scores, thr):
    from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 average_precision_score)
    p = (scores >= thr).astype(int)
    return dict(mcc=matthews_corrcoef(y, p), acc=accuracy_score(y, p),
                f1=f1_score(y, p, zero_division=0),
                precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0),
                pr_auc=average_precision_score(y, scores))


def ensemble_eval(params, seeds, val_loader, test_loader,
                  tech_dim, fund_dim, device):
    """Kayıtlı TUNED checkpoint'lerden ensemble değerlendirmesi."""
    d_model = params['n_heads'] * params['head_dim']
    y_val = val_loader.dataset.labels.astype(int)
    y_test = test_loader.dataset.labels.astype(int)

    vs, ts = [], []
    for seed in seeds:
        ckpt = os.path.join('checkpoints', f'best_tuned_gated_seed{seed}.pth')
        if not os.path.exists(ckpt):
            print(f"  ⚠️  bulunamadı: {ckpt}")
            continue
        model = DualEncoderTransformer(
            tech_dim=tech_dim, fund_dim=fund_dim, seq_len=20,
            d_model=d_model, n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            ffn_dim=d_model * params['ffn_mult'], dropout=params['dropout'],
            modality='multimodal', fusion_type=FUSION_TYPE)
        model.load_state_dict(torch.load(ckpt, weights_only=True,
                                         map_location='cpu'))
        model.to(device)
        vs.append(_infer(model, val_loader, device))
        ts.append(_infer(model, test_loader, device))
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()

    if not vs:
        return None

    os.makedirs('results/preds', exist_ok=True)
    for i, seed in enumerate(seeds[:len(vs)]):
        np.savez_compressed(f'results/preds/TUNED_gated_seed{seed}.npz',
                            val=vs[i], test=ts[i])

    # Tek modeller
    singles = []
    for v, t in zip(vs, ts):
        thr = _best_threshold_mcc(y_val, v)
        singles.append(_metrics(y_test, t, thr))
    single_avg = {k: float(np.mean([s[k] for s in singles])) for k in singles[0]}

    # Ensemble
    v_ens, t_ens = np.mean(vs, axis=0), np.mean(ts, axis=0)
    thr = _best_threshold_mcc(y_val, v_ens)
    ens = _metrics(y_test, t_ens, thr)

    return single_avg, ens, len(vs)


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--subsample', type=float, default=0.40,
                    help='Aramada kullanılacak eğitim verisi oranı')
    ap.add_argument('--search-epochs', type=int, default=8)
    ap.add_argument('--search-patience', type=int, default=3)
    ap.add_argument('--search-seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--final-seeds', type=int, nargs='+',
                    default=[42, 43, 44, 45, 46])
    ap.add_argument('--skip-search', action='store_true',
                    help='Aramayı atla, kayıtlı en iyi config ile final eğitim')
    # ── Özellik grubu seçimi ──
    # ÖNEMLİ: varsayılan artık MAKROSUZ (tech + 6 firma oranı) — yani
    # run_modality_v2.py'deki `multi_pure` ile BİREBİR aynı özellik kümesi.
    # Eskiden argüman verilmediği için get_dual_stream_dataloaders'ın legacy
    # varsayılanı devreye giriyordu (fund = firma + makro + etkileşim, 19 kolon)
    # ve tune edilen model, karşılaştırılan modelden FARKLI bir setle
    # eğitiliyordu — bu, tuning sonucunu kullanılamaz kılar.
    ap.add_argument('--tech-groups', type=str, nargs='+', default=['tech'])
    ap.add_argument('--fund-groups', type=str, nargs='+', default=['fund'],
                    help="ör: --fund-groups fund fund_xs  |  legacy için: fund macro interact")
    ap.add_argument('--fusion-type', type=str, default='cross_attention',
                    help="multi_pure ile aynı olmalı (cross_attention)")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    global FUSION_TYPE
    FUSION_TYPE = args.fusion_type

    os.makedirs('results', exist_ok=True)
    device = get_device()

    print("═" * 78)
    print("TRANSFORMER HİPERPARAMETRE ARAMASI (Optuna)")
    print("═" * 78)
    print(f"  Deneme sayısı  : {args.trials}")
    print(f"  Alt-örneklem   : %{args.subsample*100:.0f} eğitim verisi")
    print(f"  Arama epoch    : {args.search_epochs} (patience={args.search_patience})")
    print(f"  Hedef          : validation PR-AUC  (test'e DOKUNULMAZ)")
    print(f"  Cihaz          : {device}\n")

    print(f"  Özellik grupları: tech={tuple(args.tech_groups)} | "
          f"fund={tuple(args.fund_groups)}")
    print(f"  Füzyon tipi     : {args.fusion_type}\n")

    df = prepare_dataset(force_refresh=False)
    train_loader, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(df, seq_len=20, batch_size=args.batch_size,
                                    tech_groups=tuple(args.tech_groups),
                                    fund_groups=tuple(args.fund_groups))

    train_ds = train_loader.dataset
    alpha = compute_alpha(train_ds.labels)
    print(f"\n  tech_dim={len(tech_cols)} | fund_dim={len(fund_cols)} | "
          f"FocalLoss alpha={alpha:.2f}")
    print(f"  Arama seti: {int(args.subsample*len(train_ds)):,} / "
          f"{len(train_ds):,} sequence\n")

    # ── Arama ──
    if not args.skip_search:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=args.search_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                               n_warmup_steps=2),
        )
        objective = build_objective(train_ds, val_loader, len(tech_cols),
                                    len(fund_cols), alpha, device, args)

        done = {'n': 0}
        def cb(study_, trial_):
            done['n'] += 1
            st = trial_.state.name
            v = f"{trial_.value:.4f}" if trial_.value is not None else "pruned"
            print(f"  [{done['n']:>3}/{args.trials}] {st:<9} val PR-AUC={v:>8} "
                  f"| en iyi={study_.best_value:.4f}")

        print("Arama başlıyor...\n")
        study.optimize(objective, n_trials=args.trials, callbacks=[cb])

        best = study.best_params
        print("\n" + "═" * 78)
        print(f"EN İYİ CONFIG — val PR-AUC = {study.best_value:.4f}")
        print("═" * 78)
        for k, v in best.items():
            print(f"  {k:<16}: {v}")

        with open(STUDY_PATH, 'w') as f:
            json.dump({'best_params': best, 'best_value': study.best_value},
                      f, indent=2)
        print(f"\n  → {STUDY_PATH}")
    else:
        with open(STUDY_PATH) as f:
            best = json.load(f)['best_params']
        print("Arama atlandı, kayıtlı config kullanılıyor:")
        for k, v in best.items():
            print(f"  {k:<16}: {v}")

    # ── Final eğitim ──
    print("\n" + "═" * 78)
    print(f"FINAL EĞİTİM — tam veri, {len(args.final_seeds)} seed")
    print("═" * 78)
    rows = final_training(best, train_loader, val_loader, test_loader,
                          len(tech_cols), len(fund_cols), alpha, device,
                          args.final_seeds)

    import pandas as pd
    res = pd.DataFrame(rows)
    res.to_csv('results/tuned_gated_multiseed.csv', index=False)

    print("\n" + "═" * 78)
    print("SONUÇ — tuned gated cross-attention")
    print("═" * 78)
    for metric in ['test_mcc', 'test_pr_auc', 'test_f1', 'test_acc']:
        if metric in res.columns:
            print(f"  {metric:<14}: {res[metric].mean():.4f} "
                  f"(±{res[metric].std(ddof=1):.4f})")

    # ── Ensemble değerlendirmesi ──
    print("\n" + "═" * 78)
    print("ENSEMBLE DEĞERLENDİRMESİ (tuned checkpoint'lerden)")
    print("═" * 78)
    out = ensemble_eval(best, args.final_seeds, val_loader, test_loader,
                        len(tech_cols), len(fund_cols), device)

    print("\n" + "═" * 78)
    print("KARŞILAŞTIRMA — TEST SETİ (hepsi sabit MCC-optimal eşik)")
    print("═" * 78)
    print(f"{'Model':<32} {'MCC':>9} {'PR-AUC':>9} {'F1':>8} {'ACC':>8}")
    print("─" * 78)
    # Önceki koşulardan bilinen referans değerler
    print(f"{'XGBoost (tek seed)':<32} {0.1282:>9.4f} {0.1893:>9.4f} "
          f"{0.2770:>8.4f} {0.6150:>8.4f}")
    print(f"{'XGBoost (ensemble×5)':<32} {0.1409:>9.4f} {0.1942:>9.4f} "
          f"{0.2845:>8.4f} {0.5977:>8.4f}")
    print(f"{'Gated ESKİ (tek ort.)':<32} {0.1153:>9.4f} {0.1900:>9.4f} "
          f"{0.2680:>8.4f} {0.5932:>8.4f}")
    print(f"{'Gated ESKİ (ensemble×5)':<32} {0.1267:>9.4f} {0.2000:>9.4f} "
          f"{0.2706:>8.4f} {0.7101:>8.4f}")
    print("─" * 78)

    if out:
        single_avg, ens, n = out
        print(f"{'Gated TUNED (tek ort.)':<32} {single_avg['mcc']:>9.4f} "
              f"{single_avg['pr_auc']:>9.4f} {single_avg['f1']:>8.4f} "
              f"{single_avg['acc']:>8.4f}")
        print(f"{'Gated TUNED (ensemble×' + str(n) + ')':<32} {ens['mcc']:>9.4f} "
              f"{ens['pr_auc']:>9.4f} {ens['f1']:>8.4f} {ens['acc']:>8.4f}")

        # XGBoost ensemble'a göre fark
        base = dict(mcc=0.1409, pr_auc=0.1942, f1=0.2845)
        print("\n" + "─" * 78)
        print("XGBoost (ensemble×5)'e göre fark — ACC dahil edilmedi:")
        print("  (%13 pozitif oranında ACC yanıltıcı: hiçbir şeye 'riskli'")
        print("   demeyen model %87 alır. MCC/PR-AUC/F1 anlamlı metrikler.)")
        wins = 0
        for k in ['mcc', 'pr_auc', 'f1']:
            d = ens[k] - base[k]
            mark = "✓" if d > 0 else "✗"
            if d > 0:
                wins += 1
            print(f"    {k.upper():<8}: {d:>+8.4f}  {mark}")
        print(f"\n  → 3 anlamlı metrikten {wins}'inde önde")

    print("\n  → results/tuned_gated_multiseed.csv")
    print("  → results/preds/TUNED_gated_seed*.npz")
    print("═" * 78)


if __name__ == '__main__':
    main()