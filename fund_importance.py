"""
fund_importance.py
──────────────────
Auditor items #2 & #4: "Fund-only is dead (MCC 0.02). How do you know the
fusion gain comes from fundamental information rather than just architecture?"

Turnstile test WITHOUT retraining. For each trained gated checkpoint we:
  1. score the real test set                       -> baseline
  2. shuffle the fundamental stream across samples  -> fund carries no signal
  3. shuffle the technical stream across samples    -> tech carries no signal
and measure how much PR-AUC / MCC drops.

INTERPRETATION:
  - If shuffling FUND collapses performance -> the model USES fundamentals.
    (Fusion claim survives; the fund-only=0.02 result just means macro alone
     can't rank single stocks, but interacts usefully with technicals.)
  - If shuffling FUND barely changes anything -> cross-attention has learned
    to IGNORE the fundamental stream. Fusion gain is architectural, not
    informational. Then Path B (strengthen fundamentals) or drop the claim.

The tech shuffle is the sanity check: it SHOULD collapse (tech is the main
signal). If it doesn't, something is wrong with the harness.

NO RETRAINING — loads checkpoints/best_ms_gated_cross_attention_seed*.pth.

USAGE:
    python fund_importance.py
    python fund_importance.py --seeds 42 43 44 45 46 --repeats 5
"""
import os
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef

from datasets.feature_engineering import prepare_dataset, get_dual_stream_dataloaders
from models.transformer_model import DualEncoderTransformer

LADDER_CONFIG = dict(seq_len=20, d_model=64, n_heads=4, n_layers=2,
                     dropout=0.15, modality='multimodal',
                     fusion_type='gated_cross_attention')


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def score(model, tech, fund, y, device, batch=512):
    """Return probabilities for given tech/fund arrays (numpy)."""
    model.eval()
    out = []
    n = len(y)
    for i in range(0, n, batch):
        tb = torch.tensor(tech[i:i+batch]).to(device)
        fb = torch.tensor(fund[i:i+batch]).to(device)
        logit = model(x_tech=tb, x_fund=fb)
        out.append(torch.sigmoid(logit).cpu().numpy().ravel())
    return np.concatenate(out)


def best_threshold_mcc(y, s, n=200):
    lo, hi = np.percentile(s, [0.5, 99.5])
    if hi <= lo:
        lo, hi = s.min(), s.max()
    best_m, best_t = -2.0, float(np.median(s))
    for t in np.linspace(lo, hi, n):
        p = (s >= t).astype(int)
        if p.sum() in (0, len(p)):
            continue
        m = matthews_corrcoef(y, p)
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t


def evaluate(y, scores, thr):
    return dict(pr_auc=average_precision_score(y, scores),
                mcc=matthews_corrcoef(y, (scores >= thr).astype(int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--repeats', type=int, default=5,
                    help='Number of shuffle repetitions (averaged)')
    args = ap.parse_args()

    device = get_device()
    print("═" * 72)
    print("FUNDAMENTAL IMPORTANCE — permutation test (no retraining)")
    print("═" * 72)

    df = prepare_dataset(force_refresh=False)
    _, val_loader, test_loader, _, (tech_cols, fund_cols) = \
        get_dual_stream_dataloaders(df, seq_len=20, batch_size=512)

    tech = test_loader.dataset.tech_sequences   # (N, T, tech_dim)
    fund = test_loader.dataset.fund_sequences   # (N, T, fund_dim)
    y = test_loader.dataset.labels.astype(int)

    v_tech = val_loader.dataset.tech_sequences
    v_fund = val_loader.dataset.fund_sequences
    y_val = val_loader.dataset.labels.astype(int)

    print(f"  test: {len(y):,} sequences | tech_dim={tech.shape[2]} | "
          f"fund_dim={fund.shape[2]}")
    print(f"  fund_cols: {fund_cols}\n")

    rng = np.random.default_rng(0)
    rows = []

    for seed in args.seeds:
        ckpt = os.path.join('checkpoints',
                            f'best_ms_gated_cross_attention_seed{seed}.pth')
        if not os.path.exists(ckpt):
            print(f"  ⚠️  missing: {ckpt}")
            continue

        model = DualEncoderTransformer(tech_dim=len(tech_cols),
                                       fund_dim=len(fund_cols), **LADDER_CONFIG)
        model.load_state_dict(torch.load(ckpt, weights_only=True,
                                         map_location='cpu'))
        model.to(device)

        # threshold from validation (real, unshuffled)
        v_scores = score(model, v_tech, v_fund, y_val, device)
        thr = best_threshold_mcc(y_val, v_scores)

        # baseline (real test)
        base_s = score(model, tech, fund, y, device)
        base = evaluate(y, base_s, thr)

        # shuffle fundamental stream across samples (repeat & average)
        fund_pr, fund_mcc = [], []
        tech_pr, tech_mcc = [], []
        for r in range(args.repeats):
            perm = rng.permutation(len(y))
            m = evaluate(y, score(model, tech, fund[perm], y, device), thr)
            fund_pr.append(m['pr_auc']); fund_mcc.append(m['mcc'])

            perm2 = rng.permutation(len(y))
            m2 = evaluate(y, score(model, tech[perm2], fund, y, device), thr)
            tech_pr.append(m2['pr_auc']); tech_mcc.append(m2['mcc'])

        rows.append(dict(
            seed=seed,
            base_pr=base['pr_auc'], base_mcc=base['mcc'],
            fund_shuf_pr=np.mean(fund_pr), fund_shuf_mcc=np.mean(fund_mcc),
            tech_shuf_pr=np.mean(tech_pr), tech_shuf_mcc=np.mean(tech_mcc),
        ))
        print(f"  ✓ seed {seed}: baseline PR-AUC={base['pr_auc']:.4f} MCC={base['mcc']:.4f}")

        del model
        if device.type == 'mps':
            torch.mps.empty_cache()

    if not rows:
        print("\n  No checkpoints found.")
        return

    import pandas as pd
    res = pd.DataFrame(rows)

    def col(name):
        return res[name].mean(), res[name].std(ddof=1)

    print("\n" + "═" * 72)
    print("RESULT (mean over seeds)")
    print("═" * 72)
    bp, bps = col('base_pr'); bm, bms = col('base_mcc')
    fp, fps = col('fund_shuf_pr'); fm, fms = col('fund_shuf_mcc')
    tp, tps = col('tech_shuf_pr'); tm, tms = col('tech_shuf_mcc')

    print(f"\n{'Condition':<28} {'PR-AUC':>16} {'MCC':>16}")
    print("-" * 62)
    print(f"{'Baseline (real)':<28} {bp:>9.4f}±{bps:.4f} {bm:>9.4f}±{bms:.4f}")
    print(f"{'Fundamental shuffled':<28} {fp:>9.4f}±{fps:.4f} {fm:>9.4f}±{fms:.4f}")
    print(f"{'Technical shuffled':<28} {tp:>9.4f}±{tps:.4f} {tm:>9.4f}±{tms:.4f}")

    print(f"\n{'Drop when shuffled':<28} {'ΔPR-AUC':>16} {'ΔMCC':>16}")
    print("-" * 62)
    print(f"{'Fundamental':<28} {bp-fp:>+16.4f} {bm-fm:>+16.4f}")
    print(f"{'Technical':<28} {bp-tp:>+16.4f} {bm-tm:>+16.4f}")

    # ── Verdict ──
    print("\n" + "═" * 72)
    print("VERDICT")
    print("═" * 72)
    fund_drop = bp - fp
    tech_drop = bp - tp
    rel = fund_drop / tech_drop if tech_drop > 1e-6 else 0

    if tech_drop < 0.01:
        print("  ⚠️  Technical shuffle barely moved PR-AUC — harness suspect.")
        print("      (Expected: tech shuffle should collapse performance.)")
    else:
        print(f"  Technical shuffle drop : {tech_drop:+.4f} PR-AUC (sanity: large ✓)")
        print(f"  Fundamental shuffle drop: {fund_drop:+.4f} PR-AUC")
        print(f"  Fund drop as % of tech drop: {100*rel:.0f}%\n")

        if fund_drop < 0.003:
            print("  → The model essentially IGNORES the fundamental stream.")
            print("    Fusion gain is architectural, not informational.")
            print("    PATH B is warranted: strengthen fundamentals (more accounting")
            print("    ratios) OR reframe the claim away from 'fundamentals help'.")
        elif rel < 0.15:
            print("  → Fundamentals contribute WEAKLY relative to technicals.")
            print("    Defensible as 'macro context modulates technical signal',")
            print("    but a stronger fundamental set (Path B) would help the claim.")
        else:
            print("  → Fundamentals contribute MEANINGFULLY. The fusion claim")
            print("    survives: shuffling fundamental information measurably hurts")
            print("    the model, so it is using that information, not just the")
            print("    extra architecture. fund-only=0.02 reflects that macro alone")
            print("    can't rank single stocks, not that fundamentals are useless.")

    res.to_csv('results/fund_importance.csv', index=False)
    print(f"\n  → results/fund_importance.csv")
    print("═" * 72)


if __name__ == '__main__':
    main()