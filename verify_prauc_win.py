"""
verify_prauc_win.py
───────────────────
Auditor item #4: "You claim a PR-AUC win (0.2000 vs 0.1942), but you never
measured XGBoost's own seed variance. If the distributions overlap, the win
evaporates."

This script measures the per-seed PR-AUC distribution for BOTH the gated
ensemble members and XGBoost, then runs a paired test on the same seeds.

NO RETRAINING of deep models — uses results/preds/*.npz.
XGBoost IS retrained per seed (fast, ~130 trees each) and aligned to the
sequence test rows so both models are scored on identical observations.

USAGE:
    python verify_prauc_win.py
    python verify_prauc_win.py --seeds 42 43 44 45 46
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.preprocessing import RobustScaler

from datasets.feature_engineering import prepare_dataset, get_feature_groups

VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
SEQ_LEN = 20

XGB_PARAMS = dict(
    max_depth=2, learning_rate=0.05533894705931091, n_estimators=1900,
    subsample=0.612290511361587, colsample_bytree=0.5038470300093549,
    gamma=3.69714205665728, min_child_weight=1,
    reg_alpha=1.4479885716146466, reg_lambda=0.003496116877652556,
    scale_pos_weight=1.6171278021543518,
)


def sequence_keys(df, start, end, seq_len=20):
    """Rebuild the (ticker, date) keys the sequence dataset uses, in order."""
    sub = df[(df.index >= start) & (df.index <= end)]
    keys = []
    for tk, g in sub.groupby('Ticker'):
        g = g.sort_index()
        for i in range(seq_len - 1, len(g)):
            keys.append((tk, g.index[i]))
    return keys


def sequence_labels(df, start, end, seq_len=20):
    sub = df[(df.index >= start) & (df.index <= end)]
    y = []
    for tk, g in sub.groupby('Ticker'):
        g = g.sort_index()
        y.extend(g['Target'].values[seq_len - 1:])
    return np.array(y, dtype=int)


def train_xgb_per_seed(df, val_keys, test_keys, seeds):
    """Return dict seed -> (val_scores, test_scores) aligned to sequence keys."""
    import xgboost as xgb
    tech, fund = get_feature_groups(df)
    cols = tech + fund

    tr = df[(df.index >= '2010-01-01') & (df.index <= '2019-12-31')]
    va = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    te = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    med = tr[cols].median()
    sc = RobustScaler()
    Xtr = sc.fit_transform(tr[cols].fillna(med).values)
    Xva = sc.transform(va[cols].fillna(med).values)
    Xte = sc.transform(te[cols].fillna(med).values)
    ytr, yva = tr['Target'].values, va['Target'].values

    va_idx = {k: i for i, k in enumerate(zip(va['Ticker'].values, va.index))}
    te_idx = {k: i for i, k in enumerate(zip(te['Ticker'].values, te.index))}
    va_sel = [va_idx[k] for k in val_keys]
    te_sel = [te_idx[k] for k in test_keys]

    out = {}
    for s in seeds:
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=s,
                              eval_metric='logloss', early_stopping_rounds=50,
                              n_jobs=-1, tree_method='hist')
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        v = m.predict_proba(Xva)[:, 1][va_sel]
        t = m.predict_proba(Xte)[:, 1][te_sel]
        out[s] = (v, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--preddir', default='results/preds')
    args = ap.parse_args()

    print("═" * 72)
    print("PR-AUC WIN VERIFICATION — does it survive seed variance?")
    print("═" * 72)

    df = prepare_dataset(force_refresh=False)
    val_keys = sequence_keys(df, VAL_START, VAL_END)
    test_keys = sequence_keys(df, TEST_START, TEST_END)
    y_test = sequence_labels(df, TEST_START, TEST_END)
    print(f"  test observations: {len(test_keys):,}\n")

    # ── Deep model per-seed test PR-AUC (from cache) ──
    def deep_prauc(prefix):
        vals = []
        for s in args.seeds:
            p = os.path.join(args.preddir, f'{prefix}_seed{s}.npz')
            if os.path.exists(p):
                t = np.load(p)['test']
                vals.append((s, average_precision_score(y_test, t)))
        return dict(vals)

    gated = deep_prauc('MS_gated_cross_attention')
    xattn = deep_prauc('MS_cross_attention')

    # ── XGBoost per-seed (retrained, aligned) ──
    print("  Training XGBoost per seed (aligned to sequence rows)...")
    xgb_preds = train_xgb_per_seed(df, val_keys, test_keys, args.seeds)
    xgb = {s: average_precision_score(y_test, t) for s, (v, t) in xgb_preds.items()}

    # ── Per-seed table ──
    print("\n" + "═" * 72)
    print("PER-SEED TEST PR-AUC")
    print("═" * 72)
    print(f"{'seed':>6} {'Gated':>10} {'Cross-attn':>12} {'XGBoost':>10}")
    print("-" * 42)
    for s in args.seeds:
        print(f"{s:>6} {gated.get(s, float('nan')):>10.4f} "
              f"{xattn.get(s, float('nan')):>12.4f} {xgb.get(s, float('nan')):>10.4f}")

    def stats(d):
        v = np.array(list(d.values()))
        return v.mean(), v.std(ddof=1), v.min(), v.max()

    print("-" * 42)
    for name, d in [('Gated', gated), ('Cross-attn', xattn), ('XGBoost', xgb)]:
        m, sd, lo, hi = stats(d)
        print(f"{name:>16}: {m:.4f} ± {sd:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ── Ensemble PR-AUC (the number you actually reported) ──
    def deep_ens(prefix):
        ts = [np.load(os.path.join(args.preddir, f'{prefix}_seed{s}.npz'))['test']
              for s in args.seeds
              if os.path.exists(os.path.join(args.preddir, f'{prefix}_seed{s}.npz'))]
        return average_precision_score(y_test, np.mean(ts, axis=0))
    xgb_ens_t = np.mean([xgb_preds[s][1] for s in args.seeds], axis=0)

    print("\n" + "═" * 72)
    print("ENSEMBLE PR-AUC (what you reported)")
    print("═" * 72)
    print(f"  Gated ensemble  : {deep_ens('MS_gated_cross_attention'):.4f}")
    print(f"  XGBoost ensemble: {average_precision_score(y_test, xgb_ens_t):.4f}")

    # ── Paired test on the same seeds ──
    from scipy import stats as sps
    common = [s for s in args.seeds if s in gated and s in xgb]
    if len(common) >= 2:
        d = np.array([gated[s] - xgb[s] for s in common])
        t, p_two = sps.ttest_rel([gated[s] for s in common],
                                 [xgb[s] for s in common])
        p_one = p_two / 2 if d.mean() > 0 else 1 - p_two / 2
        print("\n" + "═" * 72)
        print("PAIRED TEST — Gated vs XGBoost (same seeds, per-seed PR-AUC)")
        print("═" * 72)
        print(f"  mean diff (gated - xgb): {d.mean():+.4f}")
        print(f"  per-seed wins for gated: {int((d > 0).sum())}/{len(d)}")
        print(f"  paired t p (one-sided) : {p_one:.4f}")
        ci = sps.t.interval(0.95, len(d)-1, loc=d.mean(),
                            scale=d.std(ddof=1)/np.sqrt(len(d)))
        print(f"  95% CI of difference   : [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print()
        if ci[0] > 0:
            print("  VERDICT: CI excludes 0 → win is real at single-model level.")
        elif d.mean() > 0:
            print("  VERDICT: gated leads on average but CI includes 0 →")
            print("           the per-seed PR-AUC win is NOT statistically robust.")
            print("           (The ensemble may still edge it, but the honest")
            print("            framing is 'comparable', not 'beats'.)")
        else:
            print("  VERDICT: XGBoost leads on per-seed PR-AUC. The ensemble")
            print("           win was a seed-averaging artifact.")

    pd.DataFrame({'seed': args.seeds,
                  'gated': [gated.get(s) for s in args.seeds],
                  'cross_attn': [xattn.get(s) for s in args.seeds],
                  'xgboost': [xgb.get(s) for s in args.seeds]}
                 ).to_csv('results/prauc_by_seed.csv', index=False)
    print(f"\n  → results/prauc_by_seed.csv")
    print("═" * 72)


if __name__ == '__main__':
    main()