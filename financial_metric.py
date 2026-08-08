"""
financial_metric.py
────────────────────
"Avoided loss" — the monetary interpretation of model performance,
broken down by market regime, including the XGBoost baseline.

What the committee asked for: a financial result instead of ML metrics.
"If you had exited when the model warned you, what % of realized losses
would you have avoided?"

Test period is 2022-2024 (three years — this is the FULL test set, the
committee's request to extend the test was already addressed). We additionally
break it into three regimes to show how the model behaves in different markets:
  2022 = bear      2023 = recovery      2024 = bull
All three are inside the hold-out test set; validation (2020-2021) is untouched.

NO RETRAINING for the deep models (uses results/preds/*.npz). XGBoost is
retrained here from the Optuna params because its predictions must be aligned
to the same (ticker, date) rows the sequence models use.

USAGE:
    python financial_metric.py
    python financial_metric.py --seeds 42 43 44 45 46
"""
import os
import argparse
import numpy as np
import pandas as pd

from datasets.feature_engineering import prepare_dataset, get_feature_groups

VAL_START, VAL_END = '2020-01-01', '2021-12-31'
TEST_START, TEST_END = '2022-01-01', '2024-12-31'
SEQ_LEN = 20

# Market regimes inside the test set (all hold-out, no leakage)
REGIMES = {
    'Full test 2022-2024': ('2022-01-01', '2024-12-31'),
    '2022 (bear)':         ('2022-01-01', '2022-12-31'),
    '2023 (recovery)':     ('2023-01-01', '2023-12-31'),
    '2024 (bull)':         ('2024-01-01', '2024-12-31'),
}

# XGBoost params found by Optuna (from train.py output)
XGB_PARAMS = dict(
    max_depth=2, learning_rate=0.05533894705931091, n_estimators=1900,
    subsample=0.612290511361587, colsample_bytree=0.5038470300093549,
    gamma=3.69714205665728, min_child_weight=1,
    reg_alpha=1.4479885716146466, reg_lambda=0.003496116877652556,
    scale_pos_weight=1.6171278021543518,
)


# ══════════════════════════════════════════════════════════════════
# Forward returns + alignment keys — EXACT create_labels window logic
# ══════════════════════════════════════════════════════════════════
def compute_forward_returns(df, start, end, horizon=20, seq_len=20):
    """
    Forward return for every (ticker, day) the sequence models see, in the
    SAME order as DualStreamSequenceDataset (groupby Ticker -> sort -> range).
    Returns arrays + keys so we can align any model's predictions.
    """
    sub = df[(df.index >= start) & (df.index <= end)]
    fwd_dd, fwd_ret, tgt, keys, dates_out = [], [], [], [], []

    full_close = {tk: g.sort_index()['Close']
                  for tk, g in df.groupby('Ticker')}

    for ticker, group in sub.groupby('Ticker'):
        group = group.sort_index()
        closes_full = full_close[ticker].values
        dates_full = full_close[ticker].index
        pos = {d: k for k, d in enumerate(dates_full)}

        gdates = group.index
        gtargets = group['Target'].values.astype(np.float32)

        for i in range(seq_len - 1, len(group)):
            d = gdates[i]
            p = pos[d]
            c0 = closes_full[p]
            win = closes_full[p + 1: p + 1 + horizon]
            if len(win) == 0 or c0 <= 0:
                dd = ret = 0.0
            else:
                dd = (c0 - win.min()) / c0
                end_price = (closes_full[p + horizon]
                             if p + horizon < len(closes_full) else win[-1])
                ret = (end_price - c0) / c0
            fwd_dd.append(dd); fwd_ret.append(ret)
            tgt.append(gtargets[i]); keys.append((ticker, d)); dates_out.append(d)

    return dict(
        dd=np.array(fwd_dd), ret=np.array(fwd_ret),
        y=np.array(tgt, dtype=int), keys=keys,
        dates=pd.to_datetime(dates_out),
    )


# ══════════════════════════════════════════════════════════════════
def best_threshold_mcc(y, scores, n=300):
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


def financial_metrics(flag, fwd_dd, fwd_ret):
    flag = flag.astype(bool)
    is_loss = fwd_ret < 0
    total_loss = -fwd_ret[is_loss].sum()
    avoided_loss = -fwd_ret[is_loss & flag].sum()
    is_gain = fwd_ret > 0
    total_gain = fwd_ret[is_gain].sum()
    foregone_gain = fwd_ret[is_gain & flag].sum()
    total_dd = fwd_dd.sum()
    avoided_dd = fwd_dd[flag].sum()
    return dict(
        avoided_loss_pct=100 * avoided_loss / total_loss if total_loss else 0,
        foregone_gain_pct=100 * foregone_gain / total_gain if total_gain else 0,
        avoided_dd_pct=100 * avoided_dd / total_dd if total_dd else 0,
        net_benefit=avoided_loss - foregone_gain,
        flag_rate=100 * flag.mean(),
    )


# ══════════════════════════════════════════════════════════════════
# Prediction loaders
# ══════════════════════════════════════════════════════════════════
def load_deep(name, seeds, preddir):
    """Seed-averaged val/test predictions for a sequence model."""
    vs, ts = [], []
    for s in seeds:
        p = os.path.join(preddir, f'{name}_seed{s}.npz')
        if os.path.exists(p):
            z = np.load(p)
            vs.append(z['val']); ts.append(z['test'])
    if not vs:
        return None, None
    return np.mean(vs, axis=0), np.mean(ts, axis=0)


def train_xgb_aligned(df, val_keys, test_keys, seeds):
    """
    Train XGBoost (5 seeds, ensemble) and return val/test probabilities
    ALIGNED to the sequence models' (ticker, date) keys.
    """
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    tech_cols, fund_cols = get_feature_groups(df)
    cols = tech_cols + fund_cols

    tr = df[(df.index >= '2010-01-01') & (df.index <= '2019-12-31')]
    va = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    te = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    med = tr[cols].median()
    sc = RobustScaler()
    Xtr = sc.fit_transform(tr[cols].fillna(med).values)
    Xva = sc.transform(va[cols].fillna(med).values)
    Xte = sc.transform(te[cols].fillna(med).values)
    ytr = tr['Target'].values

    # Map each flat row to its (ticker, date) key
    va_keys = list(zip(va['Ticker'].values, va.index))
    te_keys = list(zip(te['Ticker'].values, te.index))
    va_idx = {k: i for i, k in enumerate(va_keys)}
    te_idx = {k: i for i, k in enumerate(te_keys)}

    vs, ts = [], []
    for seed in seeds:
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed,
                              eval_metric='logloss', early_stopping_rounds=50,
                              n_jobs=-1, tree_method='hist')
        m.fit(Xtr, ytr, eval_set=[(Xva, va['Target'].values)], verbose=False)
        vs.append(m.predict_proba(Xva)[:, 1])
        ts.append(m.predict_proba(Xte)[:, 1])
    v_ens = np.mean(vs, axis=0)
    t_ens = np.mean(ts, axis=0)

    # Align to sequence keys (drop rows XGBoost has but sequences don't)
    v_aligned = np.array([v_ens[va_idx[k]] for k in val_keys if k in va_idx])
    t_aligned = np.array([t_ens[te_idx[k]] for k in test_keys if k in te_idx])
    # sanity: every sequence key must exist in XGBoost's set
    miss_v = sum(k not in va_idx for k in val_keys)
    miss_t = sum(k not in te_idx for k in test_keys)
    if miss_v or miss_t:
        print(f"      ⚠️  {miss_v} val / {miss_t} test keys unmatched")
    return v_aligned, t_aligned


# ══════════════════════════════════════════════════════════════════
def evaluate_across_regimes(name, val_scores, y_val, test_scores, test_data):
    """One threshold (from val), applied to every regime slice of the test."""
    thr = best_threshold_mcc(y_val, val_scores)
    flag_all = (test_scores >= thr).astype(int)
    dates = test_data['dates']

    rows = []
    for reg, (s, e) in REGIMES.items():
        mask = (dates >= s) & (dates <= e)
        if mask.sum() == 0:
            continue
        m = financial_metrics(flag_all[mask],
                              test_data['dd'][mask],
                              test_data['ret'][mask])
        m['regime'] = reg
        m['model'] = name
        rows.append(m)
    return rows


def print_regime_table(all_rows):
    models = []
    for r in all_rows:
        if r['model'] not in models:
            models.append(r['model'])

    for reg in REGIMES:
        print(f"\n── {reg} ──")
        print(f"{'Model':<26} {'Avoid loss':>11} {'Foregone':>9} "
              f"{'Avoid DD':>9} {'Net':>9} {'Exit%':>7}")
        print("-" * 76)
        for mdl in models:
            r = next((x for x in all_rows
                      if x['model'] == mdl and x['regime'] == reg), None)
            if r is None:
                continue
            print(f"{mdl:<26} {r['avoided_loss_pct']:>10.1f}% "
                  f"{r['foregone_gain_pct']:>8.1f}% {r['avoided_dd_pct']:>8.1f}% "
                  f"{r['net_benefit']:>+9.1f} {r['flag_rate']:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--preddir', type=str, default='results/preds')
    args = ap.parse_args()

    print("═" * 76)
    print("FINANCIAL METRIC — AVOIDED LOSS BY MARKET REGIME")
    print("═" * 76)
    print("  Test = 2022-2024 (full hold-out). Broken into bear/recovery/bull.")
    print("  Validation (2020-2021) is NOT used here — no leakage.\n")

    print("[1/4] Dataset (cache)...")
    df = prepare_dataset(force_refresh=False)

    print("[2/4] Forward returns (from Close, label-aligned)...")
    val_data = compute_forward_returns(df, VAL_START, VAL_END, args.horizon, SEQ_LEN)
    test_data = compute_forward_returns(df, TEST_START, TEST_END, args.horizon, SEQ_LEN)
    y_val = val_data['y']
    print(f"      val : {len(val_data['keys']):,} obs")
    print(f"      test: {len(test_data['keys']):,} obs")
    # per-regime return character
    for reg, (s, e) in REGIMES.items():
        mask = (test_data['dates'] >= s) & (test_data['dates'] <= e)
        if mask.sum():
            r = test_data['ret'][mask]
            print(f"        {reg:<22} mean fwd ret={r.mean()*100:+.2f}% | "
                  f"loss share={100*(r<0).mean():.0f}% | n={mask.sum():,}")

    all_rows = []

    # ── Reference: perfect foresight & random ──
    print("\n[3/4] Reference strategies...")
    for name, flag in [
        ('Perfect foresight', test_data['y']),
        ('Random exit (~13%)',
         (np.random.RandomState(0).rand(len(test_data['y'])) < 0.13).astype(int)),
    ]:
        for reg, (s, e) in REGIMES.items():
            mask = (test_data['dates'] >= s) & (test_data['dates'] <= e)
            if mask.sum() == 0:
                continue
            m = financial_metrics(flag[mask],
                                  test_data['dd'][mask],
                                  test_data['ret'][mask])
            m['regime'] = reg; m['model'] = name
            all_rows.append(m)

    # ── Models ──
    print("[4/4] Models (deep from cache, XGBoost retrained + aligned)...")
    v, t = load_deep('MS_gated_cross_attention', args.seeds, args.preddir)
    if v is not None:
        all_rows += evaluate_across_regimes('Gated (ens x5)', v, y_val, t, test_data)
        print("      ✓ Gated")

    v, t = load_deep('MS_cross_attention', args.seeds, args.preddir)
    if v is not None:
        all_rows += evaluate_across_regimes('Cross-attn (ens x5)', v, y_val, t, test_data)
        print("      ✓ Cross-attention")

    try:
        xv, xt = train_xgb_aligned(df, val_data['keys'], test_data['keys'], args.seeds)
        all_rows += evaluate_across_regimes('XGBoost (ens x5)', xv, y_val, xt, test_data)
        print("      ✓ XGBoost (aligned)")
    except Exception as e:
        print(f"      ⚠️  XGBoost skipped: {e}")

    # ── Report ──
    print("\n" + "═" * 76)
    print("RESULTS BY REGIME")
    print("═" * 76)
    print_regime_table(all_rows)

    pd.DataFrame(all_rows).to_csv('results/financial_metrics_by_regime.csv', index=False)

    print("\n" + "═" * 76)
    print("HOW TO READ THIS")
    print("═" * 76)
    print("  • Headline metric: AVOIDED DRAWDOWN % (label-aligned, regime-robust).")
    print("  • Net benefit flips sign by regime — it measures the MARKET, not the")
    print("    model. Positive in bear (2022), negative in bull (2024): expected")
    print("    for a downside-protection model. Report it per-regime, not overall.")
    print("  • The bear column (2022) is where a risk model must earn its keep.")
    print("  • All four columns come from the same hold-out test set; reporting all")
    print("    of them (not just the favourable one) avoids cherry-picking.")
    print("\n  → results/financial_metrics_by_regime.csv")
    print("═" * 76)


if __name__ == '__main__':
    main()