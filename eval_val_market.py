import numpy as np
from datasets.feature_engineering import prepare_dataset
import financial_metric as fm

print("======================================================")
print("EVALUATING ON 2020-2021 (INCLUDES 2020 BEAR MARKET/CRASH)")
print("======================================================")

df = prepare_dataset(force_refresh=False)
dd_val, ret_val, y_val, keys_val = fm.compute_forward_returns(df, fm.VAL_START, fm.VAL_END, 20, 20)

print(f"\nTotal observations: {len(y_val)}")
print(f"Mean forward return in this period: {ret_val.mean()*100:.2f}%")

models = {'Gated (ensemble x5)': 'MS_gated_cross_attention', 'Cross-attention (ensemble x5)': 'MS_cross_attention'}

# Perfect Foresight
print("\n--- Perfect Foresight ---")
m = fm.financial_metrics(y_val, dd_val, ret_val)
print(f"Avoided loss: {m['avoided_loss_pct']:.1f}%")
print(f"Foregone gain: {m['foregone_gain_pct']:.1f}%")
print(f"Net benefit: {m['net_benefit']:.1f}")

for label, prefix in models.items():
    v, t = fm.load_deep_preds(prefix, [42, 43, 44, 45, 46], 'results/preds')
    if v is None: continue
    thr = fm.best_threshold_mcc(y_val, v)
    flag = (v >= thr).astype(int)
    m = fm.financial_metrics(flag, dd_val, ret_val)
    print(f"\n--- {label} ---")
    print(f"Avoided loss: {m['avoided_loss_pct']:.1f}%")
    print(f"Foregone gain: {m['foregone_gain_pct']:.1f}%")
    print(f"Net benefit: {m['net_benefit']:.1f}")
