import numpy as np
from datasets.feature_engineering import prepare_dataset
import financial_metric as fm

print("======================================================")
print("EVALUATING ON 2022 (PURE BEAR MARKET)")
print("======================================================")

df = prepare_dataset(force_refresh=False)
# Sadece 2022 yılını alıyoruz
dd_test, ret_test, y_test, keys_test = fm.compute_forward_returns(df, '2022-01-01', '2022-12-31', 20, 20)

print(f"\nTotal observations: {len(y_test)}")
print(f"Mean forward return in 2022: {ret_test.mean()*100:.2f}%")

models = {'Gated (ensemble x5)': 'MS_gated_cross_attention', 'Cross-attention (ensemble x5)': 'MS_cross_attention'}

print("\n--- Perfect Foresight ---")
m = fm.financial_metrics(y_test, dd_test, ret_test)
print(f"Avoided loss: {m['avoided_loss_pct']:.1f}%")
print(f"Foregone gain: {m['foregone_gain_pct']:.1f}%")
print(f"Net benefit: {m['net_benefit']:.1f}")

# Threshold'u yine Validation setinden hesaplamamız lazım (gerçekçi olması için)
_, _, y_val, _ = fm.compute_forward_returns(df, fm.VAL_START, fm.VAL_END, 20, 20)

for label, prefix in models.items():
    v, t = fm.load_deep_preds(prefix, [42, 43, 44, 45, 46], 'results/preds')
    if v is None: continue
    thr = fm.best_threshold_mcc(y_val, v)
    
    # t array'i 2022-2024 arası tüm test setini kapsıyor. Biz sadece 2022'yi istiyoruz.
    # keys_test ile eşleşen kısımları almalıyız.
    _, _, _, full_test_keys = fm.compute_forward_returns(df, fm.TEST_START, fm.TEST_END, 20, 20)
    
    # prediction'ları align edelim
    idx_map = {k: i for i, k in enumerate(full_test_keys)}
    aligned_t = np.array([t[idx_map[k]] for k in keys_test])
    
    flag = (aligned_t >= thr).astype(int)
    m = fm.financial_metrics(flag, dd_test, ret_test)
    print(f"\n--- {label} ---")
    print(f"Avoided loss: {m['avoided_loss_pct']:.1f}%")
    print(f"Foregone gain: {m['foregone_gain_pct']:.1f}%")
    print(f"Net benefit: {m['net_benefit']:.1f}")
