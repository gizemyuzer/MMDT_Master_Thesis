import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

print("="*60)
print("STOCK PERFORMANCE CHARTS GENERATION PIPELINE")
print("="*60)

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading Data for Chart Generation...")
df = pd.read_csv('final_dataset.csv', index_col=0, parse_dates=True)

target_col  = 'Target'
ticker_col  = 'Ticker'
absolute_cols = ['Close', 'High', 'Low', 'Volume', 'MACD', 'MACD_Signal', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'BB_Std', 'ATR_14', 'SMA_50', 'SMA_200']
feature_cols = [c for c in df.columns if c not in [target_col, ticker_col] + absolute_cols]

train_mask = (df.index >= '2015-01-01') & (df.index <= '2021-12-10')
val_mask   = (df.index >= '2022-01-01') & (df.index <= '2023-12-31')
test_mask  = (df.index >= '2024-01-01') & (df.index <= '2026-03-31')

# ── Scaler ───────────────────────────────────────────────────────────────────
print("Fitting RobustScaler on train set...")
train_medians = df[train_mask][feature_cols].median()
X_train = df[train_mask][feature_cols].fillna(train_medians).values
y_train = df[train_mask][target_col].values
scaler  = RobustScaler()
scaler.fit(X_train)

# ── XGBoost Model Yükleme ──
print("Loading XGBoost model...")
xgb_model = xgb.XGBClassifier()
if os.path.exists("xgb_model.json"):
    xgb_model.load_model("xgb_model.json")
    print("  => xgb_model.json loaded successfully.")
else:
    print("  => WARNING: xgb_model.json not found!")

# ── PyTorch MLP Model Yükleme ──
mlp_available = False
if os.path.exists('best_model.pth'):
    try:
        from model import PureTabularModel
        mlp_model = PureTabularModel(tabular_dim=len(feature_cols), d_model=256)
        mlp_model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))
        mlp_model.eval()
        mlp_available = True
        print("  => PyTorch MLP loaded successfully.")
    except Exception as e:
        print(f"  => PyTorch MLP could not be loaded: {e}")

try:
    import torch
    from model import TimeSeriesTransformer
    
    # We need the feature_dim. The model was trained with feature_dim=47 (excluding Target, Ticker).
    # It's safer to read the shape from X_t.
    meta_available = True
    print("  => meta_transformer.pth logic loaded.")
except Exception as e:
    meta_available = False
    print(f"  => Meta-Model could not be loaded: {e}")

# ── Thresholds Yükleme ──
XGB_THRESH = 0.40
META_THRESH = 0.50
if os.path.exists("model_thresholds.txt"):
    with open("model_thresholds.txt", "r") as f:
        for line in f:
            if line.startswith('XGB_THRESH='):
                XGB_THRESH = float(line.strip().split('=')[1])
            elif line.startswith('META_THRESH='):
                META_THRESH = float(line.strip().split('=')[1])
    print(f"  => Thresholds loaded dynamically: XGB={XGB_THRESH:.4f}, META={META_THRESH:.4f}")
else:
    print("  => WARNING: model_thresholds.txt not found, using fallbacks.")
    print("  => WARNING: model_thresholds.txt not found, using 0.50 fallback.")

# ── Volatility Gate 65th Percentile ──
def get_vol_threshold(ticker_history, pct=0.65):
    if 'Vol_20d' in ticker_history.columns:
        return ticker_history['Vol_20d'].quantile(pct)
    return 0.0

out_dir = "stock_performance_charts"
os.makedirs(out_dir, exist_ok=True)

test_df  = df[test_mask]
tickers  = test_df['Ticker'].unique()[:20]
print(f"\nGenerating charts for {len(tickers)} tickers under validation/test period...")

# ── Per-ticker chart ─────────────────────────────────────────────────────────
for ticker in tickers:
    t_df = test_df[test_df['Ticker'] == ticker].fillna(train_medians).copy()
    if len(t_df) < 20:
        continue

    ticker_history = df[df['Ticker'] == ticker]

    # Raw probabilities
    X_t = scaler.transform(t_df[feature_cols].values)
    t_df['Prob_XGB'] = xgb_model.predict_proba(X_t)[:, 1]

    if mlp_available:
        with torch.no_grad():
            logits = mlp_model(torch.tensor(X_t, dtype=torch.float32))
            t_df['Prob_MLP'] = torch.sigmoid(logits).numpy().flatten()
    else:
        t_df['Prob_MLP'] = t_df['Prob_XGB']

    # ────────────────────────────────────────────────────────────────────────
    # FILTER 1: Cascade / Meta-Labeling Consensus (Transformer)
    # ────────────────────────────────────────────────────────────────────────
    if meta_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Instantiate model if not yet instantiated
        if 'meta_model' not in locals():
            meta_model = TimeSeriesTransformer(feature_dim=X_t.shape[1], seq_len=20, d_model=64, n_heads=4, n_layers=2)
            meta_model.load_state_dict(torch.load("meta_transformer.pth", map_location=device))
            meta_model.to(device)
            meta_model.eval()
            
        prob_meta = np.zeros(len(t_df))
        seq_len = 20
        
        # We only need to run the transformer for days where Prob_XGB >= XGB_THRESH
        stage1_alerts = np.where(t_df['Prob_XGB'] >= XGB_THRESH)[0]
        
        with torch.no_grad():
            for i in stage1_alerts:
                if i >= seq_len - 1:
                    seq = X_t[i - seq_len + 1 : i + 1]
                    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = meta_model(seq_tensor)
                    prob_meta[i] = torch.sigmoid(logits).item()
                    
        t_df['Prob_Meta'] = prob_meta
        t_df['Consensus'] = (t_df['Prob_XGB'] >= XGB_THRESH) & (t_df['Prob_Meta'] >= META_THRESH)
    else:
        t_df['Prob_Meta'] = 0.0
        t_df['Consensus'] = (t_df['Prob_XGB'] >= XGB_THRESH)

    # ────────────────────────────────────────────────────────────────────────
    # FILTER 2: N-day persistence (3 consecutive consensus days = real alert)
    # ────────────────────────────────────────────────────────────────────────
    PERSISTENCE = 3
    consec = t_df['Consensus'].astype(int)
    rolling_sum = consec.rolling(PERSISTENCE, min_periods=PERSISTENCE).sum()
    t_df['Persistent_Alert'] = (rolling_sum >= PERSISTENCE)

    # Final alert
    t_df['Is_Critical_Alert'] = t_df['Persistent_Alert']

    # ── Forward 20-day max drawdown (New Simplified Formulation) ──────────────
    forward_drawdowns = np.zeros(len(t_df))
    dynamic_thresholds = np.zeros(len(t_df))
    closes = t_df['Close'].values
    
    for i in range(len(t_df)):
        window = closes[i+1 : min(i+21, len(t_df))]
        vol_20d = t_df['Vol_20d'].iloc[i] if 'Vol_20d' in t_df.columns else 0.0
        
        if len(window) > 0:
            min_price = np.min(window)
            dd = (closes[i] - min_price) / (closes[i] + 1e-9)
            forward_drawdowns[i] = dd * 100
        else:
            forward_drawdowns[i] = 0.0
        expected_20d_vol = (vol_20d / np.sqrt(252)) * np.sqrt(20)
        dynamic_thresholds[i] = np.clip(1.5 * expected_20d_vol * 100, 10.0, 30.0)

    t_df['FMDD_20'] = forward_drawdowns
    t_df['Dynamic_Thresh'] = dynamic_thresholds
    t_df['Target'] = (t_df['FMDD_20'] >= t_df['Dynamic_Thresh']).astype(int)

    # ── Plot ──
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={'height_ratios': [3, 1, 1]},
        sharex=True,
        facecolor='#1E1E1E'
    )
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_facecolor('#1E1E1E')
        ax.grid(True, color='#2B2B2B', alpha=0.5, linestyle='--')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Panel 1: Price + Ground Truth + Risk Heatmap
    # Thin connecting line
    ax1.plot(t_df.index, t_df['Close'], color='white', lw=1, alpha=0.3, label='Close Price Path')
    
    # Heatmap Scatter (Dots colored by XGBoost Probability)
    heatmap = ax1.scatter(t_df.index, t_df['Close'], c=t_df['Prob_XGB'], 
                          cmap='RdYlGn_r', vmin=0.0, vmax=1.0, 
                          s=30, zorder=4, label='Risk Heatmap')
                          
    # Ground Truth shading (Target = 1)
    for d in t_df[t_df['Target'] == 1].index:
        ax1.axvspan(d, d + pd.Timedelta(days=1), color='red', alpha=0.15, lw=0)
    ax1.fill_between([], [], color='red', alpha=0.15, label='Ground Truth Crash (FMDD Target=1)')

    ax1.set_ylabel('Close Price ($)', color='white')
    ax1.set_title(f'{ticker} — XGBoost Risk Heatmap (Continuous)',
                  fontsize=14, weight='bold', color='white')
    ax1.legend(loc='upper left', facecolor='#1E1E1E', edgecolor='none', labelcolor='white')

    # Panel 2: Model probabilities (Heatmap Curve)
    ax2.plot(t_df.index, t_df['Prob_XGB'] * 100, color='#e76f51', lw=1.5, label='XGBoost Risk Prob (%)')
    ax2.fill_between(t_df.index, t_df['Prob_XGB'] * 100, color='#e76f51', alpha=0.2)
    ax2.axhline(XGB_THRESH * 100, color='#e76f51', linestyle=':', alpha=0.6, label=f'Optimal Thresh ({XGB_THRESH*100:.1f}%)')
    ax2.set_ylabel('Risk Prob (%)', color='white')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', facecolor='#1E1E1E', edgecolor='none', labelcolor='white')

    # Panel 3: Forward drawdown
    ax3.plot(t_df.index, t_df['FMDD_20'], color='magenta', lw=1.5, label='20-Day Forward Max Drawdown (%)')
    ax3.plot(t_df.index, t_df['Dynamic_Thresh'], color='red', linestyle='--', lw=2, label='Dynamic Threshold (1.5x Vol, 10%-30%)')
    ax3.set_ylabel('Max Drawdown (%)', color='white')
    ax3.set_xlabel('Date', color='white')
    ax3.legend(loc='lower right', facecolor='#1E1E1E', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{ticker}_performance.png'), dpi=150, facecolor='#1E1E1E')
    plt.close()
    print(f"  ✓ {ticker} chart generated (Heatmap applied).")

print(f"\n=> Tüm stock performance grafikleri '{out_dir}' klasörüne başarıyla kaydedildi.")
