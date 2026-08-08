import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Base\n(XGB)', 'Concat\nFusion', 'Bilinear\nFusion', 'Gated\nFusion', 'Focal\nLoss', 'Macro\n(VIX)', 'Textual\nGraph', 'Dynamic\nGraph', 'Price\nGraph(GAT)', 'Hawkes\nGAT']
acc_scores = [0.613, 0.880, 0.884, 0.885, 0.884, 0.870, 0.868, 0.871, 0.871, 0.869]
mcc_scores = [0.072, 0.010, 0.010, 0.014, 0.012, 0.076, 0.073, 0.069, 0.077, 0.076]

def plot_metrics(acc_scores, mcc_scores):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Accuracy Bars (Sol Eksen)
    color1 = '#3498db'
    ax1.set_ylabel('Accuracy (Doğruluk)', color=color1, fontsize=12, fontweight='bold')
    bars = ax1.bar(methods, acc_scores, color=color1, alpha=0.7, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.0)
    
    # Accuracy değerlerini barların üstüne yaz
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=10, color=color1, fontweight='bold')
        
    # MCC Line (Sağ Eksen)
    ax2 = ax1.twinx()  
    color2 = '#e74c3c'
    ax2.set_ylabel('MCC (Çöküş Yakalama Gücü)', color=color2, fontsize=12, fontweight='bold')
    line = ax2.plot(methods, mcc_scores, color=color2, marker='o', linewidth=3, markersize=10, label='MCC')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 0.1) # Maksimum MCC grafikte daha net görünsün diye 0.1 yaptık
    
    # MCC değerlerini noktaların üstüne yaz
    for i, v in enumerate(mcc_scores):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=11, color=color2, fontweight='bold')
        
    # Başlık ve Grid
    plt.title('Tez Savunması: Fusion Metotları vs. Spatio-Temporal GAT (Accuracy Paradoksu)', fontsize=14, pad=20, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Efsane (Legend)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
    print("Grafik 'fusion_comparison.png' olarak kaydedildi.")

plot_metrics(acc_scores, mcc_scores)
