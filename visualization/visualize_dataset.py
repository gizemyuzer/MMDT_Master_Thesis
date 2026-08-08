import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset for visualization...")
df = pd.read_csv('final_dataset.csv', index_col=0, parse_dates=True)

# Find a ticker with a good mix of 0 and 1 labels
ticker_counts = df.groupby('Ticker')['Target'].value_counts().unstack().fillna(0)
# Get a ticker with at least some risk events
valid_tickers = ticker_counts[ticker_counts[1.0] > 50].index
ticker_to_plot = valid_tickers[0] if len(valid_tickers) > 0 else df['Ticker'].unique()[0]

target_df = df[df['Ticker'] == ticker_to_plot].sort_index()

# Plot Styling
plt.style.use('dark_background')
fig_color = '#1E1E1E'
text_color = '#FFFFFF'

plt.figure(figsize=(14, 7), facecolor=fig_color)
ax = plt.gca()
ax.set_facecolor(fig_color)

# Plot Close Price
plt.plot(target_df.index, target_df['Close'], color='cyan', lw=1.5, label='Close Price')

# Highlight Risk Regions
# Target == 1 means a 15% drop occurs in the next 20 days.
risk_days = target_df[target_df['Target'] == 1].index

for day in risk_days:
    plt.axvspan(day, day + pd.Timedelta(days=1), color='red', alpha=0.3, lw=0)

plt.title(f"Dataset Visualization: {ticker_to_plot} Price vs. Future 15% Crash Zones (Red)", color=text_color, fontsize=14)
plt.xlabel("Date", color=text_color)
plt.ylabel("Price (USD)", color=text_color)
plt.tick_params(colors=text_color)

# Custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='cyan', lw=2, label='Stock Price'),
    Patch(facecolor='red', alpha=0.3, label='Crash Zone (Next 20 days drops > 15%)')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('dataset_visualization.png', facecolor=fig_color, dpi=150)
print("Visualization saved to dataset_visualization.png")

# Also let's save a pie chart of the overall dataset balance
plt.figure(figsize=(6, 6), facecolor=fig_color)
ax = plt.gca()
ax.set_facecolor(fig_color)

label_counts = df['Target'].value_counts()
plt.pie(label_counts, labels=['Stable (No Crash)', 'Risk (15% Crash Ahead)'], 
        autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'], textprops={'color': text_color})
plt.title("Overall Dataset Class Balance", color=text_color)

plt.tight_layout()
plt.savefig('dataset_balance.png', facecolor=fig_color, dpi=150)
print("Balance saved to dataset_balance.png")
