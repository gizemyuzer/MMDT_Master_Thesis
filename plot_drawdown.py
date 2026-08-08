import pandas as pd
import matplotlib.pyplot as plt
import os
import deneme1

def plot_yearly_drawdown():
    # Final dataset'ten alabiliriz ama SPY_Close'u fetch_macro_data içinde korumuyoruz, 
    # bu yüzden CRSP üzerinden SPY_Close verisini yeniden çekmek daha sağlıklı olabilir.
    print("WRDS üzerinden SPY verisi çekiliyor...")
    try:
        db = wrds.Connection(wrds_username='gizemyuzer')
        spy_sql = "SELECT date, vwretd FROM crsp.dsi WHERE date >= '2015-01-01' AND date <= '2026-03-31' ORDER BY date"
        spy_df = db.raw_sql(spy_sql, date_cols=['date'])
        spy_df.set_index('date', inplace=True)
        spy_df['SPY_Close'] = (1 + spy_df['vwretd']).cumprod() * 100
        
        # Calculate Drawdown
        spy_df['Cumulative_Max'] = spy_df['SPY_Close'].cummax()
        spy_df['Drawdown'] = (spy_df['SPY_Close'] - spy_df['Cumulative_Max']) / spy_df['Cumulative_Max']
        
        # Group by year and get minimum drawdown (which is the maximum drop)
        spy_df['Year'] = spy_df.index.year
        yearly_dd = spy_df.groupby('Year')['Drawdown'].min() * 100 # Convert to percentage
        
        plt.figure(figsize=(10, 6))
        ax = yearly_dd.plot(kind='bar', color='tomato', edgecolor='black')
        plt.title('SPY (Market) Yıllık Maksimum Düşüş (Yearly Maximum Drawdown)', fontsize=14)
        plt.xlabel('Yıl', fontsize=12)
        plt.ylabel('Maksimum Düşüş (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=1)
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='top', xytext=(0, -5), textcoords='offset points', fontsize=10, color='black')
        
        os.makedirs('visualization', exist_ok=True)
        save_path = os.path.join('visualization', 'yearly_drawdown.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Yıllık drawdown grafiği kaydedildi: {save_path}")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    plot_yearly_drawdown()
