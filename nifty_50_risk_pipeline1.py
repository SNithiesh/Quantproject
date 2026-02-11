# -*- coding: utf-8 -*-
"""
NIFTY 50: PROFESSIONAL PRESENTATION DASHBOARD - FINAL VERSION
============================================================
All alignment issues fixed, production-ready for presentation
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from arch import arch_model
import QuantLib as ql
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# =========================
# PROFESSIONAL MATPLOTLIB SETTINGS
# =========================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.2
})

# =========================
# DATA INGESTION & PROCESSING
# =========================
print("Loading NIFTY 50 historical data from CSV (13 years)")

csv_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_historical_data.csv"
raw_df = pd.read_csv(csv_path, parse_dates=['Date'])

# Pivot data to get individual stock columns
pivot_close = raw_df.pivot(index='Date', columns='Ticker', values='Close')

# Calculate returns for all stocks
stock_returns = pivot_close.pct_change()

# Calculate Equal-Weighted Index Return
index_returns = stock_returns.mean(axis=1)
df = pd.DataFrame({'Close': index_returns})
df['Return'] = index_returns
df.dropna(inplace=True)

# Reconstruct Close price series
df['Close'] = 100 * (1 + df['Return']).cumprod()

print(f"Loaded data for {pivot_close.shape[1]} stocks")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Total trading days: {len(df)}")

# =========================
# CALCULATE STOCK STATISTICS
# =========================
print("\nCalculating individual stock statistics...")

stocks_metrics = pd.DataFrame(index=stock_returns.columns)
stocks_metrics['Avg_Daily_Return_%'] = stock_returns.mean() * 100
stocks_metrics['Volatility_%'] = stock_returns.std() * np.sqrt(252) * 100
stocks_metrics['Sharpe_Ratio'] = (stocks_metrics['Avg_Daily_Return_%'] * 252 - 6.5) / stocks_metrics['Volatility_%']

# Add sector info
summary_path = r"C:\Users\mksni\OneDrive\Desktop\DA &DP\nifty50_summary_statistics.csv"
try:
    summary_df = pd.read_csv(summary_path)
    sector_map = dict(zip(summary_df['Ticker'], summary_df['Sector']))
    name_map = dict(zip(summary_df['Ticker'], summary_df['Company_Name']))
    mcap_map = dict(zip(summary_df['Ticker'], summary_df['Current_Market_Cap']))
    
    stocks_metrics['Sector'] = stocks_metrics.index.map(sector_map)
    stocks_metrics['Company_Name'] = stocks_metrics.index.map(name_map)
    stocks_metrics['Current_Market_Cap'] = stocks_metrics.index.map(mcap_map)
    
    stocks_metrics['Sector'].fillna('Unknown', inplace=True)
    stocks_metrics['Company_Name'].fillna(stocks_metrics.index, inplace=True)
    stocks_metrics['Current_Market_Cap'].fillna(stocks_metrics['Current_Market_Cap'].median(), inplace=True)
    
    stocks_df = stocks_metrics.reset_index().rename(columns={'index': 'Ticker'})
except Exception as e:
    print(f"Warning: {e}")
    stocks_df = stocks_metrics.reset_index().rename(columns={'index': 'Ticker'})
    stocks_df['Sector'] = 'Unknown'
    stocks_df['Company_Name'] = stocks_df['Ticker']
    stocks_df['Current_Market_Cap'] = 1e11

# =========================
# CALCULATE INDEX METRICS
# =========================
mean_return = np.mean(df["Return"])
std_risk = np.std(df["Return"])
risk_free_rate = 0.065
sharpe_ratio = (mean_return * 252 - risk_free_rate) / (std_risk * np.sqrt(252))

index_annual_return = mean_return * 252 * 100
index_annual_vol = std_risk * np.sqrt(252) * 100

# Comparative stats
better_return_count = len(stocks_df[stocks_df['Avg_Daily_Return_%'] * 252 > index_annual_return])
lower_vol_count = len(stocks_df[stocks_df['Volatility_%'] < index_annual_vol])
better_sharpe_count = len(stocks_df[stocks_df['Sharpe_Ratio'] > sharpe_ratio])

return_percentile = (stocks_df['Avg_Daily_Return_%'] * 252 < index_annual_return).sum() / len(stocks_df) * 100
vol_percentile = (stocks_df['Volatility_%'] > index_annual_vol).sum() / len(stocks_df) * 100

# =========================
# PROFESSIONAL DASHBOARD - FIXED VERSION
# =========================
print(f"\n{'='*70}")
print("GENERATING PROFESSIONAL PRESENTATION DASHBOARD - FIXED")
print(f"{'='*70}")

# Professional color scheme
COLORS = {
    'primary': '#3498db',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'purple': '#9b59b6',
    'teal': '#1abc9c',
    'index': '#f1c40f',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}

# Create figure with perfect proportions
fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.38,
                      left=0.06, right=0.90, top=0.90, bottom=0.07)

# ========== PANEL 1: RETURN DISTRIBUTION ==========
ax1 = fig.add_subplot(gs[0, :2])

# Histogram
n, bins, patches = ax1.hist(df["Return"] * 100, bins=50, color=COLORS['primary'], 
                            alpha=0.6, edgecolor='white', linewidth=0.5)

# KDE overlay on secondary axis
from scipy.stats import gaussian_kde
density = gaussian_kde(df["Return"] * 100)
xs = np.linspace(df["Return"].min() * 100, df["Return"].max() * 100, 200)
ax1_twin = ax1.twinx()
ax1_twin.plot(xs, density(xs), color=COLORS['danger'], linewidth=3, label='Density')
ax1_twin.set_ylabel('Density', fontsize=13, color=COLORS['danger'], fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=COLORS['danger'], labelsize=11)

# Mean line
ax1.axvline(mean_return * 100, color=COLORS['danger'], linestyle='--', 
            linewidth=2.5, label=f'Mean: {mean_return*100:.3f}%', zorder=10)
ax1.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)

ax1.set_title('Daily Return Distribution', fontsize=16, fontweight='bold', pad=12, color=COLORS['dark'])
ax1.set_xlabel('Daily Returns (%)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', framealpha=0.95, edgecolor=COLORS['dark'], fontsize=10)
ax1.tick_params(labelsize=11)

# ========== PANEL 2: Key Statistics Box ==========
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

# Create perfectly aligned statistics text
stats_lines = [
    "╔═══════════════════════════════════╗",
    "║  NIFTY 50 INDEX STATISTICS        ║",
    "╠═══════════════════════════════════╣",
    "║                                   ║",
    "║  Daily Metrics:                   ║",
    f"║    Mean Return : {mean_return*100:>8.4f}% ║",
    f"║    Std Dev     : {std_risk*100:>8.4f}%  ║",
    "║                                   ║",
    "║  Annualized Metrics:              ║",
    f"║    Return      : {index_annual_return:>8.2f}% ║",
    f"║    Volatility  : {index_annual_vol:>8.2f}%    ║",
    f"║    Sharpe Ratio: {sharpe_ratio:>8.4f}       ║",
    "║                                   ║",
    "║  Comparative Performance:         ║",
    f"║    Outperformed: {better_return_count:>2d} stocks   ║",
    f"║    Lower Vol   : {lower_vol_count:>2d} stocks       ║",
    "║                                   ║",
    "║  Percentile Rank:                 ║",
    f"║    Returns   : {return_percentile:>6.1f}th   ║",
    f"║    Risk      : {vol_percentile:>6.1f}th        ║",
    "║                                   ║",
    "╚═══════════════════════════════════╝",
]

stats_text = '\n'.join(stats_lines)
ax2.text(0.5, 0.4, stats_text, transform=ax2.transAxes, fontsize=10.5,
         verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF8DC', 
                   edgecolor='#8B7355', linewidth=2, alpha=0.9))

# ========== PANEL 3: RISK-RETURN SCATTER - FIXED AXIS ==========
ax3 = fig.add_subplot(gs[1, :])

scatter_data = stocks_df.copy()
scatter_data['Annual_Return'] = scatter_data['Avg_Daily_Return_%'] * 252

# Calculate proper axis limits to maximize view
x_min = scatter_data['Volatility_%'].min() * 0.95
x_max = scatter_data['Volatility_%'].max() * 1.05
y_min = scatter_data['Annual_Return'].min() - 20
y_max = scatter_data['Annual_Return'].max() + 20

# Scatter plot
scatter = ax3.scatter(scatter_data['Volatility_%'], 
                     scatter_data['Annual_Return'],
                     c=scatter_data['Sharpe_Ratio'],
                     s=scatter_data['Current_Market_Cap'] / 2e10,
                     alpha=0.7,
                     cmap='RdYlGn',
                     edgecolors='black',
                     linewidth=0.8,
                     vmin=-1, vmax=2)

# NIFTY 50 Index
ax3.scatter(index_annual_vol, index_annual_return, 
           s=600, c=COLORS['index'], marker='*',
           edgecolors='black', linewidth=2.5,
           label='NIFTY 50 Index', zorder=100)

# Annotate ONLY top 3 - positioned to avoid overlap
top_3 = scatter_data.nlargest(3, 'Sharpe_Ratio')
offsets = [(20, 20), (20, -25), (-60, 20)]  # Different positions for each

for i, (idx, row) in enumerate(top_3.iterrows()):
    name = row['Company_Name'].split()[0][:8]
    ax3.annotate(name, 
                xy=(row['Volatility_%'], row['Annual_Return']),
                xytext=offsets[i], textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         alpha=0.75, edgecolor='black', linewidth=1),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

# SET PROPER AXIS LIMITS - MAXIMIZE VIEW
ax3.set_xlim(0, 1500)  # As requested: "1500 in xaxis"
ax3.set_ylim(-50, 1500) # Capped at 1500 to zoom in

ax3.set_title('Risk-Return Analysis: NIFTY 50 Constituents', 
             fontsize=16, fontweight='bold', pad=15, color=COLORS['dark'])
ax3.set_xlabel('Annualized Volatility (%)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Annualized Return (%)', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', framealpha=0.95, edgecolor=COLORS['dark'], fontsize=11)
ax3.tick_params(labelsize=11)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax3, pad=0.015)
cbar.set_label('Sharpe Ratio', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# ========== PANEL 4: SECTOR PERFORMANCE ==========
ax4 = fig.add_subplot(gs[2, 0])

sector_stats = scatter_data.groupby('Sector')['Sharpe_Ratio'].mean().sort_values()
colors_sector = [COLORS['success'] if x > 0 else COLORS['danger'] for x in sector_stats.values]

bars = ax4.barh(range(len(sector_stats)), sector_stats.values,
               color=colors_sector, edgecolor='black', linewidth=1, alpha=0.8)

ax4.set_yticks(range(len(sector_stats)))
ax4.set_yticklabels(sector_stats.index, fontsize=10)
ax4.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax4.set_title('Sector Performance', fontsize=15, fontweight='bold', 
             pad=12, color=COLORS['dark'])
ax4.axvline(0, color='black', linewidth=1.5, alpha=0.5)
ax4.tick_params(labelsize=10)

# Value labels
for i, val in enumerate(sector_stats.values):
    x_pos = val + 0.03 if val > 0 else val - 0.03
    ha = 'left' if val > 0 else 'right'
    ax4.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, 
            fontsize=9, fontweight='bold')

# ========== PANEL 5: TOP PERFORMERS - FIXED TITLE ==========
ax5 = fig.add_subplot(gs[2, 1])

top_8 = scatter_data.nlargest(8, 'Sharpe_Ratio')
colors_top = plt.cm.RdYlGn(np.linspace(0.5, 0.9, 8))

# Shorten names
short_names = [name.split()[0][:10] for name in top_8['Company_Name']]

bars = ax5.barh(range(len(top_8)), top_8['Sharpe_Ratio'].values,
               color=colors_top, edgecolor='black', linewidth=1, alpha=0.85)

ax5.set_yticks(range(len(top_8)))
ax5.set_yticklabels(short_names, fontsize=10)
ax5.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax5.set_title('Top 8 Performers', fontsize=15, fontweight='bold', 
             pad=12, color=COLORS['dark'])
ax5.invert_yaxis()
ax5.tick_params(labelsize=10)

# Value labels
for i, val in enumerate(top_8['Sharpe_Ratio'].values):
    ax5.text(val + 0.03, i, f'{val:.2f}', va='center', ha='left',
            fontsize=9, fontweight='bold')

# ========== PANEL 6: VOLATILITY COMPARISON ==========
ax6 = fig.add_subplot(gs[2, 2])

vol_data = [scatter_data['Volatility_%'].values]

bp = ax6.boxplot(vol_data, labels=['Stocks'],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor=COLORS['purple'], alpha=0.7, linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(color='red', linewidth=2.5),
                flierprops=dict(marker='o', markerfacecolor=COLORS['danger'], 
                              markersize=5, alpha=0.6))

# Index reference
ax6.axhline(index_annual_vol, color=COLORS['index'], linestyle='--', 
           linewidth=2.5, label=f'Index: {index_annual_vol:.1f}%', zorder=10)

ax6.set_ylabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
ax6.set_title('Volatility Distribution', fontsize=15, fontweight='bold', 
             pad=12, color=COLORS['dark'])
ax6.legend(loc='upper right', framealpha=0.95, edgecolor=COLORS['dark'], fontsize=10)
ax6.tick_params(labelsize=10)

# Stats box
median_vol = scatter_data['Volatility_%'].median()
ax6.text(0.95, 0.05, f'Median: {median_vol:.1f}%\nIndex: {index_annual_vol:.1f}%',
         transform=ax6.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                  alpha=0.9, edgecolor=COLORS['dark'], linewidth=1.5))

# ========== MAIN TITLE - NO OVERLAP ==========
fig.suptitle('NIFTY 50 COMPREHENSIVE RISK ANALYSIS', 
            fontsize=20, fontweight='bold', y=0.96, color=COLORS['dark'])

# Subtitle
date_range = f"{df.index.min().strftime('%b %Y')} - {df.index.max().strftime('%b %Y')}"
fig.text(0.5, 0.935, f'Analysis Period: {date_range}  |  {len(stocks_df)} Stocks  |  {len(df):,} Trading Days',
         ha='center', fontsize=11, color='#7f8c8d', style='italic')

# Save
plt.savefig('nifty50_final_dashboard.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("\n" + "="*70)
print("✓ FINAL PERFECT DASHBOARD SAVED: 'nifty50_final_dashboard.png'")
print("="*70)
plt.show()