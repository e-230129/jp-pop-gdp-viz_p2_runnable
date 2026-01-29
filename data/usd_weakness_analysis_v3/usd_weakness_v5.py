#!/usr/bin/env python3
"""
USD Weakness Analysis v5 - 最終修正版

GPT-5.2 Pro 最終レビュー反映:
1. DXY vs Trade-weighted の区別を明確化
   - サンプル版: "DXY-like (Major Currencies)" と明記
   - 実データ版: FRB Trade-Weighted Index を使用
2. イベント注釈をオプション化（SHOW_EVENTS フラグ）
3. その他の修正は v4 から継承

依存:
  pip install pandas matplotlib numpy
  pip install japanize-matplotlib  # 日本語（推奨）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
OUTDIR = Path("output_usd_weakness")
OUTDIR.mkdir(exist_ok=True)

# イベント注釈の表示ON/OFF（実データ版では OFF 推奨）
SHOW_EVENTS = False

# イベント定義（SHOW_EVENTS=True の場合のみ使用）
EVENTS = [
    ('2024-11-05', 'Trump Elected', 'darkred'),
    ('2025-01-20', 'Inauguration', 'darkred'),
    ('2025-04-09', 'Tariffs', 'orange'),
    ('2025-12-10', 'Fed Cut', 'green'),
]

# 日本語フォント
try:
    import japanize_matplotlib
    FONT_OK = True
except ImportError:
    print("[Warning] japanize_matplotlib not found.")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    FONT_OK = False

plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# サンプルデータ生成
# =============================================================================

def generate_sample_data():
    """2020-2026のサンプルデータ（実トレンド反映）"""
    
    dates = pd.date_range('2020-01-01', '2026-01-13', freq='B')
    n = len(dates)
    np.random.seed(42)
    
    def add_noise(base, scale=0.5):
        return base + np.random.randn(len(base)) * scale
    
    # USD Index (DXY-like): 100→114(2022.10)→104(2024)→110(2025.4)→96(2026)
    usd_idx_base = np.ones(n) * 100
    for i, d in enumerate(dates):
        if d < datetime(2022, 10, 1):
            progress = (d - dates[0]).days / ((datetime(2022, 10, 1) - dates[0]).days)
            usd_idx_base[i] = 100 + progress * 14
        elif d < datetime(2024, 1, 1):
            progress = (d - datetime(2022, 10, 1)).days / ((datetime(2024, 1, 1) - datetime(2022, 10, 1)).days)
            usd_idx_base[i] = 114 - progress * 10
        elif d < datetime(2024, 11, 5):
            usd_idx_base[i] = 104
        elif d < datetime(2025, 4, 9):
            progress = (d - datetime(2024, 11, 5)).days / ((datetime(2025, 4, 9) - datetime(2024, 11, 5)).days)
            usd_idx_base[i] = 104 + progress * 6
        else:
            progress = (d - datetime(2025, 4, 9)).days / ((dates[-1] - datetime(2025, 4, 9)).days)
            usd_idx_base[i] = 110 - progress * 14
    
    usd_idx = pd.Series(add_noise(usd_idx_base, 0.8), index=dates, name='USD_Index')
    
    # Gold
    gold_base = np.ones(n) * 100
    for i, d in enumerate(dates):
        progress = (d - dates[0]).days / ((dates[-1] - dates[0]).days)
        gold_base[i] = 100 + progress * 85
        if datetime(2022, 2, 1) <= d < datetime(2022, 6, 1):
            gold_base[i] += 10
        if d >= datetime(2024, 10, 1):
            gold_base[i] += 10
    gold = pd.Series(add_noise(gold_base, 1.5), index=dates, name='Gold')
    
    # EUR/USD
    eurusd_base = np.ones(n) * 100
    for i, d in enumerate(dates):
        if d < datetime(2022, 10, 1):
            progress = (d - dates[0]).days / ((datetime(2022, 10, 1) - dates[0]).days)
            eurusd_base[i] = 100 - progress * 14
        elif d < datetime(2024, 1, 1):
            progress = (d - datetime(2022, 10, 1)).days / ((datetime(2024, 1, 1) - datetime(2022, 10, 1)).days)
            eurusd_base[i] = 86 + progress * 10
        elif d < datetime(2024, 11, 5):
            eurusd_base[i] = 96
        elif d < datetime(2025, 4, 9):
            progress = (d - datetime(2024, 11, 5)).days / ((datetime(2025, 4, 9) - datetime(2024, 11, 5)).days)
            eurusd_base[i] = 96 - progress * 6
        else:
            progress = (d - datetime(2025, 4, 9)).days / ((dates[-1] - datetime(2025, 4, 9)).days)
            eurusd_base[i] = 90 + progress * 14
    
    eurusd = pd.Series(add_noise(eurusd_base, 0.7), index=dates, name='EUR/USD')
    gbpusd = pd.Series(add_noise(eurusd_base * 0.95 + 5, 0.7), index=dates, name='GBP/USD')
    audusd = pd.Series(add_noise(eurusd_base * 0.9 + 10, 1.0), index=dates, name='AUD/USD')
    
    # USD/JPY
    usdjpy_base = np.ones(n) * 100
    for i, d in enumerate(dates):
        if d < datetime(2022, 1, 1):
            progress = (d - dates[0]).days / ((datetime(2022, 1, 1) - dates[0]).days)
            usdjpy_base[i] = 100 + progress * 5
        elif d < datetime(2022, 10, 21):
            progress = (d - datetime(2022, 1, 1)).days / ((datetime(2022, 10, 21) - datetime(2022, 1, 1)).days)
            usdjpy_base[i] = 105 + progress * 40
        elif d < datetime(2023, 1, 15):
            progress = (d - datetime(2022, 10, 21)).days / ((datetime(2023, 1, 15) - datetime(2022, 10, 21)).days)
            usdjpy_base[i] = 145 - progress * 10
        elif d < datetime(2024, 7, 3):
            progress = (d - datetime(2023, 1, 15)).days / ((datetime(2024, 7, 3) - datetime(2023, 1, 15)).days)
            usdjpy_base[i] = 135 + progress * 20
        elif d < datetime(2024, 8, 5):
            progress = (d - datetime(2024, 7, 3)).days / ((datetime(2024, 8, 5) - datetime(2024, 7, 3)).days)
            usdjpy_base[i] = 155 - progress * 15
        else:
            progress = (d - datetime(2024, 8, 5)).days / ((dates[-1] - datetime(2024, 8, 5)).days)
            usdjpy_base[i] = 140 + progress * 15
    
    usdjpy = pd.Series(add_noise(usdjpy_base, 1.2), index=dates, name='USD/JPY')
    usdchf = pd.Series(add_noise(usd_idx_base * 0.7 + 30, 0.6), index=dates, name='USD/CHF')
    usdcad = pd.Series(add_noise(usd_idx_base * 0.5 + 50, 0.5), index=dates, name='USD/CAD')
    
    return pd.DataFrame({
        'USD_Index': usd_idx, 'Gold': gold, 'EUR/USD': eurusd, 'GBP/USD': gbpusd,
        'AUD/USD': audusd, 'USD/JPY': usdjpy, 'USD/CHF': usdchf, 'USD/CAD': usdcad,
    })

# =============================================================================
# USD弱さ指数への変換
# =============================================================================

def to_usd_weakness_index(series, direction, base_value):
    if direction == +1:
        return 100.0 * (series / base_value)
    elif direction == -1:
        return 100.0 * (base_value / series)
    else:
        raise ValueError("direction must be +1 or -1")

def add_events_to_ax(ax, data_index):
    """イベント注釈を追加（SHOW_EVENTS=True の場合のみ）"""
    if not SHOW_EVENTS:
        return
    for date_str, label, color in EVENTS:
        try:
            evt = pd.to_datetime(date_str)
            if evt >= data_index.min() and evt <= data_index.max():
                ax.axvline(x=evt, color=color, linestyle='--', alpha=0.5, linewidth=1)
                ax.text(evt, ax.get_ylim()[1] * 0.98, label, rotation=90, 
                       ha='right', va='top', fontsize=8, color=color)
        except:
            pass

# =============================================================================
# メイン
# =============================================================================

print("="*70)
print("USD Weakness Analysis v5 (Final Version)")
print("="*70)
print(f"Event annotations: {'ON' if SHOW_EVENTS else 'OFF'}")

raw_data = generate_sample_data()
print(f"Data range: {raw_data.index[0].strftime('%Y-%m-%d')} ~ {raw_data.index[-1].strftime('%Y-%m-%d')}")

base_idx = raw_data.first_valid_index()
base_values = raw_data.loc[base_idx]
print(f"Base date: {base_idx.strftime('%Y-%m-%d')}")
print(f"Output dir: {OUTDIR.absolute()}")

# 列名マッピング（表示用）
DISPLAY_NAMES = {
    'USD_Index': 'USD Index (Major Currencies)',  # DXYではなく一般的な表記
    'Gold': 'Gold (XAU/USD)',
    'EUR/USD': 'EUR/USD',
    'GBP/USD': 'GBP/USD',
    'AUD/USD': 'AUD/USD',
    'USD/JPY': 'USD/JPY',
    'USD/CHF': 'USD/CHF',
    'USD/CAD': 'USD/CAD',
}

directions = {
    'USD_Index': -1, 'Gold': +1, 'EUR/USD': +1, 'GBP/USD': +1,
    'AUD/USD': +1, 'USD/JPY': -1, 'USD/CHF': -1, 'USD/CAD': -1,
}

usd_weak = pd.DataFrame(index=raw_data.index)
for col in raw_data.columns:
    if col in directions:
        usd_weak[col] = to_usd_weakness_index(raw_data[col], directions[col], base_values[col])

# =============================================================================
# 図A: メインダッシュボード
# =============================================================================

fig1, axes = plt.subplots(2, 2, figsize=(18, 14))
fig1.suptitle('USD Weakness Dashboard\n[Simulated Data for Illustration]', 
              fontsize=18, fontweight='bold', y=1.02)

# --- 左上: USD Index ---
ax = axes[0, 0]
usd_idx = usd_weak['USD_Index'].dropna()
ax.fill_between(usd_idx.index, usd_idx, 100, where=(usd_idx >= 100), color='salmon', alpha=0.4, label='USD Weak')
ax.fill_between(usd_idx.index, usd_idx, 100, where=(usd_idx < 100), color='lightgreen', alpha=0.4, label='USD Strong')
ax.plot(usd_idx.index, usd_idx, color='navy', linewidth=2.5, label='USD Index (Major Currencies)')
ax.axhline(100, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate(f'{usd_idx.iloc[-1]:.1f}', xy=(usd_idx.index[-1], usd_idx.iloc[-1]),
           xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold', color='navy')
ax.set_title('USD Index: Weakness vs Major Currencies\n(Up = USD Weaker)', fontsize=12, fontweight='bold')
ax.set_ylabel('USD Weakness Index', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(60, 130)
add_events_to_ax(ax, usd_idx.index)

# --- 右上: Gold ---
ax = axes[0, 1]
gold = usd_weak['Gold'].dropna()
ax.fill_between(gold.index, gold, 100, where=(gold >= 100), color='gold', alpha=0.5)
ax.plot(gold.index, gold, color='darkgoldenrod', linewidth=2.5, label='Gold (XAU/USD)')
ax.axhline(100, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate(f'{gold.iloc[-1]:.1f}', xy=(gold.index[-1], gold.iloc[-1]),
           xytext=(5, -15), textcoords='offset points', fontsize=12, fontweight='bold', color='darkgoldenrod')
ax.set_title('Gold: USD Real Value Indicator\n(Up = USD Weaker)', fontsize=12, fontweight='bold')
ax.set_ylabel('USD Weakness Index', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
add_events_to_ax(ax, gold.index)

# --- 左下: EUR/GBP/AUD ---
ax = axes[1, 0]
for col, color, lw in [('EUR/USD', 'blue', 2.5), ('GBP/USD', 'orange', 2), ('AUD/USD', 'green', 2)]:
    data = usd_weak[col].dropna()
    ax.plot(data.index, data, label=col, color=color, linewidth=lw)
    ax.annotate(f'{data.iloc[-1]:.1f}', xy=(data.index[-1], data.iloc[-1]),
               xytext=(5, 0), textcoords='offset points', fontsize=10, fontweight='bold', color=color)
ax.axhline(100, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.set_title('vs EUR/GBP/AUD\n(Up = USD Weaker)', fontsize=12, fontweight='bold')
ax.set_ylabel('USD Weakness Index', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(60, 130)
add_events_to_ax(ax, data.index)

# --- 右下: 全通貨重ね描き ---
ax = axes[1, 1]

other_currencies = ['USD_Index', 'EUR/USD', 'GBP/USD', 'AUD/USD', 'USD/CHF', 'USD/CAD']
colors_other = {
    'USD_Index': 'navy', 'EUR/USD': 'blue', 'GBP/USD': 'orange', 
    'AUD/USD': 'green', 'USD/CHF': 'purple', 'USD/CAD': 'brown'
}

for col in other_currencies:
    if col in usd_weak.columns:
        data = usd_weak[col].dropna()
        display_name = DISPLAY_NAMES.get(col, col)
        ax.plot(data.index, data, color=colors_other[col], linewidth=1.5, alpha=0.5, label=display_name)

jpy = usd_weak['USD/JPY'].dropna()
ax.plot(jpy.index, jpy, color='red', linewidth=4, label='USD/JPY (JPY)', zorder=10)

ax.axhline(100, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Baseline')
ax.fill_between(jpy.index, jpy, 100, where=(jpy < 100), color='red', alpha=0.12)

for col in other_currencies:
    if col in usd_weak.columns:
        data = usd_weak[col].dropna()
        if not data.empty:
            ax.annotate(f'{data.iloc[-1]:.0f}', 
                       xy=(data.index[-1], data.iloc[-1]),
                       xytext=(8, 0), textcoords='offset points', 
                       fontsize=8, color=colors_other[col], alpha=0.8)

ax.annotate(f'JPY: {jpy.iloc[-1]:.1f}', 
           xy=(jpy.index[-1], jpy.iloc[-1]),
           xytext=(8, -5), textcoords='offset points', 
           fontsize=14, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

ax.set_title('All Currencies: JPY is THE WEAKEST\n(Red = JPY, far below baseline)', 
            fontsize=12, fontweight='bold', color='darkred')
ax.set_ylabel('USD Weakness Index (Up = USD Weaker)', fontsize=11)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 130)

textbox = '''Key Observation:
Red line (JPY) is the only one
well below baseline 100.
= "USD weak vs others,
   but strong vs JPY"
= JPY-specific weakness'''

ax.text(0.02, 0.98, textbox, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', horizontalalignment='left',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='red', linewidth=2))

for a in axes.flat:
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    a.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
out_a = OUTDIR / "graph_A_dashboard_v5.png"
plt.savefig(out_a, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure A saved: {out_a}")

# =============================================================================
# 図B: 2024-2026フォーカス
# =============================================================================

recent = usd_weak[usd_weak.index >= '2024-01-01'].copy()

fig2, ax2 = plt.subplots(figsize=(16, 10))

for col in other_currencies:
    if col in recent.columns:
        data = recent[col].dropna()
        display_name = DISPLAY_NAMES.get(col, col)
        ax2.plot(data.index, data, color=colors_other[col], linewidth=2, alpha=0.6, label=display_name)

jpy_recent = recent['USD/JPY'].dropna()
ax2.plot(jpy_recent.index, jpy_recent, color='red', linewidth=4, label='USD/JPY (JPY)', zorder=10)

ax2.axhline(100, color='black', linestyle='-', alpha=0.8, linewidth=2)
ax2.fill_between(jpy_recent.index, jpy_recent, 100, where=(jpy_recent < 100), alpha=0.15, color='red')

# イベント注釈（オプション）
add_events_to_ax(ax2, jpy_recent.index)

for col in other_currencies + ['USD/JPY']:
    if col in recent.columns:
        data = recent[col].dropna()
        if not data.empty:
            color = 'red' if col == 'USD/JPY' else colors_other.get(col, 'gray')
            size = 14 if col == 'USD/JPY' else 10
            weight = 'bold' if col == 'USD/JPY' else 'normal'
            ax2.annotate(f'{data.iloc[-1]:.1f}', 
                        xy=(data.index[-1], data.iloc[-1]),
                        xytext=(8, 0), textcoords='offset points', 
                        fontsize=size, fontweight=weight, color=color)

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('USD Weakness Index (Up = USD Weaker)', fontsize=12)
ax2.set_title('2024-2026 Focus: USD weak vs majors, but strong vs JPY\n[Simulated Data]', 
             fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(50, 130)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

interpret = '''Key Observation:
- USD Index / EUR / Gold above 100 = USD broadly weakening
- JPY (red) stays well below 100 = USD strong vs JPY only
- Gap > 40pt = JPY-specific structural weakness

Conclusion:
"USD is weak, but JPY is even weaker"'''

ax2.text(0.98, 0.02, interpret, transform=ax2.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='red', linewidth=2))

plt.tight_layout()
out_b = OUTDIR / "graph_B_focus_v5.png"
plt.savefig(out_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure B saved: {out_b}")

# =============================================================================
# 図C: USD Index vs USD/JPY 直接比較
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(14, 8))

usd_idx_aligned = usd_weak['USD_Index'].dropna()
jpy_aligned = usd_weak['USD/JPY'].dropna()
common_idx = usd_idx_aligned.index.intersection(jpy_aligned.index)
usd_idx_common = usd_idx_aligned.loc[common_idx]
jpy_common = jpy_aligned.loc[common_idx]

ax3.plot(usd_idx_common.index, usd_idx_common, 
        label='USD Index (Major Currencies)', color='navy', linewidth=2.5)
ax3.plot(jpy_common.index, jpy_common, 
        label='vs JPY (reciprocal)', color='crimson', linewidth=2.5)

ax3.axhline(100, color='black', linestyle='-', alpha=0.7, linewidth=2, label='Baseline')

# 2本の線の間を塗る
mask = usd_idx_common > jpy_common
ax3.fill_between(
    common_idx,
    jpy_common,
    usd_idx_common,
    where=mask,
    interpolate=True,
    alpha=0.2,
    color='red',
    label='Gap: JPY weaker than major currencies avg'
)

ax3.annotate(f'USD Index: {usd_idx_common.iloc[-1]:.1f}', 
            xy=(usd_idx_common.index[-1], usd_idx_common.iloc[-1]),
            xytext=(-80, 10), textcoords='offset points', fontweight='bold', color='navy',
            arrowprops=dict(arrowstyle='->', color='navy'))
ax3.annotate(f'vs JPY: {jpy_common.iloc[-1]:.1f}', 
            xy=(jpy_common.index[-1], jpy_common.iloc[-1]),
            xytext=(-80, -20), textcoords='offset points', fontweight='bold', color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson'))

gap = usd_idx_common.iloc[-1] - jpy_common.iloc[-1]
textbox = f'''Current Gap:
USD Index - vs JPY = {gap:.1f} points

Interpretation:
- USD Index = {usd_idx_common.iloc[-1]:.1f}: USD {abs(usd_idx_common.iloc[-1]-100):.1f}% {"WEAKER" if usd_idx_common.iloc[-1]>100 else "STRONGER"} vs majors
- vs JPY = {jpy_common.iloc[-1]:.1f}: USD {abs(jpy_common.iloc[-1]-100):.1f}% {"WEAKER" if jpy_common.iloc[-1]>100 else "STRONGER"} vs JPY

Red area = JPY-specific weakness ({gap:.0f}pt)'''

ax3.text(0.02, 0.98, textbox, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('USD Weakness Index (Up = USD Weaker)', fontsize=12)
ax3.set_title('USD Index vs USD/JPY Direct Comparison\nRed area = "USD weak vs majors but JPY even weaker" [Simulated]', 
             fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
add_events_to_ax(ax3, common_idx)

plt.tight_layout()
out_c = OUTDIR / "graph_C_comparison_v5.png"
plt.savefig(out_c, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure C saved: {out_c}")

# =============================================================================
# 図D: 通貨ランキング
# =============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))

latest_values = {}
for col in usd_weak.columns:
    if col != 'Gold':
        latest_values[col] = usd_weak[col].dropna().iloc[-1]

sorted_currencies = sorted(latest_values.items(), key=lambda x: x[1], reverse=True)
names = [DISPLAY_NAMES.get(x[0], x[0]) for x in sorted_currencies]
values = [x[1] for x in sorted_currencies]
orig_names = [x[0] for x in sorted_currencies]

colors = ['red' if 'JPY' in n else 'steelblue' for n in orig_names]
bars = ax4.barh(names, values, color=colors, edgecolor='black', linewidth=1.5)
ax4.axvline(100, color='black', linestyle='-', linewidth=2, label='Baseline')

for bar, val, orig_name in zip(bars, values, orig_names):
    if val >= 100:
        label = f'{val:.1f} (USD {val-100:.1f}% weaker)'
        ha, offset = 'left', 3
    else:
        label = f'{val:.1f} (USD {100-val:.1f}% stronger)'
        ha, offset = 'right', -3
    
    color = 'red' if 'JPY' in orig_name else 'black'
    weight = 'bold' if 'JPY' in orig_name else 'normal'
    ax4.text(val + offset, bar.get_y() + bar.get_height()/2, label,
            ha=ha, va='center', fontsize=11, fontweight=weight, color=color)

jpy_idx = next(i for i, n in enumerate(orig_names) if 'JPY' in n)
bars[jpy_idx].set_edgecolor('red')
bars[jpy_idx].set_linewidth(3)

ax4.set_xlabel('USD Weakness Index (Higher = Currency Stronger vs USD)', fontsize=12)
ax4.set_ylabel('Currency', fontsize=12)
ax4.set_title('Currency Strength Ranking vs USD\nJPY is THE WEAKEST [Simulated]', 
             fontsize=14, fontweight='bold')
ax4.set_xlim(40, 130)
ax4.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
out_d = OUTDIR / "graph_D_ranking_v5.png"
plt.savefig(out_d, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure D saved: {out_d}")

# =============================================================================
# 図E: 乖離幅の推移
# =============================================================================

fig5, ax5 = plt.subplots(figsize=(14, 8))

divergence = usd_idx_common - jpy_common

ax5.fill_between(common_idx, 0, divergence, where=divergence >= 0, 
                color='red', alpha=0.4, label='JPY weaker than major currencies avg')
ax5.fill_between(common_idx, 0, divergence, where=divergence < 0, 
                color='blue', alpha=0.4, label='JPY stronger')
ax5.plot(common_idx, divergence, color='darkred', linewidth=2)
ax5.axhline(0, color='black', linestyle='-', linewidth=2)

latest_div = divergence.iloc[-1]
ax5.annotate(f'Current: {latest_div:.1f}pt', 
            xy=(common_idx[-1], latest_div),
            xytext=(-100, 20), textcoords='offset points',
            fontsize=14, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

ax5.set_xlabel('Date', fontsize=12)
ax5.set_ylabel('Divergence (USD Index weakness - vs JPY weakness)', fontsize=12)
ax5.set_title('JPY-specific Weakness Gap Over Time\n(Positive = "USD weak but JPY even weaker") [Simulated]', 
             fontsize=14, fontweight='bold')
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
add_events_to_ax(ax5, common_idx)

interpret = f'''Current gap: {latest_div:.1f} points
= "USD weak vs majors, but JPY is {latest_div:.0f}pt weaker still"

Gap widening since 2022
= JPY structural weakness intensifying'''

ax5.text(0.02, 0.98, interpret, transform=ax5.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

plt.tight_layout()
out_e = OUTDIR / "graph_E_divergence_v5.png"
plt.savefig(out_e, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure E saved: {out_e}")

# =============================================================================
# サマリー
# =============================================================================

print("\n" + "="*70)
print("Summary: Currency Strength vs USD (Latest)")
print("="*70)
print(f"Base: {base_idx.strftime('%Y-%m-%d')} = 100")
print(f"\n{'Rank':<6} {'Currency':<30} {'Index':>8} {'Interpretation':<25}")
print("-" * 75)

for i, (orig_name, val) in enumerate(sorted_currencies, 1):
    display_name = DISPLAY_NAMES.get(orig_name, orig_name)
    if val >= 100:
        interp = f"USD {val-100:.1f}% WEAKER"
    else:
        interp = f"USD {100-val:.1f}% STRONGER"
    
    marker = " *** WEAKEST ***" if 'JPY' in orig_name else ""
    print(f"{i:<6} {display_name:<30} {val:>8.1f} {interp:<25} {marker}")

print("\n" + "="*70)
print("KEY FINDING")
print("="*70)

usd_idx_latest = latest_values['USD_Index']
jpy_latest = latest_values['USD/JPY']

print(f"""
JPY is THE WEAKEST currency vs USD

- USD Index (Major Currencies): {usd_idx_latest:.1f} -> USD is {"WEAKER" if usd_idx_latest > 100 else "STRONGER"} vs majors
- USD/JPY:                      {jpy_latest:.1f} -> USD is {100 - jpy_latest:.1f}% STRONGER vs JPY

Gap: {usd_idx_latest - jpy_latest:.1f} points

Conclusion:
"USD is weak vs major currencies, but overwhelmingly strong vs JPY"
= JPY-specific structural weakness
""")

print("\n" + "="*70)
print("Output Files")
print("="*70)
for f in [out_a, out_b, out_c, out_d, out_e]:
    print(f"  {f}")

print("\nDone!")
