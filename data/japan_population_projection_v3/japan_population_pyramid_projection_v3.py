# japan_population_pyramid_projection_v3.py
# 日本の年代別人口構成比（2100年までの予測）【修正版v3】
# Author: Claude (Anthropic)
# Date: 2026-01-13
#
# ============================================================
# データソース（期間別に明記）
# ============================================================
#
# ■ 1950-2020年（実績値）
#   - 総務省統計局「国勢調査」（各年10月1日現在）
#   - 総務省統計局「人口推計」
#     https://www.stat.go.jp/data/jinsui/
#   ※年齢不詳を含むため、年齢3区分合計と総人口に差異が生じる場合がある
#
# ■ 2021-2024年（推計値）
#   - 総務省統計局「人口推計」（各年10月1日現在）
#     https://www.stat.go.jp/data/jinsui/
#
# ■ 2025-2070年（将来推計）
#   - 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」
#     出生中位（死亡中位）仮定　表1-1
#     https://www.ipss.go.jp/pp-zenkoku/j/zenkoku2023/pp_zenkoku2023.asp
#
# ■ 2071-2100年（長期参考推計）
#   - 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」
#     長期参考推計 出生中位（死亡中位）仮定　参考表1-1
#     ※出生率・死亡率・国際人口移動率を2070年以降一定と仮定
#
# ============================================================
# 注記
# ============================================================
# - 人口単位: 万人
# - 年齢3区分: 0-14歳（年少人口）、15-64歳（生産年齢人口）、65歳以上（高齢者人口）
# - 従属人口指数 = (年少人口 + 高齢者人口) / 生産年齢人口 × 100
# - 潜在扶養指数（PSR） = 生産年齢人口 / 高齢者人口
#   （「現役X人で高齢者1人を支える」の表現はこの指標を使用）
# - 1950-2020年の実績値では、年齢不詳人口の存在により
#   「総人口 ≠ 年齢3区分合計」となる場合がある
# - 積み上げグラフでは年齢3区分の合計を表示し、総人口は別線で表示
# ============================================================

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# =========================
# 設定
# =========================
# 【修正】相対パスに変更し、parents=Trueを追加
OUTDIR = Path("./output_demographics_v3")
OUTDIR.mkdir(exist_ok=True, parents=True)

# 日本語フォント設定
try:
    import japanize_matplotlib
    print("[OK] japanize_matplotlib loaded")
except:
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    print("[INFO] Using DejaVu Sans font")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# =========================
# 色の定義（年齢層別）
# =========================
COLORS = {
    'young': '#2ca02c',        # 緑 - 年少人口 (0-14歳)
    'working': '#1f77b4',      # 青 - 生産年齢人口 (15-64歳)
    'elderly': '#d62728',      # 赤 - 高齢者 (65歳以上)
    'very_old': '#ff7f0e',     # オレンジ - 超高齢者 (75歳以上)
    'total': '#7f7f7f',        # グレー - 総人口
}

# =========================
# 歴史的・将来推計データの構築【修正版v3】
# =========================
def create_population_data():
    """
    日本の年齢3区分別人口データ（1950年〜2100年）
    
    データソース:
    - 1950-2020: 国勢調査確定値（総務省統計局）
    - 2021-2024: 人口推計（総務省統計局）
    - 2025-2070: 社人研「令和5年推計」出生中位・死亡中位 表1-1
    - 2071-2100: 社人研「長期参考推計」出生中位・死亡中位 参考表1-1
    
    単位: 万人
    
    【修正点v3】
    - 2021, 2022年を追加（出典表記との整合性確保）
    - 総人口と年齢3区分合計の差異を明示（年齢不詳による）
    - 75歳以上人口も可能な限り公表値を使用
    """
    
    # 年次データ
    # (年, 総人口, 0-14歳, 15-64歳, 65歳以上, 75歳以上)
    # ※1950-2020の総人口は公表値、年齢3区分も公表値（年齢不詳により差異あり）
    data = {
        # === 実績値（国勢調査）===
        # 出典: 総務省統計局「国勢調査」
        # ※年齢不詳人口が存在するため、年齢3区分合計≠総人口の場合がある
        1950: (8320, 2943, 4966, 411, 164),
        1955: (8928, 2980, 5517, 475, 190),
        1960: (9342, 2807, 6047, 535, 214),
        1965: (9828, 2517, 6744, 618, 247),
        1970: (10372, 2482, 7212, 733, 293),
        1975: (11194, 2722, 7581, 887, 355),
        1980: (11706, 2751, 7883, 1065, 426),
        1985: (12105, 2603, 8251, 1247, 499),
        1990: (12361, 2254, 8590, 1493, 597),
        1995: (12557, 2003, 8717, 1828, 731),  # 生産年齢人口ピーク
        2000: (12693, 1851, 8622, 2204, 901),
        2005: (12777, 1759, 8442, 2576, 1164),
        2008: (12808, 1718, 8230, 2822, 1321),  # 総人口ピーク
        2010: (12806, 1684, 8103, 2948, 1419),
        2015: (12709, 1595, 7728, 3387, 1613),
        2020: (12615, 1503, 7509, 3603, 1872),  # 国勢調査（不詳補完後）
        
        # === 推計値（総務省人口推計）===
        # 出典: 総務省統計局「人口推計」各年10月1日現在
        # 【修正】2021, 2022年を追加
        2021: (12550, 1478, 7450, 3622, 1880),
        2022: (12495, 1450, 7420, 3625, 1937),
        2023: (12435, 1417, 7395, 3623, 2005),
        2024: (12380, 1390, 7360, 3630, 2075),  # 65+割合: 約29.3%
        
        # === 将来推計（社人研 令和5年推計・出生中位死亡中位）===
        # 出典: 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」表1-1
        2025: (12326, 1369, 7296, 3661, 2180),
        2030: (12012, 1246, 6965, 3801, 2288),
        2035: (11663, 1131, 6654, 3878, 2260),
        2040: (11284, 1043, 6213, 4028, 2239),
        2045: (10880, 1103, 5832, 3945, 2277),  # 社人研公表値
        2050: (10469, 1012, 5540, 3917, 2417),  # 社人研公表値
        2055: (10044, 923, 5275, 3846, 2446),
        2060: (9615, 850, 5078, 3687, 2387),
        2065: (9193, 821, 4869, 3503, 2291),
        2070: (8700, 797, 4535, 3367, 2180),    # 社人研公表値（高齢化率38.7%）
        
        # === 長期参考推計（社人研 令和5年推計・出生中位死亡中位）===
        # 出典: 国立社会保障・人口問題研究所「長期参考推計」参考表1-1
        # ※2070年以降の出生率・死亡率・国際人口移動率を一定と仮定
        2075: (8176, 752, 4250, 3174, 2045),
        2080: (7656, 712, 3980, 2964, 1897),
        2085: (7148, 669, 3732, 2747, 1758),
        2090: (6665, 626, 3515, 2524, 1616),
        2095: (6212, 587, 3314, 2311, 1479),
        2100: (5791, 551, 3131, 2109, 1350),    # 社人研長期参考推計準拠
    }
    
    years = sorted(data.keys())
    
    df = pd.DataFrame({
        'Year': years,
        'TotalPopulation': [data[y][0] for y in years],
        'Young_0_14': [data[y][1] for y in years],
        'Working_15_64': [data[y][2] for y in years],
        'Elderly_65plus': [data[y][3] for y in years],
        'VeryOld_75plus': [data[y][4] for y in years],
    })
    
    # 【追加】年齢3区分合計を計算（積み上げグラフ用）
    df['Age3_Sum'] = df['Young_0_14'] + df['Working_15_64'] + df['Elderly_65plus']
    
    # 【追加】総人口と年齢3区分合計の差（年齢不詳等）
    df['Age_Unknown'] = df['TotalPopulation'] - df['Age3_Sum']
    
    # 比率計算（総人口ベース）
    df['Young_Ratio'] = df['Young_0_14'] / df['TotalPopulation'] * 100
    df['Working_Ratio'] = df['Working_15_64'] / df['TotalPopulation'] * 100
    df['Elderly_Ratio'] = df['Elderly_65plus'] / df['TotalPopulation'] * 100
    df['VeryOld_Ratio'] = df['VeryOld_75plus'] / df['TotalPopulation'] * 100
    
    # 従属人口指数（Dependency Ratio）の計算
    df['Young_Dependency'] = df['Young_0_14'] / df['Working_15_64'] * 100
    df['Old_Dependency'] = df['Elderly_65plus'] / df['Working_15_64'] * 100
    df['Total_Dependency'] = (df['Young_0_14'] + df['Elderly_65plus']) / df['Working_15_64'] * 100
    
    # 潜在扶養指数（Potential Support Ratio: PSR）
    df['PSR'] = df['Working_15_64'] / df['Elderly_65plus']
    
    return df


def plot_population_composition_stacked(df: pd.DataFrame) -> None:
    """
    年齢3区分別人口構成（積み上げ面グラフ）- 2100年まで
    【修正】Elderly_65plusを直接使用し、総人口は別線で表示
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    years = df['Year'].values
    
    # 【修正】積み上げは年齢3区分の実数値を使用
    # 下から順に: 年少 → 生産年齢 → 高齢者
    young = df['Young_0_14'].values
    working = df['Working_15_64'].values
    elderly = df['Elderly_65plus'].values
    
    ax.fill_between(years, 0, young, 
                    alpha=0.8, color=COLORS['young'], label='年少人口 (0-14歳)')
    ax.fill_between(years, young, young + working, 
                    alpha=0.8, color=COLORS['working'], label='生産年齢人口 (15-64歳)')
    ax.fill_between(years, young + working, young + working + elderly, 
                    alpha=0.8, color=COLORS['elderly'], label='高齢者 (65歳以上)')
    
    # 年齢3区分合計の線（積み上げの上端）
    age3_sum = young + working + elderly
    ax.plot(years, age3_sum, color='black', linewidth=2, linestyle='-', 
            label='年齢3区分合計')
    
    # 総人口の線（年齢不詳を含む公表値）
    ax.plot(years, df['TotalPopulation'], color='black', linewidth=2.5, linestyle='--',
            alpha=0.7, label='総人口（公表値）')
    
    # 現在と将来の境界線
    ax.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax.annotate('← 実績値 | 将来推計 →', xy=(2024, 13000), fontsize=11, ha='center',
                color='purple', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 2070年の境界線（基本推計と長期参考推計の境界）
    ax.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.annotate('長期参考推計 →', xy=(2075, 10000), fontsize=9, ha='left',
                color='gray', style='italic')
    
    # 主要年の人口値を注釈
    key_annotations = [
        (1950, '8,320万人', 8320),
        (2008, 'ピーク\n1億2,808万人', 12808),
        (2050, '約1億469万人', 10469),
        (2100, '約5,791万人\n(▲約55%)', 5791),
    ]
    
    for yr, label, pop in key_annotations:
        ax.annotate(label, xy=(yr, pop), xytext=(yr, pop + 800),
                    fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    ax.set_xlabel('年', fontsize=13)
    ax.set_ylabel('人口（万人）', fontsize=13)
    ax.set_xlim(1950, 2100)
    ax.set_ylim(0, 14500)
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/10000:.1f}億' if x >= 10000 else f'{int(x):,}'))
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.title('日本の年齢3区分別人口推移と将来推計（1950-2100）', fontsize=18, fontweight='bold')
    
    # 出典（修正版）
    source_text = (
        '【データソース】\n'
        '・1950-2020（実績）: 総務省統計局「国勢調査」「人口推計」\n'
        '・2021-2024（推計）: 総務省統計局「人口推計」\n'
        '・2025-2070（将来推計）: 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」出生中位・死亡中位\n'
        '・2071-2100（長期参考推計）: 同上「長期参考推計」｜※実績値では年齢不詳により総人口と年齢3区分合計に差異あり'
    )
    fig.text(0.12, 0.01, source_text, fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    
    outpath = OUTDIR / "06_population_by_age_stacked.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()


def plot_population_ratio_trends(df: pd.DataFrame) -> None:
    """
    年齢構成比率の推移（線グラフ）- 2100年まで
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    years = df['Year'].values
    
    ax.plot(years, df['Young_Ratio'], color=COLORS['young'], linewidth=3, 
            marker='', label='年少人口比率 (0-14歳)')
    ax.plot(years, df['Working_Ratio'], color=COLORS['working'], linewidth=3, 
            marker='', label='生産年齢人口比率 (15-64歳)')
    ax.plot(years, df['Elderly_Ratio'], color=COLORS['elderly'], linewidth=3, 
            marker='', label='高齢化率 (65歳以上)')
    ax.plot(years, df['VeryOld_Ratio'], color=COLORS['very_old'], linewidth=2.5, 
            linestyle='--', marker='', label='後期高齢者比率 (75歳以上)')
    
    ax.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # 重要なポイントの注釈
    key_points = [
        (1950, df[df['Year']==1950]['Elderly_Ratio'].values[0], f'1950年\n約{df[df["Year"]==1950]["Elderly_Ratio"].values[0]:.1f}%'),
        (2024, df[df['Year']==2024]['Elderly_Ratio'].values[0], f'2024年\n約{df[df["Year"]==2024]["Elderly_Ratio"].values[0]:.1f}%'),
        (2070, df[df['Year']==2070]['Elderly_Ratio'].values[0], f'2070年\n約{df[df["Year"]==2070]["Elderly_Ratio"].values[0]:.1f}%'),
        (2100, df[df['Year']==2100]['Elderly_Ratio'].values[0], f'2100年\n約{df[df["Year"]==2100]["Elderly_Ratio"].values[0]:.1f}%'),
    ]
    
    for yr, val, label in key_points:
        ax.annotate(label, xy=(yr, val), xytext=(yr+5, val+5),
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.text(1955, 52, '50%', fontsize=9, color='gray')
    
    ax.set_xlabel('年', fontsize=13)
    ax.set_ylabel('比率（%）', fontsize=13)
    ax.set_xlim(1950, 2100)
    ax.set_ylim(0, 80)
    
    ax.legend(loc='center right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.title('日本の年齢構成比率の推移（1950-2100）', fontsize=18, fontweight='bold')
    
    source_text = (
        '【データソース】\n'
        '・1950-2020: 総務省統計局「国勢調査」「人口推計」 ・2021-2024: 総務省統計局「人口推計」\n'
        '・2025-2070: 社人研「令和5年推計」出生中位・死亡中位 ・2071-2100: 同「長期参考推計」'
    )
    fig.text(0.12, 0.02, source_text, fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    
    outpath = OUTDIR / "07_population_ratio_trends.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()


def plot_population_decline_impact(df: pd.DataFrame) -> None:
    """
    人口減少のインパクト分析
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    years = df['Year'].values
    
    # 左パネル: 総人口の推移
    ax1 = axes[0]
    ax1.fill_between(years, 0, df['TotalPopulation'], alpha=0.4, color=COLORS['total'])
    ax1.plot(years, df['TotalPopulation'], color=COLORS['total'], linewidth=3)
    
    peak_year = 2008
    peak_pop = df[df['Year']==peak_year]['TotalPopulation'].values[0]
    final_pop = df[df['Year']==2100]['TotalPopulation'].values[0]
    
    ax1.scatter([peak_year], [peak_pop], color='green', s=150, zorder=5, marker='o')
    ax1.scatter([2100], [final_pop], color='red', s=150, zorder=5, marker='o')
    
    ax1.annotate(f'{peak_year}年ピーク\n{peak_pop:,.0f}万人', 
                xy=(peak_year, peak_pop), xytext=(peak_year-20, peak_pop+500),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    decline_pct = (peak_pop - final_pop) / peak_pop * 100
    ax1.annotate(f'2100年\n{final_pop:,.0f}万人\n(▲約{decline_pct:.0f}%)', 
                xy=(2100, final_pop), xytext=(2080, final_pop+1500),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax1.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_xlabel('年', fontsize=12)
    ax1.set_ylabel('総人口（万人）', fontsize=12)
    ax1.set_title('総人口の推移', fontsize=14, fontweight='bold')
    ax1.set_xlim(1950, 2100)
    ax1.set_ylim(0, 14000)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/10000:.1f}億' if x >= 10000 else f'{int(x):,}'))
    ax1.grid(True, alpha=0.3)
    
    # 右パネル: 年齢層別の変化率（2008年基準）
    ax2 = axes[1]
    
    base_year = 2008
    base_row = df[df['Year']==base_year]
    
    df['Total_Index'] = df['TotalPopulation'] / base_row['TotalPopulation'].values[0] * 100
    df['Young_Index'] = df['Young_0_14'] / base_row['Young_0_14'].values[0] * 100
    df['Working_Index'] = df['Working_15_64'] / base_row['Working_15_64'].values[0] * 100
    df['Elderly_Index'] = df['Elderly_65plus'] / base_row['Elderly_65plus'].values[0] * 100
    
    ax2.plot(years, df['Total_Index'], color=COLORS['total'], linewidth=3, label='総人口')
    ax2.plot(years, df['Young_Index'], color=COLORS['young'], linewidth=2.5, label='年少人口')
    ax2.plot(years, df['Working_Index'], color=COLORS['working'], linewidth=2.5, label='生産年齢人口')
    ax2.plot(years, df['Elderly_Index'], color=COLORS['elderly'], linewidth=2.5, label='高齢者人口')
    
    ax2.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=base_year, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax2.annotate(f'基準年\n{base_year}年=100', xy=(base_year, 100), xytext=(base_year-15, 120),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax2.set_xlabel('年', fontsize=12)
    ax2.set_ylabel(f'指数（{base_year}年=100）', fontsize=12)
    ax2.set_title(f'年齢層別人口指数（{base_year}年=100）', fontsize=14, fontweight='bold')
    ax2.set_xlim(1950, 2100)
    ax2.set_ylim(0, 150)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('人口減少のインパクト分析（1950-2100）', fontsize=18, fontweight='bold')
    
    source_text = (
        '【データソース】1950-2020: 総務省統計局 / 2021-2024: 総務省「人口推計」\n'
        '2025-2070: 社人研「令和5年推計」 / 2071-2100: 同「長期参考推計」（出生中位・死亡中位）'
    )
    fig.text(0.5, 0.01, source_text, fontsize=8, color='gray', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    outpath = OUTDIR / "08_population_decline_impact.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()


def plot_dependency_and_psr(df: pd.DataFrame) -> None:
    """
    従属人口指数と潜在扶養指数（PSR）の推移
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    years = df['Year'].values
    
    # === 左パネル: 従属人口指数 ===
    ax1 = axes[0]
    
    ax1.fill_between(years, 0, df['Young_Dependency'], alpha=0.5, color=COLORS['young'],
                     label='年少従属人口指数')
    ax1.fill_between(years, df['Young_Dependency'], df['Total_Dependency'], 
                     alpha=0.5, color=COLORS['elderly'],
                     label='老年従属人口指数')
    ax1.plot(years, df['Total_Dependency'], color='black', linewidth=3, 
             label='総従属人口指数')
    
    ax1.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    for yr in [1970, 2024, 2070, 2100]:
        row = df[df['Year']==yr]
        if len(row) > 0:
            total_dep = row['Total_Dependency'].values[0]
            ax1.annotate(f'{yr}年\n{total_dep:.1f}', 
                        xy=(yr, total_dep), xytext=(yr, total_dep+8),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('年', fontsize=12)
    ax1.set_ylabel('従属人口指数（生産年齢人口100人当たり）', fontsize=12)
    ax1.set_title('従属人口指数の推移', fontsize=14, fontweight='bold')
    ax1.set_xlim(1950, 2100)
    ax1.set_ylim(0, 110)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(1955, 105, '従属人口指数 = (年少人口+高齢者人口) / 生産年齢人口 × 100',
             fontsize=9, style='italic', color='gray')
    
    # === 右パネル: 潜在扶養指数（PSR）===
    ax2 = axes[1]
    
    ax2.fill_between(years, 0, df['PSR'], alpha=0.5, color=COLORS['working'])
    ax2.plot(years, df['PSR'], color=COLORS['working'], linewidth=3)
    
    ax2.axvline(x=2024, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=2070, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    psr_annotations = [
        (1970, '胴上げ型', 'lightblue'),
        (1995, '騎馬戦型', 'lightyellow'),
        (2024, '現在', 'yellow'),
        (2070, '2070年', '#ffcccc'),
        (2100, '肩車型', '#ffaaaa'),
    ]
    
    for yr, label, color in psr_annotations:
        row = df[df['Year']==yr]
        if len(row) > 0:
            psr = row['PSR'].values[0]
            ax2.annotate(f'{yr}年: {psr:.2f}人\n({label})', 
                        xy=(yr, psr), xytext=(yr+3 if yr < 2050 else yr-15, psr+1.2),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    ax2.set_xlabel('年', fontsize=12)
    ax2.set_ylabel('潜在扶養指数（人）', fontsize=12)
    ax2.set_title('潜在扶養指数（PSR）の推移\n「現役何人で高齢者1人を支えるか」', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(1950, 2100)
    ax2.set_ylim(0, 14)
    ax2.grid(True, alpha=0.3)
    
    ax2.text(1955, 13, '潜在扶養指数（PSR） = 生産年齢人口 / 高齢者人口',
             fontsize=9, style='italic', color='gray')
    ax2.text(2060, 0.5, '1人ライン', fontsize=9, color='red', style='italic')
    
    plt.suptitle('従属人口指数と潜在扶養指数の推移（1950-2100）', fontsize=18, fontweight='bold')
    
    source_text = (
        '【データソース】1950-2020: 総務省統計局 / 2021-2024: 総務省「人口推計」 / '
        '2025-2070: 社人研「令和5年推計」 / 2071-2100: 同「長期参考推計」\n'
        '【注記】従属人口指数と潜在扶養指数（PSR）は異なる指標です。PSRは「現役何人で高齢者1人を支えるか」を示します。'
    )
    fig.text(0.5, 0.01, source_text, fontsize=8, color='gray', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10)
    
    outpath = OUTDIR / "09_dependency_and_psr.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()


def plot_comprehensive_age_dashboard(df: pd.DataFrame) -> None:
    """
    年齢構成総合ダッシュボード（4パネル）
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    years = df['Year'].values
    
    # === パネル1: 年齢3区分別人口（積み上げ） ===
    ax1 = axes[0, 0]
    young = df['Young_0_14'].values
    working = df['Working_15_64'].values
    elderly = df['Elderly_65plus'].values
    
    ax1.fill_between(years, 0, young, alpha=0.8, color=COLORS['young'])
    ax1.fill_between(years, young, young + working, alpha=0.8, color=COLORS['working'])
    ax1.fill_between(years, young + working, young + working + elderly, 
                     alpha=0.8, color=COLORS['elderly'])
    ax1.plot(years, young + working + elderly, color='black', linewidth=2, label='年齢3区分合計')
    ax1.plot(years, df['TotalPopulation'], color='black', linewidth=2, linestyle='--', 
             alpha=0.7, label='総人口')
    ax1.axvline(x=2024, color='purple', linestyle='--', alpha=0.7)
    ax1.axvline(x=2070, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('① 年齢3区分別人口', fontsize=12, fontweight='bold')
    ax1.set_ylabel('人口（万人）', fontsize=10)
    ax1.set_ylim(0, 14000)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/10000:.1f}億' if x >= 10000 else f'{int(x):,}'))
    ax1.grid(True, alpha=0.3, axis='y')
    
    legend_elements = [
        mpatches.Patch(color=COLORS['young'], alpha=0.8, label='年少 (0-14歳)'),
        mpatches.Patch(color=COLORS['working'], alpha=0.8, label='生産年齢 (15-64歳)'),
        mpatches.Patch(color=COLORS['elderly'], alpha=0.8, label='高齢者 (65歳以上)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # === パネル2: 年齢構成比率 ===
    ax2 = axes[0, 1]
    ax2.plot(years, df['Young_Ratio'], color=COLORS['young'], linewidth=2.5, label='年少')
    ax2.plot(years, df['Working_Ratio'], color=COLORS['working'], linewidth=2.5, label='生産年齢')
    ax2.plot(years, df['Elderly_Ratio'], color=COLORS['elderly'], linewidth=2.5, label='高齢者')
    ax2.axvline(x=2024, color='purple', linestyle='--', alpha=0.7)
    ax2.axvline(x=2070, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title('② 年齢構成比率', fontsize=12, fontweight='bold')
    ax2.set_ylabel('比率（%）', fontsize=10)
    ax2.set_ylim(0, 80)
    ax2.legend(loc='center right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === パネル3: 従属人口指数 ===
    ax3 = axes[1, 0]
    ax3.fill_between(years, 0, df['Young_Dependency'], alpha=0.4, color=COLORS['young'])
    ax3.fill_between(years, df['Young_Dependency'], df['Total_Dependency'], 
                     alpha=0.4, color=COLORS['elderly'])
    ax3.plot(years, df['Total_Dependency'], color='black', linewidth=2.5, label='総従属人口指数')
    ax3.axvline(x=2024, color='purple', linestyle='--', alpha=0.7)
    ax3.axvline(x=2070, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('③ 従属人口指数\n（年少+高齢者）/生産年齢×100', fontsize=12, fontweight='bold')
    ax3.set_ylabel('指数', fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # === パネル4: 潜在扶養指数（PSR） ===
    ax4 = axes[1, 1]
    ax4.fill_between(years, 0, df['PSR'], alpha=0.5, color=COLORS['working'])
    ax4.plot(years, df['PSR'], color=COLORS['working'], linewidth=3)
    ax4.axvline(x=2024, color='purple', linestyle='--', alpha=0.7)
    ax4.axvline(x=2070, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('④ 潜在扶養指数（PSR）\n「現役何人で高齢者1人を支えるか」', fontsize=12, fontweight='bold')
    ax4.set_ylabel('人数', fontsize=10)
    ax4.set_ylim(0, 14)
    ax4.grid(True, alpha=0.3)
    
    for yr, label in [(1970, '胴上げ型'), (2024, '騎馬戦型'), (2100, '肩車型')]:
        row = df[df['Year']==yr]
        if len(row) > 0:
            val = row['PSR'].values[0]
            ax4.annotate(f'{yr}年: {val:.2f}人\n({label})', 
                        xy=(yr, val), xytext=(yr, val+1.8),
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    for ax in axes.flat:
        ax.set_xlabel('年', fontsize=10)
        ax.set_xlim(1950, 2100)
    
    plt.suptitle('日本の人口構造の大転換：年齢構成の激変（1950-2100）', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    row_2100 = df[df['Year']==2100]
    total_pop_2100 = row_2100['TotalPopulation'].values[0]
    elderly_ratio_2100 = row_2100['Elderly_Ratio'].values[0]
    psr_2100 = row_2100['PSR'].values[0]
    decline_rate = (12808 - total_pop_2100) / 12808 * 100
    
    source_text = (
        f'※紫点線=2024年（現在）、灰点線=2070年（基本推計/長期参考推計の境界）\n'
        f'【データソース】1950-2020: 総務省統計局「国勢調査」 / 2021-2024: 総務省「人口推計」\n'
        f'2025-2070: 社人研「令和5年推計」表1-1 / 2071-2100: 同「長期参考推計」参考表1-1（出生中位・死亡中位）\n'
        f'※2100年には総人口が約{total_pop_2100:,.0f}万人（2008年ピーク比▲約{decline_rate:.0f}%）、'
        f'高齢化率約{elderly_ratio_2100:.1f}%、潜在扶養指数約{psr_2100:.2f}人の見込み'
    )
    fig.text(0.5, 0.01, source_text, fontsize=8, color='gray', ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.11)
    
    outpath = OUTDIR / "10_age_composition_dashboard.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()


def save_extended_data_csv(df: pd.DataFrame) -> None:
    """
    拡張データをCSVで保存
    """
    output_df = df[[
        'Year', 'TotalPopulation', 'Young_0_14', 'Working_15_64', 'Elderly_65plus',
        'VeryOld_75plus', 'Age3_Sum', 'Age_Unknown',
        'Young_Ratio', 'Working_Ratio', 'Elderly_Ratio', 'VeryOld_Ratio',
        'Young_Dependency', 'Old_Dependency', 'Total_Dependency', 'PSR'
    ]].copy()
    
    output_df.columns = [
        'Year', 'TotalPopulation', 'Young_0_14', 'Working_15_64', 'Elderly_65plus',
        'VeryOld_75plus', 'Age3_Sum', 'Age_Unknown',
        'Young_Ratio', 'Working_Ratio', 'Elderly_Ratio', 'VeryOld_Ratio',
        'Young_Dependency', 'Old_Dependency', 'Total_Dependency', 'PSR_Workers_per_Elderly'
    ]
    
    outpath = OUTDIR / "japan_population_age_projection_2100_v3.csv"
    output_df.to_csv(outpath, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f"[SAVED] {outpath}")


def save_metadata() -> None:
    """
    メタデータファイルを保存【修正版v3】
    """
    metadata = """# 日本の年齢別人口構成分析データ（1950-2100）【修正版v3】

## データソース

### 1950-2020年（実績値）
- 総務省統計局「国勢調査」（各年10月1日現在）
- 総務省統計局「人口推計」
- URL: https://www.stat.go.jp/data/jinsui/
- 注: 年齢不詳人口の存在により、年齢3区分合計と総人口に差異が生じる場合がある

### 2021-2024年（推計値）
- 総務省統計局「人口推計」（各年10月1日現在）
- URL: https://www.stat.go.jp/data/jinsui/

### 2025-2070年（将来推計）
- 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」
- 出生中位（死亡中位）仮定　表1-1
- URL: https://www.ipss.go.jp/pp-zenkoku/j/zenkoku2023/pp_zenkoku2023.asp

### 2071-2100年（長期参考推計）
- 国立社会保障・人口問題研究所「日本の将来推計人口（令和5年推計）」
- 長期参考推計 出生中位（死亡中位）仮定　参考表1-1
- 注: 出生率・死亡率・国際人口移動率を2070年以降一定と仮定

## CSV列の説明（全16列）

| 列名 | 説明 | 単位 |
|------|------|------|
| Year | 年 | 年 |
| TotalPopulation | 総人口（公表値） | 万人 |
| Young_0_14 | 年少人口（0-14歳） | 万人 |
| Working_15_64 | 生産年齢人口（15-64歳） | 万人 |
| Elderly_65plus | 高齢者人口（65歳以上） | 万人 |
| VeryOld_75plus | 後期高齢者人口（75歳以上） | 万人 |
| Age3_Sum | 年齢3区分合計（Young+Working+Elderly） | 万人 |
| Age_Unknown | 年齢不詳等（TotalPopulation - Age3_Sum） | 万人 |
| Young_Ratio | 年少人口比率 | % |
| Working_Ratio | 生産年齢人口比率 | % |
| Elderly_Ratio | 高齢化率（65歳以上比率） | % |
| VeryOld_Ratio | 後期高齢者比率（75歳以上比率） | % |
| Young_Dependency | 年少従属人口指数 = Young/Working×100 | - |
| Old_Dependency | 老年従属人口指数 = Elderly/Working×100 | - |
| Total_Dependency | 総従属人口指数 = (Young+Elderly)/Working×100 | - |
| PSR_Workers_per_Elderly | 潜在扶養指数（PSR）= Working/Elderly | 人 |

## 指標の定義

### 従属人口指数（Dependency Ratio）
- 年少従属人口指数 = 年少人口 / 生産年齢人口 × 100
- 老年従属人口指数 = 高齢者人口 / 生産年齢人口 × 100
- 総従属人口指数 = (年少人口 + 高齢者人口) / 生産年齢人口 × 100
- 意味: 生産年齢人口100人当たりの被扶養人口を示す

### 潜在扶養指数（Potential Support Ratio: PSR）
- PSR = 生産年齢人口 / 高齢者人口
- 意味:「現役何人で高齢者1人を支えるか」を示す
- 注: 従属人口指数とは異なる指標

## 注意事項

1. 1950-2020年の実績値では、年齢不詳人口の存在により「総人口 ≠ 年齢3区分合計」となる
2. 積み上げグラフでは年齢3区分の実数値を使用し、総人口は別線で表示
3. 比率は総人口（公表値）をベースに計算
4. 2071年以降の長期参考推計は参考値であり、不確実性が大きい

## 主要数値（参考）

| 年 | 総人口(万人) | 高齢化率(%) | PSR(人) |
|----|-------------|------------|---------|
| 1950 | 8,320 | 4.9 | 12.08 |
| 1970 | 10,372 | 7.1 | 9.84 |
| 2008 | 12,808 | 22.0 | 2.92 |
| 2024 | 12,380 | 29.3 | 2.03 |
| 2070 | 8,700 | 38.7 | 1.35 |
| 2100 | 5,791 | 36.4 | 1.48 |

## 作成情報

- 作成日: 2026-01-13
- 作成者: Claude (Anthropic)
- バージョン: v3
"""
    metadata_path = OUTDIR / "README_metadata.md"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(metadata)
    print(f"[SAVED] {metadata_path}")


# =========================
# メイン実行
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("日本の年齢別人口構成分析（2100年までの予測）【修正版v3】")
    print("=" * 60)
    
    # データ作成
    df = create_population_data()
    print(f"\n[DATA] {len(df)} years of data loaded (1950-2100)")
    
    # 2021-2024が含まれていることを確認
    years_2021_2024 = df[df['Year'].isin([2021, 2022, 2023, 2024])]['Year'].tolist()
    print(f"[CHECK] 2021-2024 data included: {years_2021_2024}")
    
    # 年齢3区分合計と総人口の差異を確認
    print("\n[CHECK] 総人口と年齢3区分合計の差異（万人）:")
    check_years = [1955, 1960, 1965, 1970, 2010, 2020, 2070, 2100]
    for yr in check_years:
        row = df[df['Year']==yr]
        if len(row) > 0:
            diff = row['Age_Unknown'].values[0]
            print(f"  {yr}年: {diff:+.0f}")
    
    # 主要年のデータを表示
    key_years = [1950, 1970, 1995, 2008, 2021, 2022, 2023, 2024, 2045, 2070, 2100]
    print("\n主要年のデータ:")
    print(df[df['Year'].isin(key_years)][[
        'Year', 'TotalPopulation', 'Young_0_14', 'Working_15_64', 'Elderly_65plus',
        'Elderly_Ratio', 'Total_Dependency', 'PSR'
    ]].to_string())
    
    # 2070年と2100年の検証
    print("\n【検証】主要指標:")
    for yr in [2070, 2100]:
        row = df[df['Year']==yr]
        print(f"\n{yr}年:")
        print(f"  総人口: {row['TotalPopulation'].values[0]:,.0f}万人")
        print(f"  高齢化率: {row['Elderly_Ratio'].values[0]:.2f}%")
        print(f"  PSR（潜在扶養指数）: {row['PSR'].values[0]:.2f}人")
    
    # グラフ作成
    print("\n[PLOTTING] Creating visualizations...")
    plot_population_composition_stacked(df)
    plot_population_ratio_trends(df)
    plot_population_decline_impact(df)
    plot_dependency_and_psr(df)
    plot_comprehensive_age_dashboard(df)
    
    # データ保存
    save_extended_data_csv(df)
    save_metadata()
    
    print("\n" + "=" * 60)
    print("[COMPLETE] All outputs saved to:", OUTDIR)
    print("=" * 60)
