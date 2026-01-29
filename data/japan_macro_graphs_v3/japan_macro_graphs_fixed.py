# -*- coding: utf-8 -*-
"""
japan_macro_graphs_fixed.py

人口・GDP・為替（REER）を「見せる」ための統合グラフ生成スクリプト（修正版）

狙い：
- 2軸の「見た目の連動」を避け、指数化・対数分解などで比較可能にする
- 「1995年で交差＝発見」にならないよう、注釈を正しくする
- 5年平均/不均等年次の成長率を、年差(dt)込みの年率(CAGR)で計算する
- 「GDP成長 = 労働投入（人口）× 生産性（1人当たり/労働者当たり）」の分解を可視化する

修正ポイント:
- 1995年の「交差」は正規化の数学的結果であることを明示
- 不均等年次データのCAGR計算にdt（年差）を正しく使用
- ログ分解で成長寄与を正確に計算
- 名目GDP vs 実質GDPの注意喚起を追加

注意：
- GDPは名目（兆円）。実質GDPを使うならデータ差し替えが必要です。
- 為替はREER（実効実質為替レート、2020=100想定）ほか、年平均USD/JPYは近似です。
- 一部年の総人口は（与えられた点から）線形補間で推計し、ドル建て1人当たりGDPの概算に使います。
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------
# 出力先
# ---------------------------------
# スクリプトと同じ場所に出力（移植性向上）
OUTDIR = Path(__file__).resolve().parent / "output_fixed"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# 日本語フォント（任意）
# ---------------------------------
def _setup_japanese_font() -> None:
    """japanize_matplotlibが無い環境でも日本語が豆腐になりにくいようにフォントを探す"""
    try:
        import japanize_matplotlib  # type: ignore
        return
    except Exception:
        pass
    try:
        from matplotlib import font_manager
        candidates = [
            "IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
            "TakaoGothic", "Yu Gothic", "Hiragino Sans", "MS Gothic"
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = [name, "sans-serif"]
                return
    except Exception:
        pass
    # 最終フォールバック
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

_setup_japanese_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

COLORS = {
    'gdp': '#d62728',
    'working': '#1f77b4',
    'total': '#2ca02c',
    'productivity': '#9467bd',
    'reer': '#2ca02c',
    'high_growth': '#ffffb3',
    'lost_decades': '#ffcccc',
    'bubble': '#cce5ff',
}

# ---------------------------------
# データ（元資料のスクリプトから移植）
#   単位：
#   - TotalPop / WorkingPop: 万人
#   - GDP: 兆円（名目）
# ---------------------------------
DATA_POP_GDP = {
    1950: (8320, 4966, 4.0),
    1955: (8928, 5517, 8.4),
    1960: (9342, 6047, 16.0),
    1965: (9828, 6744, 32.9),
    1970: (10372, 7212, 73.3),
    1973: (10650, 7420, 112.5),
    1975: (11194, 7581, 148.3),
    1980: (11706, 7883, 240.2),
    1985: (12105, 8251, 324.5),
    1990: (12361, 8590, 442.8),
    1991: (12400, 8614, 469.4),
    1995: (12557, 8717, 501.7),
    2000: (12693, 8622, 509.9),
    2005: (12777, 8442, 505.3),
    2008: (12808, 8230, 505.1),
    2010: (12806, 8103, 500.4),
    2015: (12709, 7728, 532.2),
    2020: (12571, 7449, 539.0),
    2024: (12380, 7360, 610.0),
}

# 年, 生産年齢人口(万人), 名目GDP(兆円), REER(2020=100), NEER(2020=100), USD/JPY(年平均)
DATA_TRIPLE = {
    1964: (6500, 29.5, 55, 57, 360),
    1970: (7212, 73.3, 68, 65, 360),
    1971: (7280, 80.7, 72, 70, 350),
    1973: (7420, 112.5, 85, 82, 272),
    1975: (7581, 148.3, 82, 78, 297),
    1980: (7883, 240.2, 90, 88, 227),
    1985: (8251, 324.5, 98, 100, 238),
    1988: (8450, 387.0, 125, 130, 128),
    1990: (8590, 442.8, 110, 118, 145),
    1991: (8614, 469.4, 115, 122, 135),
    1993: (8680, 476.1, 135, 140, 111),
    1995: (8717, 501.7, 150, 160, 94),
    1998: (8650, 500.2, 125, 130, 130),
    2000: (8622, 509.9, 135, 138, 108),
    2005: (8442, 505.3, 110, 108, 110),
    2007: (8312, 518.3, 105, 102, 117),
    2008: (8230, 505.1, 100, 98, 103),
    2010: (8103, 500.4, 110, 112, 88),
    2011: (8034, 491.4, 115, 118, 80),
    2012: (7948, 495.0, 110, 112, 80),
    2013: (7901, 507.5, 100, 98, 98),
    2015: (7728, 532.2, 85, 80, 121),
    2018: (7545, 556.2, 78, 75, 110),
    2020: (7449, 539.0, 100, 100, 107),
    2022: (7421, 560.7, 75, 72, 131),
    2024: (7360, 610.0, 68, 70, 151),
}

# ---------------------------------
# ユーティリティ
# ---------------------------------
def make_df_pop_gdp() -> pd.DataFrame:
    years = sorted(DATA_POP_GDP.keys())
    df = pd.DataFrame({
        "Year": years,
        "TotalPop": [DATA_POP_GDP[y][0] for y in years],
        "WorkingPop": [DATA_POP_GDP[y][1] for y in years],
        "GDP": [DATA_POP_GDP[y][2] for y in years],
    })
    # 1人あたりではなく「労働者（生産年齢人口）あたり」GDP（名目）: 万円/人（概算）
    df["GDPperWorker"] = df["GDP"] / df["WorkingPop"] * 10000  # (兆円)/(万人) = 億円/人 → ×10000で万円/人
    return df

def make_df_triple() -> pd.DataFrame:
    years = sorted(DATA_TRIPLE.keys())
    df = pd.DataFrame({
        "Year": years,
        "WorkingPop": [DATA_TRIPLE[y][0] for y in years],
        "GDP": [DATA_TRIPLE[y][1] for y in years],
        "REER": [DATA_TRIPLE[y][2] for y in years],
        "NEER": [DATA_TRIPLE[y][3] for y in years],
        "USDJPY": [DATA_TRIPLE[y][4] for y in years],
    })
    df["GDPperWorker"] = df["GDP"] / df["WorkingPop"] * 10000
    return df

def index_series(s: pd.Series, base_year: int, years: pd.Series) -> pd.Series:
    base_val = float(s.loc[years == base_year].iloc[0])
    return s / base_val * 100.0

def annualized_cagr(series: pd.Series, years: pd.Series) -> pd.Series:
    """不均等間隔の年次データに対して、区間ごとの年率CAGRを返す（前年差 dt を使用）"""
    s = series.astype(float)
    y = years.astype(float)
    ratio = s / s.shift(1)
    dt = y - y.shift(1)
    cagr = ratio ** (1.0 / dt) - 1.0
    return cagr

# ---------------------------------
# 図1：指数（1995=100）で人口・GDP・生産性（GDP/労働者）を比較
# ---------------------------------
def fig_index_pop_gdp_productivity(df: pd.DataFrame, base_year: int = 1995) -> None:
    years = df["Year"]
    idx_work = index_series(df["WorkingPop"], base_year, years)
    idx_gdp = index_series(df["GDP"], base_year, years)
    idx_prod = index_series(df["GDPperWorker"], base_year, years)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(years, idx_work, marker="o", linewidth=4, markersize=8, 
            color=COLORS['working'], label="生産年齢人口（指数）")
    ax.plot(years, idx_gdp, marker="s", linewidth=4, markersize=8,
            color=COLORS['gdp'], label="名目GDP（指数）")
    ax.plot(years, idx_prod, marker="^", linewidth=4, markersize=8,
            color=COLORS['productivity'], label="GDP/生産年齢人口（指数）")

    ax.axvspan(1955, 1973, alpha=0.25, color=COLORS['high_growth'], label="高度経済成長期（1955-1973）")
    ax.axvspan(1991, 2024, alpha=0.15, color=COLORS['lost_decades'], label="1990年代以降（参考）")
    ax.axvline(base_year, linestyle="--", linewidth=2, color='purple', alpha=0.7)
    ax.axhline(100, linestyle="-", linewidth=1, color='black', alpha=0.5)

    ax.set_title("人口・GDP・生産性の関係（指数化：1995年=100）\n※1995年で「交差」するのは正規化の数学的結果", 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("年", fontsize=14)
    ax.set_ylabel("指数（1995年=100）", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=11, frameon=True, loc='upper left')
    ax.set_xlim(1948, 2028)

    # 注釈
    ax.annotate('1995年基準点\n（全系列が100に交差）', 
                xy=(1995, 100), xytext=(1975, 130),
                fontsize=11, color='purple', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    note = "注：GDPは名目（兆円）。交差は基準年選択の結果であり、因果関係の証拠ではない。\n成長/停滞の議論は実質GDP・1人当たり/労働者当たりでも確認推奨。"
    fig.text(0.5, 0.01, note, ha="center", fontsize=10, color="gray")

    outpath = OUTDIR / "01_index_pop_gdp_productivity.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {outpath}")

# ---------------------------------
# 図2：GDP成長の分解（GDP = 労働投入 × 生産性）
#   - 期間ごとに、倍率・CAGR・ログ分解の寄与を可視化
# ---------------------------------
def fig_decomposition_periods(df: pd.DataFrame) -> None:
    periods = [
        ("高度経済成長期", 1955, 1973),
        ("安定成長期", 1973, 1991),
        ("1990年代以降", 1991, 2024),
    ]

    rows = []
    dfi = df.set_index("Year")

    for label, y0, y1 in periods:
        g0, g1 = float(dfi.loc[y0, "GDP"]), float(dfi.loc[y1, "GDP"])
        p0, p1 = float(dfi.loc[y0, "WorkingPop"]), float(dfi.loc[y1, "WorkingPop"])
        prod0, prod1 = float(dfi.loc[y0, "GDPperWorker"]), float(dfi.loc[y1, "GDPperWorker"])
        dt = y1 - y0

        g_factor = g1 / g0
        p_factor = p1 / p0
        prod_factor = prod1 / prod0

        g_cagr = g_factor ** (1 / dt) - 1
        p_cagr = p_factor ** (1 / dt) - 1
        prod_cagr = prod_factor ** (1 / dt) - 1

        # ログ分解（加法分解）
        lg = math.log(g_factor)
        lp = math.log(p_factor)
        lprod = math.log(prod_factor)

        # 寄与比（lgがほぼ0の時の暴れを避ける）
        if abs(lg) < 1e-12:
            share_p = np.nan
            share_prod = np.nan
        else:
            share_p = lp / lg
            share_prod = lprod / lg

        rows.append({
            "期間": f"{label}\n({y0}-{y1})",
            "期間ラベル": label,
            "開始年": y0,
            "終了年": y1,
            "年数": dt,
            "GDP倍率": g_factor,
            "人口倍率": p_factor,
            "生産性倍率": prod_factor,
            "GDP_CAGR": g_cagr,
            "人口_CAGR": p_cagr,
            "生産性_CAGR": prod_cagr,
            "人口寄与(ログ比)": share_p,
            "生産性寄与(ログ比)": share_prod,
        })

    tab = pd.DataFrame(rows)

    # 可視化：CAGRバー
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    x = np.arange(len(tab))
    width = 0.25

    bars1 = ax1.bar(x - width, tab["人口_CAGR"] * 100, width, 
                    color=COLORS['working'], alpha=0.9, label="人口（生産年齢）CAGR")
    bars2 = ax1.bar(x, tab["生産性_CAGR"] * 100, width, 
                    color=COLORS['productivity'], alpha=0.9, label="生産性（GDP/労働者）CAGR")
    bars3 = ax1.bar(x + width, tab["GDP_CAGR"] * 100, width, 
                    color=COLORS['gdp'], alpha=0.9, label="GDP（名目）CAGR")

    # 数値表示
    for bars, color in [(bars1, COLORS['working']), (bars2, COLORS['productivity']), (bars3, COLORS['gdp'])]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', 
                        fontsize=10, fontweight='bold', color=color)

    ax1.axhline(0, linewidth=1, color='black')
    ax1.set_title("期間別：年平均成長率（CAGR）\n※不均等年次はdt補正済み", fontsize=15, fontweight='bold')
    ax1.set_ylabel("年率（%）", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tab["期間"], fontsize=11)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(-2, 18)
    ax1.grid(True, axis="y", alpha=0.3)

    # 可視化：ログ寄与（割合）
    bars_p = ax2.bar(x - width/2, tab["人口寄与(ログ比)"] * 100, width, 
                     color=COLORS['working'], alpha=0.9, label="人口の寄与（ログ比）")
    bars_prod = ax2.bar(x + width/2, tab["生産性寄与(ログ比)"] * 100, width, 
                        color=COLORS['productivity'], alpha=0.9, label="生産性の寄与（ログ比）")
    
    # 数値表示
    for bar in bars_p:
        h = bar.get_height()
        if not np.isnan(h):
            ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['working'])
    for bar in bars_prod:
        h = bar.get_height()
        if not np.isnan(h):
            ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['productivity'])

    ax2.axhline(0, linewidth=1, color='black')
    ax2.axhline(100, linewidth=1, linestyle='--', color='gray', alpha=0.5)
    ax2.set_title("GDP成長の内訳（ログ分解：寄与比）\n※人口寄与+生産性寄与=100%", fontsize=15, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tab["期間"], fontsize=11)
    ax2.set_ylabel("寄与（%）", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=10, loc='upper right')

    note = "注：GDP = 人口（労働投入）× 生産性（GDP/労働者）の分解。倍率は掛け算、寄与はログ分解（加法）。\nGDPは名目のため、物価変動を含む。実質分析には実質GDP使用推奨。"
    fig.text(0.5, -0.02, note, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    outpath = OUTDIR / "02_decomposition_periods.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {outpath}")

    # 表もCSVで保存
    tab_out = OUTDIR / "02_decomposition_table.csv"
    tab.to_csv(tab_out, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {tab_out}")

# ---------------------------------
# 図3：人口・GDP・REER（三指標）指数化（1995=100）
# ---------------------------------
def fig_triple_index(df: pd.DataFrame, base_year: int = 1995) -> None:
    years = df["Year"]
    idx_work = index_series(df["WorkingPop"], base_year, years)
    idx_gdp = index_series(df["GDP"], base_year, years)
    idx_reer = index_series(df["REER"], base_year, years)

    peak_work_idx = df["WorkingPop"].idxmax()
    peak_work_year = int(df.loc[peak_work_idx, "Year"])
    peak_reer_idx = df["REER"].idxmax()
    peak_reer_year = int(df.loc[peak_reer_idx, "Year"])
    max_gdp_idx = df["GDP"].idxmax()
    max_gdp_year = int(df.loc[max_gdp_idx, "Year"])

    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 背景
    ax.axvspan(1964, 1973, alpha=0.25, color=COLORS['high_growth'])
    ax.axvspan(1991, 2024, alpha=0.15, color=COLORS['lost_decades'])
    
    ax.plot(years, idx_work, marker="o", linewidth=5, markersize=10,
            color=COLORS['working'], label="生産年齢人口（指数）")
    ax.plot(years, idx_gdp, marker="s", linewidth=5, markersize=10,
            color=COLORS['gdp'], label="名目GDP（指数）")
    ax.plot(years, idx_reer, marker="^", linewidth=5, markersize=10,
            color=COLORS['reer'], label="実質実効為替レート（REER, 指数）")

    ax.axvline(base_year, linestyle="--", linewidth=2, color='purple', alpha=0.7)
    ax.axhline(100, linestyle="-", linewidth=1.5, color='black', alpha=0.5)
    
    # 1995年ピークマーカー
    ax.plot(1995, 100, 'o', color='purple', markersize=20, 
            markeredgecolor='black', markeredgewidth=3, zorder=5)

    ax.set_title("人口・GDP・為替（REER）の比較（指数化：1995年=100）\n※1995年に「交差」するのは正規化の数学的結果であり、因果関係の発見ではない", 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("年", fontsize=14)
    ax.set_ylabel("指数（1995年=100）", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=12, frameon=True, loc='upper left')
    ax.set_xlim(1960, 2028)
    # 1960年代のGDP指数が小さいため、下限を0にしてデータ欠落に見えないようにする
    ax.set_ylim(0, 130)

    # ピーク注釈
    ax.annotate(f"人口ピーク: {peak_work_year}", 
                xy=(peak_work_year, idx_work.iloc[peak_work_idx]),
                xytext=(peak_work_year-15, 115),
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS['working']),
                fontsize=11, fontweight='bold', color=COLORS['working'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.annotate(f"REERピーク: {peak_reer_year}", 
                xy=(peak_reer_year, idx_reer.iloc[peak_reer_idx]),
                xytext=(peak_reer_year-18, 125),
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS['reer']),
                fontsize=11, fontweight='bold', color=COLORS['reer'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.annotate(f"名目GDP最大: {max_gdp_year}", 
                xy=(max_gdp_year, idx_gdp.iloc[max_gdp_idx]),
                xytext=(max_gdp_year-15, 130),
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS['gdp']),
                fontsize=11, fontweight='bold', color=COLORS['gdp'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 時代注釈
    ax.annotate('【高度経済成長期】\n人口↑ GDP↑ 円↑', 
                xy=(1968, 55), fontsize=13, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLORS['high_growth'], 
                         edgecolor='green', linewidth=2))
    
    ax.annotate('【1990年代以降】\n人口↓ GDP低成長 円↓', 
                xy=(2010, 75), fontsize=13, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=COLORS['lost_decades'], 
                         edgecolor='red', linewidth=2))

    note = "注：名目GDPはインフレ影響を含む。為替の議論はREER/NEERの定義・基準年も明記推奨。\n「同時ピーク」は統計的相関の可能性を示すが、因果関係の証明ではない。"
    fig.text(0.5, 0.01, note, ha="center", fontsize=10, color="gray")

    outpath = OUTDIR / "03_triple_index_pop_gdp_reer.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {outpath}")

# ---------------------------------
# 図4：ドル建て1人当たりGDP（概算）
#   - 総人口はDATA_POP_GDPから線形補間で推計
# ---------------------------------
def fig_usd_per_capita(df_triple: pd.DataFrame, df_pop_gdp: pd.DataFrame) -> None:
    # 総人口の補間（万人）
    pop_years = df_pop_gdp["Year"].values.astype(float)
    pop_vals = df_pop_gdp["TotalPop"].values.astype(float)

    def interp_total_pop(y: np.ndarray) -> np.ndarray:
        return np.interp(y.astype(float), pop_years, pop_vals)

    d = df_triple.copy()
    d["TotalPop_est"] = interp_total_pop(d["Year"].values)

    # 1人当たり（円）→（USD）
    # GDP(兆円) → 円 = *1e12
    # 人口(万人) → 人 = *1e4
    d["GDP_per_capita_yen"] = d["GDP"] * 1e12 / (d["TotalPop_est"] * 1e4)
    d["GDP_per_capita_usd"] = d["GDP_per_capita_yen"] / d["USDJPY"]

    peak_idx = d["GDP_per_capita_usd"].idxmax()
    peak_year = int(d.loc[peak_idx, "Year"])
    peak_val = float(d["GDP_per_capita_usd"].max())
    latest_idx = d["Year"].idxmax()
    latest_year = int(d.loc[latest_idx, "Year"])
    latest_val = float(d.loc[latest_idx, "GDP_per_capita_usd"])
    pct_change = (latest_val / peak_val - 1) * 100

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(d["Year"], d["GDP_per_capita_usd"], marker="o", linewidth=4, markersize=8,
            color=COLORS['gdp'])
    
    # matplotlib+pandasの組み合わせによっては fill_between に Series を渡すと落ちるため numpy化
    ax.fill_between(d["Year"].to_numpy(), 0, d["GDP_per_capita_usd"].to_numpy(),
                    alpha=0.2, color=COLORS['gdp'])
    
    ax.set_title("ドル建て1人当たりGDP（概算）\n※総人口は線形補間推計、USD/JPYは年平均", 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("年", fontsize=14)
    ax.set_ylabel("USD/人（概算）", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1960, 2028)
    ax.set_ylim(0, max(d["GDP_per_capita_usd"]) * 1.15)

    ax.axvline(peak_year, linestyle="--", linewidth=2, color='green', alpha=0.7)
    ax.annotate(f"ピーク: {peak_year}年\n約{peak_val:,.0f} USD",
                xy=(peak_year, peak_val),
                xytext=(peak_year-20, peak_val*0.8),
                arrowprops=dict(arrowstyle="->", lw=2, color='green'),
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    ax.annotate(f"{latest_year}年: 約{latest_val:,.0f} USD\nピーク比 {pct_change:.0f}%",
                xy=(latest_year, latest_val),
                xytext=(latest_year-12, latest_val*1.15),
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORS['gdp']),
                fontsize=12, fontweight='bold', color=COLORS['gdp'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    note = "注：名目GDPと年平均為替レートによる概算値。購買力平価(PPP)ベースとは異なる。"
    fig.text(0.5, 0.01, note, ha="center", fontsize=10, color="gray")

    outpath = OUTDIR / "04_gdp_per_capita_usd.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {outpath}")

# ---------------------------------
# 図5：成長率の散布図（年率CAGR、dt込み）
# ---------------------------------
def fig_scatter_growth_rates(df_triple: pd.DataFrame) -> None:
    d = df_triple.sort_values("Year").copy()
    d["pop_cagr"] = annualized_cagr(d["WorkingPop"], d["Year"]) * 100
    d["gdp_cagr"] = annualized_cagr(d["GDP"], d["Year"]) * 100
    d["reer_cagr"] = annualized_cagr(d["REER"], d["Year"]) * 100
    d["dt"] = d["Year"].diff()

    # era label
    def era(y: int) -> str:
        if y <= 1973:
            return "高度成長前後"
        if y <= 1991:
            return "安定成長〜バブル"
        return "1990年代以降"

    d["Era"] = d["Year"].astype(int).apply(era)

    era_colors = {
        "高度成長前後": COLORS['high_growth'].replace('#', ''),
        "安定成長〜バブル": COLORS['bubble'].replace('#', ''),
        "1990年代以降": COLORS['lost_decades'].replace('#', ''),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # (1) 人口成長 vs GDP成長
    for era_label in ["高度成長前後", "安定成長〜バブル", "1990年代以降"]:
        g = d[d["Era"] == era_label].dropna()
        if len(g) > 0:
            ax1.scatter(g["pop_cagr"], g["gdp_cagr"], s=(g["dt"] * 15), alpha=0.7, 
                       label=era_label, edgecolors='black', linewidth=1)
            for _, r in g.iterrows():
                ax1.annotate(str(int(r["Year"])), (r["pop_cagr"], r["gdp_cagr"]), 
                           fontsize=9, alpha=0.8, ha='center', va='bottom')

    ax1.axhline(0, linewidth=1, alpha=0.5, color='black')
    ax1.axvline(0, linewidth=1, alpha=0.5, color='black')
    ax1.set_title("人口成長率（年率） vs 名目GDP成長率（年率）\n※点は隣接データ点間のCAGR（dt補正済み）", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("生産年齢人口 CAGR（%/年）", fontsize=12)
    ax1.set_ylabel("名目GDP CAGR（%/年）", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # (2) 人口成長 vs REER成長
    for era_label in ["高度成長前後", "安定成長〜バブル", "1990年代以降"]:
        g = d[d["Era"] == era_label].dropna()
        if len(g) > 0:
            ax2.scatter(g["pop_cagr"], g["reer_cagr"], s=(g["dt"] * 15), alpha=0.7, 
                       label=era_label, edgecolors='black', linewidth=1)
            for _, r in g.iterrows():
                ax2.annotate(str(int(r["Year"])), (r["pop_cagr"], r["reer_cagr"]), 
                           fontsize=9, alpha=0.8, ha='center', va='bottom')

    ax2.axhline(0, linewidth=1, alpha=0.5, color='black')
    ax2.axvline(0, linewidth=1, alpha=0.5, color='black')
    ax2.set_title("人口成長率（年率） vs REER変化率（年率）\n※点の大きさは年差dt", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("生産年齢人口 CAGR（%/年）", fontsize=12)
    ax2.set_ylabel("REER CAGR（%/年）", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    note = "注：点は「隣接データ点間」の年率CAGR（不均等間隔はdtで補正）。点の大きさは年差dt。\n相関は因果関係を示さない。逆因果・共通要因の可能性も考慮が必要。"
    fig.text(0.5, -0.01, note, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    outpath = OUTDIR / "05_scatter_growth_cagr.png"
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {outpath}")

# ---------------------------------
# 図6：統合1枚サマリー（元のultimate_thesisの修正版）
# ---------------------------------
def fig_integrated_summary(df: pd.DataFrame) -> None:
    """
    人口とGDPの関係を示す統合図（修正版）
    - 2軸グラフの「見た目の連動」に注意喚起
    - 因果関係ではなく相関を示すことを明示
    """
    # 図のサイズを大きくしてレイアウトに余裕を持たせる
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.6, 1], hspace=0.4, wspace=0.35)
    
    years = df['Year'].values
    
    # ========================================
    # 上段: 2軸グラフ（注意喚起付き）
    # ========================================
    ax_main = fig.add_subplot(gs[0, :])
    
    # 背景ハイライト
    ax_main.axvspan(1955, 1973, alpha=0.35, color=COLORS['high_growth'], zorder=0)
    ax_main.axvspan(1991, 2024, alpha=0.2, color=COLORS['lost_decades'], zorder=0)
    
    # 左軸：生産年齢人口
    color_pop = COLORS['working']
    ax_main.set_ylabel('生産年齢人口（万人）', fontsize=14, color=color_pop, fontweight='bold')
    # matplotlib+pandasの組み合わせによっては fill_between に Series を渡すと落ちるため numpy化
    ax_main.fill_between(years, 4000, df['WorkingPop'].to_numpy(),
                         alpha=0.2, color=color_pop, zorder=1)
    line1, = ax_main.plot(years, df['WorkingPop'], color=color_pop, linewidth=5, 
                          marker='o', markersize=10, label='生産年齢人口 (15-64歳)', zorder=3)
    ax_main.tick_params(axis='y', labelcolor=color_pop, labelsize=12)
    ax_main.set_ylim(4000, 10000)  # 上限を広げて注釈スペース確保
    
    # 右軸：GDP
    ax_gdp = ax_main.twinx()
    color_gdp = COLORS['gdp']
    ax_gdp.set_ylabel('名目GDP（兆円）', fontsize=14, color=color_gdp, fontweight='bold')
    line2, = ax_gdp.plot(years, df['GDP'], color=color_gdp, linewidth=5, 
                         marker='s', markersize=10, label='名目GDP', zorder=3)
    ax_gdp.tick_params(axis='y', labelcolor=color_gdp, labelsize=12)
    ax_gdp.set_ylim(0, 750)  # 上限を広げる
    
    # 凡例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_main.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.95)
    
    # 注釈（位置を調整して重ならないように）
    ax_main.annotate('【高度経済成長期】\n人口増加とGDP成長が\n同時に進行', 
                     xy=(1964, 5800), fontsize=12, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=COLORS['high_growth'], 
                              edgecolor='green', linewidth=2))
    
    ax_main.annotate('1995年\n生産年齢人口ピーク', 
                     xy=(1995, 8717), xytext=(1998, 9500),
                     fontsize=11, color=color_pop, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=color_pop, lw=2),
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax_main.annotate('【1990年代以降】\n人口減少・GDP低成長', 
                     xy=(2012, 7500), fontsize=12, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=COLORS['lost_decades'], 
                              edgecolor='red', linewidth=2))
    
    ax_main.set_title('生産年齢人口と名目GDPの推移\n※2軸グラフのスケール選択により「連動」の印象が変わることに注意', 
                      fontsize=16, fontweight='bold', pad=10)
    ax_main.set_xlabel('年', fontsize=14)
    ax_main.set_xlim(1948, 2028)
    ax_main.grid(True, alpha=0.4, zorder=0)
    
    # ========================================
    # 下段左: 期間別の増加率比較（棒グラフ）
    # ========================================
    ax_bar = fig.add_subplot(gs[1, 0])
    
    # 期間別データ（正しいCAGR計算）
    periods_data = [
        ('高度経済成長期\n(1955-1973)', 
         ((7420/5517)**(1/18)-1)*100,  # 人口CAGR
         ((112.5/8.4)**(1/18)-1)*100), # GDP CAGR
        ('安定成長期\n(1973-1991)', 
         ((8614/7420)**(1/18)-1)*100,
         ((469.4/112.5)**(1/18)-1)*100),
        ('1990年代以降\n(1991-2024)', 
         ((7360/8614)**(1/33)-1)*100,
         ((610/469.4)**(1/33)-1)*100),
    ]
    
    x = np.arange(len(periods_data))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, [d[1] for d in periods_data], width, 
                       color=COLORS['working'], alpha=0.9, label='生産年齢人口\n年平均増加率')
    bars2 = ax_bar.bar(x + width/2, [d[2] for d in periods_data], width,
                       color=COLORS['gdp'], alpha=0.9, label='名目GDP\n年平均成長率')
    
    # 数値表示
    for bar in bars1:
        h = bar.get_height()
        ax_bar.annotate(f'{h:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 5 if h >= 0 else -15), textcoords="offset points",
                       ha='center', va='bottom' if h >= 0 else 'top', 
                       fontsize=11, fontweight='bold', color=COLORS['working'])
    for bar in bars2:
        h = bar.get_height()
        ax_bar.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLORS['gdp'])
    
    ax_bar.axhline(y=0, color='black', linewidth=1)
    ax_bar.set_title('期間別：人口増加率とGDP成長率（年率CAGR）', fontsize=14, fontweight='bold')
    ax_bar.set_ylabel('年平均成長率（%）', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([d[0] for d in periods_data], fontsize=10)
    ax_bar.legend(loc='upper right', fontsize=9)
    ax_bar.set_ylim(-2, 19)  # 少し広げる
    ax_bar.grid(True, alpha=0.3, axis='y')
    
    # ========================================
    # 下段右: 指数比較（1955年=100）
    # ========================================
    ax_idx = fig.add_subplot(gs[1, 1])
    
    base_pop = df[df['Year']==1955]['WorkingPop'].values[0]
    base_gdp = df[df['Year']==1955]['GDP'].values[0]
    
    idx_pop = df['WorkingPop'] / base_pop * 100
    idx_gdp = df['GDP'] / base_gdp * 100
    
    ax_idx.axvspan(1955, 1973, alpha=0.35, color=COLORS['high_growth'])
    ax_idx.axvspan(1991, 2024, alpha=0.2, color=COLORS['lost_decades'])
    
    ax_idx.plot(years, idx_pop, color=COLORS['working'], linewidth=4, marker='o', 
                markersize=6, label='生産年齢人口（指数）')
    
    ax_idx2 = ax_idx.twinx()
    ax_idx2.plot(years, idx_gdp, color=COLORS['gdp'], linewidth=4, marker='s', 
                 markersize=6, label='名目GDP（指数）')
    ax_idx2.set_ylabel('名目GDP指数（1955=100）', fontsize=11, color=COLORS['gdp'])
    ax_idx2.tick_params(axis='y', labelcolor=COLORS['gdp'])
    ax_idx2.set_ylim(0, 8000)
    
    ax_idx.set_ylabel('人口指数（1955=100）', fontsize=11, color=COLORS['working'])
    ax_idx.tick_params(axis='y', labelcolor=COLORS['working'])
    ax_idx.set_ylim(80, 180)
    
    ax_idx.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    ax_idx.set_title('1955年基準の累積変化\n※スケールが異なることに注意', fontsize=14, fontweight='bold')
    ax_idx.set_xlabel('年', fontsize=12)
    ax_idx.set_xlim(1948, 2028)
    ax_idx.grid(True, alpha=0.3)
    
    # メインタイトル（位置調整）
    fig.suptitle('日本の人口動態と経済成長の関係\n※相関は因果関係を証明しない。他要因（技術進歩・資本蓄積・政策等）も重要', 
                 fontsize=18, fontweight='bold', y=0.98, color='darkblue')
    
    # データソース（位置調整）
    fig.text(0.5, 0.02, 
             '※データソース: 総務省統計局「国勢調査」「人口推計」、内閣府「国民経済計算」\n'
             '※名目GDPはインフレ影響を含む。実質分析には実質GDPデータ使用推奨。',
             fontsize=10, color='gray', ha='center')
    
    # tight_layoutの代わりにsubplots_adjustで手動調整
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.92)
    
    outpath = OUTDIR / "06_integrated_summary.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {outpath}")
    plt.close()

# ---------------------------------
# 実行
# ---------------------------------
def main() -> None:
    print("=" * 60)
    print("修正版：日本のマクロ経済グラフ生成")
    print("=" * 60)
    
    df_pop_gdp = make_df_pop_gdp()
    df_triple = make_df_triple()

    print("\n[1/6] 指数化比較（人口・GDP・生産性）...")
    fig_index_pop_gdp_productivity(df_pop_gdp, base_year=1995)
    
    print("[2/6] GDP成長の分解（期間別CAGR・ログ寄与）...")
    fig_decomposition_periods(df_pop_gdp)
    
    print("[3/6] 三指標比較（人口・GDP・REER）...")
    fig_triple_index(df_triple, base_year=1995)
    
    print("[4/6] ドル建て1人当たりGDP...")
    fig_usd_per_capita(df_triple, df_pop_gdp)
    
    print("[5/6] 成長率散布図（CAGR、dt補正済み）...")
    fig_scatter_growth_rates(df_triple)
    
    print("[6/6] 統合サマリー図...")
    fig_integrated_summary(df_pop_gdp)

    print("\n" + "=" * 60)
    print(f"完了！出力先: {OUTDIR.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
