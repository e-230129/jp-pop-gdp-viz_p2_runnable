"""
usd_weakness_fred_dashboard_v2.py

FRED公開CSVを使用（APIキー不要）
全修正反映版:
- Gold部分のtypo修正（col → c）
- DXY表記を"主要通貨"に統一
- 出力先をOUTDIR変数で指定

依存:
  pip install pandas matplotlib requests
  pip install japanize-matplotlib  # 日本語（推奨）
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests


# =========================
# 設定
# =========================
START_DATE = "2020-01-01"
BASE_DATE = "2020-01-02"
FOCUS_FROM = "2024-01-01"

OUTDIR = Path("output_usd_weakness")
OUTDIR.mkdir(exist_ok=True)

# イベント（最小限に）
EVENTS = [
    # ("2025-04-09", "Tariffs"),
]


# =========================
# 日本語フォント
# =========================
try:
    import japanize_matplotlib
except ImportError:
    print("[Warning] japanize_matplotlib not found. Install: pip install japanize-matplotlib")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


# =========================
# データ定義
# =========================
@dataclass(frozen=True)
class SeriesDef:
    fred_id: str
    label: str
    direction_usd_weak: int  # +1: 上昇=USD弱い, -1: 上昇=USD強い


SERIES: Dict[str, SeriesDef] = {
    # USD index (Trade-weighted, FRB)
    "DTWEXM": SeriesDef("DTWEXM", "USD (Major Currencies) Weakness", -1),
    "DTWEXBGS": SeriesDef("DTWEXBGS", "USD (Broad) Weakness", -1),

    # FX
    "DEXUSEU": SeriesDef("DEXUSEU", "vs EUR: USD Weakness (EUR/USD)", +1),
    "DEXUSUK": SeriesDef("DEXUSUK", "vs GBP: USD Weakness (GBP/USD)", +1),
    "DEXUSAL": SeriesDef("DEXUSAL", "vs AUD: USD Weakness (AUD/USD)", +1),
    "DEXJPUS": SeriesDef("DEXJPUS", "vs JPY: USD Weakness (reciprocal)", -1),
    "DEXSZUS": SeriesDef("DEXSZUS", "vs CHF: USD Weakness (reciprocal)", -1),
    "DEXCAUS": SeriesDef("DEXCAUS", "vs CAD: USD Weakness (reciprocal)", -1),

    # Gold
    "GOLDAMGBD228NLBM": SeriesDef("GOLDAMGBD228NLBM", "Gold: USD Weakness (XAU/USD)", +1),
}


def fetch_fred_series(fred_id: str, start_date: str) -> pd.Series:
    """FREDのCSVから系列を取得（APIキー不要）"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if "DATE" not in df.columns or fred_id not in df.columns:
        raise ValueError(f"FRED format unexpected for {fred_id}")

    s = pd.to_numeric(df[fred_id], errors="coerce")
    s.index = pd.to_datetime(df["DATE"])
    s = s.loc[s.index >= pd.to_datetime(start_date)]
    s.name = fred_id
    return s


def pick_common_base_date(df: pd.DataFrame, base_date: str) -> pd.Timestamp:
    """全列に値がある最初の日を選ぶ"""
    base_ts = pd.to_datetime(base_date)
    rows = df.loc[df.index >= base_ts].dropna(how="any")
    if rows.empty:
        if "DTWEXM" in df.columns and not df["DTWEXM"].dropna().empty:
            return df["DTWEXM"].dropna().index[0]
        raise RuntimeError("No common base date found. Try later BASE_DATE.")
    return rows.index[0]


def to_usd_weakness_index(s: pd.Series, direction: int, base_value: float) -> pd.Series:
    """USD弱さ指数への変換"""
    if direction == +1:
        return 100.0 * (s / base_value)
    if direction == -1:
        return 100.0 * (base_value / s)
    raise ValueError("direction must be +1 or -1")


def annotate_last(ax, y: pd.Series, dx: int = 6, dy: int = 0, color='black'):
    if y.dropna().empty:
        return
    last_x = y.dropna().index[-1]
    last_y = float(y.dropna().iloc[-1])
    ax.annotate(f"{last_y:.1f}", xy=(last_x, last_y), xytext=(dx, dy),
                textcoords="offset points", fontsize=9, fontweight="bold", color=color)


def plot_dashboard(weak: pd.DataFrame, base_date: pd.Timestamp) -> Path:
    """2x2ダッシュボード"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"USD Weakness Dashboard (Up = USD Weaker)\nBase: {base_date.date()} = 100",
        fontsize=16, fontweight="bold"
    )

    # 1) USD index (basket)
    ax = axes[0, 0]
    basket_cols = [c for c in weak.columns if "Major" in c or "Broad" in c]
    for col in basket_cols:
        ax.plot(weak.index, weak[col], linewidth=2, label=col)
        annotate_last(ax, weak[col])
    ax.axhline(100, linewidth=1, alpha=0.4)
    ax.set_title("USD Index (Trade-Weighted) Weakness", fontweight="bold")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # 2) Gold - ★修正: col → c
    ax = axes[0, 1]
    gold_cols = [c for c in weak.columns if c.startswith("Gold")]  # 修正済み
    for col in gold_cols:
        ax.plot(weak.index, weak[col], linewidth=2, label=col, color='goldenrod')
        annotate_last(ax, weak[col], color='goldenrod')
    ax.axhline(100, linewidth=1, alpha=0.4)
    ax.set_title("Gold (Up = USD Real Value Decline)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # 3) EUR/GBP/AUD
    ax = axes[1, 0]
    majors = [c for c in weak.columns if "EUR" in c or "GBP" in c or "AUD" in c]
    for col in majors:
        ax.plot(weak.index, weak[col], linewidth=2, label=col)
        annotate_last(ax, weak[col])
    ax.axhline(100, linewidth=1, alpha=0.4)
    ax.set_title("vs EUR/GBP/AUD: Up = USD Weaker", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # 4) JPY/CHF/CAD
    ax = axes[1, 1]
    others = [c for c in weak.columns if "JPY" in c or "CHF" in c or "CAD" in c]
    colors = {'JPY': 'red', 'CHF': 'purple', 'CAD': 'brown'}
    for col in others:
        color = next((v for k, v in colors.items() if k in col), 'gray')
        lw = 3 if 'JPY' in col else 2
        ax.plot(weak.index, weak[col], linewidth=lw, label=col, color=color)
        annotate_last(ax, weak[col], color=color)
    ax.axhline(100, linewidth=1, alpha=0.4)
    ax.set_title("vs JPY/CHF/CAD: Up = USD Weaker\n(JPY highlighted)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    out = OUTDIR / "usd_weakness_dashboard_fred.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_focus_overlay(weak: pd.DataFrame, base_date: pd.Timestamp, focus_from: str) -> Path:
    """2024年以降フォーカス"""
    df = weak.loc[weak.index >= pd.to_datetime(focus_from)].copy()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(
        f"Focus ({focus_from}~): USD Weakness (Up = USD Weaker)\nBase: {base_date.date()} = 100",
        fontsize=14, fontweight="bold"
    )

    candidates = [
        "USD (Major Currencies) Weakness",
        "USD (Broad) Weakness",
        "Gold: USD Weakness (XAU/USD)",
        "vs EUR: USD Weakness (EUR/USD)",
        "vs JPY: USD Weakness (reciprocal)",
    ]
    colors = {
        "Major": "navy", "Broad": "blue", "Gold": "goldenrod", 
        "EUR": "green", "JPY": "red"
    }
    
    for col in candidates:
        if col in df.columns:
            color = next((v for k, v in colors.items() if k in col), 'gray')
            lw = 3 if 'JPY' in col else 2
            ax.plot(df.index, df[col], linewidth=lw, label=col, color=color)
            annotate_last(ax, df[col], color=color)

    ax.axhline(100, linewidth=1.2, alpha=0.5)
    ax.set_ylabel("Index (Up = USD Weaker)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    note = (
        "Interpretation:\n"
        "- Basket/Gold/EUR rising = USD broadly weakening\n"
        "- JPY (red) below 100 = USD strong vs JPY only\n"
        "  = JPY-specific weakness"
    )
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    out = OUTDIR / "usd_weakness_focus_fred.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_jpy_comparison(weak: pd.DataFrame, base_date: pd.Timestamp) -> Path:
    """FRB Trade-Weighted Index vs JPY 直接比較"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 列名を探す（DXYではなくTrade-Weighted）
    twi_col = next((c for c in weak.columns if "Major" in c), None)
    jpy_col = next((c for c in weak.columns if "JPY" in c), None)
    
    if twi_col is None or jpy_col is None:
        print("[Warning] Could not find Trade-Weighted or JPY columns")
        plt.close(fig)
        return OUTDIR / "usd_weakness_comparison_fred.png"
    
    twi = weak[twi_col].dropna()
    jpy = weak[jpy_col].dropna()
    common_idx = twi.index.intersection(jpy.index)
    twi_common = twi.loc[common_idx]
    jpy_common = jpy.loc[common_idx]
    
    ax.plot(twi_common.index, twi_common, label='FRB Trade-Weighted (Major)', color='navy', linewidth=2.5)
    ax.plot(jpy_common.index, jpy_common, label='vs JPY (reciprocal)', color='red', linewidth=2.5)
    ax.axhline(100, color='black', linestyle='-', alpha=0.7, linewidth=2)
    
    # 2本の線の間を塗る
    mask = twi_common > jpy_common
    ax.fill_between(
        common_idx,
        jpy_common,
        twi_common,
        where=mask,
        interpolate=True,
        alpha=0.2,
        color='red',
        label='Gap: JPY weaker'
    )
    
    annotate_last(ax, twi_common, color='navy')
    annotate_last(ax, jpy_common, dy=-15, color='red')
    
    gap = twi_common.iloc[-1] - jpy_common.iloc[-1]
    ax.text(0.02, 0.98, f'Current Gap: {gap:.1f}pt\n= JPY-specific weakness',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('USD Weakness Index (Up = USD Weaker)', fontsize=12)
    ax.set_title(f'FRB Trade-Weighted Index vs USD/JPY\nRed area = "USD weak vs majors but JPY even weaker"\nBase: {base_date.date()} = 100',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out = OUTDIR / "usd_weakness_comparison_fred.png"
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main() -> None:
    print("="*70)
    print("USD Weakness Dashboard (FRED-based, No API Key)")
    print("="*70)

    # データ取得
    print("\n[1] Fetching data from FRED...")
    raw: Dict[str, pd.Series] = {}
    kept: Dict[str, SeriesDef] = {}
    for _, sd in SERIES.items():
        try:
            s = fetch_fred_series(sd.fred_id, START_DATE)
            if s.dropna().empty:
                continue
            raw[sd.fred_id] = s
            kept[sd.fred_id] = sd
            print(f"  OK: {sd.fred_id}")
        except Exception as e:
            print(f"  [skip] {sd.fred_id}: {e}")

    if not raw:
        raise RuntimeError("No data retrieved. Check network connection.")

    df = pd.concat(raw.values(), axis=1).sort_index()
    df = df.ffill()  # bfillは使わない

    # 基準日
    print("\n[2] Determining base date...")
    base_date = pick_common_base_date(df, BASE_DATE)
    base_vals = df.loc[base_date]
    print(f"  Base date: {base_date.date()}")

    # 変換
    print("\n[3] Converting to USD Weakness Index...")
    weak = pd.DataFrame(index=df.index)
    for fred_id in df.columns:
        sd = kept[fred_id]
        weak[sd.label] = to_usd_weakness_index(df[fred_id], sd.direction_usd_weak, float(base_vals[fred_id]))

    # 出力
    print("\n[4] Creating charts...")
    out1 = plot_dashboard(weak, base_date)
    out2 = plot_focus_overlay(weak, base_date, FOCUS_FROM)
    out3 = plot_jpy_comparison(weak, base_date)

    # サマリー
    latest_date = weak.dropna(how="all").index.max()
    print("\n" + "="*70)
    print("USD Weakness Index Summary (Latest)")
    print("="*70)
    print(f"Base: {base_date.date()} = 100")
    print(f"Latest: {latest_date.date() if pd.notna(latest_date) else 'n/a'}")
    print()
    
    for col in weak.columns:
        v = weak[col].dropna()
        if not v.empty:
            val = v.iloc[-1]
            if val >= 100:
                interp = f"USD {val-100:.1f}% WEAKER"
            else:
                interp = f"USD {100-val:.1f}% STRONGER"
            marker = " ***" if "JPY" in col else ""
            print(f"  {col}: {val:.1f} ({interp}){marker}")

    print("\n" + "="*70)
    print("Output Files")
    print("="*70)
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")
    print("\nDone!")


if __name__ == "__main__":
    main()
