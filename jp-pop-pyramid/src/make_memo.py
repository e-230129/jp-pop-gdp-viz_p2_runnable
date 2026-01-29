# make_memo.py
"""
日本人口ピラミッド自動生成ツール v7

v6からの改善点:
- 「ピーク年齢」（ピーク年齢追跡）を削除
- コホート追跡を一般化（--cohort "名前:出生年" 形式）
- 任意の名前でコホート線を表示可能

v5からの改善点 (GPT-5.2 Pro レビューに基づく):
- 100+の配分修正（100〜max_ageに配分、95-99と重ならない）
- 年の選択を一般化（--start-year, --end-year, --year-step）
- ホイール/キーボードで年送り（interactive HTML）
- メモのレイアウト一般化（--memo-layout grid）
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
import matplotlib.colors as mcolors

# -------------------------
# 定数
# -------------------------
ACCENT_BLUE = "#1E6BFF"
ACCENT_RED = "#E74C3C"
LABEL_BG = "#E8F4FD"


# -------------------------
# ユーティリティ
# -------------------------
def setup_japanese_font() -> None:
    candidates = [
        "Noto Sans CJK JP", "IPAexGothic", "IPAGothic",
        "Yu Gothic", "Meiryo", "Hiragino Sans", "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def format_date_jp(d: dt.date) -> str:
    wd = "月火水木金土日"[d.weekday()]
    return d.strftime(f"%Y/%m/%d({wd})")


def format_jp_population(pop_persons: float, decimals_man: int = 1) -> str:
    man = pop_persons / 1e4
    if man >= 10000:
        oku = int(man // 10000)
        rest = man - oku * 10000
        width = 4 + 1 + decimals_man
        rest_str = f"{rest:0{width}.{decimals_man}f}"
        return f"{oku}億{rest_str}万"
    return f"{man:.{decimals_man}f}万"


def normalize_sex(x) -> str:
    s = str(x).strip()
    if s in {"M", "m", "Male", "male", "男", "男性", "1"}:
        return "M"
    if s in {"F", "f", "Female", "female", "女", "女性", "2"}:
        return "F"
    raise ValueError(f"Unknown sex value: {x!r}")


# -------------------------
# 年の選択（一般化）
# -------------------------
def select_years(all_years: list[int],
                 start: int | None = None,
                 end: int | None = None,
                 step: int = 1,
                 explicit: list[int] | None = None) -> list[int]:
    """
    年の選び方を一般化
    
    - explicit: 明示指定（最優先）
    - start/end/step: 範囲指定
    """
    if explicit:
        ys = [y for y in explicit if y in set(all_years)]
        return sorted(ys)
    
    s = start if start is not None else all_years[0]
    e = end if end is not None else all_years[-1]
    ys = [y for y in all_years if s <= y <= e and (y - s) % step == 0]
    return ys


# -------------------------
# データ読み込み・判定
# -------------------------
def read_population(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Input must be .csv or .xlsx/.xls")

    col_candidates = {
        "year": ["year", "Year", "年", "年度", "西暦"],
        "age": ["age", "Age", "年齢", "歳"],
        "sex": ["sex", "Sex", "性別", "男女", "性"],
        "pop": ["pop", "Pop", "population", "Population", "人口", "人数", "人口（人）"],
    }
    rename = {}
    for canon, cands in col_candidates.items():
        for c in df.columns:
            if c in cands:
                rename[c] = canon
                break
    df = df.rename(columns=rename)

    missing = {"year", "age", "sex", "pop"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[["year", "age", "sex", "pop"]].copy()
    df["year"] = df["year"].astype(int)
    df["age"] = df["age"].astype(str).str.replace("+", "", regex=False).astype(int)
    df["sex"] = df["sex"].apply(normalize_sex)
    df["pop"] = pd.to_numeric(df["pop"], errors="raise")

    return df


def detect_data_type(df: pd.DataFrame) -> Literal["single_year", "five_year"]:
    sample_year = df["year"].iloc[0]
    ages = sorted(df[df["year"] == sample_year]["age"].unique())
    ages_under_50 = [a for a in ages if a <= 50]
    
    if len(ages_under_50) >= 40:
        return "single_year"
    
    five_year_pattern_a = {4, 9, 14, 19, 24, 29, 34, 39, 44, 49}
    five_year_pattern_b = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    
    ages_set = set(ages)
    if five_year_pattern_a.issubset(ages_set) or five_year_pattern_b.issubset(ages_set):
        return "five_year"
    
    return "five_year"


# -------------------------
# 色の計算
# -------------------------
def get_age_color(age: int) -> str:
    if age <= 14:
        t = age / 14
        base = np.array(mcolors.to_rgb("#98D54B"))
        return mcolors.to_hex(base * (1 - t * 0.15))
    elif age <= 64:
        t = (age - 15) / 49
        base = np.array(mcolors.to_rgb("#5DADE2"))
        dark = np.array(mcolors.to_rgb("#1A5276"))
        return mcolors.to_hex(base * (1 - t) + dark * t)
    elif age <= 74:
        t = (age - 65) / 9
        base = np.array(mcolors.to_rgb("#F5B041"))
        dark = np.array(mcolors.to_rgb("#D68910"))
        return mcolors.to_hex(base * (1 - t) + dark * t)
    else:
        t = min((age - 75) / 25, 1.0)
        base = np.array(mcolors.to_rgb("#E74C3C"))
        dark = np.array(mcolors.to_rgb("#922B21"))
        return mcolors.to_hex(base * (1 - t) + dark * t)


# -------------------------
# データ展開（100+の配分修正）
# -------------------------
def expand_to_single_year(df_year: pd.DataFrame, max_age: int,
                          data_type: Literal["single_year", "five_year"]) -> tuple[np.ndarray, np.ndarray]:
    """
    年齢別人口を1歳刻み配列に展開
    
    v6修正: 100+は100〜max_ageに配分（95-99と重ならない）
    """
    ages = np.arange(0, max_age + 1)
    m = np.zeros(max_age + 1)
    f = np.zeros(max_age + 1)
    
    m_data = df_year[df_year["sex"] == "M"].set_index("age")["pop"].sort_index()
    f_data = df_year[df_year["sex"] == "F"].set_index("age")["pop"].sort_index()
    
    if data_type == "single_year":
        for age in ages:
            if age in m_data.index:
                m[age] = float(m_data[age])
            if age in f_data.index:
                f[age] = float(f_data[age])
    else:
        # 5歳階級データ: 均等分配
        available_ages = sorted(m_data.index.tolist())

        # 年齢表現を検出: 0が含まれていれば下端表現（0,5,10,...）
        uses_lower_bound = 0 in set(available_ages)

        for repr_age in available_ages:
            if uses_lower_bound:
                # 下端表現: 0→0-4, 5→5-9, ...
                if repr_age >= 100:
                    # 100+: 100〜max_age に配分
                    age_start = 100
                    age_end = max_age
                else:
                    age_start = repr_age
                    age_end = repr_age + 4
            else:
                # 上端表現: 4→0-4, 9→5-9, ...
                if repr_age <= 4:
                    # 0-4歳
                    age_start, age_end = 0, repr_age
                elif repr_age >= 100:
                    # 100+: 100〜max_age に配分（修正）
                    age_start = 100
                    age_end = max_age
                elif repr_age >= 95 and repr_age < 100:
                    # 95-99歳
                    age_start = repr_age - 4
                    age_end = repr_age
                else:
                    # 通常の5歳階級
                    age_start = repr_age - 4
                    age_end = repr_age

            pop_m = float(m_data.get(repr_age, 0))
            pop_f = float(f_data.get(repr_age, 0))
            n_ages = age_end - age_start + 1

            for a in range(age_start, min(age_end + 1, max_age + 1)):
                m[a] += pop_m / n_ages
                f[a] += pop_f / n_ages
    
    return m, f


# -------------------------
# コホート年齢計算
# -------------------------
def cohort_age(year: int, birth_year: int) -> int:
    """出生年から現在の年齢を計算"""
    return year - birth_year


def parse_cohorts(cohort_str: str | None) -> list[tuple[str, int]]:
    """
    コホート文字列をパース

    形式: "名前:出生年,名前:出生年,..." または "出生年,出生年,..."
    例: "団塊の世代:1947,団塊ジュニア:1971" → [("団塊の世代", 1947), ("団塊ジュニア", 1971)]
    例: "1947,1971" → [("1947年生", 1947), ("1971年生", 1971)]
    """
    if not cohort_str:
        return []

    result = []
    for item in cohort_str.split(","):
        item = item.strip()
        if ":" in item:
            name, year_str = item.split(":", 1)
            result.append((name.strip(), int(year_str.strip())))
        else:
            year = int(item)
            result.append((f"{year}年生", year))
    return result


# -------------------------
# 描画ヘルパー
# -------------------------
def add_label_box(ax, x, y, text, fontsize=10, color=ACCENT_BLUE):
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5",
                      facecolor=LABEL_BG, edgecolor=color, linewidth=1.5))


def add_cohort_lines_matplotlib(ax, xmax: float, year: int,
                                cohorts: list[tuple[str, int]], max_age: int) -> None:
    """出生年コホートの破線を追加（名前付き）"""
    for name, by in cohorts:
        a = cohort_age(year, by)
        if 0 <= a <= max_age:
            ax.axhline(a, color="black", lw=1.2, ls="--", alpha=0.35, zorder=5)
            ax.text(0.02, a, f"{name}({a}才)",
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=8, alpha=0.7)


def plot_pyramid(ax, df_year: pd.DataFrame, year: int, max_age: int,
                 data_type: Literal["single_year", "five_year"],
                 xmax: float = 130,
                 cohorts: list[tuple[str, int]] | None = None) -> None:
    ages = np.arange(0, max_age + 1)
    m, f = expand_to_single_year(df_year, max_age, data_type)
    
    m_man = m / 1e4
    f_man = f / 1e4
    
    colors = [get_age_color(a) for a in ages]
    
    ax.barh(ages, -m_man, color=colors, height=1.0, linewidth=0, edgecolor='none')
    ax.barh(ages, f_man, color=colors, height=1.0, linewidth=0, edgecolor='none')
    
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, max_age)
    
    ax.set_title(f"{year}", fontsize=12, fontweight="bold", pad=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", lw=0.3, alpha=0.4, color='gray')
    
    xticks = np.arange(-xmax, xmax + 0.1, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(abs(x))) for x in xticks], fontsize=7)
    ax.set_xlabel("人口（万人）", fontsize=8, labelpad=2)
    
    yticks = np.arange(0, max_age + 1, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y)) for y in yticks], fontsize=7)
    
    for label, y_pos in [("年少人口", 7), ("生産年齢人口", 40),
                         ("前期老年人口", 70), ("後期老年人口", 87)]:
        if y_pos <= max_age:
            ax.text(0.5, y_pos, label, ha="center", va="center",
                    fontsize=7, alpha=0.7, transform=ax.get_yaxis_transform())
    
    male_total = float(df_year[df_year["sex"] == "M"]["pop"].sum())
    female_total = float(df_year[df_year["sex"] == "F"]["pop"].sum())
    total = male_total + female_total
    
    add_label_box(ax, 0.12, 0.92, f"男 性\n{format_jp_population(male_total)}", fontsize=9)
    add_label_box(ax, 0.88, 0.92, f"女 性\n{format_jp_population(female_total)}", fontsize=9)
    
    # コホート線（黒・破線）
    if cohorts:
        add_cohort_lines_matplotlib(ax, xmax, year, cohorts, max_age)

    # 総人口表示
    ax.text(1.02, max_age * 0.5,
            f"総人口\n{format_jp_population(total)}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center",
            fontsize=12, color=ACCENT_BLUE, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------
# 静止画メモ生成（レイアウト一般化）
# -------------------------
def make_memo(df: pd.DataFrame, years: list[int], out_path: Path,
              max_age: int = 100,
              cohorts: list[tuple[str, int]] | None = None,
              layout: str = "auto") -> None:
    setup_japanese_font()

    data_type = detect_data_type(df)
    print(f"✓ データ形式を自動判定: {data_type}")

    if cohorts:
        print(f"✓ コホート線: {[f'{name}({by})' for name, by in cohorts]}")

    n_years = len(years)
    
    # レイアウト決定
    if layout == "fixed5" and n_years == 5:
        # 従来の5枚レイアウト
        cols, rows = 2, 3
        positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
        note_pos = (2, 1)
    elif layout == "auto" or layout == "grid":
        # 自動グリッドレイアウト
        cols = min(3, n_years)
        rows = math.ceil((n_years + 1) / cols)  # +1 for note
        positions = [(r, c) for r in range(rows) for c in range(cols)][:n_years]
        note_pos = positions[-1] if len(positions) > n_years else (rows - 1, cols - 1)
        if n_years < rows * cols:
            note_pos = (rows - 1, n_years % cols) if n_years % cols != 0 else (rows - 1, cols - 1)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    fig_width = 7 * cols
    fig_height = 5 * rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.30, hspace=0.35,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)
    
    for i, y in enumerate(years):
        r, c = positions[i]
        ax = fig.add_subplot(gs[r, c])
        
        d = df[df["year"] == y]
        if d.empty:
            raise ValueError(f"No data for year {y}")
        
        plot_pyramid(ax, d, year=y, max_age=max_age,
                     data_type=data_type, cohorts=cohorts)
    
    # ノートエリア
    if n_years < rows * cols:
        ax_text = fig.add_subplot(gs[note_pos[0], note_pos[1]])
        ax_text.axis("off")
        ax_text.text(0.5, 0.4,
                     "日本の人口推計\n国立社会保障・人口問題研究所、将来人口推計より",
                     ha="center", va="center", fontsize=14,
                     color=ACCENT_BLUE, fontweight="bold")
    
    today = dt.date.today()
    fig.text(0.96, 0.96, format_date_jp(today), ha="right", va="top",
             fontsize=14, color=ACCENT_BLUE, fontweight="bold")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


# -------------------------
# アニメーション生成
# -------------------------
def make_animation(df: pd.DataFrame, years: list[int], out_path: Path,
                   max_age: int = 105, fps: int = 4,
                   cohorts: list[tuple[str, int]] | None = None) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    setup_japanese_font()

    data_type = detect_data_type(df)
    print(f"✓ データ形式を自動判定: {data_type}")
    print(f"✓ 対象年: {years[0]}〜{years[-1]} ({len(years)}年分)")

    if cohorts:
        print(f"✓ コホート線: {[f'{name}({by})' for name, by in cohorts]}")

    ages = np.arange(0, max_age + 1)

    xmax = (df["pop"].max() / 1e4) * 1.15
    xmax = max(130, int(np.ceil(xmax / 10) * 10))
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, max_age)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("人口（万人）", fontsize=10)
    ax.set_ylabel("年齢", fontsize=10)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    
    y0 = years[0]
    d0 = df[df["year"] == y0]
    m0, f0 = expand_to_single_year(d0, max_age, data_type)
    colors = [get_age_color(a) for a in ages]
    
    bars_m = ax.barh(ages, -m0/1e4, color=colors, height=1.0, linewidth=0)
    bars_f = ax.barh(ages, f0/1e4, color=colors, height=1.0, linewidth=0)
    
    title = ax.set_title(f"{y0}", fontsize=16, fontweight="bold")

    # コホート線（初期）
    cohort_lines = []
    cohort_texts = []
    if cohorts:
        for name, by in cohorts:
            line = ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.35, zorder=5)
            txt = ax.text(0.02, 0, "", transform=ax.get_yaxis_transform(),
                         ha="left", va="center", fontsize=8, alpha=0.7)
            cohort_lines.append((name, by, line))
            cohort_texts.append((name, by, txt))
    
    pop_text = ax.text(0.98, 0.02, "", transform=ax.transAxes,
                       ha="right", va="bottom", fontsize=12,
                       color=ACCENT_BLUE, fontweight="bold")
    
    def update(frame):
        y = years[frame]
        d = df[df["year"] == y]
        m, f = expand_to_single_year(d, max_age, data_type)
        
        for b, v in zip(bars_m, m/1e4):
            b.set_width(-v)
        for b, v in zip(bars_f, f/1e4):
            b.set_width(v)
        
        title.set_text(f"{y}")

        # コホート線の更新
        for name, by, line in cohort_lines:
            a = cohort_age(y, by)
            if 0 <= a <= max_age:
                line.set_ydata([a, a])
                line.set_visible(True)
            else:
                line.set_visible(False)

        for name, by, txt in cohort_texts:
            a = cohort_age(y, by)
            if 0 <= a <= max_age:
                txt.set_position((0.02, a))
                txt.set_text(f"{name}({a}才)")
                txt.set_visible(True)
            else:
                txt.set_visible(False)
        
        total = d["pop"].sum()
        pop_text.set_text(f"総人口: {format_jp_population(total)}")

        return [*bars_m, *bars_f, title, pop_text]
    
    ani = FuncAnimation(fig, update, frames=len(years), interval=1000//fps, blit=False)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.suffix.lower() == ".gif":
        ani.save(out_path, writer=PillowWriter(fps=fps))
    elif out_path.suffix.lower() == ".mp4":
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-g", "1", "-crf", "20", "-pix_fmt", "yuv420p"]
        )
        ani.save(out_path, writer=writer)
    else:
        raise ValueError("Animation output must be .gif or .mp4")
    
    plt.close(fig)
    print(f"✓ Animation saved: {out_path} ({len(years)} frames, {fps} fps)")


# -------------------------
# Plotlyインタラクティブ版（ホイール/キー対応）
# -------------------------
def inject_wheel_scrub(html_path: Path, years: list[int]) -> None:
    """HTMLにホイール/キーボード/スクリーンショット機能を埋め込む"""
    html = html_path.read_text(encoding="utf-8")
    js = f"""
<style>
#screenshot-btn {{
  position: fixed;
  top: 10px;
  right: 10px;
  padding: 8px 16px;
  font-size: 14px;
  background: #1E6BFF;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  z-index: 1000;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}}
#screenshot-btn:hover {{
  background: #1557CC;
}}
#screenshot-btn:active {{
  transform: scale(0.98);
}}
</style>
<button id="screenshot-btn" title="スクリーンショット (Sキー)">📷 保存</button>
<script>
(function(){{
  const gd = document.querySelector('.js-plotly-plot');
  if(!gd) return;
  const years = {years};
  let idx = 0;

  function go(i){{
    idx = Math.max(0, Math.min(years.length-1, i));
    Plotly.animate(gd, [String(years[idx])],
      {{mode:'immediate', frame:{{duration:0, redraw:true}}, transition:{{duration:0}}}});
  }}

  function screenshot(){{
    const year = years[idx];
    Plotly.downloadImage(gd, {{
      format: 'png',
      width: 1200,
      height: 900,
      filename: 'pyramid_' + year
    }});
  }}

  // ホイール操作
  window.addEventListener('wheel', (e)=>{{
    go(idx + (e.deltaY>0?1:-1));
    e.preventDefault();
  }}, {{passive:false}});

  // キーボード操作
  window.addEventListener('keydown', (e)=>{{
    if(e.key==='ArrowRight') go(idx+1);
    if(e.key==='ArrowLeft')  go(idx-1);
    if(e.key==='Home')       go(0);
    if(e.key==='End')        go(years.length-1);
    if(e.key==='s' || e.key==='S') screenshot();
  }});

  // スクリーンショットボタン
  document.getElementById('screenshot-btn').addEventListener('click', screenshot);

  // スライダー変更を監視してidxを同期
  const observer = new MutationObserver(()=>{{
    const label = gd.querySelector('.slider-current-value');
    if(label){{
      const y = parseInt(label.textContent.replace('年: ',''));
      const i = years.indexOf(y);
      if(i>=0) idx = i;
    }}
  }});
  observer.observe(gd, {{subtree:true, childList:true, characterData:true}});
}})();
</script>
</body>
"""
    html = html.replace("</body>", js)
    html_path.write_text(html, encoding="utf-8")


def make_interactive(df: pd.DataFrame, years: list[int], out_path: Path,
                     max_age: int = 105,
                     cohorts: list[tuple[str, int]] | None = None) -> None:
    import plotly.graph_objects as go

    data_type = detect_data_type(df)
    print(f"✓ データ形式を自動判定: {data_type}")
    print(f"✓ 対象年: {years[0]}〜{years[-1]} ({len(years)}年分)")

    if cohorts:
        print(f"✓ コホート線: {[f'{name}({by})' for name, by in cohorts]}")

    ages = np.arange(0, max_age + 1)

    xmax = (df["pop"].max() / 1e4) * 1.15
    xmax = max(130, int(np.ceil(xmax / 10) * 10))
    
    def arrays_for_year(y):
        d = df[df["year"] == y]
        m, f = expand_to_single_year(d, max_age, data_type)
        total = d["pop"].sum()
        return m / 1e4, f / 1e4, total

    m0, f0, total0 = arrays_for_year(years[0])
    
    colors_m = [get_age_color(a) for a in ages]
    colors_f = [get_age_color(a) for a in ages]
    
    # コホート線のshapes
    def make_shapes(year):
        shapes = []
        if cohorts:
            for name, by in cohorts:
                a = cohort_age(year, by)
                if 0 <= a <= max_age:
                    shapes.append(dict(type="line", x0=-xmax, x1=xmax, y0=a, y1=a,
                                       line=dict(color="black", width=1.5, dash="dash")))
        return shapes

    def make_annotations(year):
        annotations = []
        if cohorts:
            for name, by in cohorts:
                a = cohort_age(year, by)
                if 0 <= a <= max_age:
                    annotations.append(
                        dict(x=-xmax*0.9, y=a+2, text=f"{name}({a}才)",
                             showarrow=False, font=dict(color="black", size=10), opacity=0.7)
                    )
        return annotations
    
    fig = go.Figure(
        data=[
            go.Bar(y=ages, x=-m0, orientation="h", name="男性",
                   marker=dict(color=colors_m)),
            go.Bar(y=ages, x=f0, orientation="h", name="女性",
                   marker=dict(color=colors_f)),
        ],
        layout=go.Layout(
            title=dict(
                text=f"<b>{years[0]}年</b>　総人口: {total0/1e4:.1f}万人",
                font=dict(size=18)
            ),
            xaxis=dict(title="人口（万人）", range=[-xmax, xmax],
                       tickvals=list(range(-int(xmax), int(xmax)+1, 20)),
                       ticktext=[str(abs(x)) for x in range(-int(xmax), int(xmax)+1, 20)]),
            yaxis=dict(title="年齢", range=[0, max_age]),
            bargap=0,
            barmode="overlay",
            shapes=make_shapes(years[0]),
            annotations=make_annotations(years[0])
        )
    )
    
    # フレーム作成
    frames = []
    for y in years:
        m, f, total = arrays_for_year(y)
        frames.append(go.Frame(
            name=str(y),
            data=[
                go.Bar(y=ages, x=-m, marker=dict(color=colors_m)),
                go.Bar(y=ages, x=f, marker=dict(color=colors_f)),
            ],
            layout=go.Layout(
                title=dict(
                    text=f"<b>{y}年</b>　総人口: {total/1e4:.1f}万人"
                ),
                shapes=make_shapes(y),
                annotations=make_annotations(y)
            )
        ))
    
    fig.frames = frames
    
    # コントロール
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0.0,
            x=0.05,
            xanchor="left",
            buttons=[
                dict(label="▶ 再生",
                     method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ 停止",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=16),
                prefix="年: ",
                visible=True,
                xanchor="center"
            ),
            pad=dict(b=10, t=50),
            len=0.85,
            x=0.1,
            y=0.0,
            steps=[dict(
                method="animate",
                args=[[str(y)], dict(mode="immediate",
                                     frame=dict(duration=0, redraw=True))],
                label=str(y)
            ) for y in years]
        )]
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, auto_play=False, include_plotlyjs=True)
    
    # ホイール/キーボード対応JSを埋め込み
    inject_wheel_scrub(out_path, years)
    
    print(f"✓ Interactive HTML saved: {out_path}")
    print(f"✓ ホイール/矢印キーで年送り可能")


# -------------------------
# レキシス面
# -------------------------
def make_lexis_surface(df: pd.DataFrame, years: list[int], out_path: Path,
                       max_age: int = 100, sex: str = "both",
                       cohorts: list[tuple[str, int]] | None = None) -> None:
    setup_japanese_font()
    
    data_type = detect_data_type(df)
    print(f"✓ データ形式を自動判定: {data_type}")
    print(f"✓ 対象年: {years[0]}〜{years[-1]} ({len(years)}年分)")
    
    ages = np.arange(0, max_age + 1)
    
    pop_matrix = np.zeros((len(ages), len(years)))
    
    for j, y in enumerate(years):
        d = df[df["year"] == y]
        if sex == "both":
            m, f = expand_to_single_year(d, max_age, data_type)
            pop = m + f
        elif sex == "M":
            m, _ = expand_to_single_year(d, max_age, data_type)
            pop = m
        else:
            _, f = expand_to_single_year(d, max_age, data_type)
            pop = f
        pop_matrix[:, j] = pop / 1e4
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(pop_matrix, aspect='auto', origin='lower',
                   extent=[years[0]-0.5, years[-1]+0.5, 0, max_age],
                   cmap='YlOrRd')
    
    ax.set_xlabel("年", fontsize=12)
    ax.set_ylabel("年齢", fontsize=12)
    
    sex_label = {"both": "男女計", "M": "男性", "F": "女性"}[sex]
    ax.set_title(f"日本の人口推移（レキシス面） - {sex_label}", fontsize=14, fontweight="bold")
    
    cbar = fig.colorbar(im, ax=ax, label="人口（万人）")
    
    # コホート線
    if cohorts:
        for name, by in cohorts:
            cohort_ages = [cohort_age(y, by) for y in years]
            valid = [(y, a) for y, a in zip(years, cohort_ages) if 0 <= a <= max_age]
            if valid:
                ax.plot([v[0] for v in valid], [v[1] for v in valid],
                       'w-', alpha=0.8, lw=2, label=name)
        ax.legend(loc="upper left", fontsize=10)
    else:
        # デフォルトのコホート線
        for start_year in range(years[0], years[-1], 10):
            if start_year in years:
                cohort_ages = [50 + (y - start_year) for y in years if start_year <= y]
                cohort_years = [y for y in years if start_year <= y]
                valid = [(y, a) for y, a in zip(cohort_years, cohort_ages) if 0 <= a <= max_age]
                if valid:
                    ax.plot([v[0] for v in valid], [v[1] for v in valid],
                           'w--', alpha=0.5, lw=1)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"✓ Lexis surface saved: {out_path}")


# -------------------------
# メイン
# -------------------------
def main():
    p = argparse.ArgumentParser(
        description="日本人口ピラミッド自動生成ツール v7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
出力モード:
  memo        静止画メモ（年数可変）
  animation   年次推移アニメーション（GIF/MP4）
  interactive Plotlyインタラクティブ版（HTML）
  lexis       レキシス面ヒートマップ

年の選択:
  --start-year 2020 --end-year 2070 --year-step 5  → 2020,2025,...,2070
  --years 2020,2030,2040,2050,2060                  → 明示指定

コホート追跡:
  --cohort "団塊の世代:1947,団塊Jr:1971"  → 名前付きコホート線
  --cohort-birth-years 1947,1971          → 簡易形式（"XXXX年生"と表示）

使用例:
  # 毎年のアニメーション
  python make_memo.py --input data.csv --out out.gif --mode animation --year-step 1

  # 5年ごとのインタラクティブ版 + 名前付きコホート線
  python make_memo.py --input data.csv --out out.html --mode interactive \\
    --year-step 5 --cohort "団塊の世代:1947,団塊Jr:1971"

  # 10年ごとの静止画メモ
  python make_memo.py --input data.csv --out out.pdf --mode memo \\
    --years 2020,2030,2040,2050,2060,2070
        """
    )
    p.add_argument("--input", required=True, help="入力CSV/Excel")
    p.add_argument("--out", required=True, help="出力ファイル")
    p.add_argument("--mode", choices=["memo", "animation", "interactive", "lexis"], default="memo")
    
    # 年の選択
    p.add_argument("--years", default=None,
                   help="描画する年（カンマ区切り、明示指定）")
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--year-step", type=int, default=1,
                   help="年の刻み（デフォルト: 1=毎年）")
    
    p.add_argument("--max-age", type=int, default=100)
    
    # コホート追跡
    p.add_argument("--cohort", default=None,
                   help="追跡するコホート（名前:出生年 形式、例: 団塊の世代:1947,団塊Jr:1971）")
    p.add_argument("--cohort-birth-years", default=None,
                   help="追跡する出生年（カンマ区切り、例: 1971,1990）※--cohortの簡易版")
    
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--sex", choices=["both", "M", "F"], default="both")
    p.add_argument("--memo-layout", choices=["auto", "fixed5", "grid"], default="auto")
    
    args = p.parse_args()
    
    df = read_population(Path(args.input))
    all_years = sorted(df["year"].unique())
    
    # 年の選択
    explicit_years = None
    if args.years:
        explicit_years = [int(x) for x in args.years.split(",")]
    
    years = select_years(
        all_years,
        start=args.start_year,
        end=args.end_year,
        step=args.year_step,
        explicit=explicit_years
    )
    
    if not years:
        raise ValueError("No years selected. Check --years or --start-year/--end-year/--year-step")
    
    # コホート（名前付き）
    cohorts = None
    if args.cohort:
        cohorts = parse_cohorts(args.cohort)
    elif args.cohort_birth_years:
        # 後方互換性: --cohort-birth-years から変換
        cohorts = parse_cohorts(args.cohort_birth_years)

    if args.mode == "memo":
        make_memo(
            df=df,
            years=years,
            out_path=Path(args.out),
            max_age=args.max_age,
            cohorts=cohorts,
            layout=args.memo_layout,
        )
    elif args.mode == "animation":
        make_animation(
            df=df,
            years=years,
            out_path=Path(args.out),
            max_age=args.max_age,
            fps=args.fps,
            cohorts=cohorts,
        )
    elif args.mode == "interactive":
        make_interactive(
            df=df,
            years=years,
            out_path=Path(args.out),
            max_age=args.max_age,
            cohorts=cohorts,
        )
    elif args.mode == "lexis":
        make_lexis_surface(
            df=df,
            years=years,
            out_path=Path(args.out),
            max_age=args.max_age,
            sex=args.sex,
            cohorts=cohorts,
        )


if __name__ == "__main__":
    main()
