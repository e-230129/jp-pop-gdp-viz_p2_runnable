"""
interactive_enhanced.py - 拡張インタラクティブ人口ピラミッド

機能拡張:
A. インタラクティブ要素
   - ユーザー年齢入力 → 該当位置マーカー
   - アニメーション制御（再生/停止/速度調整）

B. 比較・強調表示
   - 年齢層ハイライト（年少/生産年齢/前期高齢/後期高齢）
   - 前年差の可視化
   - 構成比表示

C. 違和感検出
   - 急激な変化の自動検出・ハイライト

設計方針:
- 処理（データ変換）と表示（HTML/JS生成）を分離
- 各機能をモジュール化し、段階的に有効化可能
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# -------------------------
# 設定クラス
# -------------------------
@dataclass
class PyramidConfig:
    """ピラミッド設定"""
    max_age: int = 105
    x_max: float = 130.0  # 万人単位

    # 年齢区分
    age_groups: dict = field(default_factory=lambda: {
        "young": {"name": "年少人口", "range": (0, 14), "color": "#98D54B"},
        "working": {"name": "生産年齢人口", "range": (15, 64), "color": "#5DADE2"},
        "elderly_early": {"name": "前期高齢者", "range": (65, 74), "color": "#F5B041"},
        "elderly_late": {"name": "後期高齢者", "range": (75, 200), "color": "#E74C3C"},
    })

    # コホート設定（名前, 開始年, 終了年）の形式に変更
    # 例: [("団塊の世代", 1947, 1949), ("団塊Jr", 1971, 1974)]
    cohorts: list = field(default_factory=list)

    # データソース情報
    source_info: dict = field(default_factory=lambda: {
        "name": "IPSS 日本の将来推計人口（令和5年推計）",
        "scenario": "出生中位（死亡中位）推計",
        "note": "総人口（外国人を含む）／各年10月1日現在",
    })

    # 検証モード
    verification_enabled: bool = False

    # 拡張機能
    user_age_input: bool = True
    age_group_highlight: bool = True
    year_diff_mode: bool = True
    anomaly_detection: bool = True
    composition_display: bool = True


@dataclass
class YearData:
    """年ごとのデータを保持"""
    year: int
    male_pop: np.ndarray  # 年齢別男性人口（万人）
    female_pop: np.ndarray  # 年齢別女性人口（万人）

    @property
    def total_pop(self) -> np.ndarray:
        return self.male_pop + self.female_pop

    @property
    def total_sum(self) -> float:
        return float(self.total_pop.sum())

    def composition_ratios(self, config: PyramidConfig) -> dict:
        """年齢区分別の構成比を計算"""
        total = self.total_sum
        if total == 0:
            return {}

        ratios = {}
        for key, group in config.age_groups.items():
            start, end = group["range"]
            end = min(end, len(self.total_pop) - 1)
            group_sum = self.total_pop[start:end+1].sum()
            ratios[key] = {
                "name": group["name"],
                "count": float(group_sum),
                "ratio": float(group_sum / total),
            }
        return ratios


# -------------------------
# データ処理
# -------------------------
class DataProcessor:
    """
    データ処理クラス

    元データを描画用に変換する。
    処理と表示を分離するための中間層。
    """

    def __init__(self, df: pd.DataFrame, config: PyramidConfig):
        self.df = df
        self.config = config
        self.years_data: dict[int, YearData] = {}
        self._process_all_years()

    def _normalize_sex(self, x) -> str:
        s = str(x).strip()
        if s in {"M", "m", "Male", "male", "男", "男性", "1"}:
            return "M"
        if s in {"F", "f", "Female", "female", "女", "女性", "2"}:
            return "F"
        raise ValueError(f"Unknown sex value: {x!r}")

    def _detect_data_type(self) -> Literal["single_year", "five_year"]:
        sample_year = self.df["year"].iloc[0]
        ages = sorted(self.df[self.df["year"] == sample_year]["age"].unique())
        ages_under_50 = [a for a in ages if a <= 50]

        if len(ages_under_50) >= 40:
            return "single_year"
        return "five_year"

    def _expand_to_single_year(self, df_year: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """年齢別人口を1歳刻み配列に展開"""
        max_age = self.config.max_age
        m = np.zeros(max_age + 1)
        f = np.zeros(max_age + 1)

        m_data = df_year[df_year["sex"] == "M"].set_index("age")["pop"].sort_index()
        f_data = df_year[df_year["sex"] == "F"].set_index("age")["pop"].sort_index()

        data_type = self._detect_data_type()

        if data_type == "single_year":
            for age in range(max_age + 1):
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
                        age_start = 100
                        age_end = max_age
                    else:
                        age_start = repr_age
                        age_end = repr_age + 4
                else:
                    # 上端表現: 4→0-4, 9→5-9, ...
                    if repr_age <= 4:
                        age_start, age_end = 0, repr_age
                    elif repr_age >= 100:
                        age_start = 100
                        age_end = max_age
                    else:
                        age_start = repr_age - 4
                        age_end = repr_age

                pop_m = float(m_data.get(repr_age, 0))
                pop_f = float(f_data.get(repr_age, 0))
                n_ages = age_end - age_start + 1

                for a in range(age_start, min(age_end + 1, max_age + 1)):
                    m[a] += pop_m / n_ages
                    f[a] += pop_f / n_ages

        return m, f

    def _process_all_years(self) -> None:
        """全年のデータを処理"""
        # データのクリーニング
        df = self.df.copy()
        df["sex"] = df["sex"].apply(self._normalize_sex)
        df["age"] = df["age"].astype(str).str.replace("+", "", regex=False).astype(int)

        for year in sorted(df["year"].unique()):
            df_year = df[df["year"] == year]
            m, f = self._expand_to_single_year(df_year)

            self.years_data[year] = YearData(
                year=year,
                male_pop=m / 1e4,  # 万人単位に変換
                female_pop=f / 1e4,
            )

    def get_year_data(self, year: int) -> YearData:
        return self.years_data[year]

    def get_all_years(self) -> list[int]:
        return sorted(self.years_data.keys())

    def detect_anomalies(self, threshold: float = 0.15) -> list[dict]:
        """
        急激な変化を検出

        前年との比較で閾値を超える変化があった年齢帯を特定。
        """
        anomalies = []
        years = self.get_all_years()

        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            year_diff = curr_year - prev_year

            prev_data = self.years_data[prev_year]
            curr_data = self.years_data[curr_year]

            for age in range(self.config.max_age - year_diff):
                # コホート追跡
                prev_pop = prev_data.total_pop[age]
                curr_pop = curr_data.total_pop[age + year_diff]

                if prev_pop > 0:
                    change = (curr_pop - prev_pop) / prev_pop

                    # 大幅な変化を検出
                    if abs(change) > threshold and prev_pop > 5:  # 5万人以上
                        anomalies.append({
                            "year": int(curr_year),
                            "age": int(age + year_diff),
                            "change": float(change),
                            "prev_pop": float(prev_pop),
                            "curr_pop": float(curr_pop),
                            "type": "increase" if change > 0 else "decrease",
                        })

        return anomalies

    def compute_year_diff(self, year1: int, year2: int) -> dict:
        """2年間の差分を計算"""
        if year1 not in self.years_data or year2 not in self.years_data:
            return {}

        d1 = self.years_data[year1]
        d2 = self.years_data[year2]

        return {
            "male_diff": (d2.male_pop - d1.male_pop).tolist(),
            "female_diff": (d2.female_pop - d1.female_pop).tolist(),
            "total_diff": (d2.total_pop - d1.total_pop).tolist(),
            "total_change": d2.total_sum - d1.total_sum,
        }


# -------------------------
# HTML/JS生成
# -------------------------
class InteractiveHTMLGenerator:
    """
    インタラクティブHTML生成クラス

    処理済みデータからPlotly HTMLを生成。
    """

    def __init__(self, processor: DataProcessor, config: PyramidConfig):
        self.processor = processor
        self.config = config

    def _get_age_color(self, age: int) -> str:
        """年齢に応じた色を返す"""
        import matplotlib.colors as mcolors

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

    def _format_population(self, pop_man: float) -> str:
        """人口を日本語形式でフォーマット"""
        if pop_man >= 10000:
            oku = int(pop_man // 10000)
            rest = pop_man - oku * 10000
            return f"{oku}億{rest:.1f}万人"
        return f"{pop_man:.1f}万人"

    def generate(self, output_path: Path) -> None:
        """HTMLファイルを生成"""
        import plotly.graph_objects as go

        years = self.processor.get_all_years()
        ages = np.arange(0, self.config.max_age + 1)
        xmax = self.config.x_max

        # 初期データ
        y0_data = self.processor.get_year_data(years[0])
        colors = [self._get_age_color(a) for a in ages]

        # コホート帯のshapes生成関数（線→帯に変更）
        cohort_colors = [
            "rgba(255, 99, 71, 0.25)",   # 団塊: 薄い赤
            "rgba(65, 105, 225, 0.25)",  # 団塊Jr: 薄い青
            "rgba(50, 205, 50, 0.25)",   # 追加用: 薄い緑
            "rgba(255, 165, 0, 0.25)",   # 追加用: 薄いオレンジ
        ]

        def make_shapes(year):
            shapes = []
            if self.config.cohorts:
                for i, cohort in enumerate(self.config.cohorts):
                    # 新形式: (名前, 開始年, 終了年) または旧形式: (名前, 年)
                    if len(cohort) == 3:
                        name, birth_start, birth_end = cohort
                    else:
                        name, birth_year = cohort
                        birth_start = birth_end = birth_year

                    age_start = year - birth_end
                    age_end = year - birth_start

                    # 表示範囲内かチェック
                    if age_end >= 0 and age_start <= self.config.max_age:
                        age_start = max(0, age_start)
                        age_end = min(self.config.max_age, age_end)

                        color = cohort_colors[i % len(cohort_colors)]
                        # 帯（rect）で表示
                        shapes.append(dict(
                            type="rect",
                            x0=-xmax, x1=xmax,
                            y0=age_start - 0.5, y1=age_end + 0.5,
                            fillcolor=color,
                            line=dict(color=color.replace("0.25", "0.6"), width=1),
                            layer="below",
                        ))
            return shapes

        def make_annotations(year):
            annotations = []
            if self.config.cohorts:
                for i, cohort in enumerate(self.config.cohorts):
                    if len(cohort) == 3:
                        name, birth_start, birth_end = cohort
                        birth_label = f"{birth_start}-{str(birth_end)[-2:]}年生"
                    else:
                        name, birth_year = cohort
                        birth_start = birth_end = birth_year
                        birth_label = f"{birth_year}年生"

                    age_start = year - birth_end
                    age_end = year - birth_start
                    age_mid = (age_start + age_end) / 2

                    if age_end >= 0 and age_start <= self.config.max_age:
                        age_mid = max(3, min(self.config.max_age - 3, age_mid))
                        age_display = f"{age_start}-{age_end}歳" if age_start != age_end else f"{age_start}歳"

                        annotations.append(dict(
                            x=-xmax*0.85, y=age_mid + 2,
                            text=f"<b>{name}</b><br>{birth_label}（{age_display}）",
                            showarrow=False,
                            font=dict(color="#333", size=10),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderpad=4,
                            align="left",
                        ))
            return annotations

        # 構成比テキスト
        def make_composition_text(year):
            data = self.processor.get_year_data(year)
            ratios = data.composition_ratios(self.config)

            lines = []
            for key, group in ratios.items():
                lines.append(f"{group['name']}: {group['ratio']:.1%}")

            return "<br>".join(lines)

        # ホバー用のカスタムデータを準備（男性人口, 女性人口, 合計, 総人口に対する比率）
        def make_hover_data(data: YearData):
            total_sum = data.total_sum
            male_data = []
            female_data = []
            for i in range(len(data.male_pop)):
                m = data.male_pop[i]
                f = data.female_pop[i]
                t = m + f
                pct = (t / total_sum * 100) if total_sum > 0 else 0
                male_data.append([m, f, t, pct])
                female_data.append([m, f, t, pct])
            return male_data, female_data

        y0_male_hover, y0_female_hover = make_hover_data(y0_data)

        # Figure作成
        fig = go.Figure(
            data=[
                go.Bar(
                    y=ages, x=-y0_data.male_pop, orientation="h",
                    name="男性", marker=dict(color=colors),
                    hovertemplate=(
                        "<b>%{y}歳</b><br>"
                        "男性: %{customdata[0]:.1f}万人<br>"
                        "女性: %{customdata[1]:.1f}万人<br>"
                        "合計: %{customdata[2]:.1f}万人<br>"
                        "構成比: %{customdata[3]:.2f}%"
                        "<extra></extra>"
                    ),
                    customdata=y0_male_hover,
                ),
                go.Bar(
                    y=ages, x=y0_data.female_pop, orientation="h",
                    name="女性", marker=dict(color=colors),
                    hovertemplate=(
                        "<b>%{y}歳</b><br>"
                        "男性: %{customdata[0]:.1f}万人<br>"
                        "女性: %{customdata[1]:.1f}万人<br>"
                        "合計: %{customdata[2]:.1f}万人<br>"
                        "構成比: %{customdata[3]:.2f}%"
                        "<extra></extra>"
                    ),
                    customdata=y0_female_hover,
                ),
            ],
            layout=go.Layout(
                title=dict(
                    text=(
                        f"<b>{years[0]}年</b>　総人口: {self._format_population(y0_data.total_sum)}<br>"
                        f"<span style='font-size:12px;color:#666'>"
                        f"{self.config.source_info['name']}　{self.config.source_info['scenario']}"
                        f"</span>"
                    ),
                    font=dict(size=20),
                    x=0.5,
                ),
                xaxis=dict(
                    title="人口（万人）",
                    range=[-xmax, xmax],
                    tickvals=list(range(-int(xmax), int(xmax)+1, 20)),
                    ticktext=[str(abs(x)) for x in range(-int(xmax), int(xmax)+1, 20)],
                    gridcolor="rgba(0,0,0,0.1)",
                ),
                yaxis=dict(
                    title="年齢",
                    range=[0, self.config.max_age],
                    gridcolor="rgba(0,0,0,0.1)",
                ),
                bargap=0,
                barmode="overlay",
                shapes=make_shapes(years[0]),
                annotations=make_annotations(years[0]),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
                margin=dict(l=60, r=200, t=80, b=120),
            )
        )

        # フレーム作成
        frames = []
        for y in years:
            data = self.processor.get_year_data(y)
            male_hover, female_hover = make_hover_data(data)
            frames.append(go.Frame(
                name=str(y),
                data=[
                    go.Bar(
                        y=ages, x=-data.male_pop,
                        marker=dict(color=colors),
                        customdata=male_hover,
                    ),
                    go.Bar(
                        y=ages, x=data.female_pop,
                        marker=dict(color=colors),
                        customdata=female_hover,
                    ),
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            f"<b>{y}年</b>　総人口: {self._format_population(data.total_sum)}<br>"
                            f"<span style='font-size:12px;color:#666'>"
                            f"{self.config.source_info['name']}　{self.config.source_info['scenario']}"
                            f"</span>"
                        )
                    ),
                    shapes=make_shapes(y),
                    annotations=make_annotations(y),
                )
            ))

        fig.frames = frames

        # コントロール追加
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    y=-0.15,
                    x=0,
                    xanchor="left",
                    direction="right",
                    buttons=[
                        dict(
                            label="▶ 再生",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            )],
                        ),
                        dict(
                            label="⏸ 停止",
                            method="animate",
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            )],
                        ),
                    ],
                ),
                dict(
                    type="dropdown",
                    showactive=True,
                    y=-0.15,
                    x=0.18,
                    xanchor="left",
                    buttons=[
                        dict(label="速度: 標準", method="skip", args=[]),
                        dict(label="速度: 速い", method="skip", args=[]),
                        dict(label="速度: 遅い", method="skip", args=[]),
                    ],
                ),
            ],
            sliders=[dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=16),
                    prefix="",
                    visible=True,
                    xanchor="center",
                ),
                pad=dict(b=10, t=50),
                len=0.85,
                x=0.05,
                y=0,
                steps=[dict(
                    method="animate",
                    args=[[str(y)], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                    )],
                    label=str(y),
                ) for y in years],
            )],
        )

        # HTML出力
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path, auto_play=False, include_plotlyjs=True)

        # 拡張JS注入
        self._inject_enhanced_js(output_path, years)

        print(f"[OK] Enhanced interactive HTML saved: {output_path}")

    def _inject_enhanced_js(self, html_path: Path, years: list[int]) -> None:
        """拡張機能のJavaScriptを注入"""

        # numpy int64をintに変換
        years = [int(y) for y in years]

        # 構成比データ
        composition_data = {}
        for y in years:
            data = self.processor.get_year_data(y)
            composition_data[y] = data.composition_ratios(self.config)

        # 異常検出データ
        anomalies = self.processor.detect_anomalies() if self.config.anomaly_detection else []

        js = f"""
<style>
/* コントロールパネル */
.control-panel {{
    position: fixed;
    top: 10px;
    right: 10px;
    background: rgba(255,255,255,0.95);
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    width: 280px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    z-index: 10000;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}}

.control-panel h4 {{
    margin: 0 0 10px 0;
    color: #333;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
}}

.control-group {{
    margin-bottom: 12px;
}}

.control-group label {{
    display: block;
    margin-bottom: 4px;
    color: #666;
}}

.control-group input[type="number"] {{
    width: 80px;
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
}}

.control-group input[type="checkbox"] {{
    margin-right: 8px;
}}

.birth-year-display {{
    margin-top: 8px;
    padding: 10px;
    background: linear-gradient(135deg, #fff5f5, #ffe0e0);
    border: 2px solid #E74C3C;
    border-radius: 8px;
    text-align: center;
    display: none;
}}

.birth-year-display .age-info {{
    font-size: 18px;
    font-weight: bold;
    color: #E74C3C;
}}

.birth-year-display .birth-info {{
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}}

.composition-panel {{
    margin-top: 10px;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 4px;
}}

.composition-bar {{
    height: 20px;
    display: flex;
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}}

.composition-segment {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 11px;
    font-weight: bold;
    text-shadow: 0 1px 1px rgba(0,0,0,0.3);
}}

.anomaly-alert {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px 15px;
    max-width: 300px;
    font-size: 12px;
    z-index: 10000;
    display: none;
}}

.btn-toggle {{
    background: #007bff;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}}

.btn-toggle:hover {{
    background: #0056b3;
}}

.btn-clear {{
    background: #dc3545;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    margin-left: 5px;
}}

.btn-clear:hover {{
    background: #c82333;
}}

.btn-screenshot {{
    background: #28a745;
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    width: 100%;
    margin-top: 10px;
}}

.btn-screenshot:hover {{
    background: #218838;
}}

.btn-screenshot:active {{
    background: #1e7e34;
}}

.screenshot-options {{
    margin-top: 8px;
    display: flex;
    gap: 8px;
}}

.screenshot-options select {{
    flex: 1;
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
}}
</style>

<script>
(function() {{
    const gd = document.querySelector('.js-plotly-plot');
    if (!gd) return;

    const years = {json.dumps(years)};
    const compositionData = {json.dumps(composition_data, ensure_ascii=False)};
    const anomalies = {json.dumps(anomalies)};
    const maxAge = {self.config.max_age};
    const xMax = {self.config.x_max};
    let currentIdx = 0;
    let currentYear = years[0];
    let animationSpeed = 300;

    // ユーザー出生年（追跡用）
    let userBirthYear = null;

    // コントロールパネル作成
    const panel = document.createElement('div');
    panel.className = 'control-panel';
    panel.innerHTML = `
        <h4>人口ピラミッド操作</h4>

        <div class="source-info" style="font-size:11px;color:#666;margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px;">
            <div><strong>{self.config.source_info['name']}</strong></div>
            <div>{self.config.source_info['scenario']}</div>
            <div style="font-size:10px;margin-top:4px;">{self.config.source_info['note']}</div>
        </div>

        <div class="control-group">
            <label>あなたの生まれ年</label>
            <input type="number" id="birthYear" min="1900" max="2070" placeholder="例: 1985">
            <button class="btn-toggle" id="trackBtn">追跡</button>
            <button class="btn-clear" id="clearBtn">解除</button>
        </div>

        <div class="birth-year-display" id="birthYearDisplay">
            <div class="age-info" id="ageInfo"></div>
            <div class="birth-info" id="birthInfo"></div>
        </div>

        <div class="control-group">
            <label><strong>年齢区分（凡例）</strong></label>
            <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">
                <span style="background:#98D54B;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">0-14 年少</span>
                <span style="background:#5DADE2;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">15-64 生産年齢</span>
                <span style="background:#F5B041;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">65-74 前期高齢</span>
                <span style="background:#E74C3C;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">75+ 後期高齢</span>
            </div>
        </div>

        <div class="composition-panel">
            <strong>人口構成比</strong>
            <div class="composition-bar" id="compositionBar"></div>
            <div id="compositionText"></div>
        </div>

        <div class="control-group" style="margin-top: 15px;">
            <label>X軸スケール</label>
            <select id="scaleSelect">
                <option value="fixed" selected>固定（全期間最大）</option>
                <option value="auto">年ごと自動</option>
            </select>
        </div>

        <div class="control-group">
            <label>再生速度</label>
            <select id="speedSelect">
                <option value="500">遅い</option>
                <option value="300" selected>標準</option>
                <option value="150">速い</option>
                <option value="50">高速</option>
            </select>
        </div>

        <div class="control-group" style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 15px;">
            <label>スクリーンショット</label>
            <div class="screenshot-options">
                <select id="screenshotFormat">
                    <option value="png">PNG</option>
                    <option value="jpeg">JPEG</option>
                    <option value="svg">SVG</option>
                </select>
                <select id="screenshotScale">
                    <option value="1">1x</option>
                    <option value="2" selected>2x (高解像度)</option>
                    <option value="3">3x</option>
                </select>
            </div>
            <button class="btn-screenshot" id="screenshotBtn">画面を保存</button>
        </div>

        <div class="shortcuts-info" style="font-size:10px;color:#888;margin-top:12px;padding-top:8px;border-top:1px solid #eee;">
            <strong>キーボード操作</strong><br>
            ←→: 年移動 (Shift: 5年)<br>
            Space: 再生/停止<br>
            Home/End: 最初/最後<br>
            ホイール: 年移動 (Shift: 5年)
        </div>
    `;
    document.body.appendChild(panel);

    // 異常アラート
    const alertBox = document.createElement('div');
    alertBox.className = 'anomaly-alert';
    alertBox.innerHTML = '<strong>構造変化を検出</strong><p id="anomalyText"></p>';
    document.body.appendChild(alertBox);

    // ユーザーマーカーを更新（年度変更時に呼ばれる）
    function updateUserMarker(year) {{
        if (userBirthYear === null) return;

        const age = year - userBirthYear;
        const display = document.getElementById('birthYearDisplay');
        const ageInfo = document.getElementById('ageInfo');
        const birthInfo = document.getElementById('birthInfo');

        // 年齢表示を更新
        if (age >= 0 && age <= maxAge) {{
            ageInfo.textContent = `${{age}}歳`;
            birthInfo.textContent = `${{userBirthYear}}年生まれ / ${{year}}年時点`;
            display.style.display = 'block';

            // グラフ上のマーカーを更新
            const baseShapes = (gd.layout.shapes || []).filter(s => !s._userMarker);
            const baseAnnotations = (gd.layout.annotations || []).filter(a => !a._userMarker);

            Plotly.relayout(gd, {{
                shapes: [
                    ...baseShapes,
                    {{
                        type: 'line',
                        x0: -xMax, x1: xMax,
                        y0: age, y1: age,
                        line: {{ color: '#E74C3C', width: 3, dash: 'dot' }},
                        _userMarker: true
                    }}
                ],
                annotations: [
                    ...baseAnnotations,
                    {{
                        x: xMax * 0.7, y: age,
                        xref: 'x', yref: 'y',
                        text: `<b>${{userBirthYear}}年生 ${{age}}歳</b>`,
                        showarrow: true,
                        arrowhead: 2,
                        arrowcolor: '#E74C3C',
                        ax: 40, ay: -25,
                        font: {{ color: '#E74C3C', size: 14, family: 'sans-serif' }},
                        bgcolor: 'rgba(255,255,255,0.95)',
                        bordercolor: '#E74C3C',
                        borderwidth: 2,
                        borderpad: 5,
                        _userMarker: true
                    }}
                ]
            }});
        }} else if (age < 0) {{
            ageInfo.textContent = `まだ生まれていません`;
            birthInfo.textContent = `${{userBirthYear}}年生まれ（${{-age}}年後に誕生）`;
            display.style.display = 'block';
            // マーカーを非表示
            const baseShapes = (gd.layout.shapes || []).filter(s => !s._userMarker);
            const baseAnnotations = (gd.layout.annotations || []).filter(a => !a._userMarker);
            Plotly.relayout(gd, {{ shapes: baseShapes, annotations: baseAnnotations }});
        }} else {{
            ageInfo.textContent = `${{maxAge}}歳を超えています`;
            birthInfo.textContent = `${{userBirthYear}}年生まれ`;
            display.style.display = 'block';
            // マーカーを非表示
            const baseShapes = (gd.layout.shapes || []).filter(s => !s._userMarker);
            const baseAnnotations = (gd.layout.annotations || []).filter(a => !a._userMarker);
            Plotly.relayout(gd, {{ shapes: baseShapes, annotations: baseAnnotations }});
        }}
    }}

    // 追跡開始ボタン
    document.getElementById('trackBtn').addEventListener('click', () => {{
        const birthYear = parseInt(document.getElementById('birthYear').value);
        if (isNaN(birthYear) || birthYear < 1900 || birthYear > 2070) {{
            alert('1900〜2070の間で生まれ年を入力してください');
            return;
        }}

        userBirthYear = birthYear;
        updateUserMarker(currentYear);
    }});

    // クリアボタン
    document.getElementById('clearBtn').addEventListener('click', () => {{
        userBirthYear = null;
        document.getElementById('birthYearDisplay').style.display = 'none';

        // マーカーを削除
        const baseShapes = (gd.layout.shapes || []).filter(s => !s._userMarker);
        const baseAnnotations = (gd.layout.annotations || []).filter(a => !a._userMarker);
        Plotly.relayout(gd, {{ shapes: baseShapes, annotations: baseAnnotations }});
    }});

    // 構成比バーを更新
    function updateComposition(year) {{
        const data = compositionData[year];
        if (!data) return;

        const bar = document.getElementById('compositionBar');
        const text = document.getElementById('compositionText');

        const segments = [
            {{ name: '年少', ratio: data.young?.ratio || 0, color: '#98D54B' }},
            {{ name: '生産年齢', ratio: data.working?.ratio || 0, color: '#5DADE2' }},
            {{ name: '前期高齢', ratio: data.elderly_early?.ratio || 0, color: '#F5B041' }},
            {{ name: '後期高齢', ratio: data.elderly_late?.ratio || 0, color: '#E74C3C' }},
        ];

        bar.innerHTML = segments.map(s =>
            `<div class="composition-segment" style="width:${{s.ratio*100}}%;background:${{s.color}}">${{(s.ratio*100).toFixed(0)}}%</div>`
        ).join('');

        text.innerHTML = segments.map(s =>
            `<small>${{s.name}}: ${{(s.ratio*100).toFixed(1)}}%</small>`
        ).join(' | ');
    }}

    // 異常チェック
    function checkAnomalies(year) {{
        const yearAnomalies = anomalies.filter(a => a.year === year);
        if (yearAnomalies.length > 0) {{
            const top = yearAnomalies[0];
            document.getElementById('anomalyText').textContent =
                `${{top.age}}歳付近で${{top.type === 'increase' ? '増加' : '減少'}}（${{(top.change*100).toFixed(1)}}%変化）`;
            alertBox.style.display = 'block';
            setTimeout(() => alertBox.style.display = 'none', 3000);
        }}
    }}

    // 年切り替え時の処理
    function onYearChange(year) {{
        currentYear = year;
        updateComposition(year);
        checkAnomalies(year);
        updateUserMarker(year);  // ユーザーマーカーを更新
    }}

    // ホイール/キーボード操作
    function goToYear(i) {{
        currentIdx = Math.max(0, Math.min(years.length - 1, i));
        const year = years[currentIdx];
        Plotly.animate(gd, [String(year)], {{
            mode: 'immediate',
            frame: {{ duration: 0, redraw: true }},
            transition: {{ duration: 0 }}
        }});
        onYearChange(year);
    }}

    // 再生状態管理
    let isPlaying = false;
    let playInterval = null;

    function togglePlay() {{
        isPlaying = !isPlaying;
        if (isPlaying) {{
            playInterval = setInterval(() => {{
                if (currentIdx < years.length - 1) {{
                    goToYear(currentIdx + 1);
                }} else {{
                    // 最後まで行ったら停止
                    isPlaying = false;
                    clearInterval(playInterval);
                }}
            }}, animationSpeed);
        }} else {{
            clearInterval(playInterval);
        }}
    }}

    // ホイール操作（Shift押下で±5年）
    window.addEventListener('wheel', (e) => {{
        const step = e.shiftKey ? 5 : 1;
        goToYear(currentIdx + (e.deltaY > 0 ? step : -step));
        e.preventDefault();
    }}, {{ passive: false }});

    // キーボード操作
    window.addEventListener('keydown', (e) => {{
        // テキスト入力中は無視
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'ArrowRight') goToYear(currentIdx + (e.shiftKey ? 5 : 1));
        if (e.key === 'ArrowLeft') goToYear(currentIdx - (e.shiftKey ? 5 : 1));
        if (e.key === 'Home') goToYear(0);
        if (e.key === 'End') goToYear(years.length - 1);
        if (e.key === ' ' || e.code === 'Space') {{
            e.preventDefault();
            togglePlay();
        }}
    }});

    // スライダー変更を監視
    gd.on('plotly_sliderchange', (e) => {{
        const year = parseInt(e.slider.active);
        currentIdx = years.indexOf(years.find(y => y.toString() === e.step.label));
        if (currentIdx >= 0) {{
            onYearChange(years[currentIdx]);
        }}
    }});

    // 速度変更
    document.getElementById('speedSelect').addEventListener('change', (e) => {{
        animationSpeed = parseInt(e.target.value);
    }});

    // スケール切替
    let scaleMode = 'fixed';
    document.getElementById('scaleSelect').addEventListener('change', (e) => {{
        scaleMode = e.target.value;
        if (scaleMode === 'fixed') {{
            // 固定スケール（全期間の最大値）
            Plotly.relayout(gd, {{
                'xaxis.autorange': false,
                'xaxis.range': [-xMax, xMax]
            }});
        }} else {{
            // 自動スケール（年ごと）
            Plotly.relayout(gd, {{
                'xaxis.autorange': true
            }});
        }}
    }});

    // スクリーンショット
    document.getElementById('screenshotBtn').addEventListener('click', () => {{
        const format = document.getElementById('screenshotFormat').value;
        const scale = parseInt(document.getElementById('screenshotScale').value);
        const filename = `人口ピラミッド_${{currentYear}}年`;

        Plotly.downloadImage(gd, {{
            format: format,
            width: 1200 * scale,
            height: 800 * scale,
            scale: scale,
            filename: filename
        }}).then(() => {{
            console.log('Screenshot saved:', filename);
        }}).catch((err) => {{
            console.error('Screenshot failed:', err);
            alert('スクリーンショットの保存に失敗しました');
        }});
    }});

    // 初期表示
    updateComposition(years[0]);

}})();
</script>
</body>
"""

        html = html_path.read_text(encoding="utf-8")
        html = html.replace("</body>", js)
        html_path.write_text(html, encoding="utf-8")


# -------------------------
# メイン関数
# -------------------------
def create_enhanced_interactive(
    input_path: Path,
    output_path: Path,
    years: list[int] | None = None,
    year_step: int = 1,
    max_age: int = 105,
    cohorts: list[tuple[str, int]] | None = None,
    verification: bool = False,
) -> None:
    """
    拡張版インタラクティブHTMLを生成

    Args:
        input_path: 入力CSVパス
        output_path: 出力HTMLパス
        years: 明示的な年リスト（Noneの場合は全年）
        year_step: 年の刻み
        max_age: 最大年齢
        cohorts: コホートリスト [(名前, 出生年), ...]
        verification: 検証モード有効化
    """
    # データ読み込み
    df = pd.read_csv(input_path)

    # 設定
    config = PyramidConfig(
        max_age=max_age,
        cohorts=cohorts or [],
        verification_enabled=verification,
    )

    # 年の選択
    all_years = sorted(df["year"].unique())
    if years:
        selected_years = [y for y in years if y in all_years]
    else:
        selected_years = [y for y in all_years if (y - all_years[0]) % year_step == 0]

    # 選択された年だけを処理
    df_filtered = df[df["year"].isin(selected_years)]

    # 処理
    processor = DataProcessor(df_filtered, config)

    # 検証（有効な場合）
    if verification:
        from verification import DataVerifier, VerificationReport

        verifier = DataVerifier(df_filtered, max_age)
        for year in selected_years:
            data = processor.get_year_data(year)
            verifier.run_all_verifications(
                year,
                data.male_pop * 1e4,  # 人単位に戻す
                data.female_pop * 1e4,
            )
        verifier.report.print_summary()

    # HTML生成
    generator = InteractiveHTMLGenerator(processor, config)
    generator.generate(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="拡張版インタラクティブ人口ピラミッド")
    parser.add_argument("--input", required=True, help="入力CSVパス")
    parser.add_argument("--out", required=True, help="出力HTMLパス")
    parser.add_argument("--year-step", type=int, default=1, help="年の刻み")
    parser.add_argument("--max-age", type=int, default=105, help="最大年齢")
    parser.add_argument("--cohort", help="コホート（名前:開始年-終了年 または 名前:年 形式、カンマ区切り）")
    parser.add_argument("--verify", action="store_true", help="検証モード")

    args = parser.parse_args()

    # コホートパース（新形式: 名前:開始年-終了年 または 名前:年）
    cohorts = None
    if args.cohort:
        cohorts = []
        for item in args.cohort.split(","):
            if ":" in item:
                name, years_part = item.split(":", 1)
                name = name.strip()
                years_part = years_part.strip()

                if "-" in years_part:
                    # 範囲形式: 1947-1949
                    start, end = years_part.split("-", 1)
                    cohorts.append((name, int(start.strip()), int(end.strip())))
                else:
                    # 単一年形式: 1947
                    year = int(years_part)
                    cohorts.append((name, year, year))
            else:
                # 年のみ
                year = int(item.strip())
                cohorts.append((f"{year}年生", year, year))

    create_enhanced_interactive(
        input_path=Path(args.input),
        output_path=Path(args.out),
        year_step=args.year_step,
        max_age=args.max_age,
        cohorts=cohorts,
        verification=args.verify,
    )
