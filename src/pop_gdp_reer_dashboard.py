# -*- coding: utf-8 -*-
"""
pop_gdp_reer_dashboard.py

人口構成・GDP・実効為替レート（REER）を統合したインタラクティブダッシュボード

特徴:
- 人口ピラミッド（年齢構成）の時系列表示
- GDP・REERの推移グラフ
- スライダーで年を操作すると全グラフが連動
- 主要イベント（人口ピーク、バブル崩壊等）のマーカー
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse

# データ構築モジュール（唯一のSSOT）
from build_macro_data import (
    build_macro_dataframe,
    get_data_metadata,
    get_projection_start_year,
    get_actual_data_end,
)

# ---------------------------------
# 表示設定定数
# ---------------------------------
YEAR_RANGE_MIN = 1948  # X軸の最小年
YEAR_RANGE_MAX = 2075  # X軸の最大年
POP_RANGE_MAX = 14000  # 人口Y軸の最大値（万人）
REER_RANGE_MAX = 180   # REER Y軸の最大値
RATIO_RANGE_MAX = 105  # 構成比Y軸の最大値（%）
BASE_YEAR = 1995       # 指数化の基準年
# 注: 推計境界はDataFrameの*_IsProjectionフラグから動的に取得（SSOT一本化）

# ---------------------------------
# カラースキーム
# ---------------------------------
COLORS = {
    "young": "#6AA84F",           # 緑（年少人口 0-14）
    "working": "#2196F3",         # 青（生産年齢人口 全体）
    "working_15_19": "#CFE2F3",   # 最も薄い水色（学生層 15-19歳）
    "working_20_34": "#6FA8DC",   # 明るい青（若手・形成層 20-34歳）
    "working_35_59": "#0B5394",   # 濃い青（熟練・高所得層 35-59歳）
    "working_60_64": "#3D85C6",   # くすみ青（シニア就労 60-64歳）
    "elderly": "#E69138",         # オレンジ（高齢人口 全体）
    "elderly_65_74": "#E69138",   # オレンジ（前期高齢者 65-74）
    "elderly_75plus": "#B45F06",  # 濃いオレンジ（後期高齢者 75+）
    "gdp": "#E91E63",             # ピンク（GDP）
    "reer": "#9C27B0",            # 紫（REER）
    "cursor": "#FF0000",          # 赤（カーソル線）
}

# 主要イベント
EVENTS = {
    1973: ("第一次オイルショック", "gray"),
    1985: ("プラザ合意", "orange"),
    1990: ("バブル崩壊", "red"),
    1995: ("生産年齢人口ピーク", "blue"),
    2008: ("リーマンショック", "darkred"),
    2011: ("東日本大震災", "purple"),
    2020: ("コロナ危機", "brown"),
}

# GDPシナリオ定義（推計部分のみ表示）
# (シナリオID, 表示名, 色, 線種, デフォルト透明度)
GDP_SCENARIOS = [
    ("gdp_baseline", "ベースライン（+20兆/5年）", "#E91E63", "solid", 1.0),
    ("gdp_cabinet_growth", "内閣府・成長実現ケース", "#4CAF50", "dash", 0.4),
    ("gdp_cabinet_baseline", "内閣府・ベースラインケース", "#FF9800", "dot", 0.4),
]

# REERシナリオ定義（推計部分のみ表示）
# (シナリオID, 表示名, 色, 線種, デフォルト透明度)
REER_SCENARIOS = [
    ("reer_flat", "横ばい", "#9C27B0", "solid", 1.0),
    ("reer_mean_revert", "平均回帰", "#673AB7", "dash", 0.5),
    ("reer_trend", "トレンド継続", "#3F51B5", "dot", 0.5),
]


def get_dataframe(csv_path: Path | None = None, input_unit: str = "man") -> pd.DataFrame:
    """描画用DataFrameを取得（build_macro_dataからの薄いラッパー）"""
    return build_macro_dataframe(csv_path=csv_path, input_unit=input_unit)


def create_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plotlyインタラクティブダッシュボードを生成してHTMLに保存

    Args:
        df: build_macro_dataframe()で作成したマクロデータDataFrame
            必須列: Year, TotalPop, Working, Young, Elderly, GDP, REER,
                   Working_Ratio, Young_Ratio, Elderly_Ratio
        output_path: 出力HTMLファイルのパス

    Returns:
        None（ファイルを出力）

    Raises:
        ValueError: dfに必須列が不足している場合
    """
    # メタデータ取得（データソース情報）
    metadata = get_data_metadata()
    gdp_source = metadata.get("gdp", {}).get("source", "不明")
    reer_source = metadata.get("reer", {}).get("source", "不明")
    # 推計仮定
    projection_info = metadata.get("projection", {})
    gdp_assumption = projection_info.get("gdp_assumption", "仮定値")
    reer_assumption = projection_info.get("reer_assumption", "仮定値")

    # 推計境界をDataFrameフラグから動的に取得（SSOT一本化）
    gdp_actual_end = get_actual_data_end(df, "GDP")
    reer_actual_end = get_actual_data_end(df, "REER")

    years = df["Year"].tolist()
    df_reer_all = df[df["REER"].notna()]

    # 軸範囲
    min_year = int(df["Year"].min())
    x_range_start = YEAR_RANGE_MIN
    x_range_end = YEAR_RANGE_MAX
    reer_min_year = int(df_reer_all["Year"].min()) if len(df_reer_all) > 0 else min_year
    gdp_max = float(df["GDP"].max()) * 1.1  # 10%マージン
    reer_max = float(df_reer_all["REER"].max()) * 1.2 if len(df_reer_all) > 0 else REER_RANGE_MAX
    reer_y_max = max(reer_max, REER_RANGE_MAX)

    # GDP_per_working は build_macro_data の派生指標を使用（SSOT）
    df = df.copy()

    # 年齢構成比を正規化（合計が100%になるよう調整）
    total_ratio = df["Young_Ratio"] + df["Working_Ratio"] + df["Elderly_Ratio"]
    df["Young_Ratio"] = df["Young_Ratio"] / total_ratio * 100
    df["Working_Ratio"] = df["Working_Ratio"] / total_ratio * 100
    df["Elderly_Ratio"] = df["Elderly_Ratio"] / total_ratio * 100

    # 区分データの有無を先に判定（サブプロットタイトルに反映）
    has_4_categories_for_title = "Elderly_65_74" in df.columns and "Elderly_75plus" in df.columns
    # 生産年齢の内部分割は視覚表現のみ（区分数は変わらない）
    pop_title = "人口構成（年齢4区分）" if has_4_categories_for_title else "人口構成（年齢3区分）"

    # サブプロット作成（2行2列）
    # 名目GDPパネル（row=1, col=2）に第2Y軸を追加（REER重ね合わせ用）
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "bar"}, {"secondary_y": True}],
            [{"secondary_y": True}, {}],
        ],
        subplot_titles=[
            pop_title,
            "名目GDP・REER推移",
            "年齢構成比・総人口の推移",
            "相関図: 生産年齢比率 vs 生産性",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # トレースインデックスを追跡（フレーム更新用）
    # trace 0: 人口バー
    # trace 1: GDP全期間線
    # trace 2: GDP現在年マーカー
    # trace 3-5: 年齢構成比（積み上げ）
    # trace 6: REER全期間線
    # trace 7: REER現在年マーカー

    initial_year = years[0]
    initial_row = df[df["Year"] == initial_year].iloc[0]

    # 4区分データがあるか確認
    has_4_categories = "Elderly_65_74" in df.columns and "Elderly_75plus" in df.columns
    # 生産年齢の内部分割データがあるか確認（4区分: 15-19, 20-34, 35-59, 60-64）
    has_working_breakdown = "Working_20_34" in df.columns and "Working_35_59" in df.columns

    # 1. 人口構成バー（選択年の値を表示）
    # 4区分のまま、生産年齢バーだけを内部で4色スタック表示
    bar_x = ["年少<br>(0-14)", "生産年齢<br>(15-64)", "前期高齢<br>(65-74)", "後期高齢<br>(75+)"]

    # 初期年に単年齢データがあるかチェック（年ごとに判定）
    initial_has_breakdown = (has_working_breakdown and has_4_categories and
                             pd.notna(initial_row.get("Working_20_34")))

    if initial_has_breakdown:
        # 生産年齢を4層に分割（15-19, 20-34, 35-59, 60-64）
        w_15_19 = initial_row.get("Working_15_19", 0) if pd.notna(initial_row.get("Working_15_19")) else 0
        w_20_34 = initial_row.get("Working_20_34", 0) if pd.notna(initial_row.get("Working_20_34")) else 0
        w_35_59 = initial_row.get("Working_35_59", 0) if pd.notna(initial_row.get("Working_35_59")) else 0
        w_60_64 = initial_row.get("Working_60_64", 0) if pd.notna(initial_row.get("Working_60_64")) else 0
        working_total = w_15_19 + w_20_34 + w_35_59 + w_60_64

        # テキスト（最上層トレースで表示）
        bar_text = [f"{initial_row['Young']:,.0f}", f"{working_total:,.0f}",
                    f"{initial_row['Elderly_65_74']:,.0f}", f"{initial_row['Elderly_75plus']:,.0f}"]

        # 4層スタック（下から: 15-19 → 20-34 → 35-59 → 60-64）
        # アクセシビリティ: 色だけでなくパターンでも識別可能に
        # トレース1: ベース層（年少、15-19、前期高齢、後期高齢）
        fig.add_trace(
            go.Bar(x=bar_x, y=[initial_row["Young"], w_15_19, initial_row["Elderly_65_74"], initial_row["Elderly_75plus"]],
                   marker=dict(
                       color=[COLORS["young"], COLORS["working_15_19"], COLORS["elderly_65_74"], COLORS["elderly_75plus"]],
                       pattern=dict(shape=["", "", "", ""])  # 15-19: 塗りつぶし
                   ),
                   showlegend=False, hovertemplate="%{x}: %{y:,.0f}万人<extra></extra>"),
            row=1, col=1
        )
        # トレース2: 20-34歳（若手・形成層）- 斜線パターン
        fig.add_trace(
            go.Bar(x=bar_x, y=[0, w_20_34, 0, 0],
                   marker=dict(
                       color=["rgba(0,0,0,0)", COLORS["working_20_34"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                       pattern=dict(shape=["", "/", "", ""])
                   ),
                   showlegend=False, hovertemplate="20-34歳: %{y:,.0f}万人<extra></extra>"),
            row=1, col=1
        )
        # トレース3: 35-59歳（熟練・高所得層）- クロスパターン
        fig.add_trace(
            go.Bar(x=bar_x, y=[0, w_35_59, 0, 0],
                   marker=dict(
                       color=["rgba(0,0,0,0)", COLORS["working_35_59"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                       pattern=dict(shape=["", "x", "", ""])
                   ),
                   showlegend=False, hovertemplate="35-59歳: %{y:,.0f}万人<extra></extra>"),
            row=1, col=1
        )
        # トレース4: 60-64歳（シニア就労）- 逆斜線パターン + テキスト表示
        fig.add_trace(
            go.Bar(x=bar_x, y=[0, w_60_64, 0, 0],
                   marker=dict(
                       color=["rgba(0,0,0,0)", COLORS["working_60_64"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                       pattern=dict(shape=["", "\\", "", ""])
                   ),
                   text=bar_text, textposition="outside",
                   showlegend=False, hovertemplate="60-64歳: %{y:,.0f}万人<extra></extra>"),
            row=1, col=1
        )
    elif has_4_categories:
        # 4区分表示（従来）- 単一トレース + ダミー3つ（テキストは最後のトレースで表示）
        bar_y = [initial_row["Young"], initial_row["Working"], initial_row["Elderly_65_74"], initial_row["Elderly_75plus"]]
        bar_colors = [COLORS["young"], COLORS["working"], COLORS["elderly_65_74"], COLORS["elderly_75plus"]]
        bar_text = [f"{v:,.0f}" for v in bar_y]
        fig.add_trace(go.Bar(x=bar_x, y=bar_y, marker_color=bar_colors,
                             showlegend=False), row=1, col=1)
        for i in range(3):
            if i == 2:  # 最後のダミートレースでテキスト表示
                fig.add_trace(go.Bar(x=bar_x, y=[0, 0, 0, 0], marker_color="rgba(0,0,0,0)",
                                     text=bar_text, textposition="outside", showlegend=False), row=1, col=1)
            else:
                fig.add_trace(go.Bar(x=bar_x, y=[0, 0, 0, 0], marker_color="rgba(0,0,0,0)", showlegend=False), row=1, col=1)
    else:
        # 3区分表示（後方互換）- テキストは最後のトレースで表示
        bar_x_3 = ["年少<br>(0-14)", "生産年齢<br>(15-64)", "高齢<br>(65+)"]
        bar_y = [initial_row["Young"], initial_row["Working"], initial_row["Elderly"]]
        bar_colors = [COLORS["young"], COLORS["working"], COLORS["elderly"]]
        bar_text = [f"{v:,.0f}" for v in bar_y]
        fig.add_trace(go.Bar(x=bar_x_3, y=bar_y, marker_color=bar_colors,
                             showlegend=False), row=1, col=1)
        for i in range(3):
            if i == 2:  # 最後のダミートレースでテキスト表示
                fig.add_trace(go.Bar(x=bar_x_3, y=[0, 0, 0], marker_color="rgba(0,0,0,0)",
                                     text=bar_text, textposition="outside", showlegend=False), row=1, col=1)
            else:
                fig.add_trace(go.Bar(x=bar_x_3, y=[0, 0, 0], marker_color="rgba(0,0,0,0)", showlegend=False), row=1, col=1)

    # 人口構成バーの注釈（生産年齢の内部分割説明）
    if has_working_breakdown:
        # 凡例的な注釈を人口構成パネルの右側に追加
        # パターン情報を含めてアクセシビリティ対応
        c_35_59 = COLORS["working_35_59"]
        c_20_34 = COLORS["working_20_34"]
        c_15_19 = COLORS["working_15_19"]
        c_60_64 = COLORS["working_60_64"]
        legend_text = (
            "<b>生産年齢の内訳</b><br>"
            f"<span style='color:{c_35_59}'>✕</span> 熟練層(35-59): GDPコア<br>"
            f"<span style='color:{c_20_34}'>／</span> 若手層(20-34): イノベーション<br>"
            f"<span style='color:{c_15_19}'>■</span> 学生層(15-19)<br>"
            f"<span style='color:{c_60_64}'>＼</span> シニア(60-64)"
        )
        fig.add_annotation(
            x=1.0, y=0.95,
            xref="x domain", yref="y domain",
            text=legend_text,
            showarrow=False,
            font=dict(size=9),
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )

    # 2. GDP実績線（全シナリオ共通）
    df_gdp_actual = df[~df["GDP_IsProjection"]]
    fig.add_trace(
        go.Scatter(
            x=df_gdp_actual["Year"],
            y=df_gdp_actual["GDP"],
            mode="lines",
            line=dict(color=COLORS["gdp"], width=2),
            name="GDP（実績）",
            showlegend=True,
            legendgroup="gdp",
            hovertemplate="%{x}年: %{y:.0f}兆円（実績）<extra></extra>",
        ),
        row=1, col=2
    )

    # 2a. GDP推計線（シナリオ別）
    # トレースインデックスを記録（ドロップダウン制御用）
    gdp_scenario_trace_start = len(fig.data)
    for scenario_id, scenario_name, color, dash, opacity in GDP_SCENARIOS:
        try:
            # シナリオごとにデータを取得
            df_scenario = build_macro_dataframe(gdp_scenario=scenario_id, reer_scenario="reer_flat")
            df_proj = df_scenario[df_scenario["GDP_IsProjection"]]
            if len(df_proj) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_proj["Year"],
                        y=df_proj["GDP"],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        opacity=opacity,
                        name=scenario_name,
                        showlegend=True,
                        legendgroup="gdp_scenario",
                        hovertemplate=f"{scenario_name}: %{{y:.0f}}兆円<extra></extra>",
                    ),
                    row=1, col=2
                )
        except Exception as e:
            # シナリオデータがない場合はスキップ
            print(f"[WARN] シナリオ '{scenario_id}' のデータ取得失敗: {e}")

    # 2b. GDP現在年マーカー
    fig.add_trace(
        go.Scatter(
            x=[initial_year],
            y=[initial_row["GDP"]],
            mode="markers+text",
            marker=dict(color=COLORS["gdp"], size=14, symbol="circle", line=dict(color="white", width=2)),
            text=[f"{initial_row['GDP']:.0f}兆円"],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["gdp"]),
            showlegend=False,
            hovertemplate="%{x}年: %{y:.1f}兆円<extra></extra>",
        ),
        row=1, col=2, secondary_y=False
    )

    # 2c. REER全期間線（実績）- 第2Y軸
    df_reer_actual = df[(df["REER"].notna()) & (~df["REER_IsProjection"])]
    df_reer_proj = df[(df["REER"].notna()) & (df["REER_IsProjection"])]

    if len(df_reer_actual) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_reer_actual["Year"],
                y=df_reer_actual["REER"],
                mode="lines",
                line=dict(color=COLORS["reer"], width=2),
                name="REER（実績）",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1, col=2, secondary_y=True
        )
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines", showlegend=False),
            row=1, col=2, secondary_y=True
        )

    # 2d. REER推計線（シナリオ別）- 第2Y軸
    reer_scenario_trace_start = len(fig.data)
    for scenario_id, scenario_name, color, dash, opacity in REER_SCENARIOS:
        try:
            # シナリオごとにデータを取得
            df_reer_scenario = build_macro_dataframe(gdp_scenario="gdp_baseline", reer_scenario=scenario_id)
            df_reer_proj_scenario = df_reer_scenario[df_reer_scenario["REER_IsProjection"] & df_reer_scenario["REER"].notna()]
            if len(df_reer_proj_scenario) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_reer_proj_scenario["Year"],
                        y=df_reer_proj_scenario["REER"],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        opacity=opacity,
                        name=f"REER（{scenario_name}）",
                        showlegend=True,
                        legendgroup="reer_scenario",
                        hovertemplate=f"REER {scenario_name}: %{{y:.0f}}<extra></extra>",
                    ),
                    row=1, col=2, secondary_y=True
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode="lines", showlegend=False),
                    row=1, col=2, secondary_y=True
                )
        except Exception as e:
            print(f"[WARN] REERシナリオ '{scenario_id}' のデータ取得失敗: {e}")
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="lines", showlegend=False),
                row=1, col=2, secondary_y=True
            )

    # 2e. REER現在年マーカー - 第2Y軸
    initial_reer = initial_row["REER"] if pd.notna(initial_row["REER"]) else None
    if initial_reer is not None:
        fig.add_trace(
            go.Scatter(
                x=[initial_year],
                y=[initial_reer],
                mode="markers+text",
                marker=dict(color=COLORS["reer"], size=14, symbol="diamond", line=dict(color="white", width=2)),
                text=[f"{initial_reer:.0f}"],
                textposition="bottom center",
                textfont=dict(size=10, color=COLORS["reer"]),
                showlegend=False,
                hovertemplate="%{x}年: REER %{y:.1f}<extra></extra>",
            ),
            row=1, col=2, secondary_y=True
        )
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", showlegend=False),
            row=1, col=2, secondary_y=True
        )

    # 二重軸誤読リスク軽減注釈（GDPパネル）
    fig.add_annotation(
        text="※ スケール独立。因果関係ではありません",
        xref="x2 domain", yref="y2 domain",
        x=0.98, y=0.02,
        showarrow=False,
        font=dict(size=9, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=3,
        xanchor="right", yanchor="bottom",
    )

    # GDPシナリオ仮定の注釈（GDPパネル左上）
    fig.add_annotation(
        text=(
            "<b>GDP推計シナリオ</b><br>"
            "<span style='color:#E91E63'>━</span> ベースライン: +20兆/5年<br>"
            "<span style='color:#4CAF50'>┅</span> 成長ケース: 名目3%成長<br>"
            "<span style='color:#FF9800'>┈</span> 内閣府BL: 潜在成長率"
        ),
        xref="x2 domain", yref="y2 domain",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=8),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        borderpad=4,
        align="left",
        xanchor="left", yanchor="top",
    )

    # 3. 年齢構成比（全期間積み上げ）
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255, 87, 34, 0.5)",
            line=dict(color=COLORS["elderly"], width=1),
            name="高齢者(65+)",
            hoverinfo="skip",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"] + df["Working_Ratio"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(33, 150, 243, 0.5)",
            line=dict(color=COLORS["working"], width=1),
            name="生産年齢(15-64)",
            hoverinfo="skip",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"] + df["Working_Ratio"] + df["Young_Ratio"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(76, 175, 80, 0.5)",
            line=dict(color=COLORS["young"], width=1),
            name="年少(0-14)",
            hoverinfo="skip",
        ),
        row=2, col=1, secondary_y=False
    )

    # 3b. 全人口推移（第2Y軸）
    fig.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["TotalPop"],
            mode="lines",
            line=dict(color="#555555", width=2, dash="dot"),
            name="総人口",
            hovertemplate="%{x}年: %{y:,.0f}万人<extra></extra>",
        ),
        row=2, col=1, secondary_y=True
    )

    # 4. 相関図（Connected Scatter）: Working_Ratio vs GDP_per_working
    # 実績データ（全シナリオ共通）
    df_scatter_all = df[df["GDP_per_working"].notna()].copy()
    df_scatter_actual = df_scatter_all[~df_scatter_all["GDP_IsProjection"]].copy()

    # 相関図トレース（実績）- 全シナリオ共通
    fig.add_trace(
        go.Scatter(
            x=df_scatter_actual["Working_Ratio"],
            y=df_scatter_actual["GDP_per_working"],
            mode="lines+markers",
            line=dict(color=COLORS["gdp"], width=2),
            marker=dict(size=6),
            name="相関図（実績）",
            showlegend=False,
            hovertemplate="生産年齢比率: %{x:.1f}%<br>生産性: %{y:.1f}万円/人<extra></extra>",
        ),
        row=2, col=2
    )

    # 相関図トレース（推計）- シナリオ別に作成
    # 各シナリオごとにGDP_per_workingを計算して表示
    scatter_scenario_trace_start = len(fig.data)
    scatter_scenario_data = {}  # フレーム用にデータを保存
    for scenario_id, scenario_name, color, dash, opacity in GDP_SCENARIOS:
        try:
            df_scenario = build_macro_dataframe(gdp_scenario=scenario_id, reer_scenario="reer_flat")
            df_scenario_proj = df_scenario[df_scenario["GDP_IsProjection"] & df_scenario["GDP_per_working"].notna()]
            scatter_scenario_data[scenario_id] = df_scenario_proj
            if len(df_scenario_proj) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_scenario_proj["Working_Ratio"],
                        y=df_scenario_proj["GDP_per_working"],
                        mode="lines+markers",
                        line=dict(color=color, width=2, dash=dash),
                        marker=dict(size=6, symbol="circle-open"),
                        opacity=opacity,
                        name=f"相関図（{scenario_name[:10]}）",
                        showlegend=False,
                        hovertemplate=f"{scenario_name[:10]}: %{{x:.1f}}% / %{{y:.1f}}万円/人<extra></extra>",
                    ),
                    row=2, col=2
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode="markers", showlegend=False),
                    row=2, col=2
                )
        except Exception as e:
            print(f"[WARN] 相関図シナリオ '{scenario_id}' のデータ取得失敗: {e}")
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="markers", showlegend=False),
                row=2, col=2
            )
            scatter_scenario_data[scenario_id] = pd.DataFrame()

    # 相関図現在年マーカー（初期状態）
    initial_scatter_row = df_scatter_all[df_scatter_all["Year"] == initial_year]
    if len(initial_scatter_row) > 0:
        init_scatter_x = initial_scatter_row.iloc[0]["Working_Ratio"]
        init_scatter_y = initial_scatter_row.iloc[0]["GDP_per_working"]
        fig.add_trace(
            go.Scatter(
                x=[init_scatter_x],
                y=[init_scatter_y],
                mode="markers+text",
                marker=dict(color=COLORS["cursor"], size=16, symbol="star", line=dict(color="white", width=2)),
                text=[f"{initial_year}年"],
                textposition="top center",
                textfont=dict(size=11, color=COLORS["cursor"]),
                name="現在年",
                showlegend=False,
            ),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", showlegend=False),
            row=2, col=2
        )

    # 因果注意の注記（X軸ラベルと重ならないように位置調整）
    # 注: secondary_y追加により軸番号がずれる (相関図: x4,y5)
    fig.add_annotation(
        x=0.5,
        y=-0.28,
        xref="x4 domain", yref="y6 domain",
        text="※ これは相関であり、因果関係を示すものではありません",
        showarrow=False,
        font=dict(size=9, color="gray"),
    )

    # 推計仮定カード（右上に表示）
    assumption_text = (
        f"<b>推計の仮定（{gdp_actual_end + 1}年以降）</b><br>"
        f"GDP: {gdp_assumption}"
    )
    fig.add_annotation(
        x=1.0,
        y=1.12,
        xref="paper", yref="paper",
        text=assumption_text,
        showarrow=False,
        font=dict(size=9),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
        borderpad=4,
    )

    # 相関図の軸範囲を計算（全シナリオを考慮）
    scatter_x_min = df_scatter_all["Working_Ratio"].min() - 2
    scatter_x_max = df_scatter_all["Working_Ratio"].max() + 2
    scatter_y_min = 0
    scatter_y_max = df_scatter_all["GDP_per_working"].max()
    # 全シナリオの最大値を考慮
    for sc_id, sc_df in scatter_scenario_data.items():
        if len(sc_df) > 0 and "GDP_per_working" in sc_df.columns:
            scatter_y_max = max(scatter_y_max, sc_df["GDP_per_working"].max())
    scatter_y_max = scatter_y_max * 1.1

    # フレーム作成（アニメーション用）
    frames = []

    for i, year in enumerate(years):
        row = df[df["Year"] == year].iloc[0]
        scatter_row = df_scatter_all[df_scatter_all["Year"] == year]

        frame_data = []

        # 1. 人口構成バー（選択年の値）- 4トレース構成（スタック用）
        frame_bar_x = ["年少<br>(0-14)", "生産年齢<br>(15-64)", "前期高齢<br>(65-74)", "後期高齢<br>(75+)"]

        # この年に単年齢データがあるかチェック
        frame_has_breakdown = (has_working_breakdown and has_4_categories and
                               pd.notna(row.get("Working_20_34")))

        if frame_has_breakdown:
            # 生産年齢を4層に分割（15-19, 20-34, 35-59, 60-64）
            fw_15_19 = row.get("Working_15_19", 0) if pd.notna(row.get("Working_15_19")) else 0
            fw_20_34 = row.get("Working_20_34", 0) if pd.notna(row.get("Working_20_34")) else 0
            fw_35_59 = row.get("Working_35_59", 0) if pd.notna(row.get("Working_35_59")) else 0
            fw_60_64 = row.get("Working_60_64", 0) if pd.notna(row.get("Working_60_64")) else 0
            working_total = fw_15_19 + fw_20_34 + fw_35_59 + fw_60_64

            frame_bar_text = [f"{row['Young']:,.0f}", f"{working_total:,.0f}",
                              f"{row['Elderly_65_74']:,.0f}", f"{row['Elderly_75plus']:,.0f}"]

            # トレース1: ベース層（年少、15-19、前期高齢、後期高齢）
            frame_data.append(go.Bar(x=frame_bar_x,
                y=[row["Young"], fw_15_19, row["Elderly_65_74"], row["Elderly_75plus"]],
                marker=dict(
                    color=[COLORS["young"], COLORS["working_15_19"], COLORS["elderly_65_74"], COLORS["elderly_75plus"]],
                    pattern=dict(shape=["", "", "", ""])
                ),
                showlegend=False))
            # トレース2: 20-34歳 - 斜線パターン
            frame_data.append(go.Bar(x=frame_bar_x, y=[0, fw_20_34, 0, 0],
                marker=dict(
                    color=["rgba(0,0,0,0)", COLORS["working_20_34"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                    pattern=dict(shape=["", "/", "", ""])
                ),
                showlegend=False))
            # トレース3: 35-59歳 - クロスパターン
            frame_data.append(go.Bar(x=frame_bar_x, y=[0, fw_35_59, 0, 0],
                marker=dict(
                    color=["rgba(0,0,0,0)", COLORS["working_35_59"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                    pattern=dict(shape=["", "x", "", ""])
                ),
                showlegend=False))
            # トレース4: 60-64歳 - 逆斜線パターン + テキスト
            frame_data.append(go.Bar(x=frame_bar_x, y=[0, fw_60_64, 0, 0],
                marker=dict(
                    color=["rgba(0,0,0,0)", COLORS["working_60_64"], "rgba(0,0,0,0)", "rgba(0,0,0,0)"],
                    pattern=dict(shape=["", "\\", "", ""])
                ),
                text=frame_bar_text, textposition="outside", showlegend=False))
        elif has_4_categories:
            frame_bar_y = [row["Young"], row["Working"], row["Elderly_65_74"], row["Elderly_75plus"]]
            frame_bar_colors = [COLORS["young"], COLORS["working"], COLORS["elderly_65_74"], COLORS["elderly_75plus"]]
            frame_bar_text = [f"{v:,.0f}" for v in frame_bar_y]
            frame_data.append(go.Bar(x=frame_bar_x, y=frame_bar_y, marker_color=frame_bar_colors, showlegend=False))
            for i in range(3):
                if i == 2:  # 最後のダミートレースでテキスト表示
                    frame_data.append(go.Bar(x=frame_bar_x, y=[0, 0, 0, 0], marker_color="rgba(0,0,0,0)",
                                             text=frame_bar_text, textposition="outside", showlegend=False))
                else:
                    frame_data.append(go.Bar(x=frame_bar_x, y=[0, 0, 0, 0], marker_color="rgba(0,0,0,0)", showlegend=False))
        else:
            frame_bar_x_3 = ["年少<br>(0-14)", "生産年齢<br>(15-64)", "高齢<br>(65+)"]
            frame_bar_y = [row["Young"], row["Working"], row["Elderly"]]
            frame_bar_colors = [COLORS["young"], COLORS["working"], COLORS["elderly"]]
            frame_bar_text = [f"{v:,.0f}" for v in frame_bar_y]
            frame_data.append(go.Bar(x=frame_bar_x_3, y=frame_bar_y, marker_color=frame_bar_colors, showlegend=False))
            for i in range(3):
                if i == 2:  # 最後のダミートレースでテキスト表示
                    frame_data.append(go.Bar(x=frame_bar_x_3, y=[0, 0, 0], marker_color="rgba(0,0,0,0)",
                                             text=frame_bar_text, textposition="outside", showlegend=False))
                else:
                    frame_data.append(go.Bar(x=frame_bar_x_3, y=[0, 0, 0], marker_color="rgba(0,0,0,0)", showlegend=False))

        # 2. GDP実績線（変更なし）
        df_gdp_actual_frame = df[~df["GDP_IsProjection"]]
        frame_data.append(go.Scatter(
            x=df_gdp_actual_frame["Year"],
            y=df_gdp_actual_frame["GDP"],
            mode="lines",
            line=dict(color=COLORS["gdp"], width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 2a. GDPシナリオ推計線（静的、フレーム間で変化なし）
        for scenario_id, scenario_name, color, dash, opacity in GDP_SCENARIOS:
            try:
                df_scenario = build_macro_dataframe(gdp_scenario=scenario_id, reer_scenario="reer_flat")
                df_proj = df_scenario[df_scenario["GDP_IsProjection"]]
                if len(df_proj) > 0:
                    frame_data.append(go.Scatter(
                        x=df_proj["Year"],
                        y=df_proj["GDP"],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        opacity=opacity,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                else:
                    frame_data.append(go.Scatter(x=[], y=[], mode="lines", showlegend=False))
            except Exception:
                frame_data.append(go.Scatter(x=[], y=[], mode="lines", showlegend=False))

        # 2b. GDP現在年マーカー
        frame_data.append(go.Scatter(
            x=[year],
            y=[row["GDP"]],
            mode="markers+text",
            marker=dict(color=COLORS["gdp"], size=14, symbol="circle", line=dict(color="white", width=2)),
            text=[f"{row['GDP']:.0f}兆円"],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["gdp"]),
            showlegend=False,
        ))

        # 2c. REER全期間線（実績）- 第2Y軸
        if len(df_reer_actual) > 0:
            frame_data.append(go.Scatter(
                x=df_reer_actual["Year"],
                y=df_reer_actual["REER"],
                mode="lines",
                line=dict(color=COLORS["reer"], width=2),
                showlegend=False,
                hoverinfo="skip",
            ))
        else:
            frame_data.append(go.Scatter(x=[], y=[], mode="lines", showlegend=False))

        # 2d. REER推計線（シナリオ別）- フレーム内
        for scenario_id, scenario_name, color, dash, opacity in REER_SCENARIOS:
            try:
                df_reer_scenario = build_macro_dataframe(gdp_scenario="gdp_baseline", reer_scenario=scenario_id)
                df_reer_proj_scenario = df_reer_scenario[df_reer_scenario["REER_IsProjection"] & df_reer_scenario["REER"].notna()]
                if len(df_reer_proj_scenario) > 0:
                    frame_data.append(go.Scatter(
                        x=df_reer_proj_scenario["Year"],
                        y=df_reer_proj_scenario["REER"],
                        mode="lines",
                        line=dict(color=color, width=2, dash=dash),
                        opacity=opacity,
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                else:
                    frame_data.append(go.Scatter(x=[], y=[], mode="lines", showlegend=False))
            except Exception:
                frame_data.append(go.Scatter(x=[], y=[], mode="lines", showlegend=False))

        # 2e. REER現在年マーカー - 第2Y軸
        current_reer = row["REER"] if pd.notna(row["REER"]) else None
        if current_reer is not None:
            frame_data.append(go.Scatter(
                x=[year],
                y=[current_reer],
                mode="markers+text",
                marker=dict(color=COLORS["reer"], size=14, symbol="diamond", line=dict(color="white", width=2)),
                text=[f"{current_reer:.0f}"],
                textposition="bottom center",
                textfont=dict(size=10, color=COLORS["reer"]),
                showlegend=False,
            ))
        else:
            frame_data.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))

        # 3. 年齢構成比（全期間、変更なし）
        frame_data.append(go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255, 87, 34, 0.5)",
            line=dict(color=COLORS["elderly"], width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        frame_data.append(go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"] + df["Working_Ratio"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(33, 150, 243, 0.5)",
            line=dict(color=COLORS["working"], width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        frame_data.append(go.Scatter(
            x=df["Year"],
            y=df["Elderly_Ratio"] + df["Working_Ratio"] + df["Young_Ratio"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(76, 175, 80, 0.5)",
            line=dict(color=COLORS["young"], width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 3b. 全人口推移（第2Y軸）
        frame_data.append(go.Scatter(
            x=df["Year"],
            y=df["TotalPop"],
            mode="lines",
            line=dict(color="#555555", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 4. 相関図（実績）全期間線
        frame_data.append(go.Scatter(
            x=df_scatter_actual["Working_Ratio"],
            y=df_scatter_actual["GDP_per_working"],
            mode="lines+markers",
            line=dict(color=COLORS["gdp"], width=2),
            marker=dict(size=6),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 4b. 相関図（推計）シナリオ別
        for scenario_id, scenario_name, color, dash, opacity in GDP_SCENARIOS:
            df_sc_proj = scatter_scenario_data.get(scenario_id, pd.DataFrame())
            if len(df_sc_proj) > 0:
                frame_data.append(go.Scatter(
                    x=df_sc_proj["Working_Ratio"],
                    y=df_sc_proj["GDP_per_working"],
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=6, symbol="circle-open"),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                ))
            else:
                frame_data.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))

        # 4c. 相関図現在年マーカー
        if len(scatter_row) > 0:
            scatter_x = scatter_row.iloc[0]["Working_Ratio"]
            scatter_y = scatter_row.iloc[0]["GDP_per_working"]
            frame_data.append(go.Scatter(
                x=[scatter_x],
                y=[scatter_y],
                mode="markers+text",
                marker=dict(color=COLORS["cursor"], size=16, symbol="star", line=dict(color="white", width=2)),
                text=[f"{year}年"],
                textposition="top center",
                textfont=dict(size=11, color=COLORS["cursor"]),
                showlegend=False,
            ))
        else:
            frame_data.append(go.Scatter(x=[], y=[], mode="markers", showlegend=False))

        # 垂直カーソル線をshapesで追加（GDP図と年齢構成比図のみ）
        # 注: secondary_y追加により軸番号がずれる (年齢構成比: x3,y4)
        cursor_shapes = [
            # GDP図のカーソル
            dict(type="line", x0=year, x1=year, y0=0, y1=gdp_max,
                 xref="x2", yref="y2",
                 line=dict(color=COLORS["cursor"], width=2, dash="dash")),
            # 年齢構成比図のカーソル
            dict(type="line", x0=year, x1=year, y0=0, y1=RATIO_RANGE_MAX,
                 xref="x3", yref="y4",
                 line=dict(color=COLORS["cursor"], width=2, dash="dash")),
        ]

        frames.append(go.Frame(
            data=frame_data,
            name=str(year),
            layout=go.Layout(shapes=cursor_shapes)
        ))

    # イベントマーカー（GDP図と年齢構成比図のみ）
    for event_year, (event_name, event_color) in EVENTS.items():
        if event_year in years:
            # GDP図にイベント線
            fig.add_vline(
                x=event_year,
                line=dict(color=event_color, width=1, dash="dot"),
                row=1, col=2
            )
            # 年齢構成比図にイベント線
            fig.add_vline(
                x=event_year,
                line=dict(color=event_color, width=1, dash="dot"),
                row=2, col=1
            )

    # 初期カーソル線（shapes）- GDP図と年齢構成比図のみ
    # 注: secondary_y追加により軸番号がずれる (年齢構成比: x3,y4)
    initial_cursor_shapes = [
        dict(type="line", x0=initial_year, x1=initial_year, y0=0, y1=gdp_max,
             xref="x2", yref="y2",
             line=dict(color=COLORS["cursor"], width=2, dash="dash")),
        dict(type="line", x0=initial_year, x1=initial_year, y0=0, y1=RATIO_RANGE_MAX,
             xref="x3", yref="y4",
             line=dict(color=COLORS["cursor"], width=2, dash="dash")),
    ]

    # スライダー作成
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16, "color": "#333"},
            "prefix": "年: ",
            "visible": True,
            "xanchor": "center"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 30},
        "len": 0.75,
        "x": 0.2,
        "y": -0.08,
        "steps": [
            {
                "args": [[str(year)], {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300}
                }],
                "label": str(year),
                "method": "animate"
            }
            for year in years
        ]
    }]

    # 再生/一時停止ボタン
    play_pause_buttons = {
        "buttons": [
            {
                "args": [None, {
                    "frame": {"duration": 500, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 300, "easing": "quadratic-in-out"}
                }],
                "label": "▶ 再生",
                "method": "animate"
            },
            {
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                "label": "⏸ 停止",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 30},
        "showactive": False,
        "type": "buttons",
        "x": 0.15,
        "xanchor": "right",
        "y": -0.08,
        "yanchor": "top"
    }

    # GDPシナリオ選択ドロップダウン
    # シナリオトレースの表示/非表示を切り替え
    # 注: トレース構成（REERシナリオ追加後）
    #   0-3: 人口バー(4)
    #   4: GDP実績線(1)
    #   5-7: GDPシナリオ推計線(3) ← GDP制御対象
    #   8: GDP現在年マーカー(1)
    #   9: REER実績線(1)
    #   10-12: REERシナリオ推計線(3) ← REER制御対象
    #   13: REER現在年マーカー(1)
    #   14-16: 年齢構成比(3)
    #   17: 全人口(1)
    #   18-19: 相関図実績・推計(2)
    #   20: 相関図現在年マーカー(1)
    # 合計21トレース

    num_gdp_scenarios = len(GDP_SCENARIOS)
    num_reer_scenarios = len(REER_SCENARIOS)

    def make_gdp_visibility(selected_idx):
        """選択されたGDPシナリオのみ表示するvisibilityリストを生成"""
        vis = [True] * 4  # 人口バー
        vis.append(True)   # GDP実績線
        # GDPシナリオトレース
        for i in range(num_gdp_scenarios):
            vis.append(i == selected_idx if selected_idx >= 0 else True)
        vis.append(True)   # GDP現在年マーカー
        vis.append(True)   # REER実績線
        vis.extend([True] * num_reer_scenarios)  # REERシナリオ
        vis.append(True)   # REER現在年マーカー
        vis.extend([True] * 3)  # 年齢構成比
        vis.append(True)   # 全人口
        vis.extend([True] * 3)  # 相関図
        return vis

    # ドロップダウンはHTML側で実装（Plotly updatemenusは位置調整が困難なため）
    updatemenus = [play_pause_buttons]

    # GDPシナリオのトレースインデックスを記録（HTML側のJSで使用）
    gdp_scenario_info = {
        "trace_start": 5,  # GDP実績線の次から
        "trace_count": num_gdp_scenarios,
        "scenarios": [
            {"id": sid, "name": sname}
            for sid, sname, _, _, _ in GDP_SCENARIOS
        ]
    }

    # 推計期間の背景帯（GDP図と年齢構成比図のみ、相関図は線種で区別）
    projection_shapes = [
        # GDP図（row=1, col=2 → x2, y2）
        dict(
            type="rect",
            x0=gdp_actual_end + 0.5, x1=x_range_end,
            y0=0, y1=gdp_max,
            xref="x2", yref="y2",
            fillcolor="rgba(128, 128, 128, 0.1)",
            line_width=0,
            layer="below",
        ),
        # 年齢構成比図（row=2, col=1 → x3, y4）※secondary_y追加で軸番号ずれ
        dict(
            type="rect",
            x0=gdp_actual_end + 0.5, x1=x_range_end,
            y0=0, y1=RATIO_RANGE_MAX,
            xref="x3", yref="y4",
            fillcolor="rgba(128, 128, 128, 0.1)",
            line_width=0,
            layer="below",
        ),
        # 推計境界の縦線（GDP図）
        dict(
            type="line",
            x0=gdp_actual_end + 0.5, x1=gdp_actual_end + 0.5,
            y0=0, y1=gdp_max,
            xref="x2", yref="y2",
            line=dict(color="gray", width=1, dash="dash"),
        ),
        # 推計境界の縦線（年齢構成比図）※secondary_y追加で軸番号ずれ
        dict(
            type="line",
            x0=gdp_actual_end + 0.5, x1=gdp_actual_end + 0.5,
            y0=0, y1=RATIO_RANGE_MAX,
            xref="x3", yref="y4",
            line=dict(color="gray", width=1, dash="dash"),
        ),
    ]

    # 出所表示（下部マージン領域に表示 - 相関図と重ならないよう配置）
    retrieved_at = metadata.get("retrieved_at", "不明")
    source_text = (
        f"<b>データ出所</b> | "
        f"GDP: {gdp_source}（〜{gdp_actual_end}年実績） | "
        f"REER: {reer_source}（〜{reer_actual_end}年実績） | "
        f"人口: 総務省/IPSS推計 | "
        f"取得: {retrieved_at}"
    )
    fig.add_annotation(
        text=source_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=8, color="gray"),
        align="center",
        xanchor="center", yanchor="top",
    )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text="日本の人口動態・経済・為替 統合ダッシュボード<br>"
                 f"<sup style='font-size:10px; color:gray;'>"
                 f"人口: IPSS推計 | GDP: {gdp_source} | REER: {reer_source}"
                 f"</sup>",
            font=dict(size=18),
            x=0.5,
        ),
        height=850,
        margin=dict(b=120),
        barmode="stack",  # 生産年齢バーの内部スタック表示用
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        shapes=initial_cursor_shapes + projection_shapes,
    )

    # 軸設定（データから動的計算）
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="人口（万人）", range=[0, POP_RANGE_MAX], row=1, col=1)

    fig.update_xaxes(title_text="年", range=[x_range_start, x_range_end], row=1, col=2)
    fig.update_yaxes(title_text="名目GDP（兆円）", range=[0, gdp_max], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="REER（2020=100）", range=[0, reer_y_max], row=1, col=2, secondary_y=True)

    fig.update_xaxes(title_text="年", range=[x_range_start, x_range_end], row=2, col=1)
    fig.update_yaxes(title_text="構成比（%）", range=[0, RATIO_RANGE_MAX], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="総人口（万人）", range=[0, POP_RANGE_MAX], row=2, col=1, secondary_y=True)

    # 相関図の軸設定
    fig.update_xaxes(title_text="生産年齢比率（%）", range=[scatter_x_min, scatter_x_max], row=2, col=2)
    fig.update_yaxes(title_text="生産性（万円/人）", range=[scatter_y_min, scatter_y_max], row=2, col=2)

    # フレーム追加
    fig.frames = frames

    # カスタムHTML出力（ドロップダウンをグラフ外に配置）
    plotly_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        div_id="dashboard-graph"
    )

    # シナリオ選択用HTML/CSS/JS
    scenario_options = "\n".join([
        f'<option value="{i}">{name}</option>'
        for i, (_, name, _, _, _) in enumerate(GDP_SCENARIOS)
    ])

    custom_html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>日本の人口動態・経済・為替 統合ダッシュボード</title>
    <style>
        body {{
            font-family: "Yu Gothic", "Hiragino Sans", sans-serif;
            margin: 0;
            padding: 10px;
            background: #fafafa;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .scenario-selector {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            padding: 8px 15px;
            background: linear-gradient(to right, transparent 50%, rgba(233, 30, 99, 0.05));
            border-radius: 4px;
            margin-bottom: 5px;
        }}
        .scenario-selector label {{
            font-size: 13px;
            font-weight: bold;
            color: #E91E63;
        }}
        .scenario-selector select {{
            padding: 6px 12px;
            font-size: 12px;
            border: 2px solid #E91E63;
            border-radius: 4px;
            background: white;
            color: #333;
            cursor: pointer;
            min-width: 200px;
        }}
        .scenario-selector select:hover {{
            border-color: #C2185B;
        }}
        .scenario-selector select:focus {{
            outline: none;
            box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.2);
        }}
        #dashboard-graph {{
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="scenario-selector">
            <label for="gdp-scenario">GDPシナリオ:</label>
            <select id="gdp-scenario" onchange="updateGdpScenario(this.value)">
                <option value="-1">全シナリオ表示</option>
                {scenario_options}
            </select>
        </div>
        {plotly_html}
    </div>
    <script>
        function updateGdpScenario(selectedIdx) {{
            const gd = document.getElementById('dashboard-graph');
            const numScenarios = {num_gdp_scenarios};

            // トレース構成:
            // 0-3: 人口バー(4)
            // 4: GDP実績線(1)
            // 5-7: GDPシナリオ推計線(3)
            // 8: GDP現在年マーカー(1)
            // 9: REER実績線(1)
            // 10-12: REERシナリオ推計線(3)
            // 13: REER現在年マーカー(1)
            // 14-16: 年齢構成比(3)
            // 17: 全人口(1)
            // 18: 相関図実績(1)
            // 19-21: 相関図シナリオ推計(3)
            // 22: 相関図現在年マーカー(1)

            const gdpTraceStart = 5;
            const scatterTraceStart = 19;
            const totalTraces = gd.data.length;

            // visibility配列を構築
            const visibility = [];
            for (let i = 0; i < totalTraces; i++) {{
                // GDPシナリオトレース（5-7）
                if (i >= gdpTraceStart && i < gdpTraceStart + numScenarios) {{
                    if (selectedIdx === "-1") {{
                        visibility.push(true);
                    }} else {{
                        visibility.push(i === gdpTraceStart + parseInt(selectedIdx));
                    }}
                }}
                // 相関図シナリオトレース（19-21）
                else if (i >= scatterTraceStart && i < scatterTraceStart + numScenarios) {{
                    if (selectedIdx === "-1") {{
                        visibility.push(true);
                    }} else {{
                        visibility.push(i === scatterTraceStart + parseInt(selectedIdx));
                    }}
                }}
                else {{
                    visibility.push(true);
                }}
            }}

            Plotly.restyle(gd, {{'visible': visibility}});
        }}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(custom_html)

    print(f"[SAVED] {output_path}")


def create_static_summary(df: pd.DataFrame, output_path: Path) -> None:
    """
    Matplotlibで静的サマリーPNGを生成

    Args:
        df: build_macro_dataframe()で作成したマクロデータDataFrame
        output_path: 出力PNGファイルのパス

    Returns:
        None（ファイルを出力）

    Note:
        - REER列が全NaNでもクラッシュしない（グラフ要素をスキップ）
        - 日本語フォントはjapanize_matplotlibまたはシステムフォントを使用
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # 入力dfを破壊しないようコピー
    df = df.copy()

    # メタデータ取得（推計仮定情報）
    metadata = get_data_metadata()
    gdp_source = metadata.get("gdp", {}).get("source", "内閣府")
    reer_source = metadata.get("reer", {}).get("source", "BIS")
    projection_info = metadata.get("projection", {})
    gdp_assumption = projection_info.get("gdp_assumption", "仮定値")
    reer_assumption = projection_info.get("reer_assumption", "仮定値")

    # 日本語フォント設定
    try:
        import japanize_matplotlib
    except ImportError:
        plt.rcParams["font.family"] = ["Yu Gothic", "Hiragino Sans", "MS Gothic", "sans-serif"]

    plt.rcParams["axes.unicode_minus"] = False

    # 軸範囲
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    x_range_start = YEAR_RANGE_MIN
    x_range_end = YEAR_RANGE_MAX
    df_reer_all = df[df["REER"].notna()]
    reer_min_year = int(df_reer_all["Year"].min()) if len(df_reer_all) > 0 else min_year

    # 推計境界をDataFrameフラグから動的に取得（SSOT一本化）
    actual_data_end = get_actual_data_end(df, "GDP")

    # 実績データのみ（actual_data_endまで）
    df_actual = df[df["Year"] <= actual_data_end].copy()
    df_future = df[df["Year"] > actual_data_end].copy()

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)

    # ----- 1. 人口構成の推移（積み上げ面グラフ）-----
    ax1 = fig.add_subplot(gs[0, :])

    # ★ pandas.Series → numpy配列に変換（matplotlibの互換性のため）
    years_np = df["Year"].to_numpy(dtype=float)
    elderly_np = df["Elderly"].to_numpy(dtype=float)
    working_np = df["Working"].to_numpy(dtype=float)
    young_np = df["Young"].to_numpy(dtype=float)

    # stackplotを使用（より簡潔）
    ax1.stackplot(
        years_np,
        elderly_np, working_np, young_np,
        colors=[COLORS["elderly"], COLORS["working"], COLORS["young"]],
        labels=["高齢者(65+)", "生産年齢(15-64)", "年少(0-14)"],
        alpha=0.7
    )

    # 実績/推計の境界
    ax1.axvline(actual_data_end, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax1.annotate("← 実績 | 推計 →", xy=(actual_data_end, POP_RANGE_MAX - 1000), fontsize=11, ha="center",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # 生産年齢人口ピーク
    ax1.annotate("生産年齢人口ピーク\n(1995年: 8,717万人)",
                 xy=(1995, 8717), xytext=(1975, 11500),
                 fontsize=10, fontweight="bold", color=COLORS["working"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["working"], lw=2),
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax1.set_title(f"日本の人口構成推移（{min_year}-{max_year}年）", fontsize=16, fontweight="bold")
    ax1.set_xlabel("年", fontsize=12)
    ax1.set_ylabel("人口（万人）", fontsize=12)
    ax1.set_xlim(x_range_start, x_range_end)
    ax1.set_ylim(0, POP_RANGE_MAX)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ----- 2. GDP推移 -----
    ax2 = fig.add_subplot(gs[1, 0])

    # 推計期間の背景帯
    ax2.axvspan(actual_data_end + 0.5, x_range_end, alpha=0.1, color='gray', zorder=0)
    ax2.axvline(actual_data_end + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax2.plot(df_actual["Year"], df_actual["GDP"], color=COLORS["gdp"], linewidth=3, marker="o", markersize=5, label="実績")
    ax2.plot(df_future["Year"], df_future["GDP"], color=COLORS["gdp"], linewidth=2, linestyle="--", marker="o", markersize=4, alpha=0.6, label="推計")
    # numpy配列に変換してfill_betweenに渡す
    ax2.fill_between(df_actual["Year"].to_numpy(), 0, df_actual["GDP"].to_numpy(), alpha=0.2, color=COLORS["gdp"])

    # イベントマーカー
    for event_year, (event_name, event_color) in EVENTS.items():
        if event_year <= actual_data_end:
            ax2.axvline(event_year, color=event_color, linestyle=":", linewidth=1.5, alpha=0.7)

    ax2.set_title("名目GDP推移", fontsize=14, fontweight="bold")
    ax2.set_xlabel("年", fontsize=12)
    ax2.set_ylabel("名目GDP（兆円）", fontsize=12)
    ax2.set_xlim(x_range_start, x_range_end)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ----- 3. REER推移 -----
    ax3 = fig.add_subplot(gs[1, 1])

    # 推計期間の背景帯
    ax3.axvspan(actual_data_end + 0.5, x_range_end, alpha=0.1, color='gray', zorder=0)
    ax3.axvline(actual_data_end + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    df_reer_actual = df_actual[df_actual["REER"].notna()]
    df_reer_future = df_future[df_future["REER"].notna()]

    # REERデータがある場合のみプロット
    if len(df_reer_actual) > 0:
        ax3.plot(df_reer_actual["Year"], df_reer_actual["REER"], color=COLORS["reer"], linewidth=3, marker="o", markersize=5, label="実績")
        # numpy配列に変換してfill_betweenに渡す
        ax3.fill_between(df_reer_actual["Year"].to_numpy(), 0, df_reer_actual["REER"].to_numpy(), alpha=0.2, color=COLORS["reer"])

        # REERピークを動的に取得
        reer_peak_idx = df_reer_actual["REER"].idxmax()
        reer_peak_year = int(df_reer_actual.loc[reer_peak_idx, "Year"])
        reer_peak_val = float(df_reer_actual.loc[reer_peak_idx, "REER"])

        ax3.annotate(f"REERピーク\n({reer_peak_year}年: {reer_peak_val:.0f})",
                     xy=(reer_peak_year, reer_peak_val), xytext=(1975, 140),
                     fontsize=10, fontweight="bold", color=COLORS["reer"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["reer"], lw=2),
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    if len(df_reer_future) > 0:
        ax3.plot(df_reer_future["Year"], df_reer_future["REER"], color=COLORS["reer"], linewidth=2, linestyle="--", marker="o", markersize=4, alpha=0.6, label="推計")

    ax3.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.annotate("基準(2020=100)", xy=(2065, 102), fontsize=9, color="gray")

    ax3.set_title("実質実効為替レート（REER）推移", fontsize=14, fontweight="bold")
    ax3.set_xlabel("年", fontsize=12)
    ax3.set_ylabel("REER（2020=100）", fontsize=12)
    ax3.set_xlim(reer_min_year - 2, x_range_end)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ----- 4. 年齢構成比 -----
    ax4 = fig.add_subplot(gs[2, 0])

    ax4.plot(df["Year"], df["Young_Ratio"], color=COLORS["young"], linewidth=2.5, label="年少(0-14)")
    ax4.plot(df["Year"], df["Working_Ratio"], color=COLORS["working"], linewidth=2.5, label="生産年齢(15-64)")
    ax4.plot(df["Year"], df["Elderly_Ratio"], color=COLORS["elderly"], linewidth=2.5, label="高齢者(65+)")

    ax4.axvline(actual_data_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax4.set_title("年齢構成比の推移", fontsize=14, fontweight="bold")
    ax4.set_xlabel("年", fontsize=12)
    ax4.set_ylabel("構成比（%）", fontsize=12)
    ax4.set_xlim(x_range_start, x_range_end)
    ax4.set_ylim(0, RATIO_RANGE_MAX)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ----- 5. 三指標の指数化比較（1995年=100）-----
    ax5 = fig.add_subplot(gs[2, 1])

    # 推計期間の背景帯
    ax5.axvspan(actual_data_end + 0.5, x_range_end, alpha=0.1, color='gray', zorder=0)
    ax5.axvline(actual_data_end + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    base_year = BASE_YEAR
    if base_year not in df["Year"].values:
        base_year = int(df["Year"].iloc[0])
    df_base = df[df["Year"] == base_year].iloc[0]

    df["Working_Index"] = df["Working"] / df_base["Working"] * 100
    df["GDP_Index"] = df["GDP"] / df_base["GDP"] * 100

    ax5.plot(df["Year"], df["Working_Index"], color=COLORS["working"], linewidth=2.5, label="生産年齢人口")
    ax5.plot(df["Year"], df["GDP_Index"], color=COLORS["gdp"], linewidth=2.5, label="名目GDP")

    # REER指数は基準年データがある場合のみプロット
    df_reer = df[df["REER"].notna()].copy()
    if len(df_reer) > 0 and base_year in df_reer["Year"].values:
        reer_base = float(df_reer.loc[df_reer["Year"] == base_year, "REER"].iloc[0])
        df_reer["REER_Index"] = df_reer["REER"] / reer_base * 100
        ax5.plot(df_reer["Year"], df_reer["REER_Index"], color=COLORS["reer"], linewidth=2.5, label="REER")

    ax5.axvline(base_year, color="purple", linestyle="--", linewidth=2, alpha=0.7)
    ax5.axhline(100, color="black", linestyle="-", linewidth=1, alpha=0.5)

    ax5.annotate(f"基準年\n({base_year}年=100)", xy=(base_year, 100), xytext=(base_year+10, 130),
                 fontsize=10, color="purple", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax5.set_title("三指標の比較（1995年=100で正規化）", fontsize=14, fontweight="bold")
    ax5.set_xlabel("年", fontsize=12)
    ax5.set_ylabel("指数（1995年=100）", fontsize=12)
    ax5.set_xlim(reer_min_year - 2, x_range_end)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # メインタイトル
    fig.suptitle("日本の人口動態・経済・為替の長期推移\n〜生産年齢人口ピーク(1995年)を軸とした分析〜",
                 fontsize=18, fontweight="bold", y=0.98)

    # 注釈（データソースと推計仮定を明示）
    footnote = (
        f"データソース:\n"
        f"  人口: 総務省統計局（実績）、IPSS将来人口推計（{actual_data_end + 1}年以降、出生中位・死亡中位）\n"
        f"  GDP: {gdp_source}（名目GDP、兆円、暦年）\n"
        f"  REER: {reer_source}（Broad、2020=100、年平均）\n"
        f"推計仮定（{actual_data_end + 1}年以降）:\n"
        f"  GDP: {gdp_assumption}\n"
        f"  REER: {reer_assumption}"
    )
    fig.text(0.5, 0.01, footnote,
             fontsize=8, color="gray", ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="lightgray"))

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="人口・GDP・REER統合ダッシュボード")
    parser.add_argument("--out", type=str, default="out/pop_gdp_reer_dashboard.html",
                        help="出力HTMLファイルパス")
    parser.add_argument("--png", type=str, default="out/pop_gdp_reer_summary.png",
                        help="出力PNGファイルパス（静的サマリー）")
    parser.add_argument("--csv", type=str,
                        help="人口CSVファイルパス")
    parser.add_argument(
        "--unit",
        type=str,
        choices=["person", "thousand", "man"],
        default="man",
        help="入力CSVの人口単位（person=人, thousand=千人, man=万人）"
    )
    args = parser.parse_args()

    # 出力ディレクトリ作成
    out_html = Path(args.out)
    out_png = Path(args.png)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("人口・GDP・REER統合ダッシュボード生成")
    print("=" * 60)

    # データ取得（CSVパス指定 or デフォルト）
    csv_path = Path(args.csv) if args.csv else None
    df = get_dataframe(csv_path, input_unit=args.unit)
    print(f"[INFO] データ読み込み: {csv_path or 'デフォルトパス'}")
    print(f"データ範囲: {df['Year'].min()}-{df['Year'].max()}年 ({len(df)}点)")

    # インタラクティブHTML生成
    print("\n[1/2] インタラクティブHTML生成...")
    create_dashboard(df, out_html)

    # 静的PNG生成
    print("[2/2] 静的サマリーPNG生成...")
    create_static_summary(df, out_png)

    print("\n" + "=" * 60)
    print("完了!")
    print(f"  HTML: {out_html.resolve()}")
    print(f"  PNG:  {out_png.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
