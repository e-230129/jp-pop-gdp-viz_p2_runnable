"""
interpolate_years.py - 5年刻みデータを毎年データに補間

1965〜2015年の5年刻みデータを線形補間して、
1965〜2070年の全期間を毎年データとして出力する。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate


def interpolate_population_data(
    df: pd.DataFrame,
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    5年刻みデータを補間して毎年データを生成

    Args:
        df: 元データ（year, age, sex, pop列を持つ）
        target_years: 出力する年のリスト（Noneの場合は自動生成）

    Returns:
        補間済みの毎年データ
    """
    existing_years = sorted(df["year"].unique())
    min_year = min(existing_years)
    max_year = max(existing_years)

    # 補間が必要な年を特定
    if target_years is None:
        target_years = list(range(min_year, max_year + 1))

    # 既存の年はそのまま使う
    existing_set = set(existing_years)
    years_to_interpolate = [y for y in target_years if y not in existing_set]

    print(f"既存年数: {len(existing_years)}")
    print(f"補間対象: {len(years_to_interpolate)}年")
    print(f"出力年数: {len(target_years)}年")

    if not years_to_interpolate:
        print("補間不要：すべての年が既に存在します")
        return df[df["year"].isin(target_years)].copy()

    # 年齢×性別ごとに補間
    results = []

    for sex in ["M", "F"]:
        df_sex = df[df["sex"] == sex]
        ages = sorted(df_sex["age"].unique())

        for age in ages:
            df_age = df_sex[df_sex["age"] == age].sort_values("year")

            # 既存の年と人口
            x = df_age["year"].values
            y = df_age["pop"].values

            # スプライン補間（滑らかに）
            # k=3: 3次スプライン、k=1: 線形
            # 5年刻みなので3次スプライン推奨
            try:
                if len(x) >= 4:
                    spline = interpolate.UnivariateSpline(x, y, k=3, s=0)
                else:
                    spline = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")
            except Exception:
                spline = interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate")

            # 補間値を計算
            for year in years_to_interpolate:
                if min(x) <= year <= max(x):
                    pop = float(spline(year))
                    # 人口は非負
                    pop = max(0, pop)
                    results.append({
                        "year": year,
                        "age": age,
                        "sex": sex,
                        "pop": pop,
                    })

    # 補間結果をDataFrameに
    df_interpolated = pd.DataFrame(results)

    # 元データと結合
    df_existing = df[df["year"].isin(target_years)].copy()
    df_combined = pd.concat([df_existing, df_interpolated], ignore_index=True)

    # 整形
    df_combined = df_combined.sort_values(["year", "age", "sex"]).reset_index(drop=True)

    return df_combined


def main():
    parser = argparse.ArgumentParser(description="5年刻みデータを毎年データに補間")
    parser.add_argument("--input", required=True, help="入力CSVパス")
    parser.add_argument("--out", required=True, help="出力CSVパス")
    parser.add_argument("--start-year", type=int, default=None, help="開始年")
    parser.add_argument("--end-year", type=int, default=None, help="終了年")

    args = parser.parse_args()

    # 読み込み
    df = pd.read_csv(args.input)
    print(f"入力: {args.input}")
    print(f"元データ年: {sorted(df['year'].unique())}")

    # 対象年を決定
    existing_years = sorted(df["year"].unique())
    start = args.start_year or min(existing_years)
    end = args.end_year or max(existing_years)
    target_years = list(range(start, end + 1))

    # 補間実行
    df_interpolated = interpolate_population_data(df, target_years)

    # 出力
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_interpolated.to_csv(output_path, index=False)

    print(f"\n出力: {output_path}")
    print(f"出力年: {sorted(df_interpolated['year'].unique())}")
    print(f"レコード数: {len(df_interpolated)}")


if __name__ == "__main__":
    main()
