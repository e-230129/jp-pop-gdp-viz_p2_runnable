# convert_ipss_1_9.py
"""
IPSS 表1-9（男女年齢各歳別人口）→ tidy CSV 変換

表1-9は「各歳・毎年」のデータで、人口ピラミッドの密度を最大化できる。
（GPT-5.2 Pro アドバイスに基づく実装）

使用方法:
  python convert_ipss_1_9.py --input 1-9.xlsx --output ipss_single_year.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def year_from_text(x) -> int | None:
    """テキストから年を抽出（例: "(2020)年" → 2020）"""
    m = re.search(r"\((\d{4})\)年", str(x))
    return int(m.group(1)) if m else None


def parse_age(x) -> int | None:
    """年齢文字列をパース（例: "50", "105+" → 整数）"""
    if pd.isna(x):
        return None
    s = str(x).strip().replace(" ", "")
    if s in {"総数", "総　数", "計"}:
        return None
    if s.endswith("+"):  # 105+ など
        s = s[:-1]
    try:
        return int(float(s))
    except:
        return None


def ipss_table_1_9_to_tidy(xlsx_path: str) -> pd.DataFrame:
    """
    IPSS 詳細結果表: 表1-9（男女年齢各歳別人口）を
    year, age, sex, pop(人) のロング形式に変換する。
    
    表1-9の特徴:
    - 各シートに1年分のデータ
    - 左右2ブロック構成（0〜54歳 / 55歳以上）
    - 単位は1,000人
    """
    xls = pd.ExcelFile(xlsx_path)
    records = []
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        
        # 年を抽出（先頭付近に "(2020)年" の形で入っている）
        year = None
        for i in range(min(15, len(df))):
            for j in range(min(5, len(df.columns))):
                y = year_from_text(df.iat[i, j])
                if y is not None:
                    year = y
                    break
            if year is not None:
                break
        
        if year is None:
            print(f"⚠ シート '{sheet}' から年を検出できませんでした。スキップします。")
            continue
        
        def add_block(age_col: int, male_col: int, female_col: int):
            """1ブロック（左または右）のデータを読み取り"""
            for _, row in df.iterrows():
                age = parse_age(row.iloc[age_col] if age_col < len(row) else None)
                if age is None:
                    continue
                
                try:
                    m = row.iloc[male_col] if male_col < len(row) else None
                    f = row.iloc[female_col] if female_col < len(row) else None
                    
                    if pd.isna(m) or pd.isna(f):
                        continue
                    
                    # 単位が(1,000人)なので人に直す
                    records.append({
                        "year": year,
                        "age": age,
                        "sex": "M",
                        "pop": float(m) * 1000
                    })
                    records.append({
                        "year": year,
                        "age": age,
                        "sex": "F",
                        "pop": float(f) * 1000
                    })
                except (IndexError, ValueError, TypeError):
                    continue
        
        # 表1-9は左右2ブロック構成
        # 左ブロック: 列0=年齢, 列2=男, 列3=女 (0〜54歳)
        # 右ブロック: 列5=年齢, 列7=男, 列8=女 (55歳以上)
        add_block(0, 2, 3)  # 左ブロック
        add_block(5, 7, 8)  # 右ブロック
        
        print(f"✓ {year}年: 処理完了")
    
    # 重複除去とソート
    out = (pd.DataFrame(records)
           .drop_duplicates(["year", "age", "sex"])
           .sort_values(["year", "age", "sex"])
           .reset_index(drop=True))
    
    return out


def main():
    p = argparse.ArgumentParser(
        description="IPSS 表1-9（各歳別人口）→ tidy CSV 変換"
    )
    p.add_argument("--input", required=True, help="入力Excel (1-9.xlsx)")
    p.add_argument("--output", required=True, help="出力CSV")
    args = p.parse_args()
    
    df = ipss_table_1_9_to_tidy(args.input)
    
    # 保存
    df.to_csv(args.output, index=False)
    
    # サマリー表示
    years = sorted(df["year"].unique())
    print(f"\n=== 変換完了 ===")
    print(f"出力: {args.output}")
    print(f"年: {years[0]}〜{years[-1]} ({len(years)}年分)")
    print(f"年齢: {df['age'].min()}〜{df['age'].max()}歳")
    print(f"総レコード数: {len(df)}")
    
    # 検証
    print(f"\n=== 検証（最初の年） ===")
    y0 = years[0]
    df_y0 = df[df["year"] == y0]
    male = df_y0[df_y0["sex"] == "M"]["pop"].sum()
    female = df_y0[df_y0["sex"] == "F"]["pop"].sum()
    print(f"{y0}年: 男性 {male/1e4:.1f}万人, 女性 {female/1e4:.1f}万人, 計 {(male+female)/1e4:.1f}万人")


if __name__ == "__main__":
    main()
