# -*- coding: utf-8 -*-
"""
build_macro_data.py

人口CSVからマクロデータを自動構築するユーティリティ

機能:
- IPSSの人口予測CSVから年齢3区分（年少・生産年齢・高齢）を読み込み
- GDP/REERデータはYAMLファイルから読み込み（メタデータ付き）
- DataFrameとして統合出力
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from copy import deepcopy
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # フォールバック用


# デフォルトのGDP/REERデータファイル
DEFAULT_GDP_REER_PATH = Path(__file__).parent.parent / "data" / "gdp_reer_data.yaml"

# ---------------------------------
# GDP構造モデルのデフォルトパラメータ
# ---------------------------------
# 名目GDP = 生産年齢人口(万人) × 労働生産性(万円/人) × 物価調整
#
# 2024年実績からキャリブレーション:
#   GDP: 610兆円, 生産年齢人口: 約7,400万人
#   → 労働生産性 ≈ 610兆円 / 7,400万人 ≈ 824万円/人
GDP_STRUCTURAL_DEFAULTS = {
    "base_year": 2024,
    "base_gdp": 610.0,           # 兆円
    "productivity_scenarios": {  # 生産性成長率シナリオ (年率%)
        "low": 0.5,              # 低成長: 0.5%/年
        "mid": 1.0,              # 中位: 1.0%/年
        "high": 1.5,             # 高成長: 1.5%/年
    },
    "inflation_scenarios": {     # インフレ率シナリオ (年率%)
        "deflation": -0.5,       # デフレ継続
        "stable": 1.0,           # 安定
        "target": 2.0,           # 日銀目標
    },
    "custom_scenarios": {},      # 追加シナリオ
}

# デフォルト人口CSV（パッケージングで日本語ディレクトリ名が崩れることがあるため、ファイル名で探索する）
DEFAULT_POPULATION_CSV_NAME = "japan_population_age_projection_2100_v3.csv"

# 単年齢人口CSV（生産年齢の内部分割用）
SINGLE_AGE_CSV_NAME = "ipss_interpolated_1965_2070.csv"

def get_default_population_csv_path() -> Path:
    """デフォルトの人口CSVパスを取得する。

    パッケージング環境（zip展開など）によっては日本語ディレクトリ名が文字化けする場合があるため、
    data/ 配下からファイル名で探索して見つける。
    """
    data_dir = Path(__file__).parent.parent / "data"

    # 1) ファイル名で探索（最も頑丈）
    candidates = list(data_dir.rglob(DEFAULT_POPULATION_CSV_NAME))
    if candidates:
        # 最も浅いパスを優先（候補が複数あっても安定して選べる）
        candidates = sorted(candidates, key=lambda p: len(str(p)))
        return candidates[0]

    # 2) 従来の日本語ディレクトリ名（開発環境互換）
    legacy = data_dir / "日本の人口予測03_修正版v3" / DEFAULT_POPULATION_CSV_NAME
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "デフォルト人口CSVが見つかりません。\n"
        f"探したファイル名: {DEFAULT_POPULATION_CSV_NAME}\n"
        f"探索ディレクトリ: {data_dir}\n"
        "ヒント: --csv オプションで人口CSVのパスを指定してください。"
    )


# GDP・REERデータ（YAMLが読めない場合のフォールバック）
GDP_REER_DATA: Dict[int, Tuple[float, Optional[float]]] = {
    # Year: (GDP兆円, REER)
    1950: (4.0, None),
    1955: (8.4, None),
    1960: (16.0, None),
    1964: (29.5, 55),
    1965: (32.9, 57),
    1970: (73.3, 68),
    1971: (80.7, 72),
    1973: (112.5, 85),
    1975: (148.3, 82),
    1980: (240.2, 90),
    1985: (324.5, 98),
    1988: (387.0, 125),
    1990: (442.8, 110),
    1991: (469.4, 115),
    1993: (476.1, 135),
    1995: (501.7, 150),
    1998: (500.2, 125),
    2000: (509.9, 135),
    2005: (505.3, 110),
    2007: (518.3, 105),
    2008: (505.1, 100),
    2010: (500.4, 110),
    2011: (491.4, 115),
    2012: (495.0, 110),
    2013: (507.5, 100),
    2015: (532.2, 85),
    2018: (556.2, 78),
    2020: (539.0, 100),
    2021: (541.5, 90),
    2022: (560.7, 75),
    2023: (590.0, 70),
    2024: (610.0, 68),
    # 推計値（仮定）
    2025: (625, 65),
    2030: (707, 62),
    2035: (800, 60),
    2040: (906, 58),
    2045: (1025, 55),
    2050: (1159, 53),
    2055: (1312, 50),
    2060: (1484, 48),
    2065: (1679, 45),
    2070: (1899, 43),
    2075: (2149, 40),
    2080: (2431, 38),
    2085: (2751, 36),
    2090: (3112, 34),
    2095: (3521, 32),
    2100: (3984, 30),
}


def load_gdp_reer_from_yaml(
    yaml_path: Path = DEFAULT_GDP_REER_PATH,
    gdp_scenario: str = "gdp_baseline",
    reer_scenario: str = "reer_flat"
) -> Tuple[Dict[int, Tuple[float, Optional[float], bool, bool]], Dict[str, Any]]:
    """
    YAML形式のGDP/REERデータを読み込む

    Args:
        yaml_path: YAMLファイルのパス
        gdp_scenario: GDPシナリオ名（gdp_baseline など）
        reer_scenario: REERシナリオ名（reer_flat など）

    Returns:
        (gdp_reer_dict, metadata)
        - gdp_reer_dict: {year: (gdp, reer, gdp_is_projection, reer_is_projection)} の辞書
        - metadata: 出所・仮定情報
    """
    if yaml is None:
        raise ImportError("PyYAMLがインストールされていません: pip install pyyaml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"GDP/REERデータが見つかりません: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 旧形式: data配列がある場合は後方互換で処理
    if "data" in raw:
        gdp_reer_dict = {}
        for item in raw["data"]:
            year = item["year"]
            gdp = item.get("gdp")
            reer = item.get("reer")
            gdp_is_proj = item.get("gdp_is_projection", False)
            reer_is_proj = item.get("reer_is_projection", False)
            gdp_reer_dict[year] = (gdp, reer, gdp_is_proj, reer_is_proj)
        return gdp_reer_dict, raw.get("metadata", {})

    # 新形式: actual_data + projection_data (scenarios)
    actual_data = raw.get("actual_data", [])
    projection_data = raw.get("projection_data", {})

    def get_projection_list(data: Any) -> list:
        if isinstance(data, dict):
            return data.get("data", [])
        return data or []

    gdp_proj_list = get_projection_list(projection_data.get(gdp_scenario, []))
    reer_proj_list = get_projection_list(projection_data.get(reer_scenario, []))

    gdp_proj = {item["year"]: item.get("gdp") for item in gdp_proj_list}
    reer_proj = {item["year"]: item.get("reer") for item in reer_proj_list}

    actual_map: Dict[int, Dict[str, Optional[float]]] = {}
    for item in actual_data:
        year = item["year"]
        actual_map[year] = {
            "gdp": item.get("gdp"),
            "reer": item.get("reer")
        }

    years = set(actual_map.keys()) | set(gdp_proj.keys()) | set(reer_proj.keys())
    gdp_reer_dict: Dict[int, Tuple[float, Optional[float], bool, bool]] = {}
    for year in sorted(years):
        gdp = gdp_proj.get(year, actual_map.get(year, {}).get("gdp"))
        reer = reer_proj.get(year, actual_map.get(year, {}).get("reer"))
        gdp_is_proj = year in gdp_proj
        reer_is_proj = year in reer_proj
        gdp_reer_dict[year] = (gdp, reer, gdp_is_proj, reer_is_proj)

    return gdp_reer_dict, raw.get("metadata", {})


def get_available_scenarios(yaml_path: Path = DEFAULT_GDP_REER_PATH) -> Dict[str, Any]:
    """利用可能なシナリオ一覧を取得"""
    if yaml is None:
        raise ImportError("PyYAMLがインストールされていません: pip install pyyaml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"GDP/REERデータが見つかりません: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw.get("scenarios", {})


def get_scenario_display_info(
    scenario_id: str,
    yaml_path: Path = DEFAULT_GDP_REER_PATH
) -> Dict[str, Any]:
    """
    シナリオIDからUI表示用の情報を取得

    Args:
        scenario_id: "gdp_baseline", "reer_flat" など

    Returns:
        例:
        {
            "name": "ベースライン（名目2.5%成長）",
            "type": "growth_rate",
            "growth_rate": 2.5,
            "description": "現状維持シナリオ",
        }
    """
    scenarios = get_available_scenarios(yaml_path)
    if scenario_id in scenarios:
        return scenarios[scenario_id]
    return {"name": scenario_id, "type": "unknown"}


def get_scenario_display_name(scenario_id: str) -> str:
    """
    シナリオIDからUI表示名のみを取得（簡易版）

    Args:
        scenario_id: "gdp_baseline" など

    Returns:
        "ベースライン（名目2.5%成長）"
    """
    info = get_scenario_display_info(scenario_id)
    return info.get("name", scenario_id)


def generate_gdp_from_growth_rate(
    base_year: int = 2024,
    base_gdp: float = 610.0,
    growth_rate: float = 2.5,
    end_year: int = 2100,
    interval: int = 5,
) -> list:
    """
    成長率から将来GDP推計を生成（YAMLデータ作成補助）

    Args:
        base_year: 基準年
        base_gdp: 基準年GDP（兆円）
        growth_rate: 年間成長率（%）
        end_year: 最終年
        interval: 出力間隔（年）

    Returns:
        [{"year": 2025, "gdp": 625}, ...] 形式のリスト
    """
    rate = growth_rate / 100
    projections = []

    for year in range(base_year + 1, end_year + 1, interval):
        years_diff = year - base_year
        gdp = base_gdp * ((1 + rate) ** years_diff)
        projections.append({"year": year, "gdp": int(round(gdp))})

    return projections


def get_data_metadata(yaml_path: Path = DEFAULT_GDP_REER_PATH) -> Dict[str, Any]:
    """
    メタデータのみを取得（UI表示用）

    Args:
        yaml_path: YAMLファイルのパス

    Returns:
        メタデータ辞書
    """
    try:
        _, metadata = load_gdp_reer_from_yaml(yaml_path)
        return metadata
    except (ImportError, FileNotFoundError):
        # フォールバック: デフォルトメタデータ
        return {
            "gdp": {"source": "内閣府 国民経済計算", "actual_end_year": 2024},
            "reer": {"source": "BIS 実質実効為替レート", "actual_end_year": 2024},
            "projection": {
                "gdp_assumption": "仮定値",
                "reer_assumption": "仮定値"
            }
        }


def get_projection_start_year(df: pd.DataFrame, column: str = "GDP") -> int:
    """
    指定列の推計開始年を取得（DataFrameのフラグから自動判定）

    Args:
        df: merge_gdp_reer()で作成したDataFrame
        column: "GDP" or "REER"

    Returns:
        推計開始年（フラグがTrueになる最初の年）
        フラグが存在しないか全てFalseの場合は9999を返す
    """
    proj_col = f"{column}_IsProjection"

    if proj_col not in df.columns:
        return 9999  # 推計なし

    proj_df = df[df[proj_col] == True]
    if len(proj_df) == 0:
        return 9999  # 推計なし

    return int(proj_df["Year"].min())


def get_actual_data_end(df: pd.DataFrame, column: str = "GDP") -> int:
    """
    指定列の実績データ最終年を取得

    Args:
        df: merge_gdp_reer()で作成したDataFrame
        column: "GDP" or "REER"

    Returns:
        推計開始年の1年前（実績の最終年）
    """
    proj_start = get_projection_start_year(df, column)
    if proj_start == 9999:
        # 推計なし = 全て実績
        return int(df["Year"].max())
    return proj_start - 1


# ---------------------------------
# GDP構造モデル（人口整合型推計）
# ---------------------------------

def load_structural_params(yaml_path: Path = DEFAULT_GDP_REER_PATH) -> Dict[str, Any]:
    """構造モデルパラメータをYAMLから読み込み"""
    params = deepcopy(GDP_STRUCTURAL_DEFAULTS)

    if yaml is None:
        return params

    if not yaml_path.exists():
        return params

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    yaml_params = raw.get("structural_model_params", {})
    for key, value in yaml_params.items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key].update(value)
        else:
            params[key] = value

    return params


def calculate_structural_gdp(
    pop_df: pd.DataFrame,
    productivity_growth: str = "mid",
    inflation: str = "stable",
    base_year: int = 2024,
    custom_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    構造モデルによるGDP推計を計算

    名目GDP(t) = 生産年齢人口(t) × 労働生産性(t) × 物価調整(t)

    Args:
        pop_df: 人口DataFrameー（Year, Working列を含む）
        productivity_growth: 生産性成長率シナリオ ("low", "mid", "high" or 数値)
        inflation: インフレ率シナリオ ("deflation", "stable", "target" or 数値)
        base_year: キャリブレーション基準年
        custom_params: カスタムパラメータ（上書き用）

    Returns:
        GDP_Structural列が追加されたDataFrame
    """
    params = load_structural_params()
    if custom_params:
        for key, value in custom_params.items():
            if isinstance(value, dict) and isinstance(params.get(key), dict):
                params[key].update(value)
            else:
                params[key] = value

    # シナリオ名から成長率を取得（数値の場合はそのまま使用）
    if isinstance(productivity_growth, str):
        prod_rate = params["productivity_scenarios"].get(productivity_growth, 1.0) / 100
    else:
        prod_rate = float(productivity_growth) / 100

    if isinstance(inflation, str):
        infl_rate = params["inflation_scenarios"].get(inflation, 1.0) / 100
    else:
        infl_rate = float(inflation) / 100

    # 基準年のデータでキャリブレーション
    base_row = pop_df[pop_df["Year"] == base_year]
    if len(base_row) == 0:
        # 基準年がない場合は最も近い年を使用
        closest_year = pop_df["Year"].iloc[(pop_df["Year"] - base_year).abs().argmin()]
        base_row = pop_df[pop_df["Year"] == closest_year]
        base_year = int(closest_year)

    base_working = float(base_row["Working"].iloc[0])  # 万人
    base_gdp = params["base_gdp"]  # 兆円

    # 基準年の労働生産性（万円/人）
    # GDP(兆円) / 人口(万人) = 万円/人 × 10000 → 兆円/万人
    # 610兆円 / 7400万人 = 0.0824兆円/万人 = 824万円/人
    base_productivity = base_gdp / base_working  # 兆円/万人

    df = pop_df.copy()

    def calc_gdp(row):
        year = row["Year"]
        working = row["Working"]

        if pd.isna(working) or working <= 0:
            return np.nan

        if year <= base_year:
            # 基準年以前は元のGDPを使用（存在すれば）
            return np.nan  # 後で実績値で上書き

        # 成長率を適用
        years_diff = year - base_year
        productivity = base_productivity * ((1 + prod_rate) ** years_diff)
        deflator = (1 + infl_rate) ** years_diff

        # 名目GDP = 生産年齢人口 × 労働生産性 × 物価調整
        gdp = working * productivity * deflator

        return gdp

    df["GDP_Structural"] = df.apply(calc_gdp, axis=1)

    # 成長率情報を属性として保持（UI表示用）
    df.attrs["structural_params"] = {
        "base_year": base_year,
        "base_gdp": base_gdp,
        "base_working": base_working,
        "base_productivity": base_productivity,
        "productivity_growth_rate": prod_rate * 100,
        "inflation_rate": infl_rate * 100,
        "productivity_scenario": productivity_growth,
        "inflation_scenario": inflation,
    }

    return df


def get_structural_scenarios() -> Dict[str, Dict[str, Any]]:
    """構造モデルで利用可能なシナリオ組み合わせを取得"""
    params = load_structural_params()
    scenarios: Dict[str, Dict[str, Any]] = {}

    for prod_name, prod_rate in params["productivity_scenarios"].items():
        for infl_name, infl_rate in params["inflation_scenarios"].items():
            key = f"structural_{prod_name}_{infl_name}"
            scenarios[key] = {
                "name": f"構造モデル（生産性{prod_rate}%/インフレ{infl_rate}%）",
                "productivity_growth": prod_name,
                "inflation": infl_name,
                "description": f"生産年齢人口×生産性({prod_rate}%/年)×物価({infl_rate}%/年)",
            }

    for custom_key, custom in params.get("custom_scenarios", {}).items():
        scenario_id = f"structural_custom_{custom_key}"
        scenarios[scenario_id] = {
            "name": custom.get("name", f"構造モデル（{custom_key}）"),
            "productivity_growth": custom.get("productivity", 1.0),
            "inflation": custom.get("inflation", 1.0),
            "description": custom.get("description", "カスタム構造モデル"),
        }

    return scenarios


# 内部単位: 万人
INTERNAL_UNIT = "man"

def safe_divide(numerator, denominator, scale=1.0):
    """安全な除算（ゼロ除算や欠損でNaNを返す）"""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator * scale


def get_single_age_csv_path() -> Optional[Path]:
    """単年齢人口CSVのパスを取得（存在しない場合はNone）"""
    data_dir = Path(__file__).parent.parent / "data"

    # jp-pop-pyramid/data/ 配下を探索
    pyramid_data = Path(__file__).parent.parent / "jp-pop-pyramid" / "data"
    if pyramid_data.exists():
        candidates = list(pyramid_data.glob(SINGLE_AGE_CSV_NAME))
        if candidates:
            return candidates[0]

    # data/ 配下も探索
    candidates = list(data_dir.rglob(SINGLE_AGE_CSV_NAME))
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]

    return None


def load_working_age_breakdown(single_age_csv: Path) -> pd.DataFrame:
    """
    単年齢CSVから生産年齢の内部分割データを計算

    Args:
        single_age_csv: 単年齢人口CSV（year, age, sex, pop形式）

    Returns:
        DataFrame with columns:
        - Year
        - Working_15_19: 学生層（15-19歳）万人
        - Working_20_34: 若手・形成層（20-34歳）万人
        - Working_35_59: 熟練・高所得層（35-59歳）万人
        - Working_60_64: シニア就労（60-64歳）万人
    """
    df = pd.read_csv(single_age_csv, encoding="utf-8-sig")

    # 男女合計（年・年齢別）
    df_total = df.groupby(["year", "age"])["pop"].sum().reset_index()

    # 年齢区分別に集計
    def age_sum(year_df, age_min, age_max):
        return year_df[(year_df["age"] >= age_min) & (year_df["age"] <= age_max)]["pop"].sum()

    years = sorted(df_total["year"].unique())
    result = []

    for year in years:
        year_df = df_total[df_total["year"] == year]
        result.append({
            "Year": year,
            "Working_15_19": age_sum(year_df, 15, 19) / 10000,  # 人→万人
            "Working_20_34": age_sum(year_df, 20, 34) / 10000,  # 若手・形成層
            "Working_35_59": age_sum(year_df, 35, 59) / 10000,  # 熟練・高所得層
            "Working_60_64": age_sum(year_df, 60, 64) / 10000,
        })

    return pd.DataFrame(result)


def load_population_from_csv(
    csv_path: Path,
    input_unit: str = "man"
) -> pd.DataFrame:
    """
    IPSS形式の人口CSVから年齢区分データを読み込む

    Args:
        csv_path: 人口CSVファイルのパス
        input_unit: 入力CSVの単位（"person"=人, "thousand"=千人, "man"=万人）

    Returns:
        Year, TotalPop, Young, Working, Elderly を含むDataFrame（単位: 万人）
        4区分データがある場合は Elderly_65_74, Elderly_75plus も含む
    """
    # ファイル存在チェック
    if not csv_path.exists():
        raise FileNotFoundError(
            f"人口CSVが見つかりません: {csv_path}\n"
            f"ヒント: --csv オプションでパスを指定するか、"
            f"デフォルトパスにファイルを配置してください。"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 必須カラムの確認
    required_cols = ["Year", "TotalPopulation", "Young_0_14", "Working_15_64", "Elderly_65plus"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 4区分データがあるか確認
    has_4_categories = "VeryOld_75plus" in df.columns

    # 必要なカラムのみ抽出してリネーム
    result = df[["Year", "TotalPopulation", "Young_0_14", "Working_15_64", "Elderly_65plus"]].copy()
    result.columns = ["Year", "TotalPop", "Young", "Working", "Elderly"]

    # 4区分対応: 65-74歳と75歳以上を追加
    if has_4_categories:
        result["Elderly_75plus"] = df["VeryOld_75plus"]
        result["Elderly_65_74"] = result["Elderly"] - result["Elderly_75plus"]

    # 単位変換（内部単位「万人」へ）
    unit_multiplier = {
        "person": 1 / 10000,    # 人 → 万人
        "thousand": 1 / 10,     # 千人 → 万人
        "man": 1,               # 万人 → 万人（変換なし）
    }

    if input_unit not in unit_multiplier:
        raise ValueError(f"Unknown unit: {input_unit}. Use 'person', 'thousand', or 'man'.")

    multiplier = unit_multiplier[input_unit]
    if multiplier != 1:
        pop_cols = ["TotalPop", "Young", "Working", "Elderly"]
        if has_4_categories:
            pop_cols += ["Elderly_65_74", "Elderly_75plus"]
        for col in pop_cols:
            result[col] = result[col] * multiplier

    return result


def merge_gdp_reer(
    pop_df: pd.DataFrame,
    gdp_reer: Optional[Dict[int, Tuple]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    gdp_scenario: str = "gdp_baseline",
    reer_scenario: str = "reer_flat"
) -> pd.DataFrame:
    """
    人口DataFrameにGDP・REERデータをマージ

    Args:
        pop_df: load_population_from_csv() の出力
        gdp_reer: {Year: (GDP, REER, ...)} の辞書。Noneの場合はYAML or デフォルト使用
        metadata: メタデータ辞書（実績/推計境界情報）
        gdp_scenario: GDPシナリオ名（"structural_mid_stable"形式で構造モデル使用）
        reer_scenario: REERシナリオ名

    Returns:
        GDP, REER, GDP_IsProjection, REER_IsProjection などが追加されたDataFrame
    """
    use_yaml_data = False
    use_structural_gdp = gdp_scenario.startswith("structural_")

    # 構造モデルの場合、パラメータを解析
    structural_params: Dict[str, Any] = {}
    if use_structural_gdp:
        scenario_map = get_structural_scenarios()
        if gdp_scenario in scenario_map:
            structural_params["productivity_growth"] = scenario_map[gdp_scenario]["productivity_growth"]
            structural_params["inflation"] = scenario_map[gdp_scenario]["inflation"]
        else:
            # "structural_mid_stable" → productivity="mid", inflation="stable"
            parts = gdp_scenario.split("_")
            if len(parts) >= 3:
                structural_params["productivity_growth"] = parts[1]
                structural_params["inflation"] = parts[2]
            else:
                structural_params["productivity_growth"] = "mid"
                structural_params["inflation"] = "stable"

    if gdp_reer is None:
        # YAMLからの読み込みを試行
        try:
            # 構造モデルの場合も実績データはYAMLから取得
            yaml_gdp_scenario = "gdp_baseline" if use_structural_gdp else gdp_scenario
            gdp_reer, metadata = load_gdp_reer_from_yaml(
                gdp_scenario=yaml_gdp_scenario,
                reer_scenario=reer_scenario
            )
            use_yaml_data = True
        except (ImportError, FileNotFoundError):
            # フォールバック: 旧形式の辞書を使用
            gdp_reer = GDP_REER_DATA
            metadata = None

    df = pop_df.copy()

    # GDP/REERを追加（YAML形式は4要素タプル、旧形式は2要素タプル）
    if use_yaml_data:
        # YAML形式: (gdp, reer, gdp_is_proj, reer_is_proj)
        df["GDP"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None, None, None))[0])
        df["REER"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None, None, None))[1])
        df["GDP_IsProjection"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None, False, False))[2])
        df["REER_IsProjection"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None, False, False))[3])
    else:
        # 旧形式: (gdp, reer)
        df["GDP"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None))[0])
        df["REER"] = df["Year"].map(lambda y: gdp_reer.get(y, (None, None))[1])
        # メタデータから実績/推計の境界を取得
        if metadata:
            gdp_actual_end = metadata.get("gdp", {}).get("actual_end_year", 2024)
            reer_actual_end = metadata.get("reer", {}).get("actual_end_year", 2024)
        else:
            gdp_actual_end = 2024
            reer_actual_end = 2024
        df["GDP_IsProjection"] = df["Year"] > gdp_actual_end
        df["REER_IsProjection"] = df["Year"] > reer_actual_end

    # 構造モデルによるGDP推計を適用
    if use_structural_gdp:
        df = calculate_structural_gdp(
            df,
            productivity_growth=structural_params.get("productivity_growth", "mid"),
            inflation=structural_params.get("inflation", "stable"),
        )
        # 推計年のGDPを構造モデル値で上書き
        gdp_actual_end = metadata.get("gdp", {}).get("actual_end_year", 2024) if metadata else 2024
        mask = df["Year"] > gdp_actual_end
        df.loc[mask, "GDP"] = df.loc[mask, "GDP_Structural"]
        df["GDP_IsProjection"] = df["Year"] > gdp_actual_end
        
        # 構造モデル使用フラグを追加
        df["GDP_IsStructural"] = mask

    # 派生指標を計算
    df["Working_Ratio"] = df.apply(
        lambda row: safe_divide(row["Working"], row["TotalPop"], 100),
        axis=1
    )
    df["Young_Ratio"] = df.apply(
        lambda row: safe_divide(row["Young"], row["TotalPop"], 100),
        axis=1
    )
    df["Elderly_Ratio"] = df.apply(
        lambda row: safe_divide(row["Elderly"], row["TotalPop"], 100),
        axis=1
    )

    # 4区分の構成比（データがある場合のみ）
    if "Elderly_65_74" in df.columns:
        df["Elderly_65_74_Ratio"] = df.apply(
            lambda row: safe_divide(row["Elderly_65_74"], row["TotalPop"], 100),
            axis=1
        )
    if "Elderly_75plus" in df.columns:
        df["Elderly_75plus_Ratio"] = df.apply(
            lambda row: safe_divide(row["Elderly_75plus"], row["TotalPop"], 100),
            axis=1
        )

    # GDP_per_capita / GDP_per_working は「GDP/人口 * 10000（万円/人）」で計算
    df["GDP_per_capita"] = df.apply(
        lambda row: safe_divide(row["GDP"], row["TotalPop"], 10000),
        axis=1
    )
    df["GDP_per_working"] = df.apply(
        lambda row: safe_divide(row["GDP"], row["Working"], 10000),
        axis=1
    )
    df["Support_Ratio"] = df.apply(
        lambda row: safe_divide(row["Young"] + row["Elderly"], row["Working"]),
        axis=1
    )
    df["Dependency_Ratio"] = df["Support_Ratio"] * 100

    # 生産年齢の内部分割（単年齢データがあれば追加）
    single_age_csv = get_single_age_csv_path()
    if single_age_csv is not None:
        try:
            working_breakdown = load_working_age_breakdown(single_age_csv)
            # 年でマージ（単年齢データがない年はNaN）
            df = df.merge(working_breakdown, on="Year", how="left")
        except Exception as e:
            # 単年齢データの読み込みに失敗しても続行
            print(f"[WARN] 生産年齢内部分割データ読み込み失敗: {e}")

    return df


def build_macro_dataframe(
    csv_path: Optional[Path] = None,
    years: Optional[list] = None,
    input_unit: str = "man",
    gdp_scenario: str = "gdp_baseline",
    reer_scenario: str = "reer_flat"
) -> pd.DataFrame:
    """
    マクロデータDataFrameを構築（メインエントリポイント）

    Args:
        csv_path: 人口CSVのパス。Noneの場合はデフォルトパスを使用
        years: 出力に含める年のリスト。Noneの場合は全年
        input_unit: 入力CSVの単位（"person"=人, "thousand"=千人, "man"=万人）
        gdp_scenario: GDPシナリオ名
        reer_scenario: REERシナリオ名

    Returns:
        統合されたマクロデータDataFrame
    """
    # デフォルトCSVパス
    if csv_path is None:
        csv_path = get_default_population_csv_path()

    # 人口データ読み込み
    pop_df = load_population_from_csv(csv_path, input_unit=input_unit)

    # GDP/REERマージ
    df = merge_gdp_reer(
        pop_df,
        gdp_scenario=gdp_scenario,
        reer_scenario=reer_scenario
    )

    # 年フィルタ
    if years is not None:
        df = df[df["Year"].isin(years)]

    # 年でソート
    df = df.sort_values("Year").reset_index(drop=True)

    return df


def get_available_years(csv_path: Optional[Path] = None) -> list:
    """CSVに含まれる年のリストを取得"""
    if csv_path is None:
        csv_path = get_default_population_csv_path()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return sorted(df["Year"].tolist())


def generate_scenario_comparison(
    years: list = [2030, 2050, 2070, 2100],
    output_format: str = "markdown"
) -> str:
    """
    全シナリオのGDP比較表を生成

    Args:
        years: 比較する年のリスト
        output_format: "markdown" or "csv"

    Returns:
        フォーマット済み文字列
    """
    scenarios = {
        "gdp_baseline": "ベースライン",
        "gdp_cabinet_growth_2025": "内閣府・成長（2025試算）",
        "gdp_cabinet_baseline_2025": "内閣府・基準（2025試算）",
        "structural_low_deflation": "構造・悲観",
        "structural_mid_stable": "構造・中位",
        "structural_high_target": "構造・楽観",
    }

    results: Dict[str, Dict[int, Optional[float]]] = {}
    for scenario_id, name in scenarios.items():
        try:
            df = build_macro_dataframe(gdp_scenario=scenario_id)
            results[name] = {y: df[df["Year"] == y]["GDP"].iloc[0] for y in years}
        except Exception:
            continue

    if output_format == "markdown":
        header = "| シナリオ | " + " | ".join(str(y) for y in years) + " |"
        separator = "|" + "|".join(["---"] * (len(years) + 1)) + "|"
        rows = []
        for name, vals in results.items():
            row = f"| {name} | " + " | ".join(
                f"{vals.get(y, '-'):.0f}" if isinstance(vals.get(y), (int, float, np.floating)) else "-"
                for y in years
            ) + " |"
            rows.append(row)
        return "\n".join([header, separator] + rows)

    if output_format == "csv":
        lines = ["Scenario," + ",".join(str(y) for y in years)]
        for name, vals in results.items():
            line = f"{name}," + ",".join(
                f"{vals.get(y, ''):.0f}" if isinstance(vals.get(y), (int, float, np.floating)) else ""
                for y in years
            )
            lines.append(line)
        return "\n".join(lines)

    raise ValueError("output_format must be 'markdown' or 'csv'")


# ---------------------------------
# CLI
# ---------------------------------

def main():
    """CLIエントリポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="人口CSVからマクロデータを構築")
    parser.add_argument("--csv", type=str, help="入力CSVファイルパス")
    parser.add_argument("--out", type=str, help="出力CSVファイルパス")
    parser.add_argument("--years", type=str, help="出力年（カンマ区切り、例: 1950,1995,2020,2070）")
    parser.add_argument("--list-years", action="store_true", help="利用可能な年を表示")
    parser.add_argument(
        "--unit",
        type=str,
        choices=["person", "thousand", "man"],
        default="man",
        help="入力CSVの人口単位（person=人, thousand=千人, man=万人）"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None

    if args.list_years:
        years = get_available_years(csv_path)
        print(f"利用可能な年: {min(years)}-{max(years)}年 ({len(years)}点)")
        print(f"年リスト: {years}")
        return

    # 年フィルタ
    years = None
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]

    # データ構築
    df = build_macro_dataframe(csv_path, years, input_unit=args.unit)

    print(f"データ範囲: {df['Year'].min()}-{df['Year'].max()}年 ({len(df)}点)")
    print(f"カラム: {list(df.columns)}")
    print()
    print(df.to_string(index=False))

    # CSV出力
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
