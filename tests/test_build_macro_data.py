"""build_macro_data.py のユニットテスト"""

from pathlib import Path
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_macro_data import (
    GDP_REER_DATA,
    load_population_from_csv,
    merge_gdp_reer,
    build_macro_dataframe,
    get_available_scenarios,
    get_scenario_display_name,
    generate_scenario_comparison,
)


class TestGdpReerData:
    """GDP_REER_DATA 定数のテスト"""

    def test_has_base_year_1995(self):
        """基準年1995のデータが存在する"""
        assert 1995 in GDP_REER_DATA
        gdp, reer = GDP_REER_DATA[1995]
        assert gdp > 0
        assert reer is not None and reer > 0

    def test_reer_peak_is_1995(self):
        """REERのピークが1995年である（既知の事実）"""
        years_with_reer = {y: r for y, (g, r) in GDP_REER_DATA.items() if r is not None}
        peak_year = max(years_with_reer, key=lambda y: years_with_reer[y])
        assert peak_year == 1995

    def test_future_years_exist(self):
        """将来推計年（2025以降）が含まれる"""
        future_years = [y for y in GDP_REER_DATA if y >= 2025]
        assert len(future_years) >= 5


class TestMergeGdpReer:
    """merge_gdp_reer() のテスト"""

    def test_adds_gdp_reer_columns(self):
        """GDP, REER列が追加される"""
        pop_df = pd.DataFrame({
            "Year": [1995, 2000, 2020],
            "TotalPop": [12557, 12693, 12571],
            "Young": [2003, 1851, 1503],
            "Working": [8717, 8622, 7449],
            "Elderly": [1828, 2204, 3603],
        })
        result = merge_gdp_reer(pop_df)

        assert "GDP" in result.columns
        assert "REER" in result.columns
        assert "Working_Ratio" in result.columns
        assert "GDP_per_capita" in result.columns

    def test_handles_missing_year(self):
        """GDP_REER_DATAにない年はNaNになる"""
        pop_df = pd.DataFrame({
            "Year": [9999],
            "TotalPop": [10000],
            "Young": [1000],
            "Working": [7000],
            "Elderly": [2000],
        })
        result = merge_gdp_reer(pop_df)

        assert pd.isna(result["GDP"].iloc[0])
        assert pd.isna(result["REER"].iloc[0])


class TestScenarioSupport:
    """シナリオ対応テスト"""

    def test_gdp_scenario_selection(self):
        """GDPシナリオ切替が動作する"""
        df_baseline = build_macro_dataframe(gdp_scenario="gdp_baseline")
        df_growth = build_macro_dataframe(gdp_scenario="gdp_high_growth")

        gdp_2030_baseline = df_baseline[df_baseline["Year"] == 2030]["GDP"].iloc[0]
        gdp_2030_growth = df_growth[df_growth["Year"] == 2030]["GDP"].iloc[0]

        assert gdp_2030_growth > gdp_2030_baseline

    def test_reer_scenario_selection(self):
        """REERシナリオ切替が動作する"""
        df_flat = build_macro_dataframe(reer_scenario="reer_flat")
        df_trend = build_macro_dataframe(reer_scenario="reer_trend")

        reer_2030_flat = df_flat[df_flat["Year"] == 2030]["REER"].iloc[0]
        reer_2024 = df_flat[df_flat["Year"] == 2024]["REER"].iloc[0]

        assert reer_2030_flat == reer_2024

    def test_available_scenarios(self):
        """利用可能シナリオ一覧が取得できる"""
        scenarios = get_available_scenarios()
        assert "gdp_baseline" in scenarios
        assert "reer_flat" in scenarios


class TestGrowthRateScenarios:
    """成長率ベースシナリオのテスト"""

    def test_baseline_is_growth_rate_based(self):
        """ベースラインが成長率ベースになっている"""
        df = build_macro_dataframe(gdp_scenario="gdp_baseline")
        gdp_2050 = df[df["Year"] == 2050]["GDP"].iloc[0]
        assert 1050 < gdp_2050 < 1250

    def test_low_growth_smaller_than_baseline(self):
        """低成長シナリオはベースラインより小さい"""
        df_base = build_macro_dataframe(gdp_scenario="gdp_baseline")
        df_low = build_macro_dataframe(gdp_scenario="gdp_low_growth")

        gdp_base_2100 = df_base[df_base["Year"] == 2100]["GDP"].iloc[0]
        gdp_low_2100 = df_low[df_low["Year"] == 2100]["GDP"].iloc[0]

        assert gdp_low_2100 < gdp_base_2100

    def test_high_growth_larger_than_baseline(self):
        """高成長シナリオはベースラインより大きい"""
        df_base = build_macro_dataframe(gdp_scenario="gdp_baseline")
        df_high = build_macro_dataframe(gdp_scenario="gdp_high_growth")

        gdp_base_2100 = df_base[df_base["Year"] == 2100]["GDP"].iloc[0]
        gdp_high_2100 = df_high[df_high["Year"] == 2100]["GDP"].iloc[0]

        assert gdp_high_2100 > gdp_base_2100 * 2


class TestScenarioDisplayNames:
    """シナリオ表示名のテスト"""

    def test_get_gdp_display_name(self):
        """GDP表示名が取得できる"""
        name = get_scenario_display_name("gdp_baseline")
        assert "2.5%" in name or "ベースライン" in name

    def test_get_reer_display_name(self):
        """REER表示名が取得できる"""
        name = get_scenario_display_name("reer_flat")
        assert "横ばい" in name or "flat" in name.lower()


class TestCabinetScenarios:
    """内閣府シナリオのテスト"""

    def test_cabinet_growth_2025_exists(self):
        """内閣府成長シナリオが読み込める"""
        df = build_macro_dataframe(gdp_scenario="gdp_cabinet_growth_2025")
        assert "GDP" in df.columns
        gdp_2030 = df[df["Year"] == 2030]["GDP"].iloc[0]
        assert gdp_2030 > 700

    def test_cabinet_baseline_2025_exists(self):
        """内閣府ベースラインシナリオが読み込める"""
        df = build_macro_dataframe(gdp_scenario="gdp_cabinet_baseline_2025")
        gdp_2030 = df[df["Year"] == 2030]["GDP"].iloc[0]
        assert 600 < gdp_2030 < 750


class TestScenarioComparison:
    """シナリオ比較機能のテスト"""

    def test_generate_markdown_table(self):
        """Markdown比較表が生成される"""
        result = generate_scenario_comparison(output_format="markdown")
        assert "| シナリオ |" in result
        assert "構造・中位" in result

    def test_generate_csv(self):
        """CSV比較表が生成される"""
        result = generate_scenario_comparison(output_format="csv")
        assert "Scenario," in result


class TestDerivedMetrics:
    """派生指標のテスト"""

    def test_gdp_per_working_calculation(self):
        """GDP_per_working が正しく計算される"""
        pop_df = pd.DataFrame({
            "Year": [2020],
            "TotalPop": [12571],
            "Young": [1503],
            "Working": [7449],
            "Elderly": [3603],
        })
        result = merge_gdp_reer(pop_df)

        assert "GDP_per_working" in result.columns

        gdp_per_working = result.loc[result["Year"] == 2020, "GDP_per_working"].iloc[0]
        assert gdp_per_working is not None
        assert 700 < gdp_per_working < 750

    def test_support_ratio_calculation(self):
        """Support_Ratio が正しく計算される"""
        pop_df = pd.DataFrame({
            "Year": [2020],
            "TotalPop": [12571],
            "Young": [1503],
            "Working": [7449],
            "Elderly": [3603],
        })
        result = merge_gdp_reer(pop_df)

        assert "Support_Ratio" in result.columns

        support_ratio = result.loc[result["Year"] == 2020, "Support_Ratio"].iloc[0]
        assert support_ratio is not None
        assert 0.65 < support_ratio < 0.75

    def test_zero_division_handling(self):
        """ゼロ除算でクラッシュしない"""
        pop_df = pd.DataFrame({
            "Year": [9999],
            "TotalPop": [0],
            "Young": [0],
            "Working": [0],
            "Elderly": [0],
        })
        result = merge_gdp_reer(pop_df)

        assert pd.isna(result["GDP_per_working"].iloc[0]) or result["GDP_per_working"].iloc[0] is None
        assert pd.isna(result["Support_Ratio"].iloc[0]) or result["Support_Ratio"].iloc[0] is None


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_all_reer_nan_does_not_crash(self):
        """REERが全NaNでもクラッシュしない"""
        pop_df = pd.DataFrame({
            "Year": [1950, 1955],
            "TotalPop": [8320, 8928],
            "Young": [2943, 2980],
            "Working": [4966, 5517],
            "Elderly": [411, 475],
        })
        result = merge_gdp_reer(pop_df)

        assert len(result) == 2
        assert result["REER"].isna().all()


class TestLoadPopulationFromCsv:
    """load_population_from_csv() のテスト"""

    def test_loads_and_converts_units(self):
        """CSVを読み込み、単位変換ができる"""
        df = pd.DataFrame({
            "Year": [2000, 2005],
            "TotalPopulation": [127064000, 127756000],
            "Young_0_14": [18510000, 17550000],
            "Working_15_64": [86220000, 84250000],
            "Elderly_65plus": [22040000, 26100000],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pop.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            result = load_population_from_csv(csv_path, input_unit="person")

            assert list(result.columns) == ["Year", "TotalPop", "Young", "Working", "Elderly"]
            assert result["TotalPop"].iloc[0] == pytest.approx(12706.4)

    def test_loads_four_age_categories(self):
        """4区分データがあれば65-74/75+を含む"""
        df = pd.DataFrame({
            "Year": [2000],
            "TotalPopulation": [127064000],
            "Young_0_14": [18510000],
            "Working_15_64": [86220000],
            "Elderly_65plus": [22040000],
            "VeryOld_75plus": [12000000],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pop4.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            result = load_population_from_csv(csv_path, input_unit="person")

            assert "Elderly_65_74" in result.columns
            assert "Elderly_75plus" in result.columns
            assert result["Elderly_65_74"].iloc[0] == pytest.approx(1004.0)
            assert result["Elderly_75plus"].iloc[0] == pytest.approx(1200.0)


class TestBuildMacroDataframe:
    """build_macro_dataframe() のテスト"""

    def test_builds_with_year_filter(self):
        """年フィルタが適用される"""
        df = pd.DataFrame({
            "Year": [1995, 2000, 2005],
            "TotalPopulation": [125570000, 126930000, 127756000],
            "Young_0_14": [20030000, 18510000, 17550000],
            "Working_15_64": [87170000, 86220000, 84250000],
            "Elderly_65plus": [18280000, 22040000, 26100000],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "pop.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            result = build_macro_dataframe(csv_path=csv_path, years=[1995, 2005], input_unit="person")

            assert result["Year"].tolist() == [1995, 2005]
        assert "GDP_per_capita" in result.columns

    def test_handles_missing_year(self):
        """GDP_REER_DATAにない年はNaNになる"""
        pop_df = pd.DataFrame({
            "Year": [9999],  # 存在しない年
            "TotalPop": [10000],
            "Young": [1000],
            "Working": [7000],
            "Elderly": [2000],
        })
        result = merge_gdp_reer(pop_df)

        assert pd.isna(result["GDP"].iloc[0])
        assert pd.isna(result["REER"].iloc[0])


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_all_reer_nan_does_not_crash(self):
        """REERが全NaNでもクラッシュしない"""
        pop_df = pd.DataFrame({
            "Year": [1950, 1955],  # REERがNoneの年
            "TotalPop": [8320, 8928],
            "Young": [2943, 2980],
            "Working": [4966, 5517],
            "Elderly": [411, 475],
        })
        result = merge_gdp_reer(pop_df)

        # REERが全てNaNでもDataFrameは正常に返る
        assert len(result) == 2
        assert result["REER"].isna().all()
