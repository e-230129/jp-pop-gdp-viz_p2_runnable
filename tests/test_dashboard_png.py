"""ダッシュボードPNG生成のテスト"""

import tempfile
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_macro_data import build_macro_dataframe
from pop_gdp_reer_dashboard import create_static_summary, create_dashboard


@pytest.fixture
def sample_df_with_new_metrics():
    """新しい派生指標を含むテスト用DataFrame"""
    years = [1995, 2000, 2020, 2025, 2030]
    return pd.DataFrame({
        "Year": years,
        "TotalPop": [12557, 12693, 12571, 12326, 12012],
        "Working": [8717, 8622, 7449, 7296, 6965],
        "Young": [2003, 1851, 1503, 1369, 1246],
        "Elderly": [1828, 2204, 3603, 3661, 3801],
        "GDP": [501.7, 509.9, 539.0, 620, 640],
        "REER": [150, 135, 100, 65, 62],
        "Working_Ratio": [69.4, 67.9, 59.3, 59.2, 58.0],
        "Young_Ratio": [16.0, 14.6, 12.0, 11.1, 10.4],
        "Elderly_Ratio": [14.6, 17.4, 28.7, 29.7, 31.6],
        "GDP_per_capita": [399.5, 401.6, 428.8, 503.0, 533.0],
        "GDP_per_working": [575.6, 591.3, 723.5, 849.7, 919.0],
        "Support_Ratio": [2.28, 2.13, 1.46, 1.45, 1.38],
        "GDP_IsProjection": [False, False, False, True, True],
        "REER_IsProjection": [False, False, False, True, True],
    })


class TestStaticSummary:
    """create_static_summary() のテスト"""

    def test_png_generation_does_not_crash(self):
        """PNG生成がクラッシュしない"""
        df = build_macro_dataframe()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.png"
            create_static_summary(df, out_path)

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_handles_reer_all_nan(self):
        """REER全NaNでもPNG生成が成功"""
        # REER列が全NaNのダミーデータ
        df = pd.DataFrame({
            "Year": [1950, 1960, 1970],
            "TotalPop": [8320, 9342, 10372],
            "Working": [4966, 6047, 7212],
            "Young": [2943, 2807, 2482],
            "Elderly": [411, 535, 733],
            "GDP": [4.0, 16.0, 73.3],
            "REER": [None, None, None],
            "Working_Ratio": [59.7, 64.7, 69.5],
            "Young_Ratio": [35.4, 30.1, 23.9],
            "Elderly_Ratio": [4.9, 5.7, 7.1],
            "GDP_per_capita": [48.1, 171.3, 706.8],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_reer_nan.png"
            create_static_summary(df, out_path)
            assert out_path.exists()


class TestStaticSummaryWithNewMetrics:
    """新しい派生指標を含むPNG生成テスト"""

    def test_png_generation_with_new_metrics(self, sample_df_with_new_metrics, tmp_path):
        """新しいメトリクスがあってもPNG生成がクラッシュしない"""
        out_path = tmp_path / "test_new_metrics.png"
        create_static_summary(sample_df_with_new_metrics, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestConnectedScatterPanel:
    """相関図パネルのテスト"""

    def test_correlation_panel_exists(self, sample_df_with_new_metrics, tmp_path):
        """相関図パネルの軸ラベルがHTMLに含まれる"""
        out_path = tmp_path / "test_correlation.html"
        create_dashboard(sample_df_with_new_metrics, out_path)

        assert out_path.exists()
        html_content = out_path.read_text(encoding="utf-8")
        label_x = "生産年齢比率"
        label_y = "生産性"
        escaped_x = label_x.encode("unicode_escape").decode("ascii")
        escaped_y = label_y.encode("unicode_escape").decode("ascii")
        assert label_x in html_content or escaped_x in html_content
        assert label_y in html_content or escaped_y in html_content
