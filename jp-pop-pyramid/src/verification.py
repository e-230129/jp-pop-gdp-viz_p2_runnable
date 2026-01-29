"""
verification.py - グラフ検証モジュール

データと描画の整合性を検証するユーティリティ。
描画前・描画後のデータを比較し、論理的一貫性を担保する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class VerificationResult:
    """検証結果を格納するデータクラス"""
    year: int
    check_name: str
    passed: bool
    expected: float | None = None
    actual: float | None = None
    difference: float | None = None
    message: str = ""


@dataclass
class VerificationReport:
    """検証レポート全体"""
    results: list[VerificationResult] = field(default_factory=list)

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": self.failed_count,
                "all_passed": self.all_passed,
            },
            "results": [
                {
                    "year": r.year,
                    "check": r.check_name,
                    "passed": r.passed,
                    "expected": r.expected,
                    "actual": r.actual,
                    "difference": r.difference,
                    "message": r.message,
                }
                for r in self.results
            ]
        }

    def to_json(self, path: Path) -> None:
        """JSON形式でレポートを保存"""
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def print_summary(self) -> None:
        """コンソールにサマリーを表示"""
        print("\n" + "=" * 60)
        print("検証結果サマリー")
        print("=" * 60)
        print(f"総チェック数: {len(self.results)}")
        print(f"成功: {sum(1 for r in self.results if r.passed)}")
        print(f"失敗: {self.failed_count}")

        if self.failed_count > 0:
            print("\n失敗した検証:")
            for r in self.results:
                if not r.passed:
                    print(f"  - [{r.year}] {r.check_name}: {r.message}")
        else:
            print("\n[OK] すべての検証に合格しました")
        print("=" * 60)


class DataVerifier:
    """
    データ検証クラス

    元データとグラフ描画用に変換されたデータの整合性を検証する。
    """

    def __init__(self, df: pd.DataFrame, max_age: int = 105):
        self.df = df
        self.max_age = max_age
        self.report = VerificationReport()

    def verify_population_total(self, year: int,
                                 expanded_m: np.ndarray,
                                 expanded_f: np.ndarray,
                                 tolerance: float = 1.0) -> VerificationResult:
        """
        総人口の一致を検証

        Args:
            year: 検証対象年
            expanded_m: 展開後の男性人口配列
            expanded_f: 展開後の女性人口配列
            tolerance: 許容誤差（人）
        """
        df_year = self.df[self.df["year"] == year]
        expected_total = df_year["pop"].sum()
        actual_total = expanded_m.sum() + expanded_f.sum()
        difference = abs(expected_total - actual_total)

        passed = difference <= tolerance

        result = VerificationResult(
            year=year,
            check_name="総人口一致",
            passed=passed,
            expected=expected_total,
            actual=actual_total,
            difference=difference,
            message="" if passed else f"総人口が{difference:.0f}人ずれています"
        )
        self.report.add(result)
        return result

    def verify_sex_ratio(self, year: int,
                          expanded_m: np.ndarray,
                          expanded_f: np.ndarray,
                          tolerance: float = 0.001) -> VerificationResult:
        """
        男女比の一致を検証
        """
        df_year = self.df[self.df["year"] == year]

        expected_m = df_year[df_year["sex"] == "M"]["pop"].sum()
        expected_f = df_year[df_year["sex"] == "F"]["pop"].sum()
        expected_ratio = expected_m / expected_f if expected_f > 0 else 0

        actual_m = expanded_m.sum()
        actual_f = expanded_f.sum()
        actual_ratio = actual_m / actual_f if actual_f > 0 else 0

        difference = abs(expected_ratio - actual_ratio)
        passed = difference <= tolerance

        result = VerificationResult(
            year=year,
            check_name="男女比一致",
            passed=passed,
            expected=expected_ratio,
            actual=actual_ratio,
            difference=difference,
            message="" if passed else f"男女比が{difference:.4f}ずれています"
        )
        self.report.add(result)
        return result

    def verify_age_monotonicity(self, year: int,
                                 expanded_m: np.ndarray,
                                 expanded_f: np.ndarray,
                                 age_threshold: int = 85) -> VerificationResult:
        """
        高齢部分の人口減少傾向を検証

        通常、高齢になるほど人口は減少するはず。
        急激な増加があれば配分ロジックの問題の可能性。
        """
        total = expanded_m + expanded_f
        high_age_pop = total[age_threshold:]

        # 3歳以上連続で増加していないかチェック
        increasing_streak = 0
        max_streak = 0

        for i in range(1, len(high_age_pop)):
            if high_age_pop[i] > high_age_pop[i-1] * 1.1:  # 10%以上の増加
                increasing_streak += 1
                max_streak = max(max_streak, increasing_streak)
            else:
                increasing_streak = 0

        passed = max_streak < 3

        result = VerificationResult(
            year=year,
            check_name="高齢人口減少傾向",
            passed=passed,
            expected=0,
            actual=float(max_streak),
            message="" if passed else f"{age_threshold}歳以上で{max_streak}連続の人口増加があります"
        )
        self.report.add(result)
        return result

    def verify_no_negative(self, year: int,
                            expanded_m: np.ndarray,
                            expanded_f: np.ndarray) -> VerificationResult:
        """
        負の人口値がないことを検証
        """
        has_negative = (expanded_m < 0).any() or (expanded_f < 0).any()

        result = VerificationResult(
            year=year,
            check_name="非負チェック",
            passed=not has_negative,
            message="" if not has_negative else "負の人口値が存在します"
        )
        self.report.add(result)
        return result

    def verify_age_group_totals(self, year: int,
                                 expanded_m: np.ndarray,
                                 expanded_f: np.ndarray,
                                 tolerance_ratio: float = 0.01) -> VerificationResult:
        """
        年齢区分ごとの総人口を検証

        年少人口(0-14)、生産年齢人口(15-64)、老年人口(65+)の
        各区分の合計が正しいか確認。
        """
        df_year = self.df[self.df["year"] == year]
        total = expanded_m + expanded_f

        # 年齢区分の定義
        groups = [
            ("年少人口(0-14)", 0, 14),
            ("生産年齢人口(15-64)", 15, 64),
            ("老年人口(65+)", 65, self.max_age),
        ]

        errors = []

        for name, start, end in groups:
            # 元データからの計算
            mask = (df_year["age"] >= start) & (df_year["age"] <= end)
            expected = df_year[mask]["pop"].sum()

            # 展開後データからの計算
            actual = total[start:min(end+1, len(total))].sum()

            if expected > 0:
                ratio_diff = abs(expected - actual) / expected
                if ratio_diff > tolerance_ratio:
                    errors.append(f"{name}: {ratio_diff:.2%}の誤差")

        passed = len(errors) == 0

        result = VerificationResult(
            year=year,
            check_name="年齢区分別人口",
            passed=passed,
            message="" if passed else "; ".join(errors)
        )
        self.report.add(result)
        return result

    def run_all_verifications(self, year: int,
                               expanded_m: np.ndarray,
                               expanded_f: np.ndarray) -> VerificationReport:
        """
        すべての検証を実行
        """
        self.verify_population_total(year, expanded_m, expanded_f)
        self.verify_sex_ratio(year, expanded_m, expanded_f)
        self.verify_age_monotonicity(year, expanded_m, expanded_f)
        self.verify_no_negative(year, expanded_m, expanded_f)
        self.verify_age_group_totals(year, expanded_m, expanded_f)

        return self.report


class YearTransitionVerifier:
    """
    年度間の整合性を検証

    年度切り替え時にデータの連続性が保たれているかを確認。
    """

    def __init__(self, df: pd.DataFrame, max_age: int = 105):
        self.df = df
        self.max_age = max_age
        self.report = VerificationReport()

    def verify_cohort_continuity(self, year1: int, year2: int,
                                  m1: np.ndarray, f1: np.ndarray,
                                  m2: np.ndarray, f2: np.ndarray,
                                  tolerance_ratio: float = 0.15) -> VerificationResult:
        """
        コホートの連続性を検証

        year1の年齢aの人口が、year2の年齢a+(year2-year1)と
        大きく乖離していないかを確認。
        （死亡や移民を考慮して許容幅を設ける）
        """
        year_diff = year2 - year1
        total1 = m1 + f1
        total2 = m2 + f2

        errors = []

        for age in range(0, self.max_age - year_diff):
            pop1 = total1[age]
            pop2 = total2[age + year_diff]

            # 人口が減少しているはず（死亡を考慮）
            # ただし、年齢が若いと増加もありうる（移民等）
            if pop1 > 0:
                change_ratio = (pop2 - pop1) / pop1

                # 大幅な増加は異常
                if change_ratio > tolerance_ratio and pop1 > 10000:
                    errors.append(f"年齢{age}→{age+year_diff}: {change_ratio:.1%}増加")

        passed = len(errors) <= 3  # 多少の異常は許容

        result = VerificationResult(
            year=year2,
            check_name=f"コホート連続性({year1}→{year2})",
            passed=passed,
            message="" if passed else f"{len(errors)}件の異常: " + "; ".join(errors[:3])
        )
        self.report.add(result)
        return result

    def verify_total_trend(self, years: list[int],
                           totals: list[float]) -> VerificationResult:
        """
        総人口の推移傾向を検証

        急激な変動がないかを確認。
        """
        errors = []

        for i in range(1, len(years)):
            prev_total = totals[i-1]
            curr_total = totals[i]
            year_diff = years[i] - years[i-1]

            if prev_total > 0:
                annual_change = (curr_total - prev_total) / prev_total / year_diff

                # 年間5%以上の変動は異常
                if abs(annual_change) > 0.05:
                    errors.append(
                        f"{years[i-1]}→{years[i]}: 年率{annual_change:.1%}"
                    )

        passed = len(errors) == 0

        result = VerificationResult(
            year=years[-1],
            check_name="総人口推移トレンド",
            passed=passed,
            message="" if passed else "; ".join(errors)
        )
        self.report.add(result)
        return result


def create_verification_overlay_js(verifier: DataVerifier) -> str:
    """
    検証結果をHTMLオーバーレイとして表示するJSを生成
    """
    report = verifier.report.to_dict()

    js = f"""
<script>
(function() {{
    const verificationData = {json.dumps(report, ensure_ascii=False)};

    // 検証パネルを作成
    const panel = document.createElement('div');
    panel.id = 'verification-panel';
    panel.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(255,255,255,0.95);
        border: 2px solid #333;
        border-radius: 8px;
        padding: 15px;
        max-width: 300px;
        max-height: 400px;
        overflow-y: auto;
        font-family: sans-serif;
        font-size: 12px;
        z-index: 10000;
        display: none;
    `;

    // ヘッダー
    const header = document.createElement('div');
    header.innerHTML = `
        <h3 style="margin:0 0 10px 0;">検証モード</h3>
        <p><b>総チェック:</b> ${{verificationData.summary.total_checks}}</p>
        <p style="color:green;"><b>成功:</b> ${{verificationData.summary.passed}}</p>
        <p style="color:${{verificationData.summary.failed > 0 ? 'red' : 'green'}};">
            <b>失敗:</b> ${{verificationData.summary.failed}}
        </p>
    `;
    panel.appendChild(header);

    // 詳細リスト
    if (verificationData.summary.failed > 0) {{
        const list = document.createElement('ul');
        list.style.cssText = 'margin: 10px 0; padding-left: 20px;';
        verificationData.results.filter(r => !r.passed).forEach(r => {{
            const li = document.createElement('li');
            li.style.color = 'red';
            li.textContent = `[${{r.year}}] ${{r.check}}: ${{r.message}}`;
            list.appendChild(li);
        }});
        panel.appendChild(list);
    }}

    document.body.appendChild(panel);

    // トグルボタン
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = '🔍 検証';
    toggleBtn.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        cursor: pointer;
        font-size: 14px;
        z-index: 10001;
    `;
    toggleBtn.onclick = () => {{
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        toggleBtn.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }};
    document.body.appendChild(toggleBtn);

    // 閉じるボタン
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        position: absolute;
        top: 5px;
        right: 5px;
        background: none;
        border: none;
        font-size: 16px;
        cursor: pointer;
    `;
    closeBtn.onclick = () => {{
        panel.style.display = 'none';
        toggleBtn.style.display = 'block';
    }};
    panel.insertBefore(closeBtn, panel.firstChild);
}})();
</script>
"""
    return js


if __name__ == "__main__":
    # テスト用
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "ipss_combined_1965_2070.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        verifier = DataVerifier(df, max_age=105)

        # サンプル検証（実際の展開データが必要）
        print("検証モジュールのテスト準備完了")
        print(f"データ年範囲: {df['year'].min()} - {df['year'].max()}")
