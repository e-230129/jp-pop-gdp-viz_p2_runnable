# コードレビューログ

**日時:** 2026-01-27
**レビュアー:** Claude Opus 4.5
**目的:** GPT-5.2 Pro向け事前レビュー

---

## 1. プロジェクト概要

**目的:** 日本の人口動態・GDP・REERを可視化する統合ダッシュボード + 人口ピラミッドツール

**主要ファイル:**

| ファイル | 行数 | 役割 |
|----------|------|------|
| `src/pop_gdp_reer_dashboard.py` | 813 | メインダッシュボード |
| `src/build_macro_data.py` | 221 | CSV読み込みユーティリティ |
| `jp-pop-pyramid/src/make_memo.py` | 966 | 人口ピラミッド生成 |
| `jp-pop-pyramid/src/interactive_enhanced.py` | 1246 | 拡張インタラクティブ版 |
| `tests/test_dashboard_playwright.py` | 510 | Playwrightテスト |
| `scripts/pack_for_review.py` | 181 | レビュー用パッケージング |

**総コード行数:** 約3,900行

---

## 2. 良い点（Strengths）

### 2.1 コード品質
- **docstring**: 各関数に日本語docstringが記述されており、役割が明確
- **型ヒント**: `from __future__ import annotations`を使用し、主要関数に型ヒントあり
- **定数の分離**: `DATA_MACRO`, `DATA_FUTURE`, `EVENTS`, `colors`が明確に定義されている

### 2.2 データ処理
- **pandas→numpy変換**: `stackplot()`使用時にnumpy配列へ変換（Phase 1対応済み）
- **動的計算**: REERピーク値、軸範囲がデータから動的に計算されている（Phase 2-3対応済み）
- **副作用の遮断**: `create_static_summary()`で`df.copy()`を使用

### 2.3 可視化
- **カーソル線**: `yref="paper"`で軸レンジ非依存の実装
- **カラースキーム**: 一貫した配色（young=緑、working=青、elderly=オレンジ）
- **データソース明示**: HTMLタイトルとPNG注釈にデータ出所を記載

### 2.4 ドキュメント
- **SSOT.md**: Single Source of Truth索引が整備
- **CLAUDE.md**: AI運用ルールが明確に文書化
- **ADR**: アーキテクチャ決定記録が整備
- **progress.md**: セッション状態の継承が可能

---

## 3. 問題点・改善提案

### 3.1 【重要】データの二重管理

**問題箇所:**
- `pop_gdp_reer_dashboard.py` L35-81: `DATA_MACRO`, `DATA_FUTURE`
- `build_macro_data.py` L22-73: `GDP_REER_DATA`

**問題点:** 同じGDP/REERデータが2箇所で定義されており、更新漏れのリスクがある

**推奨修正:**
```python
# build_macro_data.py を唯一のデータソースとし、
# pop_gdp_reer_dashboard.py はそれをインポートするだけにする

# pop_gdp_reer_dashboard.py
from build_macro_data import GDP_REER_DATA  # 統一

# DATA_MACRO, DATA_FUTURE は削除し、build_dataframe() を廃止
```

---

### 3.2 【中】エラーハンドリング不足

**問題箇所:** `build_macro_data.py` L76-98

```python
def load_population_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # ファイル不在時の例外なし
    # ...
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")  # OK
```

**推奨修正:**
```python
def load_population_from_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Population CSV not found: {csv_path}")
    # ...
```

---

### 3.3 【中】パフォーマンス：フレームでの冗長データ

**問題箇所:** `pop_gdp_reer_dashboard.py` L295-420

各フレームで全期間線（GDP、REER、年齢構成比）を毎回作成している：

```python
for i, year in enumerate(years):
    # ...
    # 2. GDP全期間線（変更なし、同じデータ）
    frame_data.append(go.Scatter(
        x=df["Year"],
        y=df["GDP"],
        # ... 毎フレーム同じデータを追加
    ))
```

**影響:** 36フレーム×8トレース = 288回のオブジェクト生成（軽微だが最適化可能）

**推奨修正:**
Plotlyのトレース更新は`data`に含めなければ前のフレームを維持するため、変化しないトレースはフレームから除外可能。ただし現状でも動作は正常なので優先度は低い。

---

### 3.4 【低】型ヒント・docstringの拡充

**問題箇所:** `pop_gdp_reer_dashboard.py` の内部関数

**推奨:** 戻り値の詳細（DataFrame列名など）をdocstringに追加

```python
def build_dataframe() -> pd.DataFrame:
    """
    マクロデータと将来推計を統合したDataFrameを作成

    Returns:
        DataFrame with columns:
        - Year, TotalPop, Working, Young, Elderly (万人)
        - GDP (兆円), REER
        - Working_Ratio, Young_Ratio, Elderly_Ratio (%)
        - GDP_per_capita (万円/人)
    """
```

---

### 3.5 【低】マジックナンバーの残存

**問題箇所:** `pop_gdp_reer_dashboard.py`

```python
L132: gdp_max = float(df["GDP"].max()) * 1.1  # 10%マージン → 定数化推奨
L133: reer_max = float(df_reer_all["REER"].max()) * 1.2  # 20%マージン
L539: range=[0, 10000]  # 人口バーの固定範囲
L545: range=[0, 105]    # 構成比の固定範囲
```

**推奨修正:**
```python
# 定数として定義
AXIS_MARGIN_GDP = 1.1
AXIS_MARGIN_REER = 1.2
POPULATION_BAR_MAX = 10000
COMPOSITION_MAX = 105
```

---

### 3.6 【低】未使用変数の警告

**問題箇所:** `pop_gdp_reer_dashboard.py` L424-436

```python
for event_year, (event_name, event_color) in EVENTS.items():
    # event_name は使用されていない（イベント線のみ追加、凡例なし）
```

---

## 4. セキュリティ

**問題なし:**
- ファイルI/Oは指定パスのみ
- 外部ネットワークアクセスなし
- ユーザー入力のサニタイズは不要（CLI引数のみ）

---

## 5. テストカバレッジ

**現状:**
- `tests/test_dashboard_playwright.py`: Playwrightを使用したE2Eテスト（HTML出力の確認）

**テスト項目:**
- 基本読み込みテスト（Plotlyグラフ、スライダー、再生ボタン存在確認）
- データ範囲テスト（1950-2100年）
- スライダーナビゲーションテスト
- カーソル線テスト（yref=paper確認）
- GDPマーカーテスト
- アニメーション再生テスト
- 軸範囲テスト
- サブプロット構造テスト

**推奨追加:**
1. **ユニットテスト:** `build_dataframe()`, `load_population_from_csv()`の単体テスト
2. **PNG生成テスト:** `create_static_summary()`の出力確認
3. **エッジケース:**
   - 空のCSV
   - REERがすべてNaNの場合
   - 1年分のみのデータ

---

## 6. 構造改善提案

### 6.1 ファイル構成の整理案

```
src/
├── data/
│   └── macro_data.py          # GDP_REER_DATA を統一管理
├── dashboard/
│   ├── __init__.py
│   ├── builder.py             # create_dashboard, create_static_summary
│   └── config.py              # 色、定数
├── utils/
│   ├── __init__.py
│   └── csv_loader.py          # load_population_from_csv
└── main.py                    # CLI エントリポイント
```

### 6.2 設定のクラス化案

```python
@dataclass
class DashboardConfig:
    x_range_margin: float = 5
    gdp_margin: float = 1.1
    reer_margin: float = 1.2
    colors: dict = field(default_factory=lambda: {
        "young": "#4CAF50",
        "working": "#2196F3",
        "elderly": "#FF5722",
        "gdp": "#E91E63",
        "reer": "#9C27B0",
        "cursor": "#FF0000",
    })
```

---

## 7. jp-pop-pyramid サブプロジェクトのレビュー

### 7.1 良い点
- **CLAUDE.md / SSOT.md**: AI運用ルールが明確に文書化
- **ADR**: アーキテクチャ決定記録が整備
- **コホート追跡**: 世代帯表示が柔軟（単一年/範囲年の両対応）
- **拡張機能**: ユーザー年齢入力、異常検出、構成比表示

### 7.2 問題点

**`make_memo.py` L189-238 の100+配分ロジック:**
```python
elif repr_age >= 100:
    # 100+: 100〜max_age に配分（修正）
    age_start = 100
    age_end = max_age
```
ロジックは正しいが、コメントのみで単体テストがない。

**推奨:** 境界値テストを追加
```python
def test_100plus_distribution():
    # 100+ は 100-105 に均等配分されることを確認
```

### 7.3 interactive_enhanced.py の特筆点

- **DataProcessorクラス**: 処理と表示を分離する良い設計
- **PyramidConfigクラス**: 設定のデータクラス化
- **異常検出機能**: 前年比15%超の変化を自動検出
- **拡張JS注入**: ホイール/キーボード操作、スクリーンショット機能

---

## 8. 総合評価

| 観点 | 評価 | コメント |
|------|------|----------|
| コード品質 | ★★★★☆ | docstring・型ヒントは良好、一部定数化不足 |
| データ処理 | ★★★★☆ | pandas/numpy変換は適切、二重管理が課題 |
| 可視化 | ★★★★★ | Plotlyアニメーション実装が適切 |
| パフォーマンス | ★★★☆☆ | フレーム冗長データあり（影響は軽微） |
| テスト | ★★★☆☆ | E2Eはあるがユニットテスト不足 |
| ドキュメント | ★★★★★ | SSOT/ADR/READMEが充実 |
| セキュリティ | ★★★★★ | 問題なし |

**総合:** ★★★★☆ (4.0/5.0)

---

## 9. 優先順位付き修正リスト

| 優先度 | 項目 | 影響 | 工数 |
|--------|------|------|------|
| P0 | データ二重管理の統一 | 保守性 | 中 |
| P1 | FileNotFoundError追加 | 堅牢性 | 小 |
| P1 | ユニットテスト追加 | 品質保証 | 中 |
| P2 | 定数のクラス化 | 可読性 | 小 |
| P2 | フレーム最適化 | パフォーマンス | 中 |
| P3 | 未使用変数の削除 | 警告解消 | 小 |

---

## 10. 次のアクション

1. **GPT-5.2 Pro レビュー依頼**: 本ログと `INSTRUCTIONS_GPT52_FINAL_REVIEW.md` を渡す
2. **P0修正**: データ二重管理の統一（レビュー後）
3. **テスト追加**: ユニットテスト作成（レビュー後）

---

## 付録: ファイル一覧（レビュー対象）

```
260127jp-pop-gdp-viz/
├── src/
│   ├── pop_gdp_reer_dashboard.py   ★ メインダッシュボード
│   └── build_macro_data.py         ★ CSV読み込みユーティリティ
├── jp-pop-pyramid/
│   ├── src/
│   │   ├── make_memo.py            ★ 人口ピラミッド生成
│   │   ├── interactive_enhanced.py ★ 拡張インタラクティブ版
│   │   ├── convert_ipss.py         　IPSS変換
│   │   ├── interpolate_years.py    　年補間
│   │   └── verification.py         　検証ツール
│   ├── CLAUDE.md                   　AI運用ルール
│   ├── SSOT.md                     　索引
│   ├── TASKS.md                    　タスクボード
│   └── progress.md                 　セッション状態
├── tests/
│   └── test_dashboard_playwright.py ★ Playwrightテスト
├── scripts/
│   └── pack_for_review.py          ★ パッケージングスクリプト
├── INSTRUCTIONS_GPT52_FINAL_REVIEW.md  GPT-5.2向け指示書
└── INSTRUCTIONS_CODEX_REVIEW.md    Codex向け指示書
```

---

**レビュー完了: 2026-01-27**
