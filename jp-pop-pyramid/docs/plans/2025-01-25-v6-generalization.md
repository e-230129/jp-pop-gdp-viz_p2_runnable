# Plan: v6 一般化（年選択・コホート追跡・ホイール操作）

Date: 2025-01-25
Owner: e-230129
Status: Completed

## Goal

GPT-5.2 Proのアドバイスに基づき、人口ピラミッドツールを一般化する。

## Non-Goals

- データ取得の自動化（手動でIPSSからダウンロード）
- 多言語対応

## Constraints

- Security: なし（ローカル実行のみ）
- Performance: 51年分のアニメーションが10秒以内に生成できること
- Compatibility: Python 3.10+, matplotlib, plotly
- Deadline: 2025-01-25

## Proposed Design

### 概要

v5からの改善点:
1. 100+の配分修正（100〜max_ageに配分）
2. 年の選択を一般化（start/end/step + 明示指定）
3. 出生年コホート追跡（--cohort-birth-years）
4. ホイール/キーボードで年送り（interactive HTML）
5. メモのレイアウト一般化（--memo-layout grid）

### コンポーネント

- `select_years()`: 年の選択を一般化
- `cohort_age()`: 出生年から年齢を計算
- `add_cohort_lines_matplotlib()`: Matplotlib用コホート線描画
- `inject_wheel_scrub()`: Plotly HTMLにJS埋め込み

### データフロー

1. CSVからデータ読み込み
2. 年の選択（start/end/step or explicit）
3. ピーク年齢の事前計算（初年からコホート追跡）
4. 各年のピラミッド描画（コホート線付き）
5. 出力（PNG/PDF/HTML/GIF）

### エラーハンドリング

- 存在しない年が指定された場合: 警告を出して利用可能な年のみ使用
- コホート年齢が範囲外の場合: 非表示

## File Changes

| ファイル | 変更種別 | 説明 |
|----------|----------|------|
| src/make_memo.py | Add | v6メインスクリプト |
| src/convert_ipss.py | Add | IPSS Excel→CSV変換 |

## Test Plan

- [x] Unit tests: 年選択関数のテスト
- [x] Integration tests: 全モード（memo/animation/interactive/lexis）の生成テスト
- [x] Negative cases: 存在しない年の指定、空のコホート指定

## Risks & Mitigations

| リスク | 可能性 | 影響 | 対策 |
|--------|--------|------|------|
| Plotlyが重い | 中 | UX低下 | MP4版を推奨 |
| ホイール操作がモバイルで動かない | 高 | モバイルで使えない | タッチ操作は別途検討 |

## Checklist

- [x] 人間が承認
- [x] 実装完了
- [x] テストパス
- [x] ドキュメント更新
- [ ] ADR作成（該当なし）
