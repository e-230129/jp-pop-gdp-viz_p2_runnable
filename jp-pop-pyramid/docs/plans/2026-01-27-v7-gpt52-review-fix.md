# Plan: v7 GPT-5.2 Proレビュー対応（バグ修正・品質改善）

Date: 2026-01-27
Owner: e-230129
Status: Draft（要承認）

## Goal

GPT-5.2 Proのコードレビューに基づき、`pop_gdp_reer_dashboard.py` のバグ修正と品質改善を行う。

## Non-Goals

- GDP/REERの将来シナリオ化（Phase 4.3以降で検討）
- 人口データの完全自動取得（手動CSVは継続）
- 多言語対応

## Constraints

- Security: なし（ローカル実行のみ）
- Performance: 静的PNG生成が30秒以内
- Compatibility: Python 3.10+, matplotlib 3.7+, plotly 5.x
- Deadline: 2026-01-31

## Background

GPT-5.2 Proによる2026-01-27レビューで以下が指摘された:

1. **致命的バグ**: `fill_between()` が pandas.Series を受け取ると `TypeError` でクラッシュ
2. **品質改善**: ピーク値のハードコード、データソース不明瞭、インタラクティブUXの改善余地

## Proposed Design

### Phase 1: 必須バグ修正（P0）

| 変更点 | 修正内容 |
|--------|----------|
| fill_between クラッシュ | pandas.Series → numpy配列化、stackplot統一 |

### Phase 2: コード品質改善（P1）

| 変更点 | 修正内容 |
|--------|----------|
| ハードコード撤去 | REERピーク値、基準年を動的取得 |
| データ透明性 | PNG注釈・HTMLタイトルにソース・定義明記 |
| 境界線統一 | 2024年実績/推計境界を全パネルで統一表示 |

### Phase 3: UX改善（P2、オプション）

| 変更点 | 修正内容 |
|--------|----------|
| インタラクティブ表示 | 「線が伸びる」→「全期間+縦線カーソル」 |
| 追加指標 | 生産年齢人口1人あたりGDP |

## File Changes

| ファイル | 変更種別 | 説明 |
|----------|----------|------|
| src/pop_gdp_reer_dashboard.py | Modify | バグ修正・品質改善 |
| tests/test_dashboard.py | Add | PNG生成のユニットテスト |

## Test Plan

- [ ] Unit tests: PNG生成が正常終了すること
- [ ] Integration tests: HTML/PNG両方が生成されること
- [ ] Visual check: 全サブプロットの描画確認
- [ ] Regression: 既存のHTMLインタラクティブ機能が壊れていないこと

## Risks & Mitigations

| リスク | 可能性 | 影響 | 対策 |
|--------|--------|------|------|
| matplotlib互換性 | 低 | PNG壊れ | numpy配列化で標準的なAPIを使用 |
| 動的ピーク取得の誤検出 | 低 | 表示ミス | データに複数ピークがないか確認 |

## Checklist

- [ ] 人間が承認
- [ ] Phase 1 実装完了（バグ修正）
- [ ] Phase 2 実装完了（品質改善）
- [ ] テストパス
- [ ] ドキュメント更新（TASKS.md, progress.md）
- [ ] ADR作成（該当なし - バグ修正のため）
