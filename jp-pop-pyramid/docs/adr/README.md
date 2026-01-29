# Architecture Decision Records (ADR)

重要な決定の記録。「なぜそうしたか」を未来の自分とチームに残す。

## ADR一覧

| # | Title | Status | Date |
|---|-------|--------|------|
| 0000 | [Template](0000-template.md) | Template | - |
| 0001 | [ピーク年齢の計算方法](0001-peak-calculation.md) | Accepted | 2025-01-25 |

## 書く閾値

**ADR必須:**
- 外部サービス/ライブラリ導入・変更
- 出力フォーマットの大幅変更
- 計算ロジックの変更

**ADR不要:**
- 明らかなバグ修正
- UIの微調整

## 命名規則

`NNNN-short-title.md`
- NNNN: 0001から連番
- short-title: ケバブケース

## ステータス

- `Proposed` - 提案中
- `Accepted` - 承認済み
- `Deprecated` - 非推奨
- `Superseded by ADR-XXXX` - 後続ADRで置換

## 関連

- [SSOT Index](../../SSOT.md)
- [Plans](../plans/)
