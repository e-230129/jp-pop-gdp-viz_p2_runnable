# TASKS

「次やること」の唯一の置き場。progress.mdには書かない。

## Now（作業中）

- [ ] **[P0-BUG]** PNG生成クラッシュ修正（pandas.Series→numpy配列化）
  - 対象: `src/pop_gdp_reer_dashboard.py` L466-468, L493-495, L517
  - 参照: `INSTRUCTIONS_GPT52_REVIEW_FIX.md` Phase 1

## Next（次にやる）

- [ ] **[P1-QUALITY]** ハードコード撤去（REERピーク値・基準年の動的取得）
  - 対象: L520-527, L564
  - 参照: `INSTRUCTIONS_GPT52_REVIEW_FIX.md` Phase 2

- [ ] **[P1-QUALITY]** データソース・定義のUI明示
  - 静的PNG注釈の拡充
  - HTMLタイトルへの出所サブタイトル追加
  - 参照: `INSTRUCTIONS_GPT52_REVIEW_FIX.md` Phase 3

- [ ] **[P2-FEATURE]** インタラクティブ表示を「全期間+カーソル」方式に変更
  - 参照: `INSTRUCTIONS_GPT52_REVIEW_FIX.md` Phase 4.1

- [ ] **[P2-FEATURE]** 人口3区分をピラミッドCSVから自動集計
  - DATA_MACRO/DATA_FUTUREのハードコード廃止
  - 参照: `INSTRUCTIONS_GPT52_REVIEW_FIX.md` Phase 4.2

- [ ] テストコードの追加（静的PNG生成のユニットテスト含む）

- [ ] GitHub公開準備

## Blocked（ブロック中）

（なし）

## Done（最近完了）

- [x] GPT-5.2 Pro 第1回レビュー受領（2026-01-27）
- [x] v6実装: 年選択の一般化、コホート追跡、ホイール/キー操作
- [x] v5実装: ピーク年齢の5年ローリング計算、Plotlyインタラクティブ版
- [x] v4実装: 各歳データ対応、アニメーション、レキシス面
- [x] v3実装: 5歳階級データ対応、グラデーション色
- [x] IPSS Table 1-9 データ変換（各歳×51年）
- [x] IPSS Table 1-9A データ変換（5歳階級×5年刻み）

---

**運用ルール:**
- タスク追加時は Plan または ADR へのリンクを添える
- 完了時は Done に移動し、PR/コミットリンクを追記
- Blocked は理由を必ず書く
- 優先度タグ: [P0-BUG] > [P1-QUALITY] > [P2-FEATURE]
