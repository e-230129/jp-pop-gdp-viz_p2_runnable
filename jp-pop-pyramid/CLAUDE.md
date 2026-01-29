# Claude Code Operating Constitution

## 0) Read-first（SSOT索引）
- 索引: @SSOT.md
- 現在の計画: @plan.md（1行目がパス。参照先のPlan本体も読むこと）
- タスクボード: @TASKS.md
- セッション状態: @progress.md

## 1) 絶対ルール（安全・衛生）
- 秘密情報（.env, credentials）を読み取らない、外部に出さない
- 差分は最小に。明示的な指示なしに大規模リファクタしない
- セキュリティルールを回避して「動かす」ことはしない
- ファイル削除・上書きは明示的な承認を得てから

## 2) ワークフロー（Plan → Do → Verify → Record）
- 非自明な変更: `/plan` → 人間承認 → 実装
- コード編集後: scripts/で検証（test, lint）
- テスト失敗: 3回まで修正試行。それでも失敗なら停止して報告
- セッション終了時: progress.md を更新

## 3) 書く場所（SSOT規律）
- 要件/振る舞い: docs/requirements/
- API/DB契約: docs/contracts/
- アーキテクチャ決定: docs/adr/
- 承認済み計画: docs/plans/
- タスクボード: TASKS.md
- セッションログ: docs/audits/

## 4) プロジェクト固有ルール

### ディレクトリ構成
- `src/` - Pythonソースコード
- `data/` - 入力データ（IPSS CSVなど）
- `out/` - 出力ファイル（PNG, PDF, HTML, GIF）

### 実行コマンド
- メモ生成: `python src/make_memo.py --input data/ipss_single_year.csv --out out/memo.png --mode memo`
- インタラクティブ: `python src/make_memo.py --input data/ipss_single_year.csv --out out/pyramid.html --mode interactive`
- アニメーション: `python src/make_memo.py --input data/ipss_single_year.csv --out out/pyramid.gif --mode animation`

### データソース
- IPSS（国立社会保障・人口問題研究所）将来人口推計
- Table 1-9: 各歳人口（2020-2070、51年分）
- Table 1-9A: 5歳階級人口（5年刻み）

## 5) ADRの閾値
**ADR必須:**
- 外部サービス/ライブラリ導入・変更
- 出力フォーマットの大幅変更
- 計算ロジックの変更（ピーク年齢の計算方法など）

**ADR不要:**
- 明らかなバグ修正
- UIの微調整

## 6) コマンド（優先）
- テスト: `python -m pytest tests/`
- SSOT検証: `python scripts/validate-ssot.py`
- メモ生成テスト: `python src/make_memo.py --help`

## 7) セッションライフサイクル
- 開始: `/kickoff`（SSOT検証→Plan本体読込→状態要約）
- 迷子: `/reset`（SSOTに再アンカー）
- 終了: `/update-progress`（progress.md更新）

## 8) 学習ルール（同じミスを繰り返さない）
- 5歳階級データの100+は100〜max_ageに配分（95-99と重ならないよう注意）
- ピーク年齢の計算は初年の5年ローリング最大からコホート追跡
- Plotlyのframesはリストで初期化してから代入
