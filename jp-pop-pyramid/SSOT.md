# SSOT (Single Source of Truth) Index

日本人口ピラミッド自動生成ツールの「真実の置き場」一覧。

## Project Overview

IPSS（国立社会保障・人口問題研究所）の将来人口推計データを使用して、
日本の人口ピラミッドを可視化するツール。

### 主な機能
- 静止画メモ（PNG/PDF）: 複数年を1枚に配置
- インタラクティブ版（HTML）: スライダー/ホイール/キーボードで操作
- アニメーション（GIF/MP4）: 年次推移を動画化
- レキシス面（PNG）: 年×年齢のヒートマップ
- **[NEW]** 統合ダッシュボード（HTML/PNG）: 人口×GDP×REER連動表示

## Requirements（何を満たす）

- [Requirements](docs/requirements/README.md)

## Decisions（なぜそうした：ADR）

- [ADR Index](docs/adr/README.md)

## Plan（どう作る：承認済み）

- [Plans folder](docs/plans/)
- [Current plan pointer](plan.md)

## Work State（いま何してる）

- [Task board](TASKS.md)
- [AI session state](progress.md)

## Source Code

- [Main script](src/make_memo.py) - 人口ピラミッド生成スクリプト
- [Data converter](src/convert_ipss.py) - IPSS Excel→CSV変換
- [Enhanced interactive](src/interactive_enhanced.py) - 拡張インタラクティブ版
- **[NEW]** [Dashboard](src/pop_gdp_reer_dashboard.py) - 人口×GDP×REER統合ダッシュボード

## Data

- [Input data](data/) - IPSS CSVファイル
- [Output](out/) - 生成された画像/HTML

## Commands（実行方法のSSOT）

- [SSOT validation](scripts/validate-ssot.py)

### ダッシュボード生成

```bash
# HTML + PNG 両方生成
python -X utf8 src/pop_gdp_reer_dashboard.py \
  --out out/pop_gdp_reer_dashboard.html \
  --png out/pop_gdp_reer_summary.png
```

---

**検証方法:**
```bash
python scripts/validate-ssot.py
```

**注意:** 存在しないものはリンクにしない。
将来追加したい項目は箇条書き（リンクなし）でメモ。
