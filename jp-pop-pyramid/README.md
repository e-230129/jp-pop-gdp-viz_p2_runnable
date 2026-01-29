# jp-pop-pyramid

日本の人口ピラミッド自動生成ツール

## 概要

IPSS（国立社会保障・人口問題研究所）の将来人口推計データを使用して、
日本の人口ピラミッドを可視化するツールです。

## 用語メモ

- 以前の資料で使っていた「鋭」という表現は、本リポジトリでは「ピーク年齢」に統一しています。

## 機能

- **静止画メモ（memo）**: 複数年を1枚に配置（PNG/PDF）
- **インタラクティブ版（interactive）**: スライダー/ホイール/キーボードで操作（HTML）
- **アニメーション（animation）**: 年次推移を動画化（GIF/MP4）
- **レキシス面（lexis）**: 年×年齢のヒートマップ（PNG）

## インストール

```bash
pip install matplotlib pandas numpy plotly
```

## 使用方法

### 基本的な使い方

```bash
# 静止画メモ（5年分）
python src/make_memo.py --input data/ipss_single_year.csv --out out/memo.png \
  --mode memo --years 2020,2030,2040,2050,2060

# インタラクティブ版（毎年、コホート線付き）
python src/make_memo.py --input data/ipss_single_year.csv --out out/pyramid.html \
  --mode interactive --year-step 1 --cohort-birth-years 1947,1971

# アニメーション（5年ごと）
python src/make_memo.py --input data/ipss_single_year.csv --out out/pyramid.gif \
  --mode animation --year-step 5 --fps 2

# レキシス面
python src/make_memo.py --input data/ipss_single_year.csv --out out/lexis.png \
  --mode lexis --cohort-birth-years 1947,1971
```

### 主なオプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--years` | 描画する年（カンマ区切り） | - |
| `--start-year` | 開始年 | データの最初の年 |
| `--end-year` | 終了年 | データの最後の年 |
| `--year-step` | 年の刻み | 1（毎年） |
| `--cohort-birth-years` | 追跡する出生年（カンマ区切り） | - |
| `--max-age` | 最大年齢 | 100 |
| `--fps` | アニメーションのFPS | 4 |

## データソース

- [国立社会保障・人口問題研究所 将来人口推計](https://www.ipss.go.jp/pp-zenkoku/j/zenkoku2023/pp_zenkoku2023.asp)
- Table 1-9: 各歳人口（2020-2070年、51年分）

## ディレクトリ構成

```
jp-pop-pyramid/
├── CLAUDE.md          # AI運用憲法
├── SSOT.md            # 索引
├── TASKS.md           # タスクボード
├── progress.md        # AI短期メモリ
├── plan.md            # 計画ポインタ
├── docs/
│   ├── requirements/  # 要件定義
│   ├── adr/           # アーキテクチャ決定記録
│   └── plans/         # 計画書
├── scripts/
│   └── validate-ssot.py
├── src/
│   ├── make_memo.py   # メインスクリプト
│   └── convert_ipss.py # データ変換
├── data/              # 入力データ
└── out/               # 出力ファイル
```

## 開発

### SSOT検証

```bash
python scripts/validate-ssot.py
```

### Claude Codeでの開発

```bash
claude
/kickoff
```

## ライセンス

MIT

## 作者

e-230129
