from __future__ import annotations

"""
GPT5.2pro レビュー用パッケージングスクリプト

プロジェクトをレビュー用にzipファイルにまとめる。
複数のソースディレクトリを指定可能。
不要なファイル（キャッシュ、出力ファイル、バイナリ等）は除外される。

含まれるファイル:
- src/, scripts/, tests/ - Pythonソースコード
- jp-pop-pyramid/ - サブプロジェクト全体
- data/*.yaml, data/*.csv - データ定義ファイル（画像は除外）
- docs/ - ドキュメント
- README.md, *.md (ルート) - プロジェクト説明
"""

import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path

# レビューに不要なディレクトリ
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".vite",
    ".cache",
    ".pytest_cache",
    ".claude",
    "node_modules",
    "dist",
    "out",  # 生成された出力ファイル
    "upload",  # 生成されたzipファイル
    "raw",  # 生データ（バイナリが多い）
}

# パターンで除外するディレクトリ（前方一致）
DEFAULT_EXCLUDE_DIR_PREFIXES = ["fix_files_"]

# レビューに不要なファイル拡張子
DEFAULT_EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".gif",
    ".png",
    ".jpg",
    ".jpeg",
    ".xls",
    ".xlsx",
    ".pdf",
    ".zip",
    ".html",  # 生成されたHTML
}


def should_skip(
    path: Path,
    exclude_dirs: set[str],
    exclude_extensions: set[str],
    exclude_prefixes: list[str] | None = None,
) -> bool:
    """ファイルをスキップすべきか判定"""
    # ディレクトリ名チェック（完全一致）
    parts = set(path.parts)
    if parts & exclude_dirs:
        return True
    # ディレクトリ名チェック（前方一致）
    if exclude_prefixes:
        for part in path.parts:
            for prefix in exclude_prefixes:
                if part.startswith(prefix):
                    return True
    # 拡張子チェック
    if path.suffix.lower() in exclude_extensions:
        return True
    return False


def zip_folders(
    repo_root: Path,
    source_dirs: list[Path],
    output_zip: Path,
    exclude_dirs: set[str],
    exclude_extensions: set[str],
    exclude_prefixes: list[str] | None = None,
) -> list[str]:
    """複数フォルダをzipにまとめる。含まれたファイルリストを返す"""
    if output_zip.exists():
        output_zip.unlink()

    # ZIPの最小有効タイムスタンプ (1980-01-01)
    min_date_time = (1980, 1, 1, 0, 0, 0)
    included_files = []

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_dir in source_dirs:
            for file_path in source_dir.rglob("*"):
                if file_path.is_dir():
                    continue
                if should_skip(file_path, exclude_dirs, exclude_extensions, exclude_prefixes):
                    continue

                # リポジトリルートからの相対パスを使用
                rel_path = file_path.relative_to(repo_root)
                # ZipInfoを作成して1980年以前のタイムスタンプに対応
                zinfo = zipfile.ZipInfo(rel_path.as_posix())
                mtime = file_path.stat().st_mtime
                dt = datetime.fromtimestamp(mtime)
                date_time = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                zinfo.date_time = date_time if date_time >= min_date_time else min_date_time
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zinfo, file_path.read_bytes())
                included_files.append(rel_path.as_posix())

    return included_files


# デフォルトでパッケージするディレクトリ
DEFAULT_SOURCE_DIRS = ["jp-pop-pyramid", "src", "scripts", "tests", "docs", "data"]

# ルートレベルで含めるファイルパターン
ROOT_INCLUDE_PATTERNS = ["*.md", "*.yaml", "*.yml", "*.json", ".gitignore"]


def collect_root_files(repo_root: Path, patterns: list[str]) -> list[Path]:
    """ルートレベルのファイルを収集（ディレクトリは除く）"""
    files = []
    for pattern in patterns:
        for file_path in repo_root.glob(pattern):
            if file_path.is_file():
                files.append(file_path)
    return files


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join("upload", f"project_review_{timestamp}.zip")

    parser = argparse.ArgumentParser(
        description="GPT5.2pro レビュー用にプロジェクトをパッケージング"
    )
    parser.add_argument(
        "--source",
        nargs="*",
        default=None,
        help=f"パッケージするソースディレクトリ（複数指定可） (default: {', '.join(DEFAULT_SOURCE_DIRS)})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="出力zipパス (default: upload/project_review_YYYYMMDD_HHMMSS.zip)",
    )
    parser.add_argument(
        "--exclude-dirs",
        default=",".join(sorted(DEFAULT_EXCLUDE_DIRS)),
        help="除外するディレクトリ名（カンマ区切り）",
    )
    parser.add_argument(
        "--exclude-ext",
        default=",".join(sorted(DEFAULT_EXCLUDE_EXTENSIONS)),
        help="除外する拡張子（カンマ区切り）",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="全てのファイルを含める（除外なし）",
    )
    parser.add_argument(
        "--include-fix-files",
        action="store_true",
        help="最新のfix_files_*ディレクトリを含める",
    )
    parser.add_argument(
        "--no-root-files",
        action="store_true",
        help="ルートレベルのファイル（README.md等）を含めない",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_files",
        help="含まれるファイル一覧を表示",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # ソースディレクトリのリストを構築
    source_names = args.source if args.source else DEFAULT_SOURCE_DIRS
    source_dirs: list[Path] = []
    for name in source_names:
        source_dir = (repo_root / name).resolve()
        if source_dir.exists():
            source_dirs.append(source_dir)
        else:
            print(f"警告: ディレクトリが見つかりません（スキップ）: {source_dir}")

    # 最新のfix_files_*を含める場合
    if args.include_fix_files:
        fix_dirs = sorted(repo_root.glob("fix_files_*"))
        if fix_dirs:
            latest_fix = fix_dirs[-1]
            source_dirs.append(latest_fix)
            print(f"最新の作業ファイルを含める: {latest_fix.name}")

    if not source_dirs:
        raise SystemExit("エラー: パッケージするディレクトリが1つもありません")

    output_zip = (repo_root / args.output).resolve()

    if args.include_all:
        exclude_dirs: set[str] = set()
        exclude_extensions: set[str] = set()
        exclude_prefixes: list[str] = []
    else:
        exclude_dirs = set(filter(None, (name.strip() for name in args.exclude_dirs.split(","))))
        exclude_extensions = set(
            filter(None, (ext.strip() for ext in args.exclude_ext.split(",")))
        )
        # --include-fix-files の場合はfix_files_*を除外しない
        exclude_prefixes = [] if args.include_fix_files else DEFAULT_EXCLUDE_DIR_PREFIXES

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    included_files = zip_folders(
        repo_root, source_dirs, output_zip, exclude_dirs, exclude_extensions, exclude_prefixes
    )

    # ルートレベルのファイルを追加
    root_files_added = []
    if not args.no_root_files:
        root_files = collect_root_files(repo_root, ROOT_INCLUDE_PATTERNS)
        # ZIPの最小有効タイムスタンプ (1980-01-01)
        min_date_time = (1980, 1, 1, 0, 0, 0)
        with zipfile.ZipFile(output_zip, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in root_files:
                rel_path = file_path.relative_to(repo_root)
                # 既に含まれていなければ追加
                if rel_path.as_posix() not in included_files:
                    zinfo = zipfile.ZipInfo(rel_path.as_posix())
                    mtime = file_path.stat().st_mtime
                    dt = datetime.fromtimestamp(mtime)
                    date_time = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                    zinfo.date_time = date_time if date_time >= min_date_time else min_date_time
                    zinfo.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zinfo, file_path.read_bytes())
                    root_files_added.append(rel_path.as_posix())
                    included_files.append(rel_path.as_posix())

    print(f"作成完了: {output_zip}")
    print(f"ソース: {', '.join(d.name for d in source_dirs)}")
    if root_files_added:
        print(f"ルートファイル: {', '.join(root_files_added)}")
    print(f"ファイル数: {len(included_files)}")
    print(f"サイズ: {output_zip.stat().st_size / 1024:.1f} KB")

    if args.list_files:
        print("\n含まれるファイル:")
        for f in sorted(included_files):
            print(f"  {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
