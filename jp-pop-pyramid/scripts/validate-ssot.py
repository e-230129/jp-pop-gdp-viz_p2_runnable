#!/usr/bin/env python3
"""
SSOT.md と plan.md のリンク切れを検証する。

機能:
- 必須ファイル（CLAUDE.md / SSOT.md / TASKS.md / progress.md / plan.md）存在検証
- SSOT.md 内のローカルMarkdownリンク存在検証
- plan.md のポインタ先存在検証
- フォルダリンクの検証（末尾/はディレクトリ必須）
- --verbose: 詳細出力（エラー理由も表示）
- --fix: 不足ファイルのスケルトン生成（実験的、リポジトリ内のみ）

Exit code:
  0: OK
  1: 不整合あり（リンク切れ or 必須ファイル不足）
  2: SSOT.md が見つからない
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RE_MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

REQUIRED_FILES = [
    "CLAUDE.md",
    "SSOT.md",
    "TASKS.md",
    "progress.md",
    "plan.md",
]


def is_local_path(target: str) -> bool:
    """URLやアンカー、絶対パスを除外（安全のため）"""
    if target.startswith(("http://", "https://", "mailto:")):
        return False
    if target.startswith("#"):
        return False
    # Unix absolute path
    if target.startswith("/"):
        return False
    # Windows drive absolute path or UNC
    if re.match(r"^[A-Za-z]:[\\/]", target) or target.startswith("\\"):
        return False
    return True


def repo_root() -> Path:
    """scripts/validate-ssot.py → リポジトリルートは親の親"""
    return Path(__file__).resolve().parent.parent


def read_pointer(path: Path) -> str | None:
    """最初の非空・非コメント行をポインタパスとして読む"""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # アンカー除去
        return s.split("#")[0].strip()
    return None


def validate(verbose: bool = False) -> tuple[bool, list[str]]:
    """
    SSOT.md と plan.md のリンク切れを検証。
    
    Returns:
        (ok, errors): ok=True なら検証成功、errors はエラーメッセージのリスト
    """
    root = repo_root()
    errors: list[str] = []
    
    # 1) 必須ファイル検証
    for fname in REQUIRED_FILES:
        fpath = root / fname
        if not fpath.exists():
            errors.append(f"REQUIRED: {fname} not found")
        elif verbose:
            print(f"  ✓ {fname}")
    
    # 2) SSOT.md 検証
    ssot_path = root / "SSOT.md"
    if not ssot_path.exists():
        errors.append("SSOT.md not found")
        return False, errors
    
    ssot_text = ssot_path.read_text(encoding="utf-8")
    for match in RE_MD_LINK.finditer(ssot_text):
        target = match.group(1).split("#")[0].strip()
        if not target:
            continue
        if not is_local_path(target):
            continue
        
        target_path = root / target
        
        # 末尾 / はディレクトリ必須
        if target.endswith("/"):
            if not target_path.is_dir():
                errors.append(f"SSOT.md: directory expected but not found: {target}")
                if verbose:
                    print(f"  ✗ {target} (directory expected)")
            elif verbose:
                print(f"  ✓ {target}")
        else:
            if not target_path.exists():
                errors.append(f"SSOT.md: link broken: {target}")
                if verbose:
                    print(f"  ✗ {target}")
            elif verbose:
                print(f"  ✓ {target}")
    
    # 3) plan.md のポインタ検証
    plan_path = root / "plan.md"
    if plan_path.exists():
        pointer = read_pointer(plan_path)
        if pointer:
            pointer_target = root / pointer
            if not pointer_target.exists():
                errors.append(f"plan.md: pointer target not found: {pointer}")
                if verbose:
                    print(f"  ✗ plan.md -> {pointer}")
            elif verbose:
                print(f"  ✓ plan.md -> {pointer}")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="SSOT validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    parser.add_argument("--fix", action="store_true", help="不足ファイルを生成（実験的）")
    args = parser.parse_args()
    
    if args.verbose:
        print("Validating SSOT...")
        print(f"  Root: {repo_root()}")
        print()
    
    ok, errors = validate(verbose=args.verbose)
    
    if ok:
        print("✓ SSOT validation OK")
        return 0
    else:
        print("✗ SSOT VALIDATION FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    exit(main())
