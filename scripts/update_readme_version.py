#!/usr/bin/env python3
"""README.md végén a Verziószám: blokk — GitHub URL a repo_github modulból (origin / GITHUB_REPOSITORY)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import repo_github as rg  # noqa: E402


def run_git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return (r.stdout or "").strip()


def replace_readme_version_footer(text: str, new_footer: str) -> str:
    key = "Verziószám:"
    i = text.rfind(key)
    if i == -1:
        return text.rstrip() + "\n\n" + new_footer + "\n"
    return text[:i].rstrip() + "\n\n" + new_footer + "\n"


def main() -> int:
    sha = (os.environ.get("GITHUB_SHA") or "").strip() or run_git("rev-parse", "HEAD")
    count = run_git("rev-list", "--count", sha)
    date = run_git("log", "-1", "--format=%ci", sha)
    short = run_git("rev-parse", "--short", sha)

    slug = rg.github_repo_slug_or_exit(cwd=ROOT)
    root = rg.github_repo_root_url(cwd=ROOT)
    assert root is not None
    new_block = f"Verziószám: v1.{count} ({date}; {short})\n\n**GitHub:** [{slug}]({root})\n"

    text = README.read_text(encoding="utf-8")
    nt = replace_readme_version_footer(text, new_block)

    if nt == text:
        print("README Verziószám blokk már naprakész.", file=sys.stderr)
        return 0
    README.write_text(nt, encoding="utf-8", newline="\n")
    print(new_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
