#!/usr/bin/env python3
"""Frissíti a README.md Verziószám: blokkját (v1.N + dátum + SHA; a commit link külön sorban)."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
# Egy soros (régi), kétsoros, vagy üres sorral elválasztott verzió + commit URL blokk
VERN = re.compile(r"^Verziószám:.+$(?:\n\n?https://github\.com/\S+)?", re.MULTILINE)


def run_git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return (r.stdout or "").strip()


def github_repo_slug() -> str:
    gr = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if gr:
        return gr
    try:
        url = run_git("config", "--get", "remote.origin.url")
    except subprocess.CalledProcessError as e:
        raise SystemExit("Nincs git remote (origin); állítsd GITHUB_REPOSITORY-t CI-ban.") from e
    u = url.replace(".git", "").rstrip("/")
    if "github.com:" in u:
        return u.split("github.com:", 1)[-1]
    if "github.com/" in u:
        return u.split("github.com/", 1)[-1]
    raise SystemExit(f"Nem sikerült GitHub owner/repo kiolvasni: {url!r}")


def main() -> int:
    sha = (os.environ.get("GITHUB_SHA") or "").strip() or run_git("rev-parse", "HEAD")
    count = run_git("rev-list", "--count", sha)
    date = run_git("log", "-1", "--format=%ci", sha)
    short = run_git("rev-parse", "--short", sha)
    full = run_git("rev-parse", sha)
    slug = github_repo_slug()
    link = f"https://github.com/{slug}/commit/{full}"
    new_block = f"Verziószám: v1.{count} ({date}; {short})\n\n{link}"

    text = README.read_text(encoding="utf-8")
    if VERN.search(text):
        nt = VERN.sub(new_block, text)
    else:
        nt = text.rstrip() + "\n\n" + new_block + "\n"

    if nt == text:
        print("README Verziószám blokk már naprakész.", file=sys.stderr)
        return 0
    README.write_text(nt, encoding="utf-8", newline="\n")
    print(new_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
