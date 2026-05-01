"""GitHub repó azonosító a `git remote origin` URL-ből (nincs beégetett owner/repo név)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _git_remote_origin_url(*, cwd: Path | None = None) -> str | None:
    root = cwd or ROOT
    try:
        r = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    s = (r.stdout or "").strip()
    return s or None


def github_slug_from_remote_url(url: str) -> str | None:
    u = url.replace(".git", "").rstrip("/")
    if "github.com:" in u:
        return u.split("github.com:", 1)[-1]
    if "github.com/" in u:
        return u.split("github.com/", 1)[-1]
    return None


def github_repo_slug(*, cwd: Path | None = None) -> str | None:
    """`owner/repo` vagy None (nincs origin / nem GitHub URL)."""
    gr = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if gr:
        return gr
    url = _git_remote_origin_url(cwd=cwd)
    if not url:
        return None
    return github_slug_from_remote_url(url)


def github_repo_root_url(*, cwd: Path | None = None) -> str | None:
    """`https://github.com/owner/repo` vagy None."""
    slug = github_repo_slug(cwd=cwd)
    if not slug:
        return None
    return f"https://github.com/{slug}"


def github_repo_slug_or_exit(*, cwd: Path | None = None) -> str:
    s = github_repo_slug(cwd=cwd)
    if not s:
        raise SystemExit(
            "Nem található GitHub repó: állítsd a GITHUB_REPOSITORY környezeti változót (CI), "
            "vagy add hozzá a `git remote add origin https://github.com/<owner>/<repo>.git` URL-t."
        )
    return s
